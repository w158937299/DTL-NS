from utils.parser import parse_args
from collections import defaultdict
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix, diags
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans



def compute_sparse_jaccard(train_data):

    users = [u for u, i in train_data]
    items = [i for u, i in train_data]

    user_ids = sorted(set(users))
    item_ids = sorted(set(items))

    user2idx = {u: idx for idx, u in enumerate(user_ids)}
    item2idx = {i: idx for idx, i in enumerate(item_ids)}

    num_users = len(user_ids)
    num_items = len(item_ids)


    row = [user2idx[u] for u, i in train_data]
    col = [item2idx[i] for u, i in train_data]
    data = [1] * len(train_data)

    M = csr_matrix((data, (row, col)), shape=(num_users, num_items))


    C = (M.T @ M).tocoo()

    item_degree = np.array(M.sum(axis=0)).reshape(-1)

    I = C.row
    J = C.col
    inter = C.data
    union = item_degree[I] + item_degree[J] - inter

    jac = inter / union

    mask = (I != J)
    I = I[mask]
    J = J[mask]
    jac = jac[mask]

    W_sparse = coo_matrix((jac, (I, J)), shape=(num_items, num_items))

    return W_sparse, item_ids





def graph_laplacian_embedding(
    W_sparse,
    dim=32,
    normalized=True,
    tol=1e-4,
    maxiter=500
):


    if not isinstance(W_sparse, csr_matrix):
        A = W_sparse.tocsr()
    else:
        A = W_sparse

    A = (A + A.T) * 0.5

    degree = np.array(A.sum(axis=1)).reshape(-1)
    degree[degree == 0] = 1e-12
    N = A.shape[0]

    if normalized:
        # -------- L_sym = I - D^{-1/2} A D^{-1/2} --------
        d_sqrt_inv = 1.0 / np.sqrt(degree)
        D_sqrt_inv = diags(d_sqrt_inv)
        A_norm = D_sqrt_inv @ A @ D_sqrt_inv
        L = diags(np.ones(N)) - A_norm
    else:
        D = diags(degree)
        L = D - A

    k = min(dim + 1, N - 1)
    print(f"[Laplacian] Solving eigenvectors: N={N}, k={k}")
    evals, evecs = eigsh(
        L, k=k, which='SM', tol=tol, maxiter=maxiter
    )
    print("[Laplacian] Done.")

    embeddings = evecs[:, 1:dim+1]

    return embeddings


def kway_split_balance(X, indices, k=4):

    if len(indices) <= k:
        return [[i] for i in indices]

    km = KMeans(n_clusters=k, random_state=42)
    # km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = km.fit_predict(X[indices])

    clusters = []
    for c in range(k):
        sub = indices[labels == c]
        if len(sub) > 0:
            clusters.append(sub)


    avg_size = len(indices) / k
    max_size = int(avg_size * 1.6)
    min_size = int(avg_size * 0.4)

    for i in range(len(clusters)):
        while len(clusters[i]) > max_size:
            moved = clusters[i][-1]
            clusters[i] = clusters[i][:-1]

            sizes = [len(c) for c in clusters]
            tgt = np.argmin(sizes)
            clusters[tgt] = np.append(clusters[tgt], moved)

    return clusters


def build_bktree(
    X, item_ids,
    k=4,
    min_leaf_size=20,
    level=0,
    indices=None
):

    if indices is None:
        indices = np.arange(len(item_ids))

    if len(indices) <= min_leaf_size:
        return {
            "level": level,
            "indices": indices.tolist(),
            "children": []
        }

    # ---- k-way split ----
    subclusters = kway_split_balance(X, indices, k=k)
    sizes = [len(c) for c in subclusters]
    print(f"[BK-tree] Level {level}, split into {len(subclusters)} clusters, sizes={sizes}")

    children = []
    for sub in subclusters:
        child = build_bktree(
            X, item_ids,
            k=k,
            min_leaf_size=min_leaf_size,
            level=level+1,
            indices=sub
        )
        children.append(child)

    return {
        "level": level,
        "indices": indices.tolist(),
        "children": children
    }



def extract_bktree_paths(tree, item_ids):
    result = {}
    def dfs(node, path):
        children = node["children"]
        if not children:  # leaf
            for idx in node["indices"]:
                item_id = item_ids[idx]
                result[item_id] = "-".join(map(str, path))
            return

        for cid, child in enumerate(children):
            dfs(child, path + [cid])

    dfs(tree, [])
    return result


def load_vectors_txt(path):
    vectors = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            idx, vec = line.split(":")
            idx = int(idx)
            vec = np.array([float(x) for x in vec.strip().split()])
            vectors[idx] = vec
    return vectors


if __name__ == '__main__':
    args = parse_args()
    name = args.dataset
    train_data = []
    train_ui = defaultdict(list)
    with open('./data/%s/train.txt'%(name), 'r') as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip().split()
            train_data.append((int(line[0]), int(line[1])))
            train_ui[int(line[0])].append(int(line[1]))

    W_sparse, item_ids = compute_sparse_jaccard(train_data)
    struct_emb = graph_laplacian_embedding(
        W_sparse,
        dim=32,
        normalized=True
    )
    tree_struct = build_bktree(
        X=struct_emb,
        item_ids=item_ids,
        k=4,
        min_leaf_size=30
    )
    paths_struct = extract_bktree_paths(tree_struct, item_ids)
    with open("./data/%s/%s_struct_path_encoding.txt" % (name, name), "w") as f:
        for item_id, path in paths_struct.items():
            f.write(f"{item_id}: {path}\n")


    item_embeddings = load_vectors_txt('./data/%s/%s_item_vectors.txt'%(name, name))
    item_ids = list(item_embeddings.keys())
    item_emb = np.array([item_embeddings[i] for i in item_ids])
    semantic_tree = build_bktree(
        X=item_emb,
        item_ids=item_ids,
        k=4,
        min_leaf_size=30
    )
    semantic_paths = extract_bktree_paths(semantic_tree, item_ids)
    save_path = "./data/%s/%s_semantic_path_encoding.txt" % (name, name)
    with open(save_path, "w", encoding="utf-8") as f:
        for item_id, path in semantic_paths.items():
            f.write(f"{item_id}: {path}\n")














