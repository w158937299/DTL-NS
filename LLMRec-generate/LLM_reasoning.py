from utils.parser import parse_args
from collections import defaultdict
import json
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import time
import os
from tqdm import tqdm
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def load_item_paths_from_txt(filepath):
    paths = {}
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item_str, path = line.split(":")
            except ValueError:
                continue
            paths[int(item_str.strip())] = path.strip()
    return paths

def build_user_prompt(user, train_ui, user_topk_items, co_paths, emb_paths):
    history = []
    for tid in train_ui[user]:
        history.append({
            "item_id": tid,
            "co_path": co_paths.get(tid, "None"),
            "emb_path": emb_paths.get(tid, "None")
        })

    candidates = []
    for cid in [user_topk_items[1]]:
        candidates.append({
            "item_id": cid,
            "co_path": co_paths.get(cid, "None"),
            "emb_path": emb_paths.get(cid, "None")
        })

    history_str = json.dumps(history, ensure_ascii=False)
    candidates_str = json.dumps(candidates, ensure_ascii=False)

    prompt = (
        "Below is the user's interaction history and the candidate items.\n"
        "Classify each candidate strictly according to the System Prompt.\n\n"
        "================ USER DATA BEGIN ================\n\n"
        "USER_HISTORY:\n"
        f"{history_str}\n\n"
        "CANDIDATE_ITEMS:\n"
        f"{candidates_str}\n\n"
        "================ USER DATA END =================\n\n"
        "Output only the JSON array of candidate items.\n"
    )
    return prompt

if __name__ == '__main__':
    args = parse_args()
    name = args.dataset
    user_topk_items = defaultdict(list)

    with open('data/prompt_classify.txt', "r", encoding="utf-8") as f:
        system_prompt = f.read()
    with open(f'./data/{name}/candidate_items.txt', 'r') as file:
        for line in file:
            parts = line.strip().split()
            uid = int(parts[0])
            for item in parts[1:]:
                user_topk_items[uid].append(int(item))

    test_ui = defaultdict(list)
    for filename in ["test.txt", "valid.txt"]:
        with open(f'./data/{name}/{filename}', 'r') as file:
            for line in file:
                u, it = line.strip().split()
                test_ui[int(u)].append(int(it))

    train_ui = defaultdict(list)
    with open(f'./data/{name}/train.txt', 'r') as file:
        for line in file:
            u, it = line.strip().split()
            uid = int(u)
            train_ui[uid].append(int(it))

    u_i_pais = []
    for user in user_topk_items:
        for item in user_topk_items[int(user)]:
            u_i_pais.append((int(user), int(item)))


    items = set()
    co_filepath = f'./data/{name}/{name}_struct_path_encoding.txt'
    embedding_filepath = f'./data/{name}/{name}_semantic_path_encoding.txt'


    co_paths = load_item_paths_from_txt(co_filepath)
    emb_paths = load_item_paths_from_txt(embedding_filepath)

    model_path = "/home/ai/huggingface/Meta-Llama-3.1-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, local_files_only=True)

    print("🚀 Loading model with vLLM...")
    llm = LLM(
        model=model_path,
        tensor_parallel_size=1,
        dtype="bfloat16",
        gpu_memory_utilization=0.85
    )
    print("🚀 Model loaded!")

    sampling_params = SamplingParams(
        max_tokens=2048,
        temperature=0.0,
        top_p=1.0,
        stop=["<|im_end|>"]
    )

    print("🔍 KV Cache Configuration:")
    print(llm.llm_engine.cache_config)

    batch_size = 1024
    file_path = "%s_results_classify.txt" % name
    f = open(file_path, 'w+')
    f.truncate()
    f.close()

    num_batches = (len(u_i_pais) + batch_size - 1) // batch_size
    start_batch = 0
    for batch_idx in tqdm(range(start_batch, num_batches), desc="Processing users"):

        t1 = time.time()
        i = batch_idx * batch_size
        batch_users = u_i_pais[i: i + batch_size]
        batch_prompts = []

        for user in batch_users:
            user_prompt = build_user_prompt(
                user[0],
                train_ui,
                user,
                co_paths,
                emb_paths
            )

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            prompt_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            batch_prompts.append(prompt_text)
        outputs = llm.generate(batch_prompts, sampling_params)
        t2 = time.time()
        with open(file_path, "a", encoding="utf-8") as f:
            for idx, output in enumerate(outputs):
                user_id = batch_users[idx]
                generated_text = output.outputs[0].text.strip().replace("\n", " ")
                f.write(f"{user_id}\t{generated_text}\n")


