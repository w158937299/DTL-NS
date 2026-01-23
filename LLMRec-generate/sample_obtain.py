import re
import json
import ast
from collections import defaultdict
from utils.parser import parse_args


def filter_leaky_data(raw_data):
    new_ui = defaultdict(list)
    for user, item in raw_data:
        new_ui[int(user)].append(int(item))
    filtered = []
    leaky_count = 0
    for user in new_ui:
        items = set(new_ui[int(user)])
        test_items = set(test_ui[int(user)])
        inter = items.intersection(test_items)

        leaky_count += len(inter)
        remain = items - inter
        filtered.extend((user, item) for item in remain)
    return filtered, leaky_count


def safe_load_json(s):
    s = s.strip()
    if not s.startswith("["):
        s = "[" + s
    if not s.endswith("]"):
        s = s + "]"
    s = re.sub(r'}\s*{', '},{', s)
    s = re.sub(r'\}\s*\{', '},{', s)
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        s2 = re.sub(r'([{,]\s*)([a-zA-Z_]\w*)(\s*:)', r'\1"\2"\3', s)
        return json.loads(s2)

pair_pattern = re.compile(
    r'"item_id"\s*:\s*(\d+)\s*,\s*"label"\s*:\s*"([^"]+)"'
)
def parse_line_with_regex(content):
    results = []
    for item_id, label in pair_pattern.findall(content):
        results.append({"item_id": int(item_id), "label": label})
    return results


def parse_file(path):
    result = {}
    users = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f.readlines()[:]:
            line = line.strip()
            if not line:
                continue
            try:
                user, js = line.split("\t", 1)
            except ValueError:
                print('异常行', line)
                continue

            try:
                items = safe_load_json(js)
            except:
                items = parse_line_with_regex(line)
            if len(items) == 0:
                users.append(int(user))

            user = ast.literal_eval(user)[0]
            pos_set = result.setdefault(int(user), set())
            for it in items:
                if not isinstance(it, dict):
                    continue
                item_id = it.get("item_id")
                label = it.get("label")
                if item_id is None or label is None:
                    continue
                if "positive" in label:
                    pos_set.add(int(item_id))
    return result, users


if __name__ == '__main__':
    args = parse_args()
    name = args.dataset
    user_topk_items = defaultdict(list)

    items = set()
    co_filepath = './data/%s/%s_struct_path_encoding.txt'%(name, name)
    embedding_filepath = './data/%s/%s_semantic_path_encoding.txt'%(name, name)
    enhance_filepath = './data/%s/train_enhance'%(name)

    classify_path = f"{name}_results_classify.txt"

    with open('./data/%s/candidate_items.txt'%(name), 'r') as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip().split()
            for item in line[1:]:
                user_topk_items[int(line[0])].append(int(item))

    test_ui = defaultdict(list)
    with open('./data/%s/test.txt' % (name), 'r') as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip().split()
            test_ui[int(line[0])].append(int(line[1]))
            items.add(int(line[1]))

    with open('./data/%s/valid.txt' % (name), 'r') as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip().split()
            test_ui[int(line[0])].append(int(line[1]))
            items.add(int(line[1]))

    train_ui = defaultdict(list)
    train_data = []
    with open('./data/%s/train.txt' % (name), 'r') as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip().split()
            train_ui[int(line[0])].append(int(line[1]))
            items.add(int(line[1]))
            train_data.append((int(line[0]), int(line[1])))

    new_data = []
    result, users = parse_file(classify_path)
    for user in result:
        raw_results = result[user]
        candidate_items = set(user_topk_items[int(user)])
        filtered = raw_results.intersection(candidate_items)
        for item in filtered:
            new_data.append((user, item))

    filtered, leaky_count = filter_leaky_data(new_data)
    print(len(new_data), len(filtered))
    print(len(train_data + filtered))
    with open(f"{enhance_filepath}_dtl_fni.txt", "w") as f:
        for user, item in train_data + filtered:
            f.write(f"{user}\t{item}\n")





