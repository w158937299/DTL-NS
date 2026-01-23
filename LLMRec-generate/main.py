import random
import datetime
import torch
import numpy as np
from collections import defaultdict, Counter
from time import time
from prettytable import PrettyTable
from utils.parser import parse_args
from utils.data_loader import load_data
from utils.evaluate import test
from utils.helper import early_stopping
import torch.nn.functional as F


def get_feed_dict(train_entity_pairs, train_pos_set, start, end, n_negs):

    def sampling(user_item, train_set, n):
        neg_items = []
        for user, _ in user_item.cpu().numpy():
            user = int(user)
            negitems = []
            for i in range(n):  # sample n times
                while True:
                    negitem = random.choice(range(n_items))
                    if negitem not in train_set[user]:
                        break
                negitems.append(negitem)
            neg_items.append(negitems)
        return neg_items

    feed_dict = {}
    entity_pairs = train_entity_pairs[start:end]
    feed_dict['users'] = entity_pairs[:, 0]
    feed_dict['pos_items'] = entity_pairs[:, 1]
    feed_dict['neg_items'] = torch.LongTensor(sampling(entity_pairs,
                                                       train_pos_set,
                                                       n_negs*K)).to(device)
    if args.ns == 'dtl':
        feed_dict['em_path'] = item_em_path1.to(device)
        feed_dict['co_path'] = item_co_path1.to(device)

    return feed_dict

def preprocess_tree_paths(path_dict):
    """
    返回一个矩阵 codes[item_id] = 编码向量
    """
    item_ids = list(path_dict.keys())
    max_id = max(item_ids)
    # 先确定最大深度
    max_len = 0
    for v in path_dict.values():
        nums = list(map(int, v.split('-')))
        max_len = max(max_len, len(nums))

    # 创建矩阵（全部填 -1）
    codes = torch.full((max_id + 1, max_len), -1, dtype=torch.long)

    # 逐项拷贝
    for item, v in path_dict.items():
        nums = list(map(int, v.split('-')))
        codes[item, :len(nums)] = torch.tensor(nums, dtype=torch.long)

    return codes  # shape = [num_items, max_depth]





if __name__ == '__main__':
    """fix the random seed"""
    seed = 2022
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    """read args"""
    global args, device
    args = parse_args()

    print(args.ps, args.ns, args.ps_param, args.alpha_co, args.alpha_se)

    if args.cuda and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu_id}")
    else:
        device = torch.device("cpu")

    """build dataset"""
    train_cf, user_dict, n_params, norm_mat = load_data(args)
    train_cf_size = len(train_cf)

    train_cf = torch.LongTensor(np.array([[cf[0], cf[1]] for cf in train_cf], np.int32))

    n_users = n_params['n_users']
    n_items = n_params['n_items']

    item_em_path, item_co_path = {}, {}
    if args.ns == 'dtl':
        with open(f'./data/{args.dataset}/{args.dataset}_semantic_path_encoding.txt', 'r') as file:
            lines = file.readlines()
            for line in lines:
                line = line.strip().split(": ")
                element = int(line[0])
                if element > n_users - 1:
                    item_em_path.setdefault(element - n_users, line[1])

        with open(f'./data/{args.dataset}/{args.dataset}_struct_path_encoding.txt', 'r') as file:
            lines = file.readlines()
            for line in lines:
                line = line.strip().split(": ")
                element = int(line[0])
                if element > n_users - 1:
                    item_co_path.setdefault(element - n_users, line[1])
        item_em_path1 = preprocess_tree_paths(item_em_path)
        item_co_path1 = preprocess_tree_paths(item_co_path)

    n_negs = args.n_negs
    K = args.K

    """define model"""
    from modules.LightGCN import LightGCN
    if args.gnn == 'lightgcn':
        model = LightGCN(n_params, args, norm_mat).to(device)

    """define optimizer"""
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    cur_best_pre_0 = 0
    stopping_step = 0
    should_stop = False

    path = './data/%s/%s_%s_%s_%.1f_%.1f_loss.txt'%(args.dataset, args.ps, args.ps_param, args.ns, args.alpha_co, args.alpha_se)
    f = open(path, 'w+')
    f.truncate()
    f.close()

    print("start training ...")
    for epoch in range(args.epoch):
        # shuffle training data
        train_cf_ = train_cf
        index = np.arange(len(train_cf_))
        np.random.shuffle(index)
        train_cf_ = train_cf_[index].to(device)

        """training"""
        model.train()
        loss, s = 0, 0
        hits = 0
        train_s_t = time()

        while s + args.batch_size <= len(train_cf):
            batch = get_feed_dict(train_cf_,
                                  user_dict['train_user_set'],
                                  s, s + args.batch_size,
                                  n_negs)

            batch_loss, _, _ = model(epoch, batch)
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            loss += batch_loss
            s += args.batch_size

        train_e_t = time()

        if epoch % 5 == 0:
            """testing"""

            train_res = PrettyTable()
            train_res.field_names = ["Epoch", "training time(s)", "tesing time(s)", "Loss", "recall", "ndcg", "precision", "hit_ratio"]

            model.eval()
            test_s_t = time()
            test_ret = test(model, user_dict, n_params, mode='test')
            test_e_t = time()
            train_res.add_row(
                [epoch, train_e_t - train_s_t, test_e_t - test_s_t, loss.item(), test_ret['recall'], test_ret['ndcg'],
                 test_ret['precision'], test_ret['hit_ratio']])

            if user_dict['valid_user_set'] is None:
                valid_ret = test_ret
            else:
                test_s_t = time()
                valid_ret = test(model, user_dict, n_params, mode='valid')
                test_e_t = time()
                train_res.add_row(
                    [epoch, train_e_t - train_s_t, test_e_t - test_s_t, loss.item(), valid_ret['recall'], valid_ret['ndcg'],
                     valid_ret['precision'], valid_ret['hit_ratio']])
            print(train_res)
            with open(path, 'a') as file:
                file.write(
                    str(datetime.datetime.now()) + '\t' + str(epoch).strip("'") + '\t' + str(loss.item()) + '\t' + str(
                        test_ret['recall']) + '\t' + str(test_ret['ndcg']) + '\t' + str(test_ret['precision']) + '\t' + str(test_ret['hit_ratio']) + '\n')



            # *********************************************************
            # early stopping when cur_best_pre_0 is decreasing for 10 successive steps.
            cur_best_pre_0, stopping_step, should_stop = early_stopping(valid_ret['recall'][2], cur_best_pre_0,
                                                                        stopping_step, expected_order='acc',
                                                                        flag_step=10)




            if should_stop:
                break

            """save weight"""
            if valid_ret['recall'][0] == cur_best_pre_0 and args.save:
                torch.save(model.state_dict(), args.out_dir + 'model_' + '.ckpt')

        else:
            pass


    print('early stopping at %d, recall@20:%.4f' % (epoch, cur_best_pre_0))


