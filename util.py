import sys
import copy
import random
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Queue
from tqdm import tqdm


def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t


def computeRePos(time_seq, time_span):
    size = time_seq.shape[0]
    time_matrix = np.zeros([size, size], dtype=np.int32)
    for i in range(size):
        for j in range(size):
            span = abs(time_seq[i] - time_seq[j])
            if span > time_span:
                time_matrix[i][j] = time_span
            else:
                time_matrix[i][j] = span
    return time_matrix


def Relation(user_train, usernum, maxlen, time_span):
    data_train = dict()
    for user in tqdm(range(1, usernum + 1), desc='Preparing relation matrix'):
        time_seq = np.zeros([maxlen], dtype=np.int32)
        idx = maxlen - 1
        for i in reversed(user_train[user][:-1]):
            time_seq[idx] = i[1]
            idx -= 1
            if idx == -1: break
        data_train[user] = computeRePos(time_seq, time_span)
    return data_train

# def data_partition(fname):
#     usernum = 0
#     itemnum = 0
#     User = defaultdict(list)
#     user_train = {}
#     user_valid = {}
#     user_test = {}
#     user_total = {}
#     # assume user/item index starting from 1
#     f = open('data/%s.txt' % fname, 'r')
#     for line in f:
#         u, i = line.rstrip().split()
#         u = int(u)
#         i = int(i)
#         usernum = max(u, usernum)
#         itemnum = max(i, itemnum)
#         User[u].append(i)
#
#     for user in User:
#         nfeedback = len(User[user])
#         if nfeedback < 3:
#             user_total[user] = User[user]
#             user_train[user] = User[user]
#             user_valid[user] = []
#             user_test[user] = []
#         else:
#             user_total[user] = User[user]
#             user_train[user] = User[user][:-2]
#             user_valid[user] = []
#             user_valid[user].append(User[user][-2])
#             user_test[user] = []
#             user_test[user].append(User[user][-1])
#     return [user_total, user_train, user_valid, user_test, usernum, itemnum]

def timeSlice(time_set):
    time_min = min(time_set)
    time_map = dict()
    for time in time_set:
        time_map[time] = int(round(float(time - time_min)))
    return time_map


def cleanAndsort(User, time_map):
    User_filted = dict()
    user_set = set()
    item_set = set()
    for user, items in User.items():
        user_set.add(user)
        User_filted[user] = items
        for item in items:
            item_set.add(item[0])
    user_map = dict()
    item_map = dict()
    for u, user in enumerate(user_set):
        user_map[user] = u + 1
    for i, item in enumerate(item_set):
        item_map[item] = i + 1

    for user, items in User_filted.items():
        User_filted[user] = sorted(items, key=lambda x: x[1])

    User_res = dict()
    for user, items in User_filted.items():
        User_res[user_map[user]] = list(map(lambda x: [item_map[x[0]], time_map[x[1]]], items))

    time_max = set()
    for user, items in User_res.items():
        time_list = list(map(lambda x: x[1], items))
        time_diff = set()
        for i in range(len(time_list) - 1):
            if time_list[i + 1] - time_list[i] != 0:
                time_diff.add(time_list[i + 1] - time_list[i])
        if len(time_diff) == 0:
            time_scale = 1
        else:
            time_scale = min(time_diff)
        time_min = min(time_list)
        User_res[user] = list(map(lambda x: [x[0], int(round((x[1] - time_min) / time_scale) + 1)], items))
        time_max.add(max(set(map(lambda x: x[1], User_res[user]))))

    return User_res, len(user_set), len(item_set), max(time_max)

def data_partition(fname):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    user_total = {}

    print('Preparing data...')
    f = open('data/%s.txt' % fname, 'r')
    time_set = set()

    user_count = defaultdict(int)
    item_count = defaultdict(int)
    for line in f:
        try:
            u, i, ratings, timestamp = line.rstrip().split('\t')
        except:
            u, i, timestamp = line.rstrip().split('\t')
        u = int(u)
        i = int(i)
        user_count[u] += 1
        item_count[i] += 1
    f.close()
    f = open('data/%s.txt' % fname, 'r')
    for line in f:
        try:
            u, i, ratings, timestamp = line.rstrip().split('\t')
        except:
            u, i, timestamp = line.rstrip().split('\t')
        u = int(u)
        i = int(i)
        timestamp = float(timestamp)
        # if user_count[u] < 10 or item_count[i] < 5:
        if user_count[u] < 5:
            continue
        time_set.add(timestamp)
        User[u].append([i, timestamp])
    f.close()
    time_map = timeSlice(time_set)
    User, usernum, itemnum, timenum = cleanAndsort(User, time_map)

    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_total[user] = User[user]
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_total[user] = User[user]
            user_train[user] = User[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])
    print('Preparing done...')
    return [user_total, user_train, user_valid, user_test, usernum, itemnum, timenum]


# def evaluate(model, dataset, args, sess):
#     [total, train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)
#
#     NDCG = 0.0
#     NDCG_sparse = 0.0
#     HT = 0.0
#     HT_sparse = 0.0
#     valid_user = 0.0
#
#     if usernum>10000:
#         users = random.sample(range(1, usernum + 1), 10000)
#     else:
#         users = range(1, usernum + 1)### 注意user是从0开始的还是从1开始的
#
#     P = 0.0;
#     R = 0.0;
#     MAP = 0.0;
#     MRR = 0.0;
#     MRR_sparse = 0.0;
#     sparse_user = 0.0
#     for u in users:
#         if len(train[u]) < 1 or len(test[u]) < 1: continue
#
#         seq = np.zeros([args.maxlen], dtype=np.int32)
#         idx = args.maxlen - 1
#         seq[idx] = valid[u][0]
#         idx -= 1
#         for i in reversed(train[u]):
#             seq[idx] = i
#             idx -= 1
#             if idx == -1: break
#         rated = set(train[u])
#         rated.add(0) ### ？？？
#         item_idx = [test[u][0]]
#         for _ in range(100): # 100个负例
#             t = np.random.randint(1, itemnum + 1)
#             while t in rated: t = np.random.randint(1, itemnum + 1)
#             item_idx.append(t)
#
#         predictions = -model.predict(sess, [u], [seq], item_idx)
#         # print(predictions)
#         #
#         # print(np.shape(predictions))
#         predictions = predictions[0]
#
#         rank = predictions.argsort().argsort()[0] # pos item rank
#
#         valid_user += 1
#
#         if rank < 10:
#             NDCG += 1 / np.log2(rank + 2)
#             HT += 1
#             MRR += 1 / (rank + 1)
#         if len(train[u]) < 32:
#             sparse_user += 1
#             if rank < 10:
#                 NDCG_sparse += 1 / np.log2(rank + 2)
#                 HT_sparse += 1
#                 MRR_sparse += 1 / (rank + 1)
#
#
#         top_k = 10
#
#         # trueResult = [test[u][0]]
#         # predictions = -predictions
#         # total_pro = [(item_idx[i], predictions[i]) for i in range(101)]
#         # total_pro.sort(key=lambda x: x[1], reverse=True)
#         # rankedItem = [pair[0] for pair in total_pro]
#         #
#         # right_num = 0
#         # trueNum = len(trueResult)
#         # count = 0
#         # for j in rankedItem:
#         #     if count == top_k:
#         #         P += 1.0 * right_num / count
#         #         R += 1.0 * right_num / trueNum
#         #     count += 1
#         #     if j in trueResult:
#         #         right_num += 1
#         #         MAP = MAP + 1.0 * right_num / count
#         #         if right_num == 1:
#         #             MRR += 1.0 / count
#         # if right_num != 0:
#         #     MAP /= right_num
#
#         if valid_user % 100 == 0:
#             sys.stdout.flush()
#
#     return NDCG / valid_user, HT / valid_user, MRR / valid_user, \
#            NDCG_sparse / sparse_user, HT_sparse / sparse_user, MRR_sparse / sparse_user
#
#
# def evaluate_valid(model, dataset, args, sess):
#     [total, train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)
#
#     NDCG = 0.0
#     valid_user = 0.0
#     HT = 0.0
#     if usernum>10000:
#         users = random.sample(range(1, usernum + 1), 10000)
#     else:
#         users = range(1, usernum + 1)
#
#     P = 0.0; R = 0.0; MAP = 0.0; MRR = 0.0;
#     for u in users:
#         # if len(train[u]) < 1 or len(valid[u]) < 1: continue
#
#         seq = np.zeros([args.maxlen], dtype=np.int32)
#         idx = args.maxlen - 1
#         for i in reversed(train[u]):
#             seq[idx] = i
#             idx -= 1
#             if idx == -1: break
#
#         rated = set(train[u])
#         rated.add(0)
#         item_idx = [valid[u][0]]
#         for _ in range(100):
#             t = np.random.randint(1, itemnum + 1)
#             while t in rated: t = np.random.randint(1, itemnum + 1)
#             item_idx.append(t)
#
#         predictions = -model.predict(sess, [u], [seq], item_idx)
#         predictions = predictions[0]
#
#         rank = predictions.argsort().argsort()[0]
#
#         valid_user += 1
#
#         if rank < 10:
#             NDCG += 1 / np.log2(rank + 2)
#             HT += 1
#
#         top_k = 10
#
#         trueResult = [valid[u][0]]
#         predictions = -predictions
#         total_pro = [(item_idx[i], predictions[i]) for i in range(101)]
#         total_pro.sort(key=lambda x: x[1], reverse=True)
#         rankedItem = [pair[0] for pair in total_pro]
#
#         right_num = 0
#         trueNum = len(trueResult)
#         count = 0
#         for j in rankedItem:
#             if count == top_k:
#                 P += 1.0 * right_num / count
#                 R += 1.0 * right_num / trueNum
#             count += 1
#             if j in trueResult:
#                 right_num += 1
#                 MAP = MAP + 1.0 * right_num / count
#                 if right_num == 1:
#                     MRR += 1.0 / count
#         if right_num != 0:
#             MAP /= right_num
#
#         if valid_user % 100 == 0:
#             sys.stdout.flush()
#
#     return NDCG / valid_user, HT / valid_user, R / valid_user, MRR / valid_user

def evaluate(model, dataset, args, sess):
    [total, train, valid, test, usernum, itemnum, timenum] = copy.deepcopy(dataset)

    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0

    if usernum > 10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:

        if len(train[u]) < 1 or len(test[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        time_seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1

        seq[idx] = valid[u][0][0]
        time_seq[idx] = valid[u][0][1]
        idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i[0]
            time_seq[idx] = i[1]
            idx -= 1
            if idx == -1: break
        rated = set(map(lambda x: x[0], train[u]))
        rated.add(valid[u][0][0])
        rated.add(test[u][0][0])
        rated.add(0)
        item_idx = [test[u][0][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        time_matrix = computeRePos(time_seq, args.time_span)

        predictions = -model.predict(sess, [u], [seq], [time_matrix], item_idx)
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0]

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print('.', end='')
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user


def evaluate_valid(model, dataset, args, sess):
    [total, train, valid, test, usernum, itemnum, timenum] = copy.deepcopy(dataset)

    NDCG = 0.0
    valid_user = 0.0
    HT = 0.0
    if usernum > 10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:
        if len(train[u]) < 1 or len(valid[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        time_seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i in reversed(train[u]):
            seq[idx] = i[0]
            time_seq[idx] = i[1]
            idx -= 1
            if idx == -1: break

        rated = set(map(lambda x: x[0], train[u]))
        rated.add(valid[u][0][0])
        rated.add(0)
        item_idx = [valid[u][0][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        time_matrix = computeRePos(time_seq, args.time_span)
        predictions = -model.predict(sess, [u], [seq], [time_matrix], item_idx)
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0]

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print('.', end='')
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user