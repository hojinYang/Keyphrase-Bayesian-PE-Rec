import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
import os
from scipy.sparse import csr_matrix

random_state = 20191120
np.random.seed(random_state)


def sample_n_tags(item_tag_dist, itemId, n):
    # num_total_item = item_tag_dist.shape[0]
    # num_total_tag = item_tag_dist.shape[1]

    prob = np.asarray(item_tag_dist[itemId]).squeeze()
    num_nonzero = len(prob.nonzero()[0])
    n = min(n, num_nonzero)
    if n == 0:
        return []

    num_total_tag = item_tag_dist.shape[1]
    tags = np.random.choice(num_total_tag, n, p=prob, replace=False).tolist()
    return tags


def sample_tags_for_user(item_tag_dist, itemIds, n):
    tags = list()
    items = list()
    for itemId in itemIds:
        t = sample_n_tags(item_tag_dist, itemId, n)
        tags += t
        items += len(t) * [itemId]
    return tags, items


def get_user_relevant_tags(r_ui, t_ui, item_tag_dist, n):
    user_rel_tags = []
    userIds = []
    itemIds = []
    num_user = r_ui.shape[0]

    for u in tqdm(range(num_user)):
        user_total_item = r_ui[u].nonzero()[1]
        user_tag_item = t_ui[u].nonzero()[1]
        items_without_tag = np.setdiff1d(user_total_item, user_tag_item)

        sampled_tags, items = sample_tags_for_user(item_tag_dist, items_without_tag, n)
        user_rel_tags += sampled_tags
        itemIds += items
        userIds += [u] * len(sampled_tags)
    return user_rel_tags, userIds, itemIds


def get_csr_matrix(df, rowname, colname, value=None, shape=None):
    row = df[rowname]
    col = df[colname]
    if value is None:
        value = [1] * len(row)
    return csr_matrix((value, (row, col)), shape=shape)


def main(args):
    ratings = pd.read_csv(os.path.join(args.data_dir, 'tr_ratings' + args.cv + '.csv'))
    tags = pd.read_csv(os.path.join(args.data_dir, 'tr_tags' + args.cv + '.csv'))

    if args.test:
        print("test!")
        ratings_val = pd.read_csv(os.path.join(args.data_dir, 'val_ratings' + args.cv + '.csv'))
        tags_val = pd.read_csv(os.path.join(args.data_dir, 'val_tags' + args.cv + '.csv'))
        ratings = ratings.append(ratings_val, ignore_index=True)
        tags = tags.append(tags_val, ignore_index=True)

    max_userId = max(list(tags['userId'].unique()) + list(ratings['userId'].unique()))
    max_itemId = max(list(tags['itemId'].unique()) + list(ratings['itemId'].unique()))
    r_ui = get_csr_matrix(ratings, 'userId', 'itemId', shape=(max_userId + 1, max_itemId + 1))
    t_ui = get_csr_matrix(tags, 'userId', 'itemId', shape=r_ui.shape)

    t_it = get_csr_matrix(tags, 'itemId', 'tagId', shape=(r_ui.shape[1], max(tags['tagId']) + 1)).todense()
    item_tag_dist = t_it / (t_it.sum(axis=1) + 1e-10)

    user_rel_tags, userIds, itemIds = get_user_relevant_tags(r_ui, t_ui, item_tag_dist, n=args.n)
    sampled_tags = pd.DataFrame({'userId': userIds, 'itemId': itemIds, 'tagId': user_rel_tags})

    tags_with_sample = tags.append(sampled_tags, ignore_index=True)
    if args.test:
        tags_with_sample.to_csv(os.path.join(args.data_dir, 'te_tags_s' + args.cv + '.csv'), index=False)
    else:
        tags_with_sample.to_csv(os.path.join(args.data_dir, 'tags_s' + args.cv + '.csv'), index=False)


if __name__ == "__main__":
    # Commandline arguments
    parser = argparse.ArgumentParser(description="Data Preprocess")
    parser.add_argument('--dir', dest='data_dir', default='data/lastfm')
    parser.add_argument('--cv_id', dest='cv', default='1')
    parser.add_argument('--test', dest='test', action='store_true')
    parser.add_argument('--num_sample', dest='n', type=int, default=5)
    args = parser.parse_args()

    main(args)
