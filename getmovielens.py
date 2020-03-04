import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from utils.preprocessing import preprocess
from utils.kfold import get_kfold
import argparse
import os

random_state = 20191109
np.random.seed(random_state)

def main(args):
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        
    rpath = os.path.join(args.dir,args.rname)
    tpath = os.path.join(args.dir,args.tname)
    mpath = os.path.join(args.dir,args.mname)
    ratings = pd.read_csv(rpath)
    tags = pd.read_csv(tpath)
    movies = pd.read_csv(mpath)
    
    ratings = ratings.rename(columns={'movieId':'itemId'})
    tags = tags.rename(columns={'movieId':'itemId'})
    tags['tag'] = tags['tag'].astype(str)
    tags = tags[tags.tag != 'BD-R']
    movies = movies.rename(columns={'movieId':'itemId','title':'name'})

    ratings, tags, interactions, movies, tag_tagId = preprocess(ratings = ratings, tags = tags, \
        items = movies, tag_user_threshold = args.tag_user_threshold, tag_item_threshold = args.tag_item_threshold)

    kf = KFold(n_splits = args.k, shuffle = True, random_state = random_state)
    k = 1
    tag_tagId.to_csv(os.path.join(args.save_dir, 'tag_tagId.csv'), index=False)
    movies.sort_values(by='itemId').to_csv(os.path.join(args.save_dir, 'movies.csv'), index=False)
    for train_index, test_index in kf.split(interactions):
        train, valid, test = \
            get_kfold(ratings, tags, interactions, train_index, test_index, args.val_ratio)
        
        train[0].sort_values(by='userId').to_csv(os.path.join(args.save_dir, 'tr_ratings'+str(k)+'.csv'), index=False)
        train[1].sort_values(by='userId').to_csv(os.path.join(args.save_dir, 'tr_tags'+str(k)+'.csv'), index=False)
        valid[0].sort_values(by='userId').to_csv(os.path.join(args.save_dir, 'val_ratings'+str(k)+'.csv'), index=False)
        valid[1].sort_values(by='userId').to_csv(os.path.join(args.save_dir, 'val_tags'+str(k)+'.csv'), index=False)
        test[0].sort_values(by='userId').to_csv(os.path.join(args.save_dir, 'te_ratings'+str(k)+'.csv'), index=False)
        test[1].sort_values(by='userId').to_csv(os.path.join(args.save_dir, 'te_tags'+str(k)+'.csv'), index=False)
        k+= 1
    

if __name__ == "__main__":
    # Commandline arguments
    parser = argparse.ArgumentParser(description="Data Preprocess")
    parser.add_argument('--dir', dest='dir', default="ml-latest/")
    parser.add_argument('--save', dest= 'save_dir', default='data/movielens')
    parser.add_argument('--rating', dest='rname', default='ratings.csv')
    parser.add_argument('--tag', dest='tname', default='tags.csv')
    parser.add_argument('--movie', dest='mname', default='movies.csv')
    parser.add_argument('--tu', dest='tag_user_threshold', default=10)
    parser.add_argument('--ti', dest='tag_item_threshold', default=5)
    parser.add_argument('--k', dest='k', default=5)
    parser.add_argument('--ratio', dest='val_ratio', default=0.3)
    # parser.add_argument('--rm', dest='minrating', default=0)
    args = parser.parse_args()

    main(args)
