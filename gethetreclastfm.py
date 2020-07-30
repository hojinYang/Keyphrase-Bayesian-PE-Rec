import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from utils.preprocessing import preprocess
from utils.kfold import get_kfold
import argparse
import os

random_state = 20191205
np.random.seed(random_state)

def main(args):
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    rpath = os.path.join(args.dir,args.rname)
    uatpath = os.path.join(args.dir,args.uatname) 
    tpath = os.path.join(args.dir,args.tname)
    apath = os.path.join(args.dir,args.aname)

    ratings = pd.read_csv(rpath, sep='\t')
    tags = pd.read_csv(uatpath, sep='\t')
    tags_ = pd.read_csv(tpath, sep='\t', encoding = "ISO-8859-1")
    artists = pd.read_csv(apath, sep='\t')
    
    ratings = ratings.rename(columns={'userID':'userId', 'artistID':'itemId','weight':'rating'})
    tags = pd.merge(tags, tags_, on='tagID')
    tags = tags.rename(columns={'userID':'userId', 'artistID':'itemId','tagValue':'tag'})[['userId','itemId','tag']]

    artists = artists.rename(columns={'id':'itemId'})


    ratings, tags, interactions, artists, tag_tagId = preprocess(ratings = ratings, tags = tags, \
        items = artists, tag_user_threshold = args.tag_user_threshold, tag_item_threshold = args.tag_item_threshold)

    kf = KFold(n_splits = args.k, shuffle = True, random_state = random_state)
    k = 1
    tag_tagId.to_csv(os.path.join(args.save_dir, 'tag_tagId.csv'), index=False)
    artists.sort_values(by='itemId').to_csv(os.path.join(args.save_dir, 'artists.csv'), index=False)
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
    parser.add_argument('--dir', dest='dir', default="hetrec-lastfm")
    parser.add_argument('--save', dest= 'save_dir', default='data/lastfm')
    parser.add_argument('--ua', dest='rname', default='user_artists.dat')
    parser.add_argument('--uat', dest='uatname', default='user_taggedartists.dat')
    parser.add_argument('--artist', dest='aname', default='artists.dat')
    parser.add_argument('--tag', dest='tname', default='tags.dat')
    parser.add_argument('--tu', dest='tag_user_threshold', default=3)
    parser.add_argument('--ti', dest='tag_item_threshold', default=3)
    parser.add_argument('--k', dest='k', default=5)
    parser.add_argument('--ratio', dest='val_ratio', default=0.3)
    # parser.add_argument('--rm', dest='minrating', default=0)
    args = parser.parse_args()

    main(args)
