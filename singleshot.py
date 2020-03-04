import numpy as np
import pandas as pd
import argparse
import time
import os
from utils.io import get_csr_matrix, get_test_df
from utils.modelnames import models
from utils.querynames import queries
from models.regressions import get_point_estimate
from experiment.simulator import singleshot_simulator


def check_int_positive(value):
    ivalue = int(value)
    if ivalue < 0:
         raise argparse.ArgumentTypeError("%s is an invalid positive int value" % value)
    return ivalue


def check_float_positive(value):
    ivalue = float(value)
    if ivalue < -1:
         raise argparse.ArgumentTypeError("%s is an invalid positive float value" % value)
    return ivalue

def main(args):
    print("run")
    print(args.k)
    if not os.path.exists(args.spath):
        os.makedirs(args.spath)

    output = pd.DataFrame()
    for f in args.fold:
        f= str(f)
        ratings_tr = pd.read_csv(os.path.join(args.dpath, 'tr_ratings'+f+'.csv'))
        
        if args.test:
            ratings_val = pd.read_csv(os.path.join(args.dpath, 'val_ratings'+f+'.csv'))
            ratings_tr = ratings_tr.append(ratings_val, ignore_index=True)
            tags_tr = pd.read_csv(os.path.join(args.dpath, 'te_tags_s'+f+'.csv'))
            tags_te = pd.read_csv(os.path.join(args.dpath, 'te_tags'+f+'.csv'))
        else:
            tags_tr = pd.read_csv(os.path.join(args.dpath, 'tags_s'+f+'.csv'))
            tags_te = pd.read_csv(os.path.join(args.dpath, 'val_tags'+f+'.csv'))
        
        r_ui = get_csr_matrix(ratings_tr, 'userId','itemId')
        t_ut = get_csr_matrix(tags_tr,'userId','tagId')
        t_it = get_csr_matrix(tags_tr,'itemId','tagId')

        test_df = get_test_df(ratings_tr,tags_tr,tags_te)

        del ratings_tr, ratings_val, tags_tr, tags_te

        U, It, _ = models['PureSVD'](r_ui, embeded_matrix=np.empty((0)),
                                            iteration=10, rank=args.rank,
                                            gpu_on=args.gpu, seed=args.seed)
        I = It.T
        #5,10
        t_ut[t_ut>5] = 5
        #
        Tt = get_point_estimate(X = U, Y = t_ut, lam=args.beta/args.sigma)
        T = Tt.T
        
        a_U, a_I, bias = models[args.model](r_ui, embeded_matrix=np.empty((0)),
                                            iteration=args.iter, rank=args.rank,
                                            corruption=args.corruption, gpu_on=args.gpu,
                                            lam=args.lamb, seed=args.seed, root=args.root)

        #singleshot_simulator(df_UIT, user_emb, item_emb, tag_emb, inf_user, inf_item, inf_bias, r_ui, t_it, k, threshold=5, gpu=False)
        result = singleshot_simulator(test_df, U, I, T, a_U, a_I.T, bias, r_ui, t_it, k=args.k)
        
        temp = {}
        temp['step'] = np.arange(1)
        temp['query'] = np.array([args.model]*1)
        for k in args.k:
            avg = np.mean(result[k],axis=0)
            temp['hr@'+str(k)]=avg

        output = output.append(pd.DataFrame(temp),ignore_index=True)
    output.to_csv(os.path.join(args.spath,args.sname),index=False)




if __name__ == "__main__":
    # Commandline arguments
    parser = argparse.ArgumentParser(description="p")

    parser.add_argument('--iter', dest='iter', type=check_int_positive, default=50)
    parser.add_argument('--lamb', dest='lamb', type=check_float_positive, default=0.0001)
    parser.add_argument('--sigma', dest='sigma', type=check_float_positive, default=5)
    parser.add_argument('--beta', dest='beta', type=check_float_positive, default=5)
    parser.add_argument('--model', dest='model', default='PureSVD')
    parser.add_argument('--rank', dest='rank', default=128)
    parser.add_argument('--dpath', dest='dpath', default="data/movielens20m")
    parser.add_argument('--fold', dest='fold', type=str, nargs= '+', default=[1,2,3,4,5])
    parser.add_argument('--test', dest='test', action='store_true')
    parser.add_argument('--k', dest='k', type=int, nargs= '+', default=[1,5,10,15,20])
    parser.add_argument('--gpu', dest='gpu', action='store_true')
    parser.add_argument('--corruption', dest='corruption', default=0.3)
    parser.add_argument('--seed', dest='seed', default=0)
    parser.add_argument('--alpha', dest='alpha', type=check_float_positive,default=0)
    parser.add_argument('--root', dest='root', default=0)
    parser.add_argument('--spath', dest='spath',type=str,default='table/movielens')
    parser.add_argument('--sname', dest='sname',type=str,default='singleshot.csv')
    args = parser.parse_args()

    main(args)

