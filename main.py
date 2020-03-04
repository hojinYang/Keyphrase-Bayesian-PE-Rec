import numpy as np
import pandas as pd
import argparse
import time
import os
from utils.io import get_csr_matrix, get_test_df
from utils.modelnames import models
from utils.querynames import queries
from models.regressions import get_point_estimate
from experiment.simulator import simulator


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
    if not os.path.exists(args.spath):
        os.makedirs(args.spath)
    
    if not os.path.exists(os.path.join(args.spath,args.sname)):
        output = pd.DataFrame({"fold":[],"step":[],"query":[]})
        output.to_csv(os.path.join(args.spath,args.sname),index=False)
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

        del ratings_tr, tags_tr, tags_te

        U, It, _ = models[args.model](r_ui, embeded_matrix=np.empty((0)),
                                            iteration=args.iter, rank=args.rank,
                                            corruption=args.corruption, gpu_on=args.gpu,
                                            lam=args.lamb, seed=args.seed, root=args.root)
        I = It.T
        #5,10
        #t_ut[t_ut>5] = 5
        t_ut[t_ut>3] = 3

        Tt = get_point_estimate(X = U, Y = t_ut, lam=args.lamb)
        T = Tt.T
        #U,I,T = [num_data,dim]
        for q in args.query:
            result = simulator(test_df, I, T, args.prec_W, args.prec_item, args.prec_tag, args.alpha,\
                r_ui, t_ut, t_it, query_ranker=queries[q], k=args.k, step=args.step)

            temp = {}
            temp['fold'] = np.array([f]*(args.step + 1))
            temp['step'] = np.arange(args.step + 1)
            temp['query'] = np.array([q]*(args.step + 1))
            for k in args.k:
                avg = np.mean(result[k],axis=0)
                temp['hr@'+str(k)]=avg

            output = pd.read_csv(os.path.join(args.spath,args.sname))
            output = output.append(pd.DataFrame(temp),ignore_index=True)
            output.to_csv(os.path.join(args.spath,args.sname),index=False)




if __name__ == "__main__":
    # Commandline arguments
    parser = argparse.ArgumentParser(description="p")

    parser.add_argument('--step', dest='step', type=check_int_positive, default=7, help='Num of steps for elicitation process')
    parser.add_argument('--iter', dest='iter', type=check_int_positive, default=10, help='fast SVD iteraton')
    parser.add_argument('--lamb', dest='lamb', type=check_float_positive, default=1, help='weight for regularization')
    parser.add_argument('--prec_W', dest='prec_W', type=check_float_positive, default=0.1, help='precision used in weight matrix')
    parser.add_argument('--prec_item', dest='prec_item', type=check_float_positive, default=0.001, help='precision used in item likelihood')
    parser.add_argument('--prec_tag', dest='prec_tag', type=check_float_positive, default=10, help='precision used in tag likelihood')
    parser.add_argument('--model', dest='model', default='PureSVD')
    parser.add_argument('--rank', dest='rank', default=128, help='hidden dim')
    parser.add_argument('--dpath', dest='dpath', default="data/movielens")
    parser.add_argument('--fold', dest='fold', type=str, nargs= '+', default=[1,2,3,4,5])
    parser.add_argument('--test', dest='test', action='store_true')
    parser.add_argument('--k', dest='k', type=int, nargs= '+', default=[1,5,10,15,20])
    parser.add_argument('--gpu', dest='gpu', action='store_true')
    parser.add_argument('--corruption', dest='corruption', default=0)
    parser.add_argument('--seed', dest='seed', default=0)
    parser.add_argument('--alpha', dest='alpha', type=check_float_positive,default=1, help='leverage on varince used in UCB')
    parser.add_argument('--root', dest='root', default=0)
    parser.add_argument('--query', dest='query',type=str, nargs= '+', default=['UCB','Var','Mean','Random'])
    parser.add_argument('--spath', dest='spath',type=str,default='table/movielens')
    parser.add_argument('--sname', dest='sname',type=str,default='main.csv')
    args = parser.parse_args()

    main(args)

