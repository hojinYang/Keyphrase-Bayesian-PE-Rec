import numpy as np

def rank_by_variance(pred_v, threshold=30, **unused):
    candidate_index = np.argpartition(-pred_v, threshold)[:threshold]
    rank = candidate_index[pred_v[candidate_index].argsort()[::-1]]
    return rank

def rank_by_mean(pred_m, threshold=30, **unused):
    candidate_index = np.argpartition(-pred_m, threshold)[:threshold]
    rank = candidate_index[pred_m[candidate_index].argsort()[::-1]]
    return rank

def rank_randomly(pred_v, threshold=30,**unused):
    num_tags = len(pred_v)
    tags = np.arange(num_tags)
    np.random.shuffle(tags)
    return tags[:threshold]

def rank_by_UCB(pred_v, pred_m, alpha=1, threshold=30, **unused):
    pred_v = np.sqrt(pred_v)
    #print(pred_m)

    #pred_v = (pred_v-min(pred_v))/(max(pred_v)-min(pred_v))
    #pred_m = (pred_m-min(pred_m))/(max(pred_m)-min(pred_m))

    score = alpha*pred_v + pred_m
    candidate_index = np.argpartition(-score, threshold)[:threshold]
    rank = candidate_index[score[candidate_index].argsort()[::-1]]

    return rank

def rank_by_TS(pred_v, pred_m, threshold=50, **unused):
    pred_v = np.sqrt(pred_v)

    #pred_v = ((max(pred_m)-min(pred_m))/(max(pred_v)-min(pred_v)))*pred_v

    score = np.random.normal(pred_m, pred_v)
    candidate_index = np.argpartition(-score, threshold)[:threshold]
    rank = candidate_index[score[candidate_index].argsort()[::-1]]
    return rank

def rank_by_POP(global_sum, threshold=50,**unused):
    score = global_sum
    candidate_index = np.argpartition(-score, threshold)[:threshold]
    rank = candidate_index[score[candidate_index].argsort()[::-1]]
    return rank

def rank_by_PPOP(s_ut, uid, threshold=50,**unused):
    score = np.asarray(s_ut[uid].todense()).squeeze()
    candidate_index = np.argpartition(-score, threshold)[:threshold]
    rank = candidate_index[score[candidate_index].argsort()[::-1]]
    return rank

from scipy.stats import norm
from models.regressions import get_predictive_dist, update_posterior,get_user_dist 
import numpy as np

def rank_by_EVOI(pred_v, pred_m, item_emb, tag_emb, tau, S, m, prec_tag, user_history,threshold=50,**unused):
    train_index = user_history.nonzero()[1]
    # calculate probability > or <
    ss = (3 - pred_m) / np.sqrt(pred_v)
    
    cdfs = norm.cdf(ss)
    #print(cdfs)

    '''
    c,r = tag_emb.shape
    X = tag_emb.reshape((c,r,1))

    Y = np.array([[tau]])

    _, new_m =  update_posterior(X, Y, S, m, prec_tag)
    new_m = new_m.reshape(50,1153)
    vector_predict = item_emb @ new_m

    vector_predict[train_index] = -99
    pos = np.max(vector_predict,axis=0)


    Y = np.array([[0]])

    _, new_m =  update_posterior(X, Y, S, m, prec_tag)
    new_m = new_m.reshape(50,1153)
    vector_predict = item_emb @ new_m

    vector_predict[train_index] = -99
    neg = np.max(vector_predict,axis=0)

    '''
    pos = []
    for query in range(len(tag_emb)):
        X = tag_emb[query].reshape((-1,1))
        Y = np.array([[tau]])

        _, new_m =  update_posterior(X, Y, S, m, prec_tag)

        vector_predict = item_emb.dot(new_m).flatten() 
        vector_predict[train_index] = -99
        maxval = np.max(vector_predict)
        pos.append(maxval)
    
    neg = []
    for query in range(len(tag_emb)):
        X = tag_emb[query].reshape((-1,1))
        Y = np.array([[0]])

        _, new_m =  update_posterior(X, Y, S, m, prec_tag)

        vector_predict = item_emb.dot(new_m).flatten() 
        vector_predict[train_index] = -99
        maxval = np.max(vector_predict)
        neg.append(maxval)
    

    score = (1-cdfs)*np.array(pos) + cdfs*np.array(neg)
    #score = np.array(pos) + np.array(neg)
    candidate_index = np.argpartition(-score, threshold)[:threshold]
    rank = candidate_index[score[candidate_index].argsort()[::-1]]
    
    return rank


def rank_by_EVOI2(pred_v, pred_m, item_emb, tag_emb, tau, S, m, prec_tag, user_history,threshold=50,**unused):
    train_index = user_history.nonzero()[1]
    # calculate probability > or <
    ss = (1 - pred_m) / np.sqrt(pred_v)
    
    cdfs = np.array([0.5]*len(pred_m))
    print(cdfs)

    '''
    c,r = tag_emb.shape
    X = tag_emb.reshape((c,r,1))

    Y = np.array([[tau]])

    _, new_m =  update_posterior(X, Y, S, m, prec_tag)
    new_m = new_m.reshape(50,1153)
    vector_predict = item_emb @ new_m

    vector_predict[train_index] = -99
    pos = np.max(vector_predict,axis=0)


    Y = np.array([[0]])

    _, new_m =  update_posterior(X, Y, S, m, prec_tag)
    new_m = new_m.reshape(50,1153)
    vector_predict = item_emb @ new_m

    vector_predict[train_index] = -99
    neg = np.max(vector_predict,axis=0)

    '''
    pos = []
    for query in range(len(tag_emb)):
        X = tag_emb[query].reshape((-1,1))
        Y = np.array([[tau]])

        _, new_m =  update_posterior(X, Y, S, m, prec_tag)

        vector_predict = item_emb.dot(new_m).flatten() 
        vector_predict[train_index] = -99
        maxval = np.max(vector_predict)
        pos.append(maxval)
    
    neg = []
    for query in range(len(tag_emb)):
        X = tag_emb[query].reshape((-1,1))
        Y = np.array([[0]])

        _, new_m =  update_posterior(X, Y, S, m, prec_tag)

        vector_predict = item_emb.dot(new_m).flatten() 
        vector_predict[train_index] = -99
        maxval = np.max(vector_predict)
        neg.append(maxval)
    

    score = np.array(pos) + np.array(neg)
    candidate_index = np.argpartition(-score, threshold)[:threshold]
    rank = candidate_index[score[candidate_index].argsort()[::-1]]
    
    return rank
