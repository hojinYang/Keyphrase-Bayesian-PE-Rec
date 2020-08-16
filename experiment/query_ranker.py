import numpy as np

def rank_by_variance(pred_v, threshold=50, **unused):
    candidate_index = np.argpartition(-pred_v, threshold)[:threshold]
    rank = candidate_index[pred_v[candidate_index].argsort()[::-1]]
    return rank

def rank_by_mean(pred_m, threshold=50, **unused):
    candidate_index = np.argpartition(-pred_m, threshold)[:threshold]
    rank = candidate_index[pred_m[candidate_index].argsort()[::-1]]
    return rank

def rank_randomly(pred_v, threshold=50,**unused):
    num_tags = len(pred_v)
    tags = np.arange(num_tags)
    np.random.shuffle(tags)
    return tags[:threshold]

def rank_by_UCB(pred_v, pred_m, alpha=1, threshold=50, **unused):
    pred_v = np.sqrt(pred_v)

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
