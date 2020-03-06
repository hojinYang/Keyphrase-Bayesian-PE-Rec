import numpy as np
from tqdm import tqdm 
from sklearn.metrics.pairwise import cosine_similarity
from models.regressions import get_predictive_dist, update_posterior,get_user_dist 

def simulator(df_UIT, item_emb, tag_emb, prec_W, prec_item, prec_tag, alpha, r_ui, s_ut, t_it, sim_type, k,step, query_ranker, threshold=5, gpu=False):
    #3, 5
    result = dict()
    for n in k:
        result[n] = []
        
    for uit in tqdm(df_UIT.itertuples()):
        res = user_simulator(uit, item_emb, tag_emb, prec_W, prec_item, prec_tag, alpha, r_ui, s_ut, t_it, sim_type, k,step,query_ranker, threshold, gpu)
        if res is not None:
            for n in k:
                result[n].append(res[n])
    return result

def user_simulator(uit, item_emb, tag_emb, prec_W, prec_item, prec_tag, alpha, r_ui, s_ut, t_it, sim_type, k,step,query_ranker, threshold, gpu=False):
    result = dict()
    for n in k:
        result[n] = []
    redun_queries = []
    item_tags = t_it[uit.itemId].nonzero()[1]
    if sim_type == 'subset':
        relevant_tags = get_similar_tags(uit.tagIds, tag_emb)
    elif sim_type == 'all':
        relevant_tags = item_tags

    if len(relevant_tags) <threshold:
        return None
    
    #banned_queries = np.setdiff1d(item_tags, relevant_tags).tolist()
    #banned_queries = []
    #redun_queries += banned_queries
    redun_queries = []
    vector_train = r_ui[uit.userId]
    
    #max_val = float(s_ut[uit.userId].max(axis=1).todense())
    S, m = get_user_dist(uit.userId, r_ui, item_emb, prec_W=prec_W, prec_y=prec_item)

    pred_v, pred_m = get_predictive_dist(tag_emb.T, S, m, prec_tag)
    max_val = 3
    #max_val = np.max(pred_m)
    #print(np.max(pred_m))    
    #m = np.mean(user_emb,axis=0).reshape((-1,1))
    vector_predict = sub_routine(m, item_emb, vector_train, max(k), gpu=gpu)
    for n in k:
        result[n].append(hr_k(vector_predict,uit.itemId, n))
    for _ in range(step):
        pred_v, pred_m = get_predictive_dist(tag_emb.T, S, m, prec_tag)

        query_rank = query_ranker(pred_v=pred_v, pred_m=pred_m, alpha=alpha)
        query = select_single_question(query_rank, redun_queries)
        redun_queries.append(query)

        user_response = get_user_reponse(query, relevant_tags)
        X = tag_emb[query].reshape((-1,1))
        Y = np.array([[user_response*max_val]])

        S, m =  update_posterior(X, Y, S, m, prec_tag)

        vector_predict = sub_routine(m, item_emb, vector_train, max(k), gpu=gpu)
        for n in k:
            result[n].append(hr_k(vector_predict,uit.itemId, n))

    return result

def select_single_question(rank, redun_tags):
    rank = np.delete(rank, np.isin(rank, redun_tags).nonzero()[0])
    return rank[0]

def get_user_reponse(query_tags, relevant_tags):
    return np.isin(query_tags, relevant_tags)

def get_similar_tags(given_tags, tag_emb, sim=0.95):
    #sim 0.90
    sim_matrix = cosine_similarity(tag_emb[given_tags], tag_emb)
    similar_tags = np.unique((sim_matrix >= sim).nonzero()[1])
    return similar_tags

def sub_routine(vector_u, item_emb, vector_train, topK, gpu=False):
    train_index = vector_train.nonzero()[1]
    vector_predict = item_emb.dot(vector_u).flatten()

    if gpu:
        import cupy as cp
        candidate_index = cp.argpartition(-vector_predict, topK+len(train_index))[:topK+len(train_index)]
        vector_predict = candidate_index[vector_predict[candidate_index].argsort()[::-1]]
        vector_predict = cp.asnumpy(vector_predict).astype(np.float32)
    else:
        candidate_index = np.argpartition(-vector_predict, topK+len(train_index))[:topK+len(train_index)]
        vector_predict = candidate_index[vector_predict[candidate_index].argsort()[::-1]]
    vector_predict = np.delete(vector_predict, np.isin(vector_predict, train_index).nonzero()[0])
    return vector_predict

def hr_k(preds,target, k):
    if target in set(preds[:k]):
        return 1
    else:
        return 0

def singleshot_simulator(df_UIT, user_emb, item_emb, tag_emb, inf_user, inf_item, inf_bias, r_ui, t_it, k, threshold=5, gpu=False):
    #3, 5
    result = dict()
    for n in k:
        result[n] = []
    
    for uit in tqdm(df_UIT.itertuples()):
        res = singleshot_user_simulator(uit, user_emb, item_emb, tag_emb, inf_user, inf_item, inf_bias, r_ui, t_it, k, threshold, gpu)
        if res is not None:
            for n in k:
                result[n].append(res[n])
    print(len(result[1]))
    return result

def singleshot_user_simulator(uit, user_emb, item_emb, tag_emb, inf_user, inf_item, inf_bias, r_ui, t_it, k, threshold, gpu=False):
    result = dict()
    for n in k:
        result[n] = []

    item_tags = t_it[uit.itemId].nonzero()[1]
    relevant_tags = get_similar_tags(uit.tagIds, tag_emb)
    #relevant_tags = item_tags

    if len(relevant_tags) <threshold:
        return None
    
    vector_train = r_ui[uit.userId]
    
    m = inf_user[uit.userId].reshape((-1,1))

    vector_predict = singleshot_sub_routine(m, inf_item, inf_bias, vector_train, max(k), gpu=gpu)
    for n in k:
        result[n].append(hr_k(vector_predict,uit.itemId, n))

    return result

def singleshot_sub_routine(vector_u, item_emb, bias, vector_train, topK, gpu=False):
 
    train_index = vector_train.nonzero()[1]
    vector_predict = item_emb.dot(vector_u).flatten() +bias

    if gpu:
        import cupy as cp
        candidate_index = cp.argpartition(-vector_predict, topK+len(train_index))[:topK+len(train_index)]
        vector_predict = candidate_index[vector_predict[candidate_index].argsort()[::-1]]
        vector_predict = cp.asnumpy(vector_predict).astype(np.float32)
    else:
        candidate_index = np.argpartition(-vector_predict, topK+len(train_index))[:topK+len(train_index)]
        vector_predict = candidate_index[vector_predict[candidate_index].argsort()[::-1]]
    vector_predict = np.delete(vector_predict, np.isin(vector_predict, train_index).nonzero()[0])
    return vector_predict
