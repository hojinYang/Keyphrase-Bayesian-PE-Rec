import numpy as np
from tqdm import tqdm 
from sklearn.metrics.pairwise import cosine_similarity
from models.regressions import get_predictive_dist, update_posterior,get_user_dist 


class Simulator:
    def __init__(self, item_emb, tag_emb, r_ui, t_it, k,tau, step, prec_W, prec_item, prec_tag, query_ranker, alpha):
        
        self.item_emb  = item_emb
        self.tag_emb = tag_emb
        self.r_ui = r_ui
        self.t_it = t_it
        self.k = k
        self.tau = tau
        self.step = step
        self.alpha = alpha

        self.prec_W = prec_W
        self.prec_item = prec_item
        self.prec_tag = prec_tag
        self.query_ranker = query_ranker

        self.global_sum = np.squeeze(np.asarray(self.t_it.sum(axis=0)))

    def user_simulator(self, user, target_item, tags):
        
        result = {k:[] for k in self.k}

        #tags = get_similar_tags(tags,self.tag_emb)
        
        #TODO: how should we model user's response? Should we ban unpredictable queries?
        #item_tags = (self.t_it[target_item]>=10).nonzero()[1].tolist() + tags
        item_tags = (self.t_it[target_item]>=5).nonzero()[1].tolist() + tags
        #item_tags = self.t_it[target_item].nonzero()[1]
        #banned_queries = np.setdiff1d(item_tags, tags).tolist()
        
        user_history = self.r_ui[user]

        #queries in redun_queires will not be asked 
        #redun_queries = [] + banned_queries
        redun_queries = []

        #S, m = get_user_dist(user, self.r_ui, self.item_emb, prec_W=self.prec_W, prec_y=self.prec_item)
        #S, m = (1/self.prec_W)*np.identity(self.item_emb.shape[1]), np.zeros((self.item_emb.shape[1],1))
        S, m = (1/self.prec_W)*np.identity(self.item_emb.shape[1]), np.mean(self.item_emb,axis=0,keepdims=True).T
        vector_predict = sub_routine(m, self.item_emb, user_history, max(self.k))
        
        #initial prediction
        for n in self.k:
            result[n].append(hr_k(vector_predict,target_item, n))
        alpha = self.alpha
        for _ in range(self.step):
            
            pred_v, pred_m = get_predictive_dist(self.tag_emb.T, S, m, self.prec_tag)
            #query_rank = self.query_ranker(pred_v=pred_v, pred_m=pred_m, alpha=self.alpha, global_sum=self.global_sum)
            query_rank = self.query_ranker(global_sum=self.global_sum,pred_v=pred_v, pred_m=pred_m, alpha=alpha, item_emb=self.item_emb, tag_emb=self.tag_emb,\
                tau = self.tau, S = S, m = m, prec_tag = self.prec_tag, user_history=user_history)
            
            #alpha = alpha * 0.5
            #print(alpha)
            
            query = self.select_single_question(query_rank, redun_queries)
            redun_queries.append(query)
            

            user_response = self.get_user_reponse(query, item_tags)
            X = self.tag_emb[query].reshape((-1,1))
            Y = np.array([[user_response * self.tau]])

            S, m =  update_posterior(X, Y, S, m, self.prec_tag)
            
            #updated prediction
            vector_predict = sub_routine(m, self.item_emb, user_history, max(self.k))
            for n in self.k:
                result[n].append(hr_k(vector_predict,target_item, n))

        return result

    def select_single_question(self, rank, redun_tags):
        rank = np.delete(rank, np.isin(rank, redun_tags).nonzero()[0])
        return rank[0]

    def get_user_reponse(self, query_tags, relevant_tags):
        return np.isin(query_tags, relevant_tags)


def sub_routine(vector_u, item_emb, user_history, topK):
    train_index = user_history.nonzero()[1]
    vector_predict = item_emb.dot(vector_u).flatten() 

    candidate_index = np.argpartition(-vector_predict, topK+len(train_index))[:topK+len(train_index)]
    vector_predict = candidate_index[vector_predict[candidate_index].argsort()[::-1]]
    vector_predict = np.delete(vector_predict, np.isin(vector_predict, train_index).nonzero()[0])

    return vector_predict


def hr_k(preds,target, k):
    if target in set(preds[:k]):
        return 1

    else:
        return 0


def get_similar_tags(given_tags, tag_emb, sim=0.95):
    #sim 0.90
    sim_matrix = cosine_similarity(tag_emb[given_tags], tag_emb)
    similar_tags = np.unique((sim_matrix >= sim).nonzero()[1])
    return similar_tags