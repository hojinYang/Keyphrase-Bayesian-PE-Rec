from experiment.query_ranker import rank_by_mean, rank_by_variance, rank_by_UCB, rank_randomly,rank_by_TS,rank_by_POP,rank_by_PPOP

queries = {
    "Mean" : rank_by_mean,
    "Var": rank_by_variance,
    "UCB" : rank_by_UCB,
    "Random" : rank_randomly,
    "TS" : rank_by_TS,
    "POP" : rank_by_POP,
    "PPOP" : rank_by_PPOP
}

