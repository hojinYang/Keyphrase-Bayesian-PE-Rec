import pandas as pd
import numpy as np

def get_kfold(ratings, tags, interactions, tr_index, te_index, val_percent):
    np.random.shuffle(tr_index)
    tr_offset = int(len(tr_index) * (1-val_percent))
    tr_interaction = interactions.iloc[tr_index[:tr_offset]]
    val_interaction = interactions.iloc[tr_index[tr_offset:]]
    te_interaction = interactions.iloc[te_index]

    train = extract(ratings, tags, tr_interaction)
    valid = extract(ratings, tags, val_interaction)
    test = extract(ratings, tags, te_interaction)

    return train, valid, test


def extract(ratings, tags, interaction):
    ratings = ratings.merge(interaction, on=['userId', 'itemId'])
    tags = tags.merge(interaction, on=['userId', 'itemId'])
    return ratings, tags


