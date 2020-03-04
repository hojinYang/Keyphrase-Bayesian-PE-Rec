from scipy.sparse import csr_matrix

def get_csr_matrix(df, rowname, colname, value=None, shape=None):
    row = df[rowname]
    col = df[colname]
    if value == None:
        value = [1]*len(row)
    return csr_matrix((value, (row,col)), shape=shape)


def get_test_df(ratings_tr, tags_tr, tags_val):
    valid_user = ratings_tr['userId'].unique()
    valid_item = ratings_tr['itemId'].unique()
    valid_tag = tags_tr['tagId'].unique()

    tags_val = tags_val.loc[tags_val['userId'].isin(valid_user) &
                                tags_val['itemId'].isin(valid_item) &
                                tags_val['tagId'].isin(valid_tag)]

    return tags_val.groupby(['userId','itemId'])['tagId'].apply(list).reset_index(name='tagIds')