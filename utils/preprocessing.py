import pandas as pd
import numpy as np

def get_rating_summary(df,num_users = None, num_items = None):
    '''
    print summary of user-item matrix
    args: 
        df: data frame which contains userId & itemId columns
    '''
    if num_users == None:
        num_users = len(df['userId'].unique())
    if num_items == None:
        num_items = len(df['itemId'].unique())
    num_values = len(df)
    
    sparsity = 1 - (num_values/(num_users * num_items))
    print('# users: {0}, # items: {1}, # vals: {2}, sparsity: {3:.7f}'.
          format(num_users, num_items, num_values, sparsity))
    
def get_tag_summary(df, num_users = None, num_items = None, tagcol = 'tagId'):
    '''
    print summary of user-item-tag matrix
    args: 
        df: data frame which contains userId & itemId & tagId columns
    '''
    if num_users == None:
        num_users = len(df['userId'].unique())
    if num_items == None:
        num_items = len(df['itemId'].unique())
    num_tags = len(df[tagcol].unique())

    tagnum_per_interaction = df.groupby(['userId','itemId'])[tagcol].apply(lambda x:len(set(x))).reset_index()[tagcol]
    num_interaction = len(tagnum_per_interaction)
    sparsity = 1 - (num_interaction/(num_users * num_items))
    tagged_items_per_user = df.groupby('userId')['itemId'].apply(lambda x:len(set(x))).reset_index()['itemId']

    tag_count = df.groupby(tagcol)['itemId'].apply(len).reset_index()['itemId']
    
    print('# users: {0}, # items: {1}, # tags: {2}, #interaction: {3}, sparsity: {4:.7f}'.
          format(num_users, num_items, num_tags, num_interaction, sparsity))
    print("summary for the number of tags per interation")
    print(tagnum_per_interaction.describe())
    print("summary for the number of tagged items per users")
    print(tagged_items_per_user.describe())
    print("summary for the occurence per tag")
    print(tag_count.describe())    

def preprocess_ratings(ratings, min_rating):
    if min_rating > 1:
        ratings = ratings[ratings['rating'] >= min_rating]
    return ratings[['userId','itemId']]

def preprocess_tags(tags, tag_user_threshold, tag_item_threshold):
    '''
    stemming tags and remove rare tags. 
    '''
    tags = tags[['userId','itemId','tag']]
    tt = tags['tag'].apply(lambda x: x.lower().replace('.', ''))
    tags.loc[:,'tag'] = tt
    if tag_item_threshold > 1:
        #limit the vocabulary of tags to those that have been applied by at least "tag_item_threshold" items
        counter = tags.groupby('tag')['itemId'].apply(lambda x: len(set(x))).to_frame('count').reset_index()
        counter = counter[counter['count']>=tag_item_threshold]
        tags = pd.merge(tags,counter,on='tag')[['userId','itemId','tag']]

    if tag_user_threshold > 1:
        #limit the vocabulary of tags to those that have been applied by at least "tag_user_threshold" users
        counter = tags.groupby('tag')['userId'].apply(lambda x: len(set(x))).to_frame('count').reset_index()
        counter = counter[counter['count']>=tag_user_threshold]
        tags = pd.merge(tags,counter,on='tag')[['userId','itemId','tag']]

    return tags

def set_tagId(tags):
    '''
    set uinque tag id for tags.
    '''
    tag_list = list(tags['tag'].unique())
    tagId_list = list(range(len(tag_list)))
    tag_tagId = pd.DataFrame({'tag':tag_list,'tagId':tagId_list})
    tags = pd.merge(tags,tag_tagId, on='tag')[['userId','itemId','tagId']]
    return tags, tag_tagId

def _update_id(ratings, tags):

    old_itemId = ratings['itemId'].unique()
    new_itemId = np.arange(len(old_itemId))
    updated_itemId = pd.DataFrame({'itemId':old_itemId,'new_itemId':new_itemId})
    
    old_userId = ratings['userId'].unique()
    new_userId = np.arange(len(old_userId))
    updated_userId = pd.DataFrame({'userId':old_userId,'new_userId':new_userId})

    ratings = pd.merge(ratings,updated_itemId,on='itemId')[['userId','new_itemId','rating']].rename(columns={'new_itemId':'itemId'})
    ratings = pd.merge(ratings,updated_userId,on='userId')[['new_userId','itemId','rating']].rename(columns={'new_userId':'userId'})
    
    # remove items only in tag interacitons and users only in tag interactions: before: 310,041 after: 305,437    
    tags = pd.merge(tags,updated_itemId,on='itemId')[['userId','new_itemId','tagId']].rename(columns={'new_itemId':'itemId'})
    tags = pd.merge(tags,updated_userId,on='userId')[['new_userId','itemId','tagId']].rename(columns={'new_userId':'userId'})
     
    return ratings, tags, updated_itemId

def update_id(ratings, tags):
    # consider items both has tag and appear in ratings(this is to remove items with small number)
    old_itemId = list(set(tags['itemId'].unique()) & set(ratings['itemId'].unique()))
    new_itemId = np.arange(len(old_itemId))
    updated_itemId = pd.DataFrame({'itemId':old_itemId,'new_itemId':new_itemId})
    
    old_userId = ratings['userId'].unique()
    new_userId = np.arange(len(old_userId))
    print(new_userId[-1])
    updated_userId = pd.DataFrame({'userId':old_userId,'new_userId':new_userId})

    ratings = pd.merge(ratings,updated_itemId,on='itemId')[['userId','new_itemId','rating']].rename(columns={'new_itemId':'itemId'})
    ratings = pd.merge(ratings,updated_userId,on='userId')[['new_userId','itemId','rating']].rename(columns={'new_userId':'userId'})
    
    # remove items only in tag interacitons and users only in tag interactions: before: 310,041 after: 305,437    
    tags = pd.merge(tags,updated_itemId,on='itemId')[['userId','new_itemId','tagId']].rename(columns={'new_itemId':'itemId'})
    tags = pd.merge(tags,updated_userId,on='userId')[['new_userId','itemId','tagId']].rename(columns={'new_userId':'userId'})
     
    return ratings, tags, updated_itemId


def preprocess(ratings, tags, items, tag_user_threshold, tag_item_threshold):

    """
    args: 
        ratings: pd.DataFrame which contains 3 columns=['userId','itemId','rating']
        tags: pd.DataFrame which contains 3 columns=['userId','itemId','tag']
    """
    
    num_users = len(ratings['userId'].unique())
    num_items = len(ratings['itemId'].unique())
    
    print("-"*5 + 'before preprocessing' + '-'*5)
    print("rating summary")
    get_rating_summary(ratings, num_users=num_users, num_items=num_items)
    print("tag summary")
    get_tag_summary(tags, num_users=num_users, num_items=num_items, tagcol='tag')    
    
    tags = preprocess_tags(tags, tag_user_threshold, tag_item_threshold)
    tags, tag_tagId = set_tagId(tags)

    ratings, tags, updated_itemId = update_id(ratings, tags)
    # items = pd.merge(items,updated_itemId,on='itemId')[['new_itemId','title','genres']].rename(columns={'new_itemId':'itemId'})
    items = pd.merge(items,updated_itemId,on='itemId')[['new_itemId','name']].rename(columns={'new_itemId':'itemId'})

    num_users = len(ratings['userId'].unique())
    num_items = len(ratings['itemId'].unique())
    
    print("-"*5 + 'after preprocessing' + '-'*5)
    print("rating summary")
    get_rating_summary(ratings, num_users=num_users, num_items=num_items)
    print("tag summary")
    get_tag_summary(tags, num_users=num_users, num_items=num_items)

    interactions = ratings[['userId','itemId']]
    interactions = interactions.append(tags[['userId','itemId']],ignore_index=True)
    interactions.drop_duplicates(inplace=True)    

    return ratings, tags, interactions, items, tag_tagId
