from sklearn.metrics.pairwise import cosine_similarity

def get_cosine_sim(input_tags, tag_tagId, matrix, num):
    for tag in input_tags:
        input_tagId = tag_tagId[tag_tagId['tag']==tag]['tagId'].values[0]
        input_vec = matrix[[input_tagId],:]
        cs = cosine_similarity(input_vec,matrix )
        cands = cs[0].argsort()[-1*num:][::-1]
        results = ""
        for i in cands[1:]:
            result = tag_tagId[tag_tagId['tagId']==i]['tag'].values[0]
            results += result+", "
        print(tag+': '+results[:-2])
        
def get_cosine_sim_item_tag(input_items, item_itemId, item_matrix, tag_tagId, tag_matrix, num):
    for item in input_items:
        input_itemId = item_itemId[item_itemId['item']==item]['itemId'].values[0]
        input_vec = item_matrix[[input_itemId],:]
        cs = cosine_similarity(input_vec,tag_matrix )
        cands = cs[0].argsort()[-1*num:][::-1]
        results = ""
        for i in cands:
            result = tag_tagId[tag_tagId['tagId']==i]['tag'].values[0]
            results += result+"//"
        print(item+': '+results)