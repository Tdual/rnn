import boto3
import pandas as pd
import MeCab
import itertools as itt
from collections import Counter

def get_data(max_num=0):
    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table('qiita2')
    print("item num: ",table.item_count)
    response = table.scan()
    data = response['Items']

    while 'LastEvaluatedKey' in response:
        response = table.scan(ExclusiveStartKey=response['LastEvaluatedKey'])
        data.extend(response['Items'])
        print(len(data))
        if max_num !=0 and len(data) > max_num:
            break
    return data


def get_bodies_tags(data):
    body_data = []
    tag_data = []
    for d in data:
        json = d["json"]
        body = json["raw_body"]
        tags = json["tags"]
        for tag in tags:        
            body_data.append(body)
            tag_data.append(tag["name"])
    return body_data, tag_data


def get_tag_dic(filepath="tags.json"):
    df = pd.read_json(filepath) 
    return df

def create_one_hots(df, dim=100, include_other=True):
    if include_other:
        name_list = df.ix[:,"id"].tolist()[:dim-1]
        name_list.append("other")
    else:
        name_list = df.ix[:,"id"].tolist()[:dim]
    one_hots = pd.get_dummies(name_list)
    print("dim", len(one_hots))
    return one_hots

def tags_to_one_hots(tags, dim=100, tag_dic=None):
    if not tag_dic:
        td = get_tag_dic()
        tag_dic = create_one_hots(td, dim)
    name_list = tag_dic.columns.tolist()
    one_hot_data = []
    for t in tags:
        if t not in name_list:
            t = "other"
        d = tag_dic.ix[:,t]
        one_hot_data.append(d)
    return one_hot_data

def wakati(data, one_hots, max_seq_size=2000):
    tagger = MeCab.Tagger('-Owakati -d /usr/local/lib/mecab/dic/mecab-ipadic-neologd')
    contents = []
    new_one_hots = []
    for b,oh in zip(data,one_hots):
        word = tagger.parse(b).split(' ')
        word = [w.strip() for w in word ]
        if len(word) > max_seq_size:
            word = word[:max_seq_size]
        contents.append(word)
        new_one_hots.append(oh)
    return contents, new_one_hots

def pad_seq(seqs, max_size=2000):
    return [s + ["<PAD/>"] * (max_size - len(s)) for s in seqs]

def create_num_vecs(seqs):
    ctr = Counter(itt.chain(*seqs))
    word_dic = { c[0]: i for i, c in enumerate(ctr.most_common()) }
    data = [[word_dic[w] for w in seq ] for seq in seqs]
    dic_size = len(word_dic)
    return data, dic_size

def get_tf_data(max_seq_size=2000, class_num=100, max_data_num=0):
    json_data = get_data(max_data_num)
    bodies, tags = get_bodies_tags(json_data)
    one_hots = tags_to_one_hots(tags, class_num)
    data, oh = wakati(bodies, one_hots, max_seq_size)
    padded_data = pad_seq(data, max_seq_size)
    data, dic_size = create_num_vecs(padded_data)
    return data, oh, dic_size
    
    



