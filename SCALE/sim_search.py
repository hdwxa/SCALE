# encoding: utf-8

import numpy as np
from utils import *
from collections import Counter, defaultdict
import math

def get_emb_data(emb_file):
    emb_data = load_emb_file(emb_file)

    emb_data = emb_data.split('\n')

    dimension = int(emb_data[0].split(' ')[1])       # embedding 数据维度

    d_vec_dict = defaultdict(float)

    for line in range(1, len(emb_data)):
        parts = emb_data[line].split(' ')

        if(len(parts) > dimension):
            mim = parts[0]  # 学生编号
            emb_vec = [parts[i] for i in range(1, dimension + 1)]  # 学生向量
            emb_vec = np.array(emb_vec).astype(float)   # 转换为浮点型

            d_vec_dict[mim] = emb_vec       # 学生向量

    print("emb dict :", len(d_vec_dict))

    return d_vec_dict

def cal_euclidean_distance(vec1, vec2):
    dist = np.sqrt(np.sum(np.square(vec1 - vec2)))
    return dist


def cal_cosine_sim(vec1, vec2):
    multiple = 0
    norm_vec1 = 0
    norm_vec2 = 0
    for v1, v2 in zip(vec1, vec2):
        multiple += v1 * v2
        norm_vec1 += v1 ** 2
        norm_vec2 += v2 ** 2
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0
    else:
        return multiple / ((norm_vec1 * norm_vec2) ** 0.5)


def cal_pearson_distance(vector1, vector2):
    sum1 = sum(vector1)
    sum2 = sum(vector2)

    sum1Sq = sum([pow(v, 2) for v in vector1])
    sum2Sq = sum([pow(v, 2) for v in vector2])

    pSum = sum([vector1[i] * vector2[i] for i in range(len(vector1))])

    num = pSum - (sum1 * sum2 / len(vector1))
    den = math.sqrt((sum1Sq - pow(sum1, 2) / len(vector1)) * (sum2Sq - pow(sum2, 2) / len(vector1)))

    if den == 0: return 0.0
    return 1.0 - num / den

def sort_dict(dic):
    ''' 将字典转化为列表 '''

    lst = [(key, value) for key, value in dic.items()]
    # 对列表元素排序
    # sorted_list = sorted(lst, key=lambda x: x[1])  # 距离升序排序
    sorted_list = sorted(lst, key=lambda x: x[1], reverse = True)  # 相似度降序排序

    return sorted_list


def top_k_sim_search(embed_file, query_d_list, k):

    d_vec_dict = get_emb_data(embed_file)  # get embeddings

    sim_d_dict = defaultdict(float)  # 存储当前top-k个相似学生+相似度值
    sim_d_list_dict = defaultdict(list)     # 存储query_d_list中每个学生的相似学生list

    # dict_len = len(sim_d_dict)
    # dict_len = 0

    for query_d in query_d_list:
        sim_d_dict = defaultdict(float)  # 存储当前top-k个相似学生+相似度值
        dict_len = 0
        query_d_vec = d_vec_dict[query_d]  # 查询学生的向量表示

        for candidate_mim, candi_vec in d_vec_dict.items():

            # print(query_d_vec_array)
            # print(candi_vec_array)

            candi_vec_array = d_vec_dict[candidate_mim]

            dis_candi_query = cal_cosine_sim(query_d_vec, candi_vec_array)  # 向量的余弦相似度

            print(query_d, candidate_mim, dis_candi_query)

            dis_candi_query = float(dis_candi_query)  # 转换为浮点型

            if dict_len < k:  # 未满k个直接添加
                sim_d_dict[candidate_mim] = dis_candi_query
                dict_len += 1
            elif dict_len == k:
                # cur_max_dis = max(sim_d_dict)  # 当前候选学生中最大距离值
                # cur_max_dis_mim = max(sim_d_dict, key=sim_d_dict.get)
                cur_max_dis_mim = min(sim_d_dict, key=sim_d_dict.get)

                if dis_candi_query > float(sim_d_dict[cur_max_dis_mim]):
                    sim_d_dict.pop(cur_max_dis_mim)  # 删除当前候选结果中距离最大的学生键值对
                    sim_d_dict[candidate_mim] = dis_candi_query  # 添加新的候选学生
                else:
                    continue
        top_k_sim_d_sorted = sort_dict(sim_d_dict)
        sim_d_list_dict[query_d] = top_k_sim_d_sorted

    save_sim_search(sim_d_list_dict)

    # return top_k_sim_d_sorted
    return sim_d_list_dict


def save_sim_search(topk_sim):
    for query_key, sim_list in topk_sim.items():
        i = 1
        for min, value in sim_list:
            save_sim('topk.csv', [str(min), str(value), i])
            i = i + 1
    return