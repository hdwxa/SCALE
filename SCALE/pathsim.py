# encoding: utf-8
"""
calculates the meta path based similarity for pairwise nodes
"""

import os
import sys
import time
import pickle
import argparse
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations

from collections import defaultdict
from utils import *


def cal_pairSim(G, edgefile, pairs, alpha):
    lis = []
    for pair in pairs:
        liP = []
        mim1 = pair[0]  # student_id_1
        mim2 = pair[1]  # student_id_2
        related_elem_list_1 = G[mim1]  # student_id_1 对应的相关信息 list
        related_elem_list_2 = G[mim2]

        counter1 = defaultdict(lambda: 0)
        for kw in related_elem_list_1:
            counter1[kw] += 1

        counter2 = defaultdict(lambda: 0)
        for kw in related_elem_list_2:
            counter2[kw] += 1

        intersect_elem_list_has = list(set(related_elem_list_1).intersection(set(related_elem_list_2)))  # 两个list的交集
        intersect_elem_list = list()
        for elem in intersect_elem_list_has:
            ratio1 = counter1[elem] / len(related_elem_list_1)
            ratio2 = counter2[elem] / len(related_elem_list_2)
            if ratio1 > ratio2:
                w = ratio1 / ratio2
            else:
                w = ratio2 / ratio1

            if w <= alpha:
                intersect_elem_list.append(elem)

        path_count_mim1_mim2 = len(intersect_elem_list)  # 元素交集个数，pathsim公式分子部分
        # print(pair, " # of shared elem: ", path_count_mim1_mim2)

        path_count_self_mim1 = len(related_elem_list_1)  # d1的相关元素个数
        path_count_self_mim2 = len(related_elem_list_2)  # d2的相关元素个数

        # 计算pathsim值
        score = (2 * path_count_mim1_mim2) / (path_count_self_mim1 + path_count_self_mim2)
        liP.append(mim1)
        liP.append(mim2)
        liP.append(score)
        lis.append(liP)
    return lis


def cal_pathSim_all(G, edgefile, alpha):
    """
    calculate the meta path-based similarity score for pairwise nodes
    store the results in .pkl file
    :param G:
    :return:
    """

    # # 笛卡尔积
    # l = [1, 2, 3, 4, 5]
    # print(list(product(l, l)))
    # # 排列组合
    # print(list(combinations([1, 2, 3, 4, 5], 2)))

    if load_from_disk(edgefile + "_d_pair_pathsim") != None:
        print("file[" + edgefile + "_d_pair_pathsim.pkl] exists, load directly!")
        return edgefile + "_d_pair_pathsim"   # 直接获取文件
    else:

        mimnumber_set = G.keys()  # all student_ids
        print("# of students:", len(mimnumber_set))  # number of all students

        d_combination_list = list(combinations(mimnumber_set, 2))  # d_pair_list: 对全部学生进行两两排列组合(没有自己和自己)，组合结果存入list中
        print("# of d_combination_list:", len(d_combination_list))
        # print(d_combination_list)

        d_pair_sim_dict = {}  # 存储所有学生对之间的pathsim值

        count = 0
        #cui - 添加
        pairlist = []
        p = int(len(d_combination_list)/10)
        for i in range(9):
            pairlist.append(d_combination_list[i*p:(i+1)*p])
        pairlist.append(d_combination_list[9 * p:len(d_combination_list)])

        with ProcessPoolExecutor(max_workers=10) as executor:
            future = []
            for pairs in pairlist:
                temp = executor.submit(cal_pairSim, G, edgefile, pairs, alpha)
                future.append(temp)

            for i in range(len(future)):
                lis = future[i].result()  # key: (did_1, did_2), val: score
                for i in lis:
                    d_pair_sim_dict[(i[0], i[1])] = i[2]

        print("# of students:", len(mimnumber_set))  # number of all students
        print("Total number of d pairs:", len(d_combination_list))
        print("There are ", count, " d-pairs whose score > 0.")

        # 输出为.pkl文件
        outfile = edgefile+'_d_pair_pathsim'
        save_on_disk(d_pair_sim_dict, outfile)
        print("dump PathSim .pkl file successfully!")

        # 输出为.xlsx文件
        # outfile = 'd_pair_semsim'
        # save_on_disk_matrix(d_pair_sim_dict, outfile)
        # print("dump SemSim matrix (xlsx) file successfully!")

        # reference_relations_file = os.path.join('save/d_pair_pathsim.pkl')

        # if os.path.exists(reference_relations_file):
        #     print("find dumped reference relations file, skip pathsim calculating.")
        # else:
        #     # 存储计算结果
        #     with open(reference_relations_file, 'wb') as f:
        #         pickle.dump(d_pair_sim_dict, f)
        #         print("dump PathSim file successfully!")

        return outfile   # 存储了计算结果的文件名称


def process_semsim(filename):
    semsim_dict = load_from_disk(filename)

    vertices = load_from_disk(filename + "_vertices")

    all_ver_pairs = list(combinations(vertices, 2))  # d_pair_list: 对全部学生进行两两排列组合(没有自己和自己)，组合结果存入list中

    for i in range(len(all_ver_pairs)):
        if all_ver_pairs[i] in semsim_dict.keys():
            continue
        else:
            semsim_dict[all_ver_pairs[i]] = 0

    proc_filename = "combined_file_name_" + filename

    save_on_disk(semsim_dict, proc_filename)

    return proc_filename
