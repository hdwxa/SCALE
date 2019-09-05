# -*- coding: utf-8 -*-

"""
Provides some general functions.

"""
import pickle
import argparse
import os
import datetime
import time
from collections import Counter, defaultdict
from itertools import product
from itertools import combinations
import logging
import inspect
import pandas
import numpy as np
import graph
import sys

src_f = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
dir_f = os.path.dirname(src_f)



folder_save = dir_f + "/save/"              # 数据保存路径
folder_data = dir_f + "/data/student/"   # 数据集存储路径
folder_save_matrix = dir_f + "/matrix/"      # 数据保存路径

def read_graph(edgefile):
    '''
    Reads an edge file and builds a HIN (CBIN).
    '''
    logging.info(" - Loading an edge file and build a HIN...")

    time_start = time.time()

    if load_from_disk(edgefile + "_d_other_dict") != None:
        print("file[" + edgefile + "_vertices.pkl] exists, load directly!")
        G = load_from_disk(edgefile + "_d_other_dict")
        return G  # 直接获取文件

    G = graph.load_edge_file(edgefile)  # 读取边数据文件，构建图G，用一个dict(list)类型的字典存储

    time_end = time.time()
    print('total time cost for read_graph[', edgefile,']: ', time_end - time_start)

    logging.info(" - HIN loaded.")

    print(edgefile + " - Graph length(number of nodes):", len(G))

    return G


def load_csv_file(filename):

    csv_file = folder_data + filename + '.csv'

    data = pandas.read_csv(csv_file, sep='\t')    # 打开csv文件

    return data


def readcsv3(filename):
    readStart = time.time()
    df2 = pandas.read_csv(folder_data + filename + ".csv", sep="\t", encoding="UTF-8")
    readEnd = time.time()
    print('read csv cost %2.f seconds' % (readEnd - readStart))
    return df2

def save_time(where_time):
    df = pandas.DataFrame([where_time])
    df.to_csv("time_cost.csv", mode='a', header=False)


def save_sim(filename, result):
    df = pandas.DataFrame([result])
    df.to_csv(filename, mode='a', header=False)


def load_emb_file(filename):
    logging.info('load emb data from disk ...')
    val = None

    with open(dir_f + filename, 'r') as f:
        val = f.read()
    
    return val


def load_from_disk(filename):

    logging.info('load data from disk ...')
    val = None
    filename = folder_save + filename + '.pkl'

    print(filename)
    if os.path.exists(filename):
        print("find dumped file...")
        with open(filename, 'rb') as handle:
            val = pickle.load(handle)
    
    return val


def load_matrix_from_disk(filename):
    logging.info('load matrix from disk ...')
    val = None

    with open(folder_save + filename + '.csv', 'rb') as handle:
        val = pickle.load(handle)

    return val


def save_on_disk(data, filename):  # data: a 'dict'
    logging.info('save data on disk')

    file_output = os.path.join(folder_save + filename + '.pkl')

    # if os.path.exists(file_output):
    #     print("find dumped file, skip output.")
    # else:
        # 存储结果文件
    with open(file_output, 'wb') as handle:
        pickle.dump(data, handle)
    
    logging.info('Data has been saved.')




def save_on_disk_matrix(matr, filename):  # data: a 'dict' saving SemSim
    logging.info('save data as a matrix on disk')

    file_output = os.path.join(folder_save_matrix + filename + '.csv')

    np.savetxt(file_output, matr, delimiter=',')

    logging.info('Matrix', filename, ' has been saved.')




def partition(lst, n):
    division = len(lst) / float(n)  # 总数/分块数 = 每一部分存的元素个数
    # 返回每一部分包含的元素，一共n部分，每一部分大小为division
    return [ lst[int(round(division * i)): int(round(division * (i + 1)))] for i in range(n) ]

def get_sim_path():
    return dir_f

def is_pkl(fname):
    return os.path.isfile(folder_save+fname+'.pkl')



def alias_draw(J, q):
    '''
    Draw sample from a non-uniform discrete distribution using alias sampling.
    '''
    K = len(J)

    kk = int(np.floor(np.random.rand()*K))
    if np.random.rand() < q[kk]:
        return kk
    else:
        return J[kk]


def alias_setup(probs):
    '''
    Compute utility lists for non-uniform sampling from discrete distributions. 离散分布的非均匀采样。
    Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    for details
    '''
    K = len(probs)
    q = np.zeros(K)
    J = np.zeros(K, dtype=np.int)

    smaller = []
    larger = []
    for kk, prob in enumerate(probs):
        q[kk] = K * prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)

    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()

        J[small] = large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return J, q
