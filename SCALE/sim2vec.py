# encoding: utf-8

import numpy as np
import random, sys, logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager
from time import time
from utils import *
from algorithm_walk import *
import graph


def construct_multi_sim_network(file_list, workers=4):
    '''构建多层带权完全图'''
    with ProcessPoolExecutor(max_workers=1) as executor:
        job = executor.submit(generate_similarity_network, file_list, workers)

        job.result()

    return


def preprocess_parameters_random_walk():
    '''初始化随机游走的参数'''
    with ProcessPoolExecutor(max_workers=1) as executor:
        job = executor.submit(generate_parameters_random_walk)
        job.result()

    return


def simulate_walks(num_walks, walk_length, filename, workers=20):
    '''
        随机游走
        @:param num_walks: 重复游走的次数
        @:param walk_length: 单次游走的序列长度
    '''
    time_start = time.time()
    vertices = load_from_disk(filename + '_vertices')  # 全部学生节点
    timm_end = time.time()

    # for large graphs, it is serially executed, because of memory use.
    if (len(vertices) > 500000):

        with ProcessPoolExecutor(max_workers=1) as executor:
            job = executor.submit(generatenets_weights_alias_method_j_random_walks_large_graphs, num_walks, walk_length, workers, vertices)

            job.result()

    else:  # 非大规模，适用于本实验

        with ProcessPoolExecutor(max_workers=1) as executor:
            job = executor.submit(generate_random_walks, num_walks, walk_length, workers, vertices)   # 对全部节点随机游走num_walks轮，保存输出产生的序列

            job.result()

    return
