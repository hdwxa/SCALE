# -*- coding: utf-8 -*-
import sys
import argparse, logging
import numpy as np
from time import time
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from concurrent.futures import ProcessPoolExecutor, as_completed
from graph import *
from pathsim import *
from sim2vec import *
from pathsim import *
from sim_search import *
import utils

logging.basicConfig(filename='SimpSim.log', filemode='w', level=logging.DEBUG, format='%(asctime)s %(message)s')


def parse_args():
    '''
    Parses the SimpSim arguments.
    '''
    parser = argparse.ArgumentParser(description="Run SimpSim.")

    # 输入文件（节点、边文件）
    parser.add_argument('--input', nargs='?', default='data/simpsim_public_hmdd_d_rna.csv', help='Input graph path')

    parser.add_argument('--output', nargs='?', default='/emb/student', help='Embeddings path')

    parser.add_argument('--alpha', type=float, default=0.5, help='Decaying weight factor when computing structural distance')

    parser.add_argument('--maxhop', type=int, default=2, help='Max hop when computing the distance.')  # 计算到哪一层邻居

    parser.add_argument('--dimensions', type=int, default=32, help='Number of dimensions in skip-gram model. Default is 128.')

    parser.add_argument('--alpha-w', type=float, default=1.4, help='Weight ratio threshold when count the instance of meta-path')

    parser.add_argument('--walk-length', type=int, default=40, help='Length of walk per source. Default is 80.')  # 单次游走长度

    parser.add_argument('--num-walks', type=int, default=128, help='Number of walks per source. Default is 10.')  # 重复游走的次数

    parser.add_argument('--data', type=int, default=1, help='')

    parser.add_argument('--window-size', type=int, default=10, help='Context size for optimization (used in Skip-Gram). Default is 10.')

    parser.add_argument('--until-layer', type=int, default=None, help='Calculate the StrucSim until which layer.')  # 计算结点到第n层的度序列

    parser.add_argument('--iter', default=5, type=int, help='Number of epochs in SGD')

    parser.add_argument('--workers', type=int, default=4, help='Number of parallel workers. Default is 4.')

    parser.add_argument('--weighted', dest='weighted', action='store_true', help='Boolean specifying (un)weighted. Default is unweighted.')
    parser.add_argument('--unweighted', dest='unweighted', action='store_false')
    parser.set_defaults(weighted=False)

    parser.add_argument('--directed', dest='directed', action='store_true', help='Graph is (un)directed. Default is undirected.')
    parser.add_argument('--undirected', dest='undirected', action='store_false')
    parser.set_defaults(directed=False)

    parser.add_argument('--OPT1', default=False, type=bool, help='optimization 1')
    parser.add_argument('--OPT2', default=False, type=bool, help='optimization 2')
    parser.add_argument('--OPT3', default=False, type=bool, help='optimization 3')
    return parser.parse_args()


def exec_SimpSim(args, filelist):
    """
        对单个数据集计算学生相似度，得到单视角下的学生相似网络 (egdefile -> CBIN-> SSN)
        1. read_graph: read file and build a graph (a CBIN, stored by a 'dict')
        2. exec_pathSim: calc meta-path-based similarity for pairwise nodes (represents semantic similarity)
    """

    #  这里需要写一个循环，从参数列表里面读取一共有多少个文件，依次对每个数据文件计算得到学生相似网络并输出保存
    sim_file_list = []  # 存储对各个数据集最终计算得到的相似文件名称
    with ProcessPoolExecutor(max_workers=4) as executor:
        pathSim_file_name_list = []
        for edgefile in filelist:  # 构建多个相互独立的SSN
            G = read_graph(edgefile)  # step 1: 读取边文件，构建图，(a CBIN, stored by a 'dict')
            pathSim_file_name = executor.submit(exec_pathSim, G, edgefile)
            pathSim_file_name_list.append(pathSim_file_name)

        for i in range(len(filelist)):
            combined_sim = load_from_disk(pathSim_file_name_list[i].result())
            combined_file_name = 'combined_file_name_' + filelist[i]
            save_on_disk(combined_sim, combined_file_name)  # 保存最终相似度计算结果


            sim_file_list.append(combined_file_name)  # 将对单个数据集计算得到的最终相似结果（SSN）存储到文件列表中
            print("Only_semsim: Single SSN for [", filelist[i], "] constructed!")
            print("---------------------------------------------------")
    return sim_file_list


def exec_pathSim(G, edgefile):
    """
    :param G: original d_other dict (d_id is the key)
    :return: the name of the file that stores the dict with pathsim score
    """
    # 基础搜索算法
    logging.info(" - Processing pruned PathSim calculation...")

    if load_from_disk(edgefile + "_d_pair_pathsim") != None:
        print("file[" + edgefile + "_d_pair_pathsim.pkl] exists, load directly!")
        return edgefile + "_d_pair_pathsim"  # 直接获取文件

    pathsim_file_name = cal_pathSim_all(G, edgefile, args.alpha_w)

    logging.info(" - PathSim calculation finished.")

    return pathsim_file_name


def exec_construct_multi_sim_network(sim_file_list, filename):
    """
    @:param sim_file_list: SSN file list
        将多个相似网络关联在一起，构建一个多层相似网络。在该网络上对每个节点开展随机游走，获得上下文（eg. 迭代游走5次，将产生的5段序列拼接起来作为该节点的上下文）
        1. associate all the similarity network together, then we get a weighted multiplex similarity network.
        2. for every node, conduct the biased random walk for 5 times to get 5 independent node sequences as the context of this node.
        3. output the walk results onto disk. eg.walk_result.txt
        4. conduct Random Walk on multiplex network M and generate node sequences for each node as the context
    """

    print("begin constructing multi-layer SSN...")
    print("SSN:", sim_file_list)
    construct_multi_sim_network(sim_file_list)  # 构建多层SSN网络(一个存储边权，一个存储邻接信息，产生随机游走使用到的2个字典j,q)
    print("construct multi-layer SSN finished")

    print("begin conducting random walks over the multi-layered SSN...")
    simulate_walks(args.num_walks, args.walk_length, filename)

    print("random walks over the multi-layered SSN finished")
    return


def exec_embedding(index, walk_length):
    '''
    Learn embeddings by optimizing the Skip-gram objective using SGD.
    '''
    print("begin embedding...")
    logging.info("Initializing creation of the representations...")

    walks = LineSentence('walk_result_num=' + str(index) + '_walk_len=' + str(walk_length) + '.txt' )

    # for i in range(4, 10):
    dimension = args.dimensions
    model = Word2Vec(walks, size=dimension, window=args.window_size, min_count=0, hs=1, sg=1,
                 workers=args.workers, iter=args.iter)
    args_filename = "_w=" + str(args.alpha_w) + "_num=" + \
                    str(index) + '_dim=' + str(dimension)
    print(get_sim_path() + args.output + args_filename + ".emb")

    model.wv.save_word2vec_format(get_sim_path() + args.output + args_filename + ".emb")

    logging.info("Representations for all student nodes created.")
    print("skip-gram embedding finished.")

    return


def exec_sim_search(args, query_mim_list, k):
    '''
        searching top-k similar students for query student
    '''
    logging.info("searching top-k similar students...")

    # embedding file
    args_filename = "_w=" + str(args.alpha_w) + "_num=" + \
                    str(args.num_walks) + '_dim=' + str(args.dimensions)
    embedFile = args.output + args_filename + ".emb"

    sim_d_list = top_k_sim_search(embedFile, query_mim_list, k)

    logging.info("Similarity search for top-k students finished.")

    return


def main(args, filelist, query_mim_list, k):
    """
    Main Steps of RADAR:
        1. for each data set, calculate the similarity(structural and semantic) between pairwise nodes and get one SSN
        2. build a multi-layered similarity network by associating every counterparts(same nodes) in each network
            --> each layer of network is a SSN, with each edge referring to the similarity between two nodes
        3. conduct the biased random walk for every node for several times, to generate independent sequences as its context
        4. use the word2vec package to learning the embeddings for every node by its context.
        5. similarity search over embeddings
    """
    sim_file_list = exec_SimpSim(args, filelist)      # 1. 依次对所有数据集计算得到对应的SSN，返回存储了各个SSN的文件名称列表
    sim_file_list_group = ['combined_file_name_'+ filelist[0], 'combined_file_name_' + filelist[1]]
    exec_construct_multi_sim_network(sim_file_list_group, filelist[0])   # 在单层SSN上执行带偏随机游走，产生序列并保存
    exec_embedding(args.num_walks, args.walk_length)  # 3. 调用skip-gram模型，根据游走出来的序列学习节点的embedding
    exec_sim_search(args, query_mim_list, k)    # 4. 相似搜索


if __name__ == "__main__":
    args = parse_args()

    query_mim_list = ['']  # query student by id
    k = 20  # top-k result

    filenamelist = ['201504_6mResult', '201504_6m_mjResult']

    main(args, filenamelist, query_mim_list, k)
