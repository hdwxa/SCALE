# encoding: utf-8
"""
Provides the detailed implementations of all the distance functions.

"""
from time import time
from collections import deque
import numpy as np
import math, logging
from fastdtw import fastdtw
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict
from utils import *
import os


def exec_bfs(G_d_other, filename, until_layer, workers = 4):
    ''' 运行BFS算法计算节点的度，输出的字典受到until_layer的影响'''

    futures = {}        # 存储线程
    degreeList = {}     # 存储结点的度序列

    G_other_d = load_from_disk(filename+'_other_d_dict')

    vertices = list(sorted(G_d_other.keys()))     # all student_ids
    parts = workers            # 一共分成workers部分，分块是为了让CPU并行处理、提高执行效率
    chunks = partition(vertices, parts)     # 把全部结点分成4部分，chunks存储了4个结点块

    with ProcessPoolExecutor(max_workers = workers) as executor:
        part = 1
        for c in chunks:  # 依次取出一个结点块（一个节点集合）
            # get the degree list for each of the vertices (from layer 1 to layer 'until_layer')
            job = executor.submit(getDegreeListsVertices, G_d_other, G_other_d, until_layer, c)
            futures[job] = part
            part += 1

        for job in as_completed(futures):
            dl = job.result()       # 获取进程结果
            degreeList.update(dl)   # 对degreeList进行更新

    logging.info("saving degreeList on dist...")
    save_on_disk(degreeList, filename + '_degreeList_maxhop=' + str(until_layer))   # 输出度序列
    logging.info("saving degreeList successfully")


def getDegreeListsVertices(G_d_other, G_other_d, until_layer, vertices):
    ''' 获取图G中所有vertices顶点的度序列 '''
    degreeList = {}  # key: vertice, val: degreeList

    for v in vertices:   # 依次获取每个结点的度序列
        # 获取结点v在整个图G中前until_layer层邻域内的度序列信息，返回的是一个字典
        degreeList[v] = getDegreeLists_new(G_d_other, G_other_d, until_layer, v)   # 节点v在前until_layer层邻域内的度序列
    return degreeList  # 返回一群结点的度列表


''' --------- 新版本做法，邻居考虑了全部节点 ---------'''
def getDegreeLists_new(G_d_other, G_other_d, until_layer, root):
    '''
    获取结点root在图G中的前until_layer层的度序列
    定义各层的度模式为：deg(student), deg(other), 构成一个二元组
    args:
        G_d_other: 图G, key: d, value: other
        G_other_d: 图G, key: other, value: d
        root: 起始学生结点
    return:
        deg_seq_dict: 目标节点root在前until_layer层邻域内的节点度序列
            = layer 0: (deg0(student), deg0(other))
            = layer 1: (deg1(student), deg1(other))
    '''
    deg_dict = {}

    vector_access = defaultdict(int)

    node_queue = deque()  # 存储将要访问并计算其度序列的【结点】

    # 往右边添加一个元素
    node_queue.append((root, 0))  # start scanning from the source node, # 区分节点类型的标记，0：表示学生节点，1：表示其他类型节点

    vector_access[root] = 1  # 标记一个结点是否已经被访问，1：已访问，0：没访问。最开始的起始结点被访问，标记为1

    deg_seq_queue = deque()  # 存储某一层的节点【度序列】

    depth = 0  # 表示结点的第几层邻域，最开始从第0层开始，表示自身
    numOfNodesNewlyAddedToQueue = 0  # 记录每一层中的邻居个数
    numOfNodesToBePoped = 1  # 每一轮node_queue队列中剩余的结点数，初始状态下只有起始节点

    ''' 每次while循环是针对node_queue中的一个节点，计算其在until_layer层之前的各层的度，得到一个度序列 '''
    while node_queue:
        vertex_info = node_queue.popleft()  # 获取最左边的一个元素
        numOfNodesToBePoped -= 1

        if vertex_info[1] == 0:  # 学生节点
            deg_seq_queue.append(len(G_d_other[vertex_info[0]]))  # 学生节点的度
            # l.append(str(len(G_d_other[vertex])))          # 将顶点的 度(字符串) 存储到l队列中

            # 以vertex为中心, 存储vertex的邻域
            for other_id in G_d_other[vertex_info[0]]:  # 与当前节点关联的其他类型结点的编号
                if vector_access[other_id] == 0:
                    vector_access[other_id] = 1  # 标记为已访问
                    node_queue.append((other_id, 1))  # 将当前节点关联到的所有【其他类型】节点添加到待访问队列中
                    numOfNodesNewlyAddedToQueue += 1  # 队列中新加入的节点数目+1

        elif vertex_info[1] == 1:  # 其他类型节点
            deg_seq_queue.append(len(G_other_d[vertex_info[0]]))  # 其他节点的度
            # l.append(str(len(G_d_other[vertex])))          # 将顶点的 度(字符串) 存储到l队列中

            # 以vertex为中心, 存储vertex的邻域
            for d_id in G_other_d[vertex_info[0]]:  # 与当前节点关联的其他类型结点的编号
                if vector_access[d_id] == 0:
                    vector_access[d_id] = 1  # 标记为已访问
                    node_queue.append((d_id, 0))  # 将学生当前节点关联到的所有【学生】节点添加到待访问队列中
                    numOfNodesNewlyAddedToQueue += 1

        # 队列中还有节点时，会继续访问并计算节点的度序列
        # 队列中无结点时, 表示该层结点度数已经计算完成, 存储度序列
        if numOfNodesToBePoped == 0:  # 保证每次一组同类型节点（某一层）全部出队列之后再输出这一层的度序列
            deg_in_one_hop = np.array(deg_seq_queue, dtype='float')
            deg_in_one_hop = np.sort(deg_in_one_hop)

            # deg_sum = 0
            # for i in deg_in_one_hop:
            #     deg_sum += i   # 对某一层的度数累加求和
            # deg_seq_dict[depth] = deg_sum  # 在这一层的度数之和

            deg_dict[depth] = deg_in_one_hop
            deg_seq_queue = deque()  # 度序列队列清空

            if until_layer == depth:
                break
            depth += 1  # 向外扩展到下一层
            numOfNodesToBePoped = numOfNodesNewlyAddedToQueue
            numOfNodesNewlyAddedToQueue = 0
        else:
            continue

    return deg_dict  # 目标节点root在前until_layer层邻域内的度序列


''' ------------ 新版本结构相似度计算 -----------'''
def calc_struDis_all_new(vertice_pairs, degreeList, alpha, part, filename):
    ''' 计算vertices中的全部节点对儿之间的结构距离。 利用DTW算法计算度序列的差值，作为两个顶点的距离 '''
    ''' 
        @:param degreeList: 度序列字典，key: d_id, val: 学生节点在各层的度，是一个dict
        @:param vertices: 一组节点
        @:param list_vertices: vertices中的每个节点需要进行比较的节点id（在全部节点集合中）list_vertices=[[d2,d3,...], [d3,d4,..], ...]
        @:param part:     
    '''
    distances = {}  # 存储vertices中每个节点在各层的结构距离。是个二维字典。外层字典中，key为结点对。内层字典中，key为层数，val为该结点对在各层的结构距离

    # 定义序列距离的计算函数
    dist_func = cost


    for v_pair in vertice_pairs:
        v_dis_in_each_layer_dict = {}
        v1 = v_pair[0]
        v2 = v_pair[1]

        v1_deg_dict = degreeList[v1]  # 结点v1的所有层的度序列, 是一个dict； key: layer, val: degree sequence
        v2_deg_dict = degreeList[v2]  # 结点v2的所有层的度序列

        num_of_deg1_hops = len(v1_deg_dict)  # 邻域层数
        num_of_deg2_hops = len(v2_deg_dict)

        max_layer = min(num_of_deg1_hops, num_of_deg2_hops)  # 此处取两个节点共同的【最大层】，因为有些边缘节点的层数小于until_layer

        dis_sum = 0
        for layer in range(0, max_layer):
            # 利用DTW 求 v1 和 v2 在第layer层的度序列距离dist
            dist, path = fastdtw(v1_deg_dict[layer], v2_deg_dict[layer], radius=1, dist=dist_func)  # 单层距离

            # dis_sum += math.pow(alpha, layer) * dist  # 累加前面各层的距离，得到当前节点对之间的总距离
            v_dis_in_each_layer_dict[layer] = dist  # 存储当前节点对在各个层的单层距离

        distances[(v1, v2)] = v_dis_in_each_layer_dict

        # distances[(v1, v2)] = dis_sum  # 节点对在前layer层构成的邻域内的总结构距离

        # print((v1, v2), ", dis:", dis_sum)

        # for i in range(max_layer):
        #     dis = dist_func(v1_deg_dict[i], v2_deg_dict[i])  # 每一层元素只有1个，所以直接调用距离函数
        #     dis_sum += math.pow(alpha, i) * dis  # 乘以权重衰减因子

    # preprocess_consolides_distances(distances)  # 对每层距离进行逐层结果合并，得到各节点对在前maxhop层的【累加结构距离】

    save_on_disk(distances, filename + '-distances-' + str(part))  # 分块内的节点对之间的结构距离计算完毕

    return


''' 旧版本做法，邻居只考虑学生节点，跨过了中间其他类型节点 '''
def getDegreeLists(G_d_other, G_other_d, until_layer, root):
    '''
    获取结点root在图G中的前until_layer层的度序列

    args:
        G_d_other: 图G, key: d, value: other
        G_other_d: 图G, key: other, value: d
        root: 目标结点
    return:
        deg_seq_dict: 目标节点root在前until_layer层邻域内的度序列
    '''
    deg_seq_dict = {}
    # vector_access = [0] * (max(G_d_other) + 1)          # 初始化list, 初始化值为0, 长度为(max(g) + 1)
    vector_access = defaultdict(int)

    node_queue = deque()     # 存储将要访问并计算其度序列的【结点】
    node_queue.append(root)  # start scanning from the source node
    vector_access[root] = 1     # 标记一个结点是否已经被访问，1：已访问，0：没访问。最开始的起始结点被访问，标记为1

    deg_seq_queue = deque()         # 存储某一层的节点【度序列】

    depth = 0           # 表示结点的第几层邻域，最开始从第0层开始，表示自身
    pendingDepthIncrease = 0   #
    timeToDepthIncrease = 1    # node_queue队列中剩余的结点数，初始状态下只有起始节点


    '''每次while循环是针对node_queue中的一个节点，计算其在until_layer层之前的各层的度，得到一个度序列'''
    while node_queue:
        vertex = node_queue.popleft()  # get a node, which is going to be scanned and get the deg sequence
        timeToDepthIncrease -= 1

        deg_seq_queue.append(len(G_d_other[vertex]))     # len(G_d_other[vertex]): 学生关联的其他类型元素个数，也就是顶点的【度】，将其存储到度序列队列中
        # l.append(str(len(G_d_other[vertex])))          # 将顶点的 度(字符串) 存储到l队列中

        # 以vertex为中心, 存储vertex的邻域
        for other_id in G_d_other[vertex]:               # other_id 为与vertex学生关联的其他类型结点的编号
            for indirect_did in G_other_d[other_id]:     # indirect_did 为与其他类型结点other_id相关联的学生编号
                if(vector_access[indirect_did] == 0):    # 若该学生编号尚未被访问过
                    vector_access[indirect_did] = 1      # 标记为已访问
                    node_queue.append(indirect_did)      # 添加到待访问队列中
                    pendingDepthIncrease += 1


        # 队列中还有节点时，会继续访问并计算节点的度序列
        # 队列中无结点时, 表示该层结点度数已经计算完成, 存储度序列
        if timeToDepthIncrease == 0:
            lp = np.array(deg_seq_queue, dtype='float')  # lp存储节点在当前层的度序列
            lp = np.sort(lp)         # 按照从小到大顺序排列
            # lp = sorted(l)
            deg_seq_dict[depth] = lp       # 存储第depth层的度序列
            deg_seq_queue = deque()  # 度序列队列清空

            # OPT3 优化
            if (until_layer == depth):
                break

            depth += 1   # 向外扩展到下一层
            timeToDepthIncrease = pendingDepthIncrease
            pendingDepthIncrease = 0

    return deg_seq_dict  # 目标节点root在前until_layer层邻域内的度序列


''' 旧方法中，计算结构距离 '''
def calc_distances_all(vertices, list_vertices, degreeList, part, filename):
    ''' 计算vertices中的全部节点对儿之间的结构距离。 利用DTW算法计算度序列的差值，作为两个顶点的距离 '''
    ''' 
        @:param degreeList: 度序列字典，key: d_id, val: 度序列，是一个list
        @:param vertices: 一组节点
        @:param list_vertices: vertices中的每个节点需要进行比较的节点id（在全部节点集合中）list_vertices=[[d2,d3,...], [d3,d4,..], ...]
        @:param part:     
    '''
    distances = {}  # 存储vertices中每个节点在各层的结构距离。是个二维字典。外层字典中，key为结点对。内层字典中，key为层数，val为该结点对在各层的结构距离

    cont = 0   # 待比较节点的下标

    # 定义序列距离的计算函数
    dist_func = cost

    for v1 in vertices:  # 取出一个节点
        lists_v1 = degreeList[v1]       # 结点v1的所有层的度序列

        for v2 in list_vertices[cont]:  # 依次取出需要和节点v1进行比较的节点id
            lists_v2 = degreeList[v2]       # 结点v2的所有层的度序列

            max_layer = min(len(lists_v1), len(lists_v2))  # 此处取两个节点的【最大层】，因为有些边缘节点的层数小于until_layer
            distances[v1, v2] = {}    # 存储v1和v2在各层的度序列距离

            # time0 = time.time()
            # 利用DTW算法计算度序列的差值
            for layer in range(0, max_layer):
                # 利用DTW 求 v1 和 v2 在第layer层的度序列距离dist
                dist, path = fastdtw(lists_v1[layer], lists_v2[layer], radius=1, dist=dist_func)
                
                # 利用Levenshtein距离, 求 v1 和 v2 第layer层度序列距离
                # dist = Levenshtein.distance(''.join(lists_v1[layer].astype(str)), ''.join(lists_v2[layer].astype(str)))
                # dist = Levenshtein.distance(''.join(lists_v1[layer]), ''.join(lists_v2[layer]))

                # 权重衰减因子
                alpha = 0.5
                dist = math.pow(alpha, layer) * dist

                distances[v1, v2][layer] = dist  # 节点对的在第layer层的结构距离（单层距离）
            # time1 = time.time()
            # print(time1 - time0)

        cont += 1
    
    preprocess_consolides_distances(distances)      # 对每层距离进行逐层结果合并，得到各节点对在各层的【累加结构距离】

    # save_on_disk(distances, 'distances-' + str(part))
    save_on_disk(distances, filename+'-distances-' + str(part))
    return


def preprocess_consolides_distances(distances, startLayer=1):
    ''' 旧方法：结构距离合并预处理 , 从第0层开始，直到最外层，将各层距离求和'''
    logging.info('Consolidating distances...')

    # vertices 为一个结点对, layers为结点对在每层的结构距离(是一个字典，key为层数, value为该层距离)
    for vertices, layer_dis_dict in distances.items():
        keys_layers = list(sorted(layer_dis_dict.keys()))       # 层数list，第0,1,2, ...层
        startLayer = min(len(keys_layers), startLayer)  # 开始层
        for layer in range(0, startLayer):
            keys_layers.pop(0)          # 将第一层先出队, 因为后续累加时, 第一层前面为空

        for layer in keys_layers:
            layer_dis_dict[layer] += layer_dis_dict[layer - 1]      # 从初始层数开始, 将每一层距离累加

    logging.info('Distances consolidated.')


# 距离计算函数，作为参数传入给DTW
def cost(a, b):
    ep = 0.5
    m = max(a, b) + ep
    mi = min(a, b) + ep
    return ((m / mi) - 1)


