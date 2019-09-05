# encoding: utf-8
from collections import deque
import numpy as np
import math, random, logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from collections import defaultdict
from time import time
from utils import *


def generate_similarity_network(file_list, workers):
    '''
        construct the multi-layered SSN by associating every student node with its counterparts in other SSNs.
    '''
    t0 = time.time()
    logging.info('Creating similarity network...')

    # 生成每层结点的相似度网络, 每层单独存储, 并构建多层网络
    os.system("rm " + get_sim_path() + "/save/weighted_multi_ssn-layer-*.pkl")
    os.system("rm " + get_sim_path() + "/save/graphs-layer-*.pkl")

    with ProcessPoolExecutor(max_workers=1) as executor:
        job = executor.submit(generate_similarity_network_part1, file_list)  # 将多个SSN进行关联，构建多层SSN，各层之间尚无关联
        job.result()
    t1 = time.time()
    t = t1 - t0

    logging.info('- Time - part 1: {}s'.format(t))


    t0 = time.time()
    os.system("rm " + get_sim_path() + "/save/similarity_nets_weights-layer-*.pkl")
    os.system("rm " + get_sim_path() + "/save/alias_method_j-layer-*.pkl")
    os.system("rm " + get_sim_path() + "/save/alias_method_q-layer-*.pkl")
    with ProcessPoolExecutor(max_workers=1) as executor:
        job = executor.submit(generate_similarity_network_part3)   # 关联
        job.result()
    t1 = time.time()
    t = t1 - t0
    logging.info('- Time - part 3: {}s'.format(t))


    # 合并多层网络
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=1) as executor:
        job = executor.submit(generate_similarity_network_part4)  # 合并存储
        job.result()
    t1 = time.time()
    t = t1 - t0
    logging.info('- Time - part 4: {}s'.format(t))


    t0 = time.time()
    with ProcessPoolExecutor(max_workers=1) as executor:
        job = executor.submit(generate_similarity_network_part5)
        job.result()
    t1 = time.time()
    t = t1 - t0
    logging.info('- Time - part 5: {}s'.format(t))


    t0 = time.time()
    with ProcessPoolExecutor(max_workers=1) as executor:
        job = executor.submit(generate_similarity_network_part6)
        job.result()
    t1 = time.time()
    t = t1 - t0
    logging.info('- Time - part 6: {}s'.format(t))

    return


def generate_similarity_network_part1(file_list):
    '''
        构建2个多层SSN，一个存储边权，一个存储邻接信息
        各层之间相同节点尚无边进行关联
        按层存储每个结点间相似度，一层代表由一个数据源计算得到的SSN
        针对每个SSN, 先将其添加到多层SSN中，然后初始化各边的权值（weighted_multi_SSN[layer]），以及节点之间的互连情况（graphs[layer]）
    '''
    # parts = workers
    weighted_multi_ssn = {}   # 多层带权SSN，记录权值。 key：layer, 节点对，val:相似度值
    graphs = {}   # 多层SSN中每一层的网络，是一个邻接矩阵，记录结点互连情况

    for layer in range(0, len(file_list)):      # 依次取出一个SSN

        print('generate_multi_SSN_layer_', layer, '...')
        ssn = load_from_disk(file_list[layer])  # 根据文件名读取存储的相似度文件

        # 初始化该层相似度存储字典
        if (layer not in weighted_multi_ssn):
            weighted_multi_ssn[layer] = {}    # 将当前的SSN添加到多层网络中，从下往上进行构建

        # 初始化每一层的图
        if (layer not in graphs):
            graphs[layer] = {}

        for vertices, value in ssn.items():   # ssn是一个字典，key：学生对，val: 相似度值
            vx = vertices[0]    # d_id_1
            vy = vertices[1]    # d_id_2
            weighted_multi_ssn[layer][vx, vy] = value       # 在多层SSN中的对应层的网络存储相似度，给边赋权

            if (vx not in graphs[layer]):     # 记录这个网络的邻接信息
                graphs[layer][vx] = []
            if (vy not in graphs[layer]):
                graphs[layer][vy] = []
            graphs[layer][vx].append(vy)      # 将连接的结点添加进图
            graphs[layer][vy].append(vx)

        logging.info('Layer {} executed.'.format(layer))
    time_start =  time.time()
    for layer, values in weighted_multi_ssn.items():
        save_on_disk(values, 'weighted_multi_ssn-layer-' + str(layer))   # 将多层SSN按照各层进行存储. 此时不同层之间的节点无边连接。有多少层/SSN, 就存多少个临时文件
    for layer, values in graphs.items():
        save_on_disk(values, 'graphs-layer-' + str(layer))  # 按层进行存储(邻接信息，可视为邻接矩阵)
    time_end = time.time()
    return


def part3(layer):
    graphs = load_from_disk('graphs-layer-' + str(layer))  # 节点之间的互连状况，无权值
    weights_similarity = load_from_disk('weighted_multi_ssn-layer-' + str(layer))  # 边权

    logging.info('Executing layer {}...'.format(layer))
    alias_method_j = {}  # 针对节点v存储的一个数（随机游走时会使用到）
    alias_method_q = {}  # 针对节点v存储的一个数（随机游走时会使用到）
    weights = {}

    # 根据边权进行离散非均匀采样
    for v, neighbors in graphs.items():  # key: layer,  value: 在layer层网络中的学生邻接信息 eg. {'d1': ['d2', 'd3', 'd7'], 'd2': ['d1'], 'd3': ['d1'], 'd7': ['d1']}
        e_list = deque()  # 当前节点v的边权队列
        sum_w = 0.0

        # 同层边赋予权重
        for n in neighbors:  # neighbors: SSN中包含的全部学生节点；n: 单个节点
            if (v, n) in weights_similarity:
                w = weights_similarity[v, n]  # 获得当前节点个每个邻居之间的权值
            else:
                w = weights_similarity[n, v]  # 获得当前节点个每个邻居之间的权值
            # w = np.exp(-float(wd))
            e_list.append(w)
            sum_w += w  # 边权之和

        avg_w = sum_w / len(e_list)  # 节点v在这一层网络中的平均边权

        i = 0
        j = 0
        length = len(e_list)  # 邻居个数
        while i < length:
            if e_list[j] < avg_w:
                e_list.remove(e_list[j])  # 去掉边权小于平均值的边
                graphs[v].pop(j)  # 将边权小于平均值的邻居剔除
                j -= 1
            else:
                if sum_w != 0:
                    e_list[j] = e_list[j] / sum_w  # 将边权更新为：原来的边权占总边权的比重
            i += 1
            j += 1

        if len(e_list) != len(graphs[v]):  # 一定会剔除掉一部分节点（边权小于平均值的）
            print(False)
        # e_list = [x / sum_w for x in e_list]
        weights[v] = e_list
        J, q = alias_setup(e_list)  # 根据新的边权列表（各个元素之和为1），进行离散非均匀采样
        alias_method_j[v] = J  # 针对节点v存储的一个数（随机游走时会使用到）
        alias_method_q[v] = q  # 针对节点v存储的一个数（随机游走时会使用到）

    os.system("rm " + get_sim_path() + "/save/graphs-layer-" + str(layer) + ".pkl")
    save_on_disk(graphs, 'graphs-layer-' + str(layer))

    save_on_disk(weights, 'similarity_nets_weights-layer-' + str(layer))
    save_on_disk(alias_method_j, 'alias_method_j-layer-' + str(layer))
    save_on_disk(alias_method_q, 'alias_method_q-layer-' + str(layer))
    logging.info('Layer {} executed.'.format(layer))

def generate_similarity_network_part3():
    '''同层之间权重赋值。针对各层的邻接信息和边权，得到 alias_method_j 和 alias_method_q '''
    layer = 0
    with ProcessPoolExecutor(max_workers=10) as executor:
        future = []
        while is_pkl('graphs-layer-' + str(layer)):  # 每次从多层SSN中取出一个SSN
            temp = executor.submit(part3, layer)
            future.append(temp)
            layer += 1

    logging.info('Weights created.')

    return


def generate_similarity_network_part4():
    '''将多层网络（各层图中的节点间的邻接信息）存储到一起，构建多层邻接图'''
    logging.info('Consolidating graphs...')
    graphs_c = {}    # 多层SSN
    layer = 0
    while (is_pkl('graphs-layer-' + str(layer))):
        logging.info('Executing layer {}...'.format(layer))
        graphs = load_from_disk('graphs-layer-' + str(layer))
        graphs_c[layer] = graphs
        logging.info('Layer {} executed.'.format(layer))
        layer += 1

    logging.info("Saving similarityNets on disk...")
    time_start = time.time()
    save_on_disk(graphs_c, 'similarity_nets_graphs')   # 整个多层SSN (每一层存储的是邻接信息)
    time_end  = time.time()

    logging.info('Graphs consolidated.')
    return



'''
    组合所有层的 alias_method_j 字典 (随机游走时使用到)
'''
def generate_similarity_network_part5():
    alias_method_j_c = {}
    layer = 0
    while (is_pkl('alias_method_j-layer-' + str(layer))):
        logging.info('Executing layer {}...'.format(layer))
        alias_method_j = load_from_disk('alias_method_j-layer-' + str(layer))
        alias_method_j_c[layer] = alias_method_j
        logging.info('Layer {} executed.'.format(layer))
        layer += 1

    logging.info("Saving nets_weights_alias_method_j on disk...")
    save_on_disk(alias_method_j_c, 'nets_weights_alias_method_j')

    return


'''
    组合所有层的alias_method_q 字典(随机游走时使用到)
'''
def generate_similarity_network_part6():
    alias_method_q_c = {}
    layer = 0
    while (is_pkl('alias_method_q-layer-' + str(layer))):
        logging.info('Executing layer {}...'.format(layer))
        alias_method_q = load_from_disk('alias_method_q-layer-' + str(layer))
        alias_method_q_c[layer] = alias_method_q
        logging.info('Layer {} executed.'.format(layer))
        layer += 1

    logging.info("Saving nets_weights_alias_method_q on disk...")
    save_on_disk(alias_method_q_c, 'nets_weights_alias_method_q')

    return





def generate_parameters_random_walk():

    logging.info('Loading similarity_nets from disk...')

    sum_weights = {}
    amount_edges = {}

    layer = 0
    while (is_pkl('similarity_nets_weights-layer-' + str(layer))):
        logging.info('Executing layer {}...'.format(layer))
        weights = load_from_disk('similarity_nets_weights-layer-' + str(layer))

        for k, list_weights in weights.items():
            if (layer not in sum_weights):
                sum_weights[layer] = 0
            if (layer not in amount_edges):
                amount_edges[layer] = 0

            for w in list_weights:
                sum_weights[layer] += w
                amount_edges[layer] += 1

        logging.info('Layer {} executed.'.format(layer))
        layer += 1

    average_weight = {}
    for layer in list(sum_weights.keys()):
        average_weight[layer] = sum_weights[layer] / amount_edges[layer]

    logging.info("Saving average_weights on disk...")
    save_on_disk(average_weight, 'average_weight')

    amount_neighbours = {}

    layer = 0
    while (is_pkl('similarity_nets_weights-layer-' + str(layer))):
        logging.info('Executing layer {}...'.format(layer))
        weights = load_from_disk('similarity_nets_weights-layer-' + str(layer))

        amount_neighbours[layer] = {}

        for k, list_weights in weights.items():
            cont_neighbours = 0
            for w in list_weights:
                if (w > average_weight[layer]):
                    cont_neighbours += 1
            amount_neighbours[layer][k] = cont_neighbours

        logging.info('Layer {} executed.'.format(layer))
        layer += 1

    logging.info("Saving amount_neighbours on disk...")
    save_on_disk(amount_neighbours, 'amount_neighbours')


def generate_random_walks_large_graphs(num_walks, walk_length, workers, vertices):

    logging.info('Loading similarity_nets from disk...')

    graphs = load_from_disk('similarity_nets_graphs')
    alias_method_j = load_from_disk('nets_weights_alias_method_j')
    alias_method_q = load_from_disk('nets_weights_alias_method_q')
    amount_neighbours = load_from_disk('amount_neighbours')

    logging.info('Creating RWs...')
    t0 = time.time()

    walks = deque()
    initialLayer = 0

    parts = workers

    with ProcessPoolExecutor(max_workers = workers) as executor:

        for walk_iter in range(num_walks):
            print('-- walk_iteration:', walk_iter)
            random.shuffle(vertices)
            logging.info("Execution iteration {} ...".format(walk_iter))
            # walk = exec_random_walks_for_chunck(vertices, graphs, alias_method_j, alias_method_q, walk_length, amount_neighbours)
            walk = exec_random_walks_for_chunck(vertices, graphs, alias_method_j, alias_method_q, walk_length)
            walks.extend(walk)
            logging.info("Iteration {} executed.".format(walk_iter))

    t1 = time.time()
    logging.info('RWs created. Time : {}m'.format((t1 - t0) / 60))
    logging.info("Saving Random Walks on disk...")
    save_random_walks(walks)


''' 获取各个节点在所有SSN中取得最大权值所在的网络层数 '''
def get_max_sim_layer(vertices, numOfNetworks, flag):
    '''
    :param vertices:  全部节点
    :param numOfNetworks:  SSN个数
    :param flag: 选层策略 -- 1: 最大相似度 2: 最大中位数
    :return: 最大边权所在层数
    '''

    if load_from_disk('d_max_sim_layer_dict') != None:
        print("[d_max_sim_layer_dict] already exists! Load directly!")
        d_max_sim_layer_dict = load_from_disk('d_max_sim_layer_dict')
        return d_max_sim_layer_dict

    time1 = time.time()
    print("Calculating the d_max_sim_layer_dict for", numOfNetworks, "SSNs...")
    print("# of vertices:", len(vertices))

    weighted_SSN = {}

    for l in range(numOfNetworks):
        weighted_DSN[l] = load_from_disk('weighted_multi_ssn-layer-' + str(l))
        print("# of dpairs in DSN:", l, "is:", len(weighted_DSN[l]))

    d_max_sim_layer_dict = {}   # 各个节点在所有DSN中取得最大权值所在的网络层数

    if flag == 1:
        print("[Strategy: max sim]...")
        logging.info('Creating d_max_sim_layer_dict [Strategy: max sim] ...')
        for ver in vertices:
            max_sim = -1
            max_sim_layer = -99
            print("processing ver", ver)
            for layer in range(numOfNetworks):
                for dpair in weighted_DSN[layer].keys():
                    if ver in dpair:
                        # temp[dpair] = weighted_DSN[layer][dpair]
                        sim = weighted_DSN[layer][dpair]
                        if sim > max_sim:
                            max_sim = sim
                            max_sim_layer = layer
            d_max_sim_layer_dict[ver] = max_sim_layer
            time2 = time.time()
            save_on_disk(d_max_sim_layer_dict, "d_max_sim_layer_dict")

    elif flag == 2:
        print("max median...")
        logging.info('Creating d_max_sim_layer_dict [max median] ...')
        for ver in vertices:
            max_median = -1
            max_median_layer = -99
            sim_list = []
            print("median - processing ver", ver)
            for layer in range(numOfNetworks):
                for dpair in weighted_DSN[layer].keys():
                    if ver in dpair:
                        sim_list.append(weighted_DSN[layer][dpair])
                median = np.median(sim_list)  # 节点ver在这一层的相似度大小的中位数
                if median > max_median:
                    max_median = median
                    max_median_layer = layer
            d_max_sim_layer_dict[ver] = max_median_layer
            time2 = time.time()
            save_on_disk(d_max_sim_layer_dict, "d_max_sim_layer_dict")

    print("Generating d_max_sim_layer_dict:", len(d_max_sim_layer_dict), ", cost time:", time2 - time1)

    return d_max_sim_layer_dict


def generate_random_walks(num_walks, walk_length, workers, vertices):
    '''
       随机游走
       @:param num_walks: 重复游走的次数
       @:param walk_length: 单次游走的序列长度
       @:param vertices: 全部学生节点id
   '''

    logging.info('Loading similarity_nets on disk...')
    graphs = load_from_disk('similarity_nets_graphs')  # 多层DSN
    alias_method_j = load_from_disk('nets_weights_alias_method_j')  # 随机游走需要使用到的字典
    alias_method_q = load_from_disk('nets_weights_alias_method_q')  # 随机游走需要使用到的字典
    # amount_neighbours = load_from_disk('amount_neighbours')

    flag = 1  # 1: max， 2：max(median)  3: max(avg)
    # flag = 2

    '''1. 获取每个节点取得最大边权的所在网络层数'''
    d_max_sim_layer_dict = {}
    # 获取各个节点在所有DSN中取得最大权值所在的网络层数
    #cui--注释
    # ''' 单层网络随机游走不需要获取 '''
    # if (len(graphs)) > 1:
    #     print("# of DSNs>1, now getting the d_max_sim_layer_dict...")
    #     d_max_sim_layer_dict = get_max_sim_layer(vertices, len(graphs), flag)
    # else:
    #     print("only single layer of DSN, skipping getting d_max_sim_layer_dict.")
        #cuiEnd

    logging.info('Creating Random Walks...')
    t0 = time.time()

    walks = deque()  # 游走序列，使用队列存储
    initialLayer = 0

    if (workers > num_walks):
        workers = num_walks

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {}

        for walk_iter in range(num_walks):  # 随机游走num_walks轮，每次对全部节点产生一组游走序列，最后将多轮游走的序列拼接在一起
            random.shuffle(vertices)  # 打乱节点
            print('-- walk_iteration:', walk_iter)
            # job = executor.submit(exec_ramdom_walks_for_chunck, vertices, graphs, alias_method_j, alias_method_q, walk_length, amount_neighbours)

            # alias_method_j 和 alias_method_q 在采样时会使用到，返回对全部节点产生的游走序列
            job = executor.submit(exec_random_walks_for_chunck, d_max_sim_layer_dict, vertices, graphs,
                                  alias_method_j, alias_method_q, walk_length)  # 随机游走
            futures[job] = walk_iter
            # part += 1
        logging.info("Receiving results...")
        for job in as_completed(futures):
            walk = job.result()
            r = futures[job]
            logging.info("Iteration {} executed.".format(r))
            walks.extend(walk)   # 将多轮游走产生的序列拼接在一起
            del futures[job]

    t1 = time.time()
    logging.info('RWs created. Time: {}m'.format((t1 - t0) / 60))
    logging.info("Saving Random Walks on disk...")
    save_random_walks(walks, num_walks, walk_length)


# def exec_ramdom_walks_for_chunck(vertices, graphs, alias_method_j, alias_method_q, walk_length, amount_neighbours):
def exec_random_walks_for_chunck(d_max_sim_layer_dict, vertices, graphs, alias_method_j, alias_method_q, walk_length):
    '''
        将vertices(已被打乱)中每个结点作为起始点, 进行随机游走
        @:param vertices: 全部学生节点
        @:param graphs: 多层DSN
        @:param walk_length: 单次游走产生的序列长度
        @:param alias_method_j, alias_method_q: 随机游走时会使用到的，是一个字典，key:d_id,  val: a number
        @:return walks: 块中所有节点的游走序列
    '''
    walks = deque()  # 全部节点的游走序列

    # # 策略3会使用到weighted_DSN
    # weighted_DSN = {}
    # for l in range(len(graphs)):
    #     weighted_DSN[l] = load_from_disk('weighted_multi_ssn-layer-' + str(l))
    #
    # d_max_sim_layer_dict = load_from_disk('d_max_sim_layer_dict')

    count = 0

    for v in vertices:  # 依次对每个节点随机游走，将产生的序列添加到walks中

        # 策略1, 2（已被注释掉），以及在【单网络，不换层，直接随机游走】
        # print("Walk for single DSN...")
        # logging.info('Walking for [single DSN]... ...')
        # walks.append(exec_random_walk_for_node(graphs, alias_method_j, alias_method_q, v, walk_length))

        # 策略3【跨网络，选择性换层】
        # cui -注释
        # print("Walk for [multi-layered DSN]...")
        logging.info('Walking for [multi-layered DSN]...')
        # cui-修改
        walks.append(exec_random_walk_for_node(graphs, alias_method_j, alias_method_q, v, walk_length))
        # cuiEnd
        count += 1

    return walks  # 块中全部节点的游走序列



''' 【游走策略3：选择最大边权最大的网络，进行随机游走】'''
def exec_random_walk_for_node_via_max_sim(graphs, alias_method_j, alias_method_q, v, walk_length, d_max_sim_layer_dict):
    '''
    针对【单个节点】，随机游走产生序列
    :param graphs:  多层DSN
    :param alias_method_j: 随机游走时会使用到的，是一个字典，key:d_id,  val: a number
    :param alias_method_q: 随机游走时会使用到的，是一个字典，key:d_id,  val: a number
    :param v: 单个节点
    :param walk_length: 单次游走产生的序列长度
    :return path: 游走路径
    '''
    t0 = time.time()
    original_v = v

    path = deque()  # 本轮游走对节点v产生的路径
    path.append(v)  # 游走最开始从自身出发

    num_graphs = len(graphs)  # DSN个数，层数

    while len(path) < walk_length:
        v_before_walk = v

        v = chooseNeighbor(v_before_walk, graphs, alias_method_j, alias_method_q, d_max_sim_layer_dict[v_before_walk])
        path.append(v)  # 添加到路径
        continue

    t1 = time.time()
    logging.info('RW - vertex {}. Time : {}s'.format(original_v, (t1 - t0)))

    return path  # 单轮游走中，针对单个节点产生的序列


# def exec_random_walk(graphs, alias_method_j, alias_method_q, v, walk_length, amount_neighbours):
def exec_random_walk_for_node(graphs, alias_method_j, alias_method_q, v, walk_length):
    '''
    针对【单个节点】，随机游走产生序列
    :param graphs:  多层DSN
    :param alias_method_j: 随机游走时会使用到的，是一个字典，key:d_id,  val: a number
    :param alias_method_q: 随机游走时会使用到的，是一个字典，key:d_id,  val: a number
    :param v: 单个节点
    :param walk_length: 单次游走产生的序列长度
    :return path: 游走路径
    '''

    original_v = v

    t0 = time.time()
    initialLayer = 0  # 最开始从第0层开始游走
    layer = initialLayer

    path = deque()  # 本轮游走对节点v产生的路径
    path.append(v)  # 游走最开始从自身出发

    num_graphs = len(graphs)  # DSN个数，层数

    prob_move = 1 / num_graphs        # 每层网络设置平均概率值（当网络个数>2）
    # prob_move = 0.3           # 游走到每层结点的概率值（层与层之间的边权）

    layer_to_go = 0   # 最开始从第0层开始游走

    count = 0

    #cui - 添加
    if num_graphs == 1:
        while len(path) < walk_length:
            # 仅运行【单层网络】(实验中，测试只使用单个数据集会用到) ，此时，注释掉上面的[for - if - else]
            # >>> 当运行多层网络随机游走时，需将以下这两行代码注释掉 , 设置 layer=0或1 <<<
            v = chooseNeighbor(v, graphs, alias_method_j, alias_method_q, layer)
            path.append(v)
    else:
        while len(path) < walk_length:
            r = random.random()    # 每次游走之前先随机产生一个数,在[0,1)范围，通过和换层概率进行比较来确定本次游走需不需要换层
            current_layer = layer_to_go  # 当前层
            if r < 0.5:  # 换层
                while current_layer == layer_to_go:
                    layer_to_go = random.randint(0, num_graphs - 1)  # 随机生成一个整数，它在[x,y]范围内
            v = chooseNeighbor(v, graphs, alias_method_j, alias_method_q, layer_to_go)  # 随机在layer层选择一个邻居（graph提供邻接信息）
            path.append(v)  # 添加到路径

        #cuiEnd


        # r = random.random()    # 每次游走之前先随机产生一个数,在[0,1)范围，通过和换层概率进行比较来确定本次游走需不需要换层
        # current_layer = layer_to_go  # 当前层

        # 【策略1：等概论换层】支持多层(>2)的随机游走
        # for l in range(num_graphs):   # l=0,1,2,3... 只要换了层就游走
        #     if r < (l + 1) * prob_move:
        #         count += 1
        #
        #         # 跳转到l层
        #         layer_to_go = l  # 即将换到的层数
        #         # 如果当前所在层就是将要换到的层数, 则在这一层随机游走，添加一个新结点
        #         if current_layer == layer_to_go:
        #             print('walk...step-', len(path))
        #             v = chooseNeighbor(v, graphs, alias_method_j, alias_method_q, layer)  # 随机在layer层选择一个邻居（graph提供邻接信息）
        #             path.append(v)  # 添加到路径
        #             break
        #         else:
        #             print('change layer...no walking...')
        #             break  # 只进行换层，不游走

        # 【策略2：等概论停留或随机换层】
        # for l in range(num_graphs):  # l=0,1,2,3... 只要换了层就游走
        #     if r < 0.5:  # 换层
        #         while current_layer == layer_to_go:
        #             layer_to_go = random.randint(0, num_graphs - 1)  # 随机生成一个整数，它在[x,y]范围内
        #         break
        #     else:       # 停留，游走一步
        #         print('walk...step-', len(path))
        #         v = chooseNeighbor(v, graphs, alias_method_j, alias_method_q, layer)  # 随机在layer层选择一个邻居（graph提供邻接信息）
        #         path.append(v)  # 添加到路径
        #         break

    t1 = time.time()
    logging.info('RW - vertex {}. Time : {}s'.format(original_v, (t1 - t0)))

    return path  # 单轮游走中，针对单个节点产生的序列


''' >>> 只在单层网络中游走（实验中的跨网络对比测试会用到）'''
def chooseNeighbor(v, graphs, alias_method_j, alias_method_q, layer):
    v_list = graphs[layer][v]  # 节点v在layer层与其他节点的连接信息
    idx = alias_draw(alias_method_j[layer][v], alias_method_q[layer][v])


    v = v_list[idx]

    return v


def prob_moveup(amount_neighbours):
    '''返回游走到k+1层的概率。 struc2vec中，向上层游走的概率计算如下'''
    x = math.log(amount_neighbours + math.e)
    p = (x / (x + 1))
    return p


def save_random_walks(walks, num_walks, walk_length):
    filename = 'walk_result_num=' + str(num_walks) + '_walk_len=' + str(walk_length) + '.txt'

    with open(filename, 'w') as file:
        for walk in walks:
            line = ''
            for v in walk:
                line += str(v) + ' '
            line += '\n'
            file.write(line)
    return