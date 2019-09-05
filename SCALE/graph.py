# -*- coding: utf-8 -*-

"""Graph utilities."""
import logging
import sys
import math
from io import open
from os import path
from time import time
from utils import *
from collections import Counter, defaultdict
from itertools import product
from itertools import combinations



"""
store nodes(2 types) into a dict:

dict = {
            v1:[u1, u2]
            v2:[u2, u3, u4]
            ...
            vn:[u1, u2, u3, ..., uk]
    
        }

"""

# filename = "data/mim_dname_pid_des_cui.xlsx"

def make_consistent(g):
    '''
    为图中每个结点的邻居结点排序
    args:
        g: 图g
    '''
    for key in g.keys():
        g[key] = list(sorted(set(g[key])))  # 对字典每个键对应的值（是一个list）按从小到大排序
    
    return g


def load_edge_file(filename):

    d_other_dict = defaultdict(list)  # student-other dict; key: student_id, val: related list
    other_d_dict = defaultdict(list)  # other-student

    data = load_csv_file(filename)    # 打开csv文件
    # data = xlrd.open_workbook(filename)  # 打开xls文件

    # data = load_from_disk(filename, 'xls')

    rows_num = len(data)  # 数据规模/记录总数
    print(filename, " - total rows (# of d-other relationships):", rows_num)

    column1 = data.columns[0]  # 第0列的属性名称
    column2 = data.columns[1]  # 第1列的属性名称

    for row in range(len(data)):
        mimnumber = data[column1][row]     # 学生编号
        other = data[column2][row]         # 其他编号

        d_other_dict[mimnumber].append(other)   # 学生_其他_字典
        other_d_dict[other].append(mimnumber)   # 其他_学生_字典

    print(filename, "- d_other_dict length (# of students): ", len(d_other_dict))
    print(filename, "- other_d_dict length (# of other student-related objs)：", len(other_d_dict))

    # 输出为.pkl文件
    save_on_disk(d_other_dict, filename+'_d_other_dict')
    save_on_disk(other_d_dict, filename+'_other_d_dict')

    vertices = list(d_other_dict.keys())  # save student_id into the list
    vertices = sorted(vertices)     # 按照学生编号排序
    save_on_disk(vertices, filename + '_vertices')   # 保存顶点

    # other_name = filename.split('_')[1]
    # save_on_disk(d_other_dict, 'd_' + other_name + '_dict')
    # save_on_disk(other_d_dict, other_name + '_d_dict')

    return d_other_dict     # 以student_id 作为 key 的字典

