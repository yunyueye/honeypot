# -*- coding: utf-8 -*-
"""

@author: ztong

"""

import numpy as np
import pandas as pd
from collections import Counter
import math


# 计算标签数据熵
def compute_label_entropy(train_label_array):
    # 标签熵计算  标签熵不变 始终一个值
    positive_example_index = np.where(train_label_array == 1.0)[0]
    c_1_number = positive_example_index.shape[0]
    negative_example_index = np.where(train_label_array == 0.0)[0]
    c_0_number = negative_example_index.shape[0]
    d_number = train_label_array.shape[0]
    h_d = -1 * (
        (c_1_number / d_number) * np.log(c_1_number / d_number) 
        + (c_0_number / d_number) * np.log(c_0_number / d_number)
        )
    return h_d, positive_example_index, negative_example_index


# 统计特征数量 Dik
def compute_sample_number(train_feature_array, train_label_array, used_index):
    # 特征集合
    feature_set = []
    positive_feature_count_list = []
    negative_feature_count_list = []
    for feature_index in range(train_feature_array.shape[0]):
        # 新的特征向量
        if list(train_feature_array[feature_index, [used_index]][0]) not in feature_set:
            feature_set.append(list(train_feature_array[feature_index, [used_index]][0]))
            label_ = train_label_array[feature_index]
            if label_ == 1.0:
                positive_feature_count_list.append(1)
                negative_feature_count_list.append(0)
            else:
                positive_feature_count_list.append(0)
                negative_feature_count_list.append(1)
        # 已有特征向量
        else:
            label_ = train_label_array[feature_index]
            count_index = feature_set.index(list(train_feature_array[feature_index, [used_index]][0]))
            if label_ == 1.0:
                positive_feature_count_list[count_index] += 1
            else:
                negative_feature_count_list[count_index] += 1
    return negative_feature_count_list, positive_feature_count_list, feature_set, 


# 计算特征数据的条件熵
def compute_condition_entropy(feature_count_list, d_):
    # 不同特征向量数量  d_ : 样本数量
    feature_combin_number = len(feature_count_list[0])
    k = 2
    neg_h_d_a = 0
    for n_ in range(feature_combin_number):
        d_i = feature_count_list[0][n_] + feature_count_list[1][n_]
        inner = 0
        print(inner)
        for k_ in range(k):
            inner = inner + ((feature_count_list[k_][n_] + 1) / (d_i + 1)) * np.log(
                (feature_count_list[k_][n_] + 1) / (d_i + 1))
        print(inner)
        neg_h_d_a = neg_h_d_a + d_i / d_ * inner
    h_d_a = -1 * neg_h_d_a
    return h_d_a
    
    
if __name__ == "__main__":
    # 读取数据
    feature_data = np.loadtxt("numpy_array_data_no_OSaccuracy")
    label_data = np.loadtxt("numpy_label_data")
    
    # 固定numpy随机数种子
    np.random.seed(1024)
    # 划分成训练集与测试集
    shuffle_index = [i for i in range(2000)]
    np.random.shuffle(shuffle_index)
    train_feature_array = feature_data[shuffle_index[0:1400]]
    train_label_array = label_data[shuffle_index[0:1400]]
    test_feature_array = feature_data[shuffle_index[1400:]]
    test_label_array = label_data[shuffle_index[1400:]]
    
    # label_series = pd.Series(label_data)
    
    # 计算标签数据熵值
    h_d, positive_example_index, negative_example_index = compute_label_entropy(train_label_array)
    
    
    """
    计算条件熵   逐次增加特征维度
    """
    # 所有特征列表 0 - 14
    all_feature_index = [i for i in range(14)]
    # 使用过的特征索引列表
    selected_feature_index = []
    # 样本数量
    d_ = train_feature_array.shape[0]
    # # 特征集合
    # used_index = [1, 2]
    # # 统计特征数据数量
    # feature_count_list = compute_sample_number(
    #     train_feature_array, 
    #     train_label_array, 
    #     used_index)
    # # 计算该特征集合中特征的条件熵
    # conditional_entropy = compute_condition_entropy(feature_count_list, d_)
    
    
    
    
    
    
    
    
    
    
    # feature_set = []
    # positive_feature_count_list = []
    # negative_feature_count_list = []
    # for feature_index in range(train_feature_array.shape[0]):
    #     # 假设现在的特征选择1,2
    #     used_index = [1,2]
    #     # 新的特征向量
    #     if list(train_feature_array[feature_index, [used_index]][0]) not in feature_set:
    #         feature_set.append(list(train_feature_array[feature_index, [used_index]][0]))
    #         label_ = train_label_array[feature_index]
    #         if label_ == 1.0:
    #             positive_feature_count_list.append(1)
    #             negative_feature_count_list.append(0)
    #         else:
    #             positive_feature_count_list.append(0)
    #             negative_feature_count_list.append(1)
    #     # 已有特征向量
    #     else:
    #         label_ = train_label_array[feature_index]
    #         count_index = feature_set.index(list(train_feature_array[feature_index, [used_index]][0]))
    #         if label_ == 1.0:
    #             positive_feature_count_list[count_index] += 1
    #         else:
    #             negative_feature_count_list[count_index] += 1
    
    
    """
    #计算信息增益
    K = 2
    D = 2000
    # 以第一个特征为例
    A = feature_data[:,0]
    A_series = pd.Series(A)
    # 不同标签的序号
    positive_example_index = np.where(label_data == 1.0)[0]
    C_1_num = positive_example_index.shape[0]
    negative_example_index = np.where(label_data == 0.0)[0]
    C_0_num = negative_example_index.shape[0]
    # compute H(D)
    h_d = -1 * ((C_1_num / D) * np.log(C_1_num / D) 
                + (C_0_num / D) * np.log(C_0_num / D))
    # 划分D   求di
    d_n = A_series.value_counts()
    d_n_index = d_n.index
    # 求dik
    # d_i_1 = []
    # d_i_0 = []
    dik = [[],[]]
    for i in d_n_index:
        dik[0].append(np.intersect1d(np.where(A == i)[0], np.where(label_data == 0)[0]))
        dik[1].append(np.intersect1d(np.where(A == i)[0], np.where(label_data == 1)[0]))
    # 计算H(D|A)
    """
        





























