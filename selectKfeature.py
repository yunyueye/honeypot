# -*- coding: utf-8 -*-
"""
Created on Sun May 22 22:13:20 2022

@author: Admin
"""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from collections import Counter
import math
import copy
import algorithm_packet
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.decomposition import PCA


if __name__ == "__main__":
# 读取数据
    feature_data = np.loadtxt("data/numpy_array_data_atg")
    label_data = np.loadtxt("data/numpy_label_data_atg")
    # 样本数量，特征维数
    sample_number, feature_number = feature_data.shape
    
    # 固定numpy随机数种子
    np.random.seed(2048)
    # 划分成训练集与测试集
    shuffle_index = [i for i in range(sample_number)]
    np.random.shuffle(shuffle_index)
    division_bound = int(np.floor(sample_number / 3 * 2))
    train_feature_array = feature_data[shuffle_index[0:division_bound]]
    train_label_array = label_data[shuffle_index[0:division_bound]]
    test_feature_array = feature_data[shuffle_index[division_bound:]]
    test_label_array = label_data[shuffle_index[division_bound:]]
    
    X = pd.DataFrame(train_feature_array)
    Y = pd.DataFrame(train_label_array)
    K = train_feature_array.shape[1]
    
    #PCA
    pca = PCA(n_components=K)
    pca.fit(X)
    ev = pca.explained_variance_ 
    s = np.argsort(ev) # 从小到大排序后的序号
    
    feature_index = list(reversed(s.tolist()))
    
    prior_p_0, prior_p_1, poster_p_list = algorithm_packet.computer_probability(
                                train_label_array,
                                selected_feature_index,
                                train_feature_array)
    
    
    
    
    
    
    
    
    
   