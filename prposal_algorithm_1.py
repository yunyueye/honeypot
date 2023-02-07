# -*- coding: utf-8 -*-
"""

@author: ztong

"""

import numpy as np
import pandas as pd
from collections import Counter
import math
import copy
import algorithm_packet
from sklearn.tree import DecisionTreeClassifier


   
    

if __name__ == "__main__":
    # 读取数据
    feature_data = np.loadtxt("data/numpy_array_data_atg")
    label_data = np.loadtxt("data/numpy_label_data_atg")
    # 样本数量，特征维数
    sample_number, feature_number = feature_data.shape
    
    
    
    # 固定numpy随机数种子
    # np.random.seed(1024)
    # 划分成训练集与测试集
    shuffle_index = [i for i in range(sample_number)]
    np.random.shuffle(shuffle_index)
    division_bound = int(np.floor(sample_number / 3 * 2))
    train_feature_array = feature_data[shuffle_index[0:division_bound]]
    train_label_array = label_data[shuffle_index[0:division_bound]]
    test_feature_array = feature_data[shuffle_index[division_bound:]]
    test_label_array = label_data[shuffle_index[division_bound:]]
    
    # label_series = pd.Series(label_data)
    
    # 计算标签数据熵值
    h_d, positive_example_index, negative_example_index = algorithm_packet.compute_label_entropy(
                                                                        train_label_array)   
    
    """
    计算条件熵   逐次增加特征维度
    """
    # 所有特征列表 0 - feature_number
    all_feature_index = [i for i in range(feature_number)]
    # 使用过的特征索引列表
    selected_feature_index = []
    # 训练集样本数量
    d_ = train_feature_array.shape[0]

    # 选择第一个特征
    g_d_a = []
    for f in range(feature_number):
        used_index = [f]
        feature_count_list = algorithm_packet.compute_sample_number(
            train_feature_array, 
            train_label_array, 
            used_index)
        h_d_a = algorithm_packet.compute_condition_entropy(feature_count_list, d_)
        g_d_a.append(h_d - h_d_a)
        # print(used_index)
    # print(g_d_a)
    selected_feature_index = [g_d_a.index(max(g_d_a))]   # 此处序号即为特征索引
    
    used_index = copy.deepcopy(selected_feature_index)
    # 计算其他的特征 g_d_a
    for f in range(feature_number - 1):
        unused_feature = list(set(all_feature_index) - set(selected_feature_index))
        g_d_a = []
        for f_index in unused_feature:
            used_index.append(f_index)   # 增加新的特征 
            feature_count_list = algorithm_packet.compute_sample_number(
                train_feature_array, 
                train_label_array, 
                used_index)
            h_d_a = algorithm_packet.compute_condition_entropy(feature_count_list, d_)
            g_d_a.append(h_d - h_d_a)
            # print(g_d_a)
        selected_feature_index.append(unused_feature[g_d_a.index(max(g_d_a))])
        used_index = copy.deepcopy(selected_feature_index)
    
    # print(selected_feature_index)   [0, 3, 2, 1, 4, 5]
    # 读取attention计算的特征相关性矩阵
    attention_w = np.loadtxt("aver_A_atg")
    # 使用attention权重矩阵计算selected_eature_index
    all_feature_index = [i for i in range(feature_number)]
    attention_index = [selected_feature_index[0]]
    unused_feature_list = list(set(all_feature_index) - set(attention_index))
    for i in range(feature_number-1):
        # next_feature = np.argmax(attention_w[attention_index[i]][unused_feature_list])
        # 选择最大的值的序号，如果是自身，将该值置零，选第二大的  选到了已有的，也置零
        next_feature = np.argmax(attention_w[attention_index[i]])
        if next_feature == attention_index[i] or next_feature in attention_index:
            attention_w[attention_index[i]][next_feature] = 0
            next_feature = np.argmax(attention_w[attention_index[i]])
            while next_feature in attention_index:
                attention_w[attention_index[i]][next_feature] = 0
                next_feature = np.argmax(attention_w[attention_index[i]])
        attention_index.append(next_feature)
        unused_feature_list = list(set(all_feature_index) - set(attention_index))
    
    
    
    # 计算laplace先验概率和贝叶斯laplace条件概率  返回p_0 p_1 feature_combin_list
    prior_p_0, prior_p_1, poster_p_list = algorithm_packet.computer_probability(
                                train_label_array,
                                # selected_feature_index,
                                attention_index,
                                train_feature_array)
    
    
    """   计算训练集等数据的效果  """
    # 计算训练集的分类数据
    train_output = [[] for i in range(len(selected_feature_index))]
    for i in range(len(selected_feature_index)):
        optimal_i = selected_feature_index[0:i+1]          # 选取optimal_i对应的特征index
        feature_data = train_feature_array[:, optimal_i]   # 提取特征数据
        feature_size = feature_data.shape
        for sample_i in range(feature_size[0]):
            f_sample = list(feature_data[sample_i])        # 当前数据特征
            for f_i in range(len(poster_p_list[i])):       # 查找特征条件概率
                if f_sample == poster_p_list[i][f_i][0]:
                    f_p_0 = poster_p_list[i][f_i][1][0]    # 该特征y=0的条件概率
                    f_p_1 = poster_p_list[i][f_i][1][1]    # 该特征y=1的条件概率
                    break
            sample_p_1 = prior_p_1 * f_p_1 / (prior_p_1 * f_p_1 + prior_p_0 * f_p_0)                           
            train_output[i].append(sample_p_1)
    # 计算训练集的各项指标
    train_output_array = np.array(train_output).T   # 输出概率转成numpy array
    train_output_array[train_output_array >= 0.5] = 1
    train_output_array[train_output_array < 0.5] = 0
    measure = [[] for i in range(len(selected_feature_index))]
    for i in range(len(selected_feature_index)):
        model_output = train_output_array[:, i]
        golden_l = train_label_array
        measure[i].append(algorithm_packet.measure(golden_l, model_output))
    # 输出训练集评价指标
    print("/*******训练集*****/")
    for m in measure:
        print(m)
    
    # 计算测试集的分类数据
    test_output = [[] for i in range(len(selected_feature_index))]
    for i in range(len(selected_feature_index)):
        optimal_i = selected_feature_index[0:i+1]
        feature_data = test_feature_array[:, optimal_i]
        feature_size = feature_data.shape
        for sample_i in range(feature_size[0]):
            f_sample = list(feature_data[sample_i])        # 当前数据特征
            for f_i in range(len(poster_p_list[i])):       # 查找特征条件概率
                if f_sample == poster_p_list[i][f_i][0]:
                    f_p_0 = poster_p_list[i][f_i][1][0]    # 该特征y=0的条件概率
                    f_p_1 = poster_p_list[i][f_i][1][1]    # 该特征y=1的条件概率
                    break
            sample_p_1 = prior_p_1 * f_p_1 / (prior_p_1 * f_p_1 + prior_p_0 * f_p_0)                           
            test_output[i].append(sample_p_1)
    # 计算测试集的各项指标
    test_output_array = np.array(test_output).T   # 输出概率转成numpy array
    test_output_array[test_output_array >= 0.6] = 1
    test_output_array[test_output_array < 0.6] = 0
    test_measure = [[] for i in range(len(selected_feature_index))]
    for i in range(len(selected_feature_index)):
        model_output = test_output_array[:, i]
        golden_l = train_label_array
        test_measure[i].append(algorithm_packet.measure(golden_l, model_output))
    # 输出测试集评价指标
    print("/*******测试集*****/")
    for m in test_measure:
        print(m)
    
    # 计算准确率
    accuracy = [[] for i in range(len(selected_feature_index))]
    for i in range(train_output_array.shape[1]):
        correct = 0
        for sample in range(train_output_array.shape[0]):
            if train_output_array[sample, i] == train_label_array[sample]:
                correct = correct + 1
        acc = correct / train_label_array.shape[0]
        accuracy[i].append(acc)
    print("---各轮次准确率---")
    for ac in accuracy:
        print(ac)
        
        
    """
        具体的迭代式识别效果测试
    """
    # 新的测试  训练集  将识别成功的不再放入识别集合中  只识别以前未识别成功的
    recongnition_log, neg_sample, model_out = algorithm_packet.iteration_computer(
                                            selected_feature_index, train_feature_array, 
                                            poster_p_list, prior_p_0, prior_p_1)
    # 计算迭代识别的评估指标
    print("---------新评估-----------")
    measure_log = []    # 记录结果
    old_iteration_sample = []     # 用于记录第i-1轮中的样本  初始化为空
    for i in range(len(selected_feature_index)):
        i_sample_index = ([recongnition_log[i][j][0] for j in range(len(recongnition_log[i]))]
                          + old_iteration_sample)
        i_sample_index.sort()
        identification = np.zeros(len(train_label_array))
        identification[i_sample_index] = 1
        m = algorithm_packet.measure(train_label_array, identification)
        print(m)
        old_iteration_sample = i_sample_index
        measure_log.append(m)
    
    # 测试集的效果
    print("--------------测试集---------------")
    test_recongnition, test_neg_sample, test_model = algorithm_packet.iteration_computer(
                                                    selected_feature_index, test_feature_array,
                                                    poster_p_list, prior_p_0, prior_p_1)
    
    test_meas = algorithm_packet.iteration_measure(
                        selected_feature_index, test_recongnition, 
                        test_label_array)
    
    
    
    

    # attention
    # tp fn fp tn 45 11 0 60
    # ((45, 11, 0, 60), (1.0, 0.8035714285714286, 0.8910891089108911), (105, 0.9051724137931034))
    # tp fn fp tn 45 11 0 60
    # ((45, 11, 0, 60), (1.0, 0.8035714285714286, 0.8910891089108911), (105, 0.9051724137931034))
    # tp fn fp tn 45 11 0 60
    # ((45, 11, 0, 60), (1.0, 0.8035714285714286, 0.8910891089108911), (105, 0.9051724137931034))
    # tp fn fp tn 45 11 0 60
    # ((45, 11, 0, 60), (1.0, 0.8035714285714286, 0.8910891089108911), (105, 0.9051724137931034))
    # tp fn fp tn 45 11 0 60
    # ((45, 11, 0, 60), (1.0, 0.8035714285714286, 0.8910891089108911), (105, 0.9051724137931034))
    # tp fn fp tn 45 11 0 60
    # ((45, 11, 0, 60), (1.0, 0.8035714285714286, 0.8910891089108911), (105, 0.9051724137931034))


    
    
    
    """
    对剩余的样本使用决策树模型
    """
    # print("----------dtc----------")
    # # 训练模型
    # dtc = DecisionTreeClassifier()
    # dtc.fit(train_feature_array, train_label_array)
    # # y_predict = dtc.predict(test_feature_array)
    # # print(algorithm_packet.measure(test_label_array, y_predict)) 0.8864~=0.89
    # neg_i = [i[0] for i in test_neg_sample]  # 取出未识别的样本
    # neg_feature = test_feature_array[neg_i]  # 未识别样本特征
    # neg_out_dtc = dtc.predict(neg_feature)   # 识别iteration未识别的样本
    # pos_i = set([i for i in range(test_label_array.shape[0])]) - set(neg_i) # iteration识别样本
    # all_pos = list(pos_i) + list(np.array(neg_i)[np.where(neg_out_dtc==1)[0]])
    # all_pos.sort()
    # final_predict = np.zeros(test_label_array.shape[0])
    # final_predict[all_pos] = 1
    # final_m = algorithm_packet.measure(test_label_array, final_predict)
    # print(final_m)
    
    
    
    
    # test_measure_log = []
    # old_sample = []
    # for i in range(len(selected_feature_index)):
    #     i_sample_index = ([test_recongnition[i][j][0] for j in range(len(test_recongnition[i]))]
    #                       + old_sample)
    #     i_sample_index.sort()
    #     identification = np.zeros(len(test_label_array))
    #     identification[i_sample_index] = 1
    #     m_t = algorithm_packet.measure(test_label_array, identification)
    #     print(m_t)
    #     old_sample = i_sample_index
    #     test_measure_log.append(m_t)
        
        
        
   # model_out = [[] for i in range(len(selected_feature_index))]
   #  recongnition_log = [[] for i in range(len(selected_feature_index))]   # 记录识别情况
   #  sample_index_list = [i for i in range(train_feature_array.shape[0])]  # 每次计算的样本特征的序号 初始化为所有样本
   #  for i in range(len(selected_feature_index)):
   #      # print(i)
   #      optimal_i = selected_feature_index[0:i+1]
   #      feature_data = train_feature_array[sample_index_list, :][:, optimal_i]
   #      feature_size = feature_data.shape
   #      for sample_i in range(feature_size[0]):            # 求每个样本的后验概率
   #          f_sample = list(feature_data[sample_i])
   #          for f_i in range(len(poster_p_list[i])):
   #              if f_sample == poster_p_list[i][f_i][0]:   # 在特征集合中找对应特征
   #                  f_p_0 = poster_p_list[i][f_i][1][0]    # 该特征y=0的条件概率
   #                  f_p_1 = poster_p_list[i][f_i][1][1]    # 该特征y=1的条件概率
   #                  break
   #          sample_p_1 = prior_p_1 * f_p_1 / (prior_p_1 * f_p_1 + prior_p_0 * f_p_0)   
   #          model_out[i].append(sample_p_1)
   #          # break
   #      # break
   #      # 筛选小于阈值的样本  H = 0.9  最后一次直接输出对应结果
   #      if i != len(selected_feature_index) - 1:
   #          next_sample_index = []    # 下次计算样本序号
   #          H = 0.8                   # 阈值
   #          for j in range(len(model_out[i])):
   #              if model_out[i][j] >= H:
   #                  sample_log_tuple = (sample_index_list[j], model_out[i][j], i)   # 单个样本的输出情况， 样本序号， 概率， 识别次数
   #                  recongnition_log[i].append(sample_log_tuple)
   #              else:
   #                  next_sample_index.append(sample_index_list[j])
   #          sample_index_list = next_sample_index   # 更新本次样本序号
   #      else:
   #          neg_sample = []
   #          H = 0.9
   #          for j in range(len(model_out[i])):
   #              if model_out[i][j] >= H:
   #                  sample_log_tuple = (sample_index_list[j], model_out[i][j], i)
   #                  recongnition_log[i].append(sample_log_tuple)
   #              else:
   #                  sample_log = (sample_index_list[j], model_out[i][j])
   #                  neg_sample.append(sample_log)     
                
            
        
    
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # # # 特征集合
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
        
    """
    使用贝叶斯定理计算各条件概率
    """
    # # 统计标签数量 I(yi = ck)
    # pos_label_number = np.where(train_label_array == 1.0)[0].shape[0]
    # neg_label_number = np.where(train_label_array == 0.0)[0].shape[0]
    # label_number = train_label_array.shape[0]
    # # 先验概率的贝叶斯估计
    # p_0 = (neg_label_number + 1) / (label_number + 2 * 1)
    # p_1 = (pos_label_number + 1) / (label_number + 2 * 1)
    # # 条件概率存储list  feature_probability_list
    # feature_probability_list = [[] for i in range(len(selected_feature_index))]     # 多维数组  d1:特征数量作为索引 d2:具体特征和特征条件概率
    # """
    # 计算具体特征以及其条件概率
    # """
    # for feature_used_number in range(len(selected_feature_index)):   
    #     combin_feature = selected_feature_index[0:feature_used_number+1]    # 本次使用特征索引list
    #     # 求得本次使用特征的各种统计量
    #     feature_statistics = algorithm_packet.compute_sample_number(
    #                             train_feature_array,
    #                             train_label_array,
    #                             combin_feature)
    #     this_feature_range = len(feature_statistics[0])   # 该项特征的取值数量   Sj
    #     # 计算条件概率的贝叶斯估计
    #     feature_store_list = []
    #     for feature_index in range(this_feature_range):
    #         # lamda = 1
    #         condition_p_0 = (feature_statistics[0][feature_index] + 1) / (
    #                             neg_label_number + this_feature_range * 1)
    #         condition_p_1 = (feature_statistics[1][feature_index] + 1) / (
    #                             pos_label_number + this_feature_range * 1)
    #         print("condition_p_0:", condition_p_0)
    #         print("condition_p_1:", condition_p_1)
    #         print("--------")
    #         store_params = (feature_statistics[2][feature_index], (condition_p_0, condition_p_1))
    #         feature_store_list.append(store_params)
    #     feature_probability_list[feature_used_number] = feature_store_list




    """
    生成测试数据
    
    output_label = np.ones(800)
    output_zeros = np.zeros(200)
    output = np.hstack((output_label, output_zeros))
    np.random.shuffle(output)
    
    label = np.ones(850)
    zeros = np.zeros(150)
    l = np.hstack((label, zeros))
    np.random.shuffle(l)
    
    
    tp = 0
    for i in range(output.shape[0]):
        if l[i] == 1 and output[i] == 1:
            tp = tp + 1
    fn = 0
    for i in range(output.shape[0]):
        if l[i] == 1 and output[i] == 0:
            fn = fn + 1
    fp = 0
    for i in range(output.shape[0]):
        if l[i] == 0 and output[i] == 1:
            fp = fp + 1 
    tn = 0 
    for i in range(output.shape[0]):
        if l[i] == 0 and output[i] == 0:
            tn = tn + 1
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * tp / (2 * tp + fp + fn)
    print(precision, recall, f1)

    """





















