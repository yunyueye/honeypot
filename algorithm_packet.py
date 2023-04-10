# -*- coding: utf-8 -*-
"""

@author: ztong
组合特征的信息增益
"""

import numpy as np
import pandas as pd
from collections import Counter
import math
import copy


# Calculate label data entropy
def compute_label_entropy(train_label_array):
    # Calculation of label entropy. The label entropy remains constant
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


# Count the number of statistical features 
def compute_sample_number(train_feature_array, train_label_array, used_index):
    feature_set = []
    positive_feature_count_list = []
    negative_feature_count_list = []
    for feature_index in range(train_feature_array.shape[0]):
        if list(train_feature_array[feature_index, [used_index]][0]) not in feature_set:
            feature_set.append(list(train_feature_array[feature_index, [used_index]][0]))
            label_ = train_label_array[feature_index]
            if label_ == 1.0:
                positive_feature_count_list.append(1)
                negative_feature_count_list.append(0)
            else:
                positive_feature_count_list.append(0)
                negative_feature_count_list.append(1)
        else:
            label_ = train_label_array[feature_index]
            count_index = feature_set.index(list(train_feature_array[feature_index, [used_index]][0]))
            if label_ == 1.0:
                positive_feature_count_list[count_index] += 1
            else:
                negative_feature_count_list[count_index] += 1
    return negative_feature_count_list, positive_feature_count_list, feature_set


# Calculate the conditional entropy of feature data
def compute_condition_entropy(feature_count_list, d_):

    feature__number = len(feature_count_list[0])
    k = 2
    neg_h_d_a = 0
    for n_ in range(feature__number):
        d_i = feature_count_list[0][n_] + feature_count_list[1][n_]
        inner = 0
        # print(inner)
        for k_ in range(k):
            inner = inner + ((feature_count_list[k_][n_] + 1) / (d_i + 1)) * np.log(
                (feature_count_list[k_][n_] + 1) / (d_i + 1))
        # print(inner)
        neg_h_d_a = neg_h_d_a + d_i / d_ * inner
    h_d_a = -1 * neg_h_d_a
    return h_d_a


# Calculate the prior probability and conditional probability
def computer_probability(train_label_array, selected_feature_index, train_feature_array):
    """
    Calculate the conditional probabilities using Bayesian theorem 
    train_label_array, selected_feature_index, train_feature_array
    """
    # the number of labels I(yi = ck)
    pos_label_number = np.where(train_label_array == 1.0)[0].shape[0]
    neg_label_number = np.where(train_label_array == 0.0)[0].shape[0]
    label_number = train_label_array.shape[0]
    # prior 
    p_0 = (neg_label_number + 1) / (label_number + 2 * 1)
    p_1 = (pos_label_number + 1) / (label_number + 2 * 1)
    # conditional probability  store in "feature_probability_list"
    feature_probability_list = [[] for i in range(len(selected_feature_index))]     
    # array  d1:Use the number of features as an index; d2:Specific features and conditional probabilities of features
    """
    Calculate specific features and their conditional probabilities
    """
    for feature_used_number in range(len(selected_feature_index)):   
        combin_feature = selected_feature_index[0:feature_used_number+1]    

        feature_statistics = compute_sample_number(
                                train_feature_array,
                                train_label_array,
                                combin_feature)
        this_feature_range = len(feature_statistics[0])   # 该项特征的取值数量   Sj

        feature_store_list = []
        for feature_index in range(this_feature_range):
            # lamda = 1
            condition_p_0 = (feature_statistics[0][feature_index] + 1) / (
                                neg_label_number + this_feature_range * 1)
            condition_p_1 = (feature_statistics[1][feature_index] + 1) / (
                                pos_label_number + this_feature_range * 1)

            store_params = (feature_statistics[2][feature_index], (condition_p_0, condition_p_1))
            feature_store_list.append(store_params)
        feature_probability_list[feature_used_number] = feature_store_list
    return p_0, p_1, feature_probability_list
        
        
def measure(l, output): 
    """
        input:  l, output  numpy array type
        output: return ((tp, fn, fp, tn), (precision, recall, f1))
    """
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
    print("tp fn fp tn", tp, fn, fp, tn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * tp / (2 * tp + fp + fn)
    # print(precision, recall, f1)   
    
    ac = 0
    for i in range(output.shape[0]):
        if l[i] == output[i]:
            ac = ac + 1
    accuracy = ac / output.shape[0]
    
    
    return ((tp, fn, fp, tn), (precision, recall, f1), (ac, accuracy)) 


def iteration_computer(selected_feature_index, train_feature_array, 
                       poster_p_list, prior_p_0, prior_p_1):

    model_out = [[] for i in range(len(selected_feature_index))]
    recongnition_log = [[] for i in range(len(selected_feature_index))]   # Record recognition results
    sample_index_list = [i for i in range(train_feature_array.shape[0])]  # The index of sample features calculated each time is initialized to all samples.
    for i in range(len(selected_feature_index)):
        # print(i)
        optimal_i = selected_feature_index[0:i+1]
        feature_data = train_feature_array[sample_index_list, :][:, optimal_i]
        feature_size = feature_data.shape
        for sample_i in range(feature_size[0]):            # poster P
            f_sample = list(feature_data[sample_i])
            for f_i in range(len(poster_p_list[i])):
                if f_sample == poster_p_list[i][f_i][0]:   
                    f_p_0 = poster_p_list[i][f_i][1][0]    # p(y=0|X)
                    f_p_1 = poster_p_list[i][f_i][1][1]    # p(y=1|X)
                    break
            sample_p_1 = prior_p_1 * f_p_1 / (prior_p_1 * f_p_1 + prior_p_0 * f_p_0)   
            model_out[i].append(sample_p_1)
            # break
        # break
        # Filter out samples below a certain threshold  H = 0.8  Directly output the corresponding results for the last iteration
        if i != len(selected_feature_index) - 1:
            next_sample_index = []    
            H = 0.8                 # threshold
            for j in range(len(model_out[i])):
                if model_out[i][j] >= H:
                    # The output information for a single sample:
                    # index of sample, p(y=1|X), the number of iteration
                    sample_log_tuple = (sample_index_list[j], model_out[i][j], i)   
                    recongnition_log[i].append(sample_log_tuple)
                else:
                    next_sample_index.append(sample_index_list[j])
            sample_index_list = next_sample_index   
        else:
            neg_sample = []
            H = 0.8
            for j in range(len(model_out[i])):
                if model_out[i][j] >= H:
                    sample_log_tuple = (sample_index_list[j], model_out[i][j], i)
                    recongnition_log[i].append(sample_log_tuple)
                else:
                    sample_log = (sample_index_list[j], model_out[i][j])
                    neg_sample.append(sample_log)    
    return (recongnition_log, neg_sample, model_out)
        

def iteration_measure(selected_feature_index, test_recongnition, test_label_array):

    test_measure_log = []
    old_sample = []
    for i in range(len(selected_feature_index)):
        i_sample_index = ([test_recongnition[i][j][0] for j in range(len(test_recongnition[i]))]
                          + old_sample)
        i_sample_index.sort()
        identification = np.zeros(len(test_label_array))
        identification[i_sample_index] = 1
        m_t = measure(test_label_array, identification)
        print(m_t)
        old_sample = i_sample_index
        test_measure_log.append(m_t)
    return test_measure_log
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        