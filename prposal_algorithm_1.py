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
    # load dataset
    feature_data = np.loadtxt("data/numpy_array_data_atg")
    label_data = np.loadtxt("data/numpy_label_data_atg")
    # number of samples, feature dimension. 
    sample_number, feature_number = feature_data.shape
    
    
    
    # generate train dataset and test dataset
    shuffle_index = [i for i in range(sample_number)]
    np.random.shuffle(shuffle_index)
    division_bound = int(np.floor(sample_number / 3 * 2))
    train_feature_array = feature_data[shuffle_index[0:division_bound]]
    train_label_array = label_data[shuffle_index[0:division_bound]]
    test_feature_array = feature_data[shuffle_index[division_bound:]]
    test_label_array = label_data[shuffle_index[division_bound:]]
    
    
    # Calculate the entropy value of labeled data
    h_d, positive_example_index, negative_example_index = algorithm_packet.compute_label_entropy(
                                                                        train_label_array)   
    
    """
    Calculate conditional entropy by gradually increasing the feature dimension.
    """
    # list of feature 0 - feature_number
    all_feature_index = [i for i in range(feature_number)]
    # List of used feature indices.
    selected_feature_index = []
    # the number of train samples
    d_ = train_feature_array.shape[0]

    # Select the first feature
    g_d_a = []
    for f in range(feature_number):
        used_index = [f]
        feature_count_list = algorithm_packet.compute_sample_number(
            train_feature_array, 
            train_label_array, 
            used_index)
        h_d_a = algorithm_packet.compute_condition_entropy(feature_count_list, d_)
        g_d_a.append(h_d - h_d_a)

    selected_feature_index = [g_d_a.index(max(g_d_a))]   # 此处序号即为特征索引
    
    used_index = copy.deepcopy(selected_feature_index)
    # Calculate the other features g_d_a
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
    
    # Where "selected_feature_index" is a list of feature importance sorted in descending order.

    
    # Calculate Laplace prior probability and Bayesian Laplace conditional probability
    prior_p_0, prior_p_1, poster_p_list = algorithm_packet.computer_probability(
                                train_label_array,
                                selected_feature_index,
                                # attention_index,
                                train_feature_array)
    
    
    """   Evaluate the performance on the training set  """
    train_output = [[] for i in range(len(selected_feature_index))]
    for i in range(len(selected_feature_index)):
        optimal_i = selected_feature_index[0:i+1]          # Select the index of the i-th feature
        feature_data = train_feature_array[:, optimal_i]   # the i-th feature data
        feature_size = feature_data.shape
        for sample_i in range(feature_size[0]):
            f_sample = list(feature_data[sample_i])        
            for f_i in range(len(poster_p_list[i])):       
                if f_sample == poster_p_list[i][f_i][0]:
                    f_p_0 = poster_p_list[i][f_i][1][0]    # Probability of belonging to class 0
                    f_p_1 = poster_p_list[i][f_i][1][1]    # Probability of belonging to class 1
                    break
            sample_p_1 = prior_p_1 * f_p_1 / (prior_p_1 * f_p_1 + prior_p_0 * f_p_0)                           
            train_output[i].append(sample_p_1)
    
    train_output_array = np.array(train_output).T   
    train_output_array[train_output_array >= 0.5] = 1
    train_output_array[train_output_array < 0.5] = 0
    measure = [[] for i in range(len(selected_feature_index))]
    for i in range(len(selected_feature_index)):
        model_output = train_output_array[:, i]
        golden_l = train_label_array
        measure[i].append(algorithm_packet.measure(golden_l, model_output))


    """
    The testing method used here is to calculate the posterior probability for the device 
    even if it has been recognized before, in the subsequent iterations, 
    and to use the last recognition result as the final result.
    """
    print("/*******train dataset*****/")
    for m in measure:
        print(m)
    
    # Process test dataset
    test_output = [[] for i in range(len(selected_feature_index))] 
    for i in range(len(selected_feature_index)):   # Iterating over the optimal feature list
        optimal_i = selected_feature_index[0:i+1]  # Combining the first i features: (f1, f2, ..,fi)
        feature_data = test_feature_array[:, optimal_i]  # Fetching data for the corresponding feature
        feature_size = feature_data.shape          
        for sample_i in range(feature_size[0]):    # index of sample; Calculating the probability for the i-th sample  
            f_sample = list(feature_data[sample_i])   # Feature data for the i-th sample      
            for f_i in range(len(poster_p_list[i])):  # Looking up the table in the calculated posterior probability table
                if f_sample == poster_p_list[i][f_i][0]: # Finding the posterior probability for the current feature combination
                    f_p_0 = poster_p_list[i][f_i][1][0]  # p(y=0|(f1, f2, ..,fi))  
                    f_p_1 = poster_p_list[i][f_i][1][1]  # p(y=1|(f1, f2, ..,fi))  
                    break
            # Using Bayes' formula to calculate the probability of p(y=1) for the sample
            sample_p_1 = prior_p_1 * f_p_1 / (prior_p_1 * f_p_1 + prior_p_0 * f_p_0)                           
            test_output[i].append(sample_p_1)  # Saving the results
            """test_outout: The i-th sublist of the result represents the probability 
            of each sample being a honeypot in the i-th calculation; 
            The i-th calculation of probability uses the first i features (f1, f2, ..,fi)"""
            

    test_output_array = np.array(test_output).T   
    test_output_array[test_output_array >= 0.6] = 1
    test_output_array[test_output_array < 0.6] = 0
    test_measure = [[] for i in range(len(selected_feature_index))]
    for i in range(len(selected_feature_index)):
        model_output = test_output_array[:, i]
        golden_l = train_label_array
        test_measure[i].append(algorithm_packet.measure(golden_l, model_output))

    print("/*******test dataset*****/")
    for m in test_measure:
        print(m)
    
    # Accuracy
    accuracy = [[] for i in range(len(selected_feature_index))]
    for i in range(train_output_array.shape[1]):
        correct = 0
        for sample in range(train_output_array.shape[0]):
            if train_output_array[sample, i] == train_label_array[sample]:
                correct = correct + 1
        acc = correct / train_label_array.shape[0]
        accuracy[i].append(acc)
    print("---Accuracy rate for each iteration---")
    for ac in accuracy:
        print(ac)
        
        
    """
        Specific iterative recognition performance testing
        Do not include the successfully recognized samples in the recognition set anymore, 
        only recognize the previously unrecognized ones  
    """ 
    # Train dataset
    recongnition_log, neg_sample, model_out = algorithm_packet.iteration_computer(
                                            selected_feature_index, train_feature_array, 
                                            poster_p_list, prior_p_0, prior_p_1)
    print("--------------------")
    measure_log = []    
    old_iteration_sample = []     
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

    print("--------------test dataset---------------")
    test_recongnition, test_neg_sample, test_model = algorithm_packet.iteration_computer(
                                                    selected_feature_index, test_feature_array,
                                                    poster_p_list, prior_p_0, prior_p_1)
    
    test_meas = algorithm_packet.iteration_measure(
                        selected_feature_index, test_recongnition, 
                        test_label_array)
    
    # test_recongnition
    # The output information for a single sample:
    # index of sample, p(y=1|X), the number of iteration
    

 