# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 11:38:55 2021

@author: ztong
"""

import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import KBinsDiscretizer
from collections import Counter


# load all protocol ICS honeypot data
data = pd.read_csv("./data_no_ip.csv")

"""
    Extracting ATG protocol data
"""
atg_row_index = data[data["IsAtg"]==True].index.tolist()
atg_data_ = data.loc[atg_row_index, :]
# ATG label
atg_label = atg_data_["IsHoneypot"]
# Filling in missing values with -1
atg_data_ = atg_data_.fillna(value="-1")   


"""
    Processing column data related to the ATG protocol
    Objective: Encoding non-integer variables as integers
"""
# column name: IPwhoisNetsDescription 
col_name = "IPwhoisNetsDescription"
N = 5
elem_number = atg_data_[col_name].value_counts()
name_list = list(elem_number.index)
less_than_list = []
for i, elem in enumerate(elem_number):
    if elem <=N:
        less_than_list.append(name_list[i])
IPwhoisNetsDescription_dict = {}
for elem in less_than_list:
   IPwhoisNetsDescription_dict[elem] = 0   
k = 1
for elem in (set(name_list) - set(less_than_list)):
    IPwhoisNetsDescription_dict[elem] = k
    k = k + 1
IPwhoisNetsDescription_array = np.zeros((348, 1))
for i, elem in enumerate(atg_data_[col_name]):
    IPwhoisNetsDescription_array[i] = IPwhoisNetsDescription_dict[elem]
    
    
# column name: OS 
col_name = "OS"
N = 1
elem_number = atg_data_[col_name].value_counts()
name_list = list(elem_number.index)
less_than_list = []
for i, elem in enumerate(elem_number):
    if elem <=N:
        less_than_list.append(name_list[i])
OS_dict = {}
for elem in less_than_list:
   OS_dict[elem] = 0   
k = 1
for elem in (set(name_list) - set(less_than_list)):
    OS_dict[elem] = k
    k = k + 1
OS_array = np.zeros((348, 1))
for i, elem in enumerate(atg_data_[col_name]):
    OS_array[i] = OS_dict[elem]    
    
    
# column name: OpenPortNum 
col_name = "OpenPortNum"
col_data = atg_data_[col_name]
N = 1
elem_number = atg_data_[col_name].value_counts()
name_list = list(elem_number.index)
OpenPortNum_array = np.zeros((348, 1))
openport_dict = {}
k = 0
for elem in name_list:
    openport_dict[elem] = k
    k = k + 1
for i, elem in enumerate(atg_data_[col_name]):
    OpenPortNum_array[i] = openport_dict[elem]    
    
    
# column name: hopNum    bin=5
col_name = "hopNum"
N = 1
elem_number = atg_data_[col_name].value_counts()
name_list = list(elem_number.index)
less_than_list = []
kbd = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
col_data = atg_data_[col_name]
discrete_hopNum_data = kbd.fit_transform(col_data.values.reshape(-1,1))    
    

# column name: atgname     atgSUPER atgUNLEAD  atgDIESEL  atgPREMIUM
col_name = ["atgSUPER", "atgUNLEAD", "atgDIESEL", "atgPREMIUM"]
col_data = atg_data_[col_name]
str_col_data = []
for i in col_data.index:
    temp = ''
    for ele in col_data.loc[i].values:
        temp += str(ele)
    str_col_data.append(temp)
str_col_data = pd.Series(str_col_data)
elem_number = str_col_data.value_counts()
atg_product_dict = {}
for i, elem in enumerate(elem_number.index):
    atg_product_dict[elem] = i
atg_product_array = np.zeros((348, 1))
for i, elem in enumerate(str_col_data):
    atg_product_array[i] = atg_product_dict[elem]


# colnum names: 
col_name = ["atgProduct1FirstTimeVolumeTC", "atgProduct1SecondTimeVolumeTC", 
            "atgProduct1FirstTimeULLAGE", "atgProduct1SecondTimeULLAGE",
            "atgProduct2FirstTimeVolumeTC", "atgProduct2SecondTimeVolumeTC",
            "atgProduct2FirstTimeULLAGE", "atgProduct2SecondTimeULLAGE",
            "atgProduct3FirstTimeVolumeTC", "atgProduct3SecondTimeVolumeTC",
            "atgProduct3FirstTimeULLAGE", "atgProduct3SecondTimeULLAGE", 
            "atgProduct4FirstTimeVolumeTC", "atgProduct4SecondTimeVolumeTC",
            "atgProduct4FirstTimeULLAGE", "atgProduct4SecondTimeULLAGE"]
col_name_compare = "atgTwoTimesCompare"
col_data = atg_data_[col_name]
col_data_compare = atg_data_[col_name_compare]
v11tc, v12tc = atg_data_[col_name[0]], atg_data_[col_name[1]]
v11tc_list, v12tc_list = [], []
for i,j in zip(v11tc, v12tc):
    v11tc_list.append(float(i))
    v12tc_list.append(float(j))
v11tc_array = np.array(v11tc_list)
v12tc_array = np.array(v12tc_list)
u11, u12 = atg_data_[col_name[2]], atg_data_[col_name[3]]
u11_list, u12_list = [], []
for i, j in zip(u12, u12):
    u11_list.append(float(i))
    u12_list.append(float(j))
u11_array = np.array(u11_list)
u12_array = np.array(u12_list)    
v21tc, v22tc = atg_data_[col_name[4]], atg_data_[col_name[5]]
v21tc_list, v22tc_list = [], []
for i,j in zip(v21tc, v22tc):
    v21tc_list.append(float(i))
    v22tc_list.append(float(j))
v21tc_array = np.array(v21tc_list)
v22tc_array = np.array(v22tc_list)
u21, u22 = atg_data_[col_name[6]], atg_data_[col_name[7]]
u21_list, u22_list = [], []
for i, j in zip(u21, u22):
    u21_list.append(float(i))
    u22_list.append(float(j))
u21_array = np.array(u21_list)
u22_array = np.array(u22_list)
v31tc, v32tc = atg_data_[col_name[8]], atg_data_[col_name[9]]
v31tc_list, v32tc_list = [], []
for i,j in zip(v31tc, v32tc):
    v31tc_list.append(float(i))
    v32tc_list.append(float(j))
v31tc_array = np.array(v31tc_list)
v32tc_array = np.array(v32tc_list)
u31, u32 = atg_data_[col_name[10]], atg_data_[col_name[11]]
u31_list, u32_list = [], []
for i, j in zip(u31, u32):
    u31_list.append(float(i))
    u32_list.append(float(j))
u31_array = np.array(u31_list)
u32_array = np.array(u32_list)
v41tc, v42tc = atg_data_[col_name[12]], atg_data_[col_name[13]]
v41tc_list, v42tc_list = [], []
for i,j in zip(v41tc, v42tc):
    v41tc_list.append(float(i))
    v42tc_list.append(float(j))
v41tc_array = np.array(v41tc_list)
v42tc_array = np.array(v42tc_list)
u41, u42 = atg_data_[col_name[14]], atg_data_[col_name[15]]
u41_list, u42_list = [], []
for i, j in zip(u41, u42):
    u41_list.append(float(i))
    u42_list.append(float(j))
u41_array = np.array(u41_list)
u42_array = np.array(u42_list)
# divided into 4 classes  1.Equal difference  2.Unequal difference   3.Contains 0 values   4.all 0
atg_vu_array = np.zeros((348,1))
threshold = 1000
# col_data = col_data.replace("-1", -1)
for i in range(348):
    row_index = atg_row_index[i]
    # all 0
    if col_data.loc[row_index].values.sum() == 0:
        atg_vu_array[i] = 0
    # contains 0
    elif 0 in col_data.loc[row_index].values:
        atg_vu_array[i] = 1
    else:
        # computer difference
        # 1
        dis_1_tc = np.abs(v12tc_array[i] - v11tc_array[i])
        dis_1_u = np.abs(u12_array[i] - u11_array[i])
        com_1 = np.abs(dis_1_tc - dis_1_u)
        # 2
        dis_2_tc = np.abs(v22tc_array[i] - v21tc_array[i])
        dis_2_u = np.abs(u22_array[i] - u21_array[i])
        com_2 = np.abs(dis_2_u - dis_2_tc)
        # 3
        dis_3_tc = np.abs(v31tc_array[i] - v32tc_array[i])
        dis_3_u = np.abs(u31_array[i] - u32_array[i])
        com_3 = np.abs(dis_3_tc - dis_3_u)
        # 4
        dis_4_tc = np.abs(v41tc_array[i] - v42tc_array[i])
        dis_4_u = np.abs(u41_array[i] - u42_array[i])
        com_4 = np.abs(dis_4_tc - dis_4_u)
        com_array = np.array([com_1<threshold, com_2<threshold, com_3<threshold, com_4<threshold]) + 0
        # sum greater than 2 indicate that there are two within the threshold
        if com_array.sum() >= 2:
            atg_vu_array[i] = 2
        else:
            atg_vu_array[i] = 3

# Merging data
atg_label_np = atg_label.to_numpy()
com_data = np.hstack((IPwhoisNetsDescription_array, OS_array))
com_data = np.hstack((com_data, OpenPortNum_array))
com_data = np.hstack((com_data, discrete_hopNum_data))
com_data = np.hstack((com_data, atg_product_array))
com_data = np.hstack((com_data, atg_vu_array))    
    
# save
np.savetxt("numpy_array_data_atg", com_data)
np.savetxt("numpy_label_data_atg", atg_label_np)   

"""
After running the file, you will get two files: numpy_array_data_atg and numpy_label_data_atg. 
Put these two files into the "data" folder and then you can run proposal_algorithm_1.py.
"""

