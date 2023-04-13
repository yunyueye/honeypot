# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 14:19:57 2021

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
    Extracting S7 protocol data
"""
s7_row_index = data[data["IsS7"]==True].index.tolist()
s7_data_ = data.loc[s7_row_index, :]
# S7 label
s7_label = s7_data_["IsHoneypot"]
# Filling in missing values with -1
s7_data_ = s7_data_.fillna(value="-1")

"""
    Processing column data related to the S7 protocol
    Objective: Encoding non-integer variables as integers
"""
# column name: s7nameofplc 
# Encoding names as integers, and encoding all names with less than 5 devices as one class
col_name = "s7NameOfThePLC"
N = 5
elem_number = s7_data_[col_name].value_counts()
name_list = list(elem_number.index)
less_than_list = []
# Finding names with less than 5 devices
for i, elem in enumerate(elem_number):
    if elem <= N:
        less_than_list.append(name_list[i])
less_device_sum = 0
for i in less_than_list:
    less_device_sum += elem_number.loc[i]
# dict  key: name, value: integer
plc_name_dict = {}
for elem in less_than_list:
    plc_name_dict[elem] = 0   
k = 1
for elem in (set(name_list) - set(less_than_list)):
    plc_name_dict[elem] = k
    k = k + 1
# Encoding names as integers
s7nameofplc_array = np.zeros((555, 1))
for i, elem in enumerate(s7_data_[col_name]):
    s7nameofplc_array[i] = plc_name_dict[elem]

# column name: s7plantidentification  
col_name = "s7PlantIdentification"
N = 5
elem_number = s7_data_[col_name].value_counts()
name_list = list(elem_number.index)
s7PlantIdentification_dict = {}
for i, elem in enumerate(name_list):
    s7PlantIdentification_dict[elem] = i
s7PlantIdentification_array = np.zeros((555, 1))
for i, elem in enumerate(s7_data_[col_name]):
    s7PlantIdentification_array[i] = s7PlantIdentification_dict[elem]
    
# column name: s7SerialNumberOfModule 
col_name = "s7SerialNumberOfModule"  
N = 1
elem_number = s7_data_[col_name].value_counts()
name_list = list(elem_number.index)
less_than_list = []
for i, elem in enumerate(elem_number):
    if elem <= N:
        less_than_list.append(name_list[i])
s7SerialNumberOfModule_dict = {}
for elem in less_than_list:
    s7SerialNumberOfModule_dict[elem] = 0   
k = 1
for elem in (set(name_list) - set(less_than_list)):
    s7SerialNumberOfModule_dict[elem] = k
    k = k + 1
s7SerialNumberOfModule_array = np.zeros((555, 1))
for i, elem in enumerate(s7_data_[col_name]):
    s7SerialNumberOfModule_array[i] = s7SerialNumberOfModule_dict[elem]
    
# column name: IPwhoisNetsDescription 
col_name = "IPwhoisNetsDescription"
N = 5
elem_number = s7_data_[col_name].value_counts()
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
IPwhoisNetsDescription_array = np.zeros((555, 1))
for i, elem in enumerate(s7_data_[col_name]):
    IPwhoisNetsDescription_array[i] = IPwhoisNetsDescription_dict[elem]
    
# column name: s7time5After False 0 True 1 
col_name = "s7time5After"
col_data = s7_data_[col_name]
s7time5After_array = np.zeros((555, 1))
for i, elem in enumerate(s7_data_[col_name]):
    if elem == True:
        s7time5After_array[i] = 1

# column name: s7ResponseTime  Using the k-bins algorithm, where bin=20
col_name = "s7ResponseTime"
col_data = s7_data_[col_name]
col_data.replace("-1", 2, inplace=True)
# sklearn.KBins algorithm
kbd = KBinsDiscretizer(n_bins=20, encode='ordinal', strategy='uniform')
discrete_s7ResponseTime_data = kbd.fit_transform(col_data.values.reshape(-1,1))

# column name: OS 
col_name = "OS"
N = 1
elem_number = s7_data_[col_name].value_counts()
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
OS_array = np.zeros((555, 1))
for i, elem in enumerate(s7_data_[col_name]):
    OS_array[i] = OS_dict[elem]


# column name:  OpenPortNum 
col_name = "OpenPortNum"
col_data = s7_data_[col_name]
N = 1
elem_number = s7_data_[col_name].value_counts()
name_list = list(elem_number.index)
OpenPortNum_array = np.zeros((555, 1))
openport_dict = {}
k = 0
for elem in name_list:
    openport_dict[elem] = k
    k = k + 1
for i, elem in enumerate(s7_data_[col_name]):
    OpenPortNum_array[i] = openport_dict[elem]
    

# column name:  hopNum Using the k-bins algorithm, where bin=5
col_name = "hopNum"
N = 1
elem_number = s7_data_[col_name].value_counts()
name_list = list(elem_number.index)
less_than_list = []
kbd = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
col_data = s7_data_[col_name]
discrete_hopNum_data = kbd.fit_transform(col_data.values.reshape(-1,1))


# Merging data
s7_label_np = s7_label.to_numpy()
com_data = np.hstack((s7nameofplc_array, s7PlantIdentification_array))
com_data = np.hstack((com_data, s7SerialNumberOfModule_array))
com_data = np.hstack((com_data, IPwhoisNetsDescription_array))
com_data = np.hstack((com_data, s7time5After_array))
com_data = np.hstack((com_data, discrete_s7ResponseTime_data))
com_data = np.hstack((com_data, OS_array))
com_data = np.hstack((com_data, OpenPortNum_array))
com_data = np.hstack((com_data, discrete_hopNum_data))

# save S7 data 
np.savetxt("numpy_array_data_s7", com_data)
np.savetxt("numpy_label_data_s7", s7_label_np)

"""
After running the file, you will get two files: numpy_array_data_s7 and numpy_label_data_s7. 
Put these two files into the "data" folder and then you can run proposal_algorithm_1.py.
"""

