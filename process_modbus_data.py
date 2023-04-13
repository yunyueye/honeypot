# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 16:38:37 2021

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
    Extracting Modbus protocol data
"""
modbus_row_index = data[data["IsModbus"]==True].index.tolist()
modbus_data_ = data.loc[modbus_row_index, :]
# Modbus label
modbus_label = modbus_data_["IsHoneypot"]
# Filling in missing values with -1
modbus_data_ = modbus_data_.fillna(value="-1")

"""
    Processing column data related to the Modbus protocol
    Objective: Encoding non-integer variables as integers
"""
# column name: modbusErrorRequestTime Using the k-bins algorithm, where bin=20
col_name = "modbusErrorRequestTime"
col_data = modbus_data_[col_name].values
# Replace with 20 if greater than 20
col_data[col_data>20]=20
kbd = KBinsDiscretizer(n_bins=20, encode='ordinal', strategy='uniform')
discrete_modbusErrorRequestTime_array = kbd.fit_transform(col_data.reshape(-1,1))


# column name: modbusReadRegister 
col_name = "modbusReadRegister"
col_data = modbus_data_[col_name]
elem_number = modbus_data_[col_name].value_counts()
# Divided into 4 categories: 1.connection failed  2.0   3.65535    4.other
modbusReadRegister_dict = {}
modbusReadRegister_dict["connection failed"] = 0
modbusReadRegister_dict["0\t0\t0"] = 1
modbusReadRegister_dict["65535\t65535\t65535"] = 2
# other
modbusReadRegister_array = np.zeros((90, 1))
for i, elem in enumerate(col_data):
    if elem in modbusReadRegister_dict:
        modbusReadRegister_array[i] = modbusReadRegister_dict[elem]
    else:
        modbusReadRegister_array[i] = 3
        

# column name: IPwhoisNetsDescription 
col_name = "IPwhoisNetsDescription"
N = 5
elem_number = modbus_data_[col_name].value_counts()
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
IPwhoisNetsDescription_array = np.zeros((90, 1))
for i, elem in enumerate(modbus_data_[col_name]):
    IPwhoisNetsDescription_array[i] = IPwhoisNetsDescription_dict[elem]
    
    
# column name: OS 
col_name = "OS"
N = 1
elem_number = modbus_data_[col_name].value_counts()
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
OS_array = np.zeros((90, 1))
for i, elem in enumerate(modbus_data_[col_name]):
    OS_array[i] = OS_dict[elem]
    
    
# column name:  OpenPortNum 
col_name = "OpenPortNum"
col_data = modbus_data_[col_name]
N = 1
elem_number = modbus_data_[col_name].value_counts()
name_list = list(elem_number.index)
OpenPortNum_array = np.zeros((90, 1))
openport_dict = {}
k = 0
for elem in name_list:
    openport_dict[elem] = k
    k = k + 1
for i, elem in enumerate(modbus_data_[col_name]):
    OpenPortNum_array[i] = openport_dict[elem]
    

# column name:  hopNum    Using the k-bins algorithm, where bin=5
col_name = "hopNum"
N = 1
elem_number = modbus_data_[col_name].value_counts()
name_list = list(elem_number.index)
less_than_list = []
kbd = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
col_data = modbus_data_[col_name]
discrete_hopNum_data = kbd.fit_transform(col_data.values.reshape(-1,1))
        

# Merging data
modbus_label_np = modbus_label.to_numpy()
com_data = np.hstack((discrete_modbusErrorRequestTime_array, modbusReadRegister_array))
com_data = np.hstack((com_data, IPwhoisNetsDescription_array))
com_data = np.hstack((com_data, OS_array))
com_data = np.hstack((com_data, OpenPortNum_array))
com_data = np.hstack((com_data, discrete_hopNum_data))


# save data
np.savetxt("numpy_array_data_modbus", com_data)
np.savetxt("numpy_label_data_modbus", modbus_label_np)

"""
After running the file, you will get two files: numpy_array_data_modbus and numpy_label_data_modbus. 
Put these two files into the "data" folder and then you can run proposal_algorithm_1.py.
"""







