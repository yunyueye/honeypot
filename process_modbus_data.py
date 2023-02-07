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


# 读取所有数据
data = pd.read_csv("./2000.csv")

"""
    提取modbus数据
"""
modbus_row_index = data[data["IsModbus"]==True].index.tolist()
modbus_data_ = data.loc[modbus_row_index, :]
# s7标签数据
modbus_label = modbus_data_["IsHoneypot"]
# 填补空缺值
modbus_data_ = modbus_data_.fillna(value="-1")


# 处理modbusErrorRequestTime 这一列没有空值  分箱
col_name = "modbusErrorRequestTime"
col_data = modbus_data_[col_name].values
# 大于20的替换成20
col_data[col_data>20]=20
# 使用sklearn分箱
kbd = KBinsDiscretizer(n_bins=20, encode='ordinal', strategy='uniform')
# kbd.bin_edges_  查看离散分界
discrete_modbusErrorRequestTime_array = kbd.fit_transform(col_data.reshape(-1,1))


# 处理modbusReadRegister 本身就是离散型的 按照字典编码
col_name = "modbusReadRegister"
col_data = modbus_data_[col_name]
elem_number = modbus_data_[col_name].value_counts()
# 分成4类 1.connection failed 2.全0 3.全65535 4.其他
modbusReadRegister_dict = {}
modbusReadRegister_dict["connection failed"] = 0
modbusReadRegister_dict["0\t0\t0"] = 1
modbusReadRegister_dict["65535\t65535\t65535"] = 2
# 其他不在上面三个中的编号为3
modbusReadRegister_array = np.zeros((90, 1))
for i, elem in enumerate(col_data):
    if elem in modbusReadRegister_dict:
        modbusReadRegister_array[i] = modbusReadRegister_dict[elem]
    else:
        modbusReadRegister_array[i] = 3
        

# 处理IPwhoisNetsDescription 将数量少于N=1的设为一个标签
col_name = "IPwhoisNetsDescription"
N = 5
elem_number = modbus_data_[col_name].value_counts()
name_list = list(elem_number.index)
less_than_list = []
# 统计出哪些数量小于N
for i, elem in enumerate(elem_number):
    if elem <=N:
        less_than_list.append(name_list[i])
# 将名字 - 数量变成字典
IPwhoisNetsDescription_dict = {}
for elem in less_than_list:
   IPwhoisNetsDescription_dict[elem] = 0   # less_than_list中元素编号为0
k = 1
for elem in (set(name_list) - set(less_than_list)):
    IPwhoisNetsDescription_dict[elem] = k
    k = k + 1
# 将IPwhoisNetsDescription列都转化成数字
IPwhoisNetsDescription_array = np.zeros((90, 1))
for i, elem in enumerate(modbus_data_[col_name]):
    IPwhoisNetsDescription_array[i] = IPwhoisNetsDescription_dict[elem]
    
    
# 处理OS 将数量少于N=1的设为一个标签
col_name = "OS"
N = 1
elem_number = modbus_data_[col_name].value_counts()
name_list = list(elem_number.index)
less_than_list = []
# 统计出哪些数量小于N
for i, elem in enumerate(elem_number):
    if elem <=N:
        less_than_list.append(name_list[i])
# 将名字 - 数量变成字典
OS_dict = {}
for elem in less_than_list:
   OS_dict[elem] = 0   # less_than_list中元素编号为0
k = 1
for elem in (set(name_list) - set(less_than_list)):
    OS_dict[elem] = k
    k = k + 1
# 将OS列都转化成数字
OS_array = np.zeros((90, 1))
for i, elem in enumerate(modbus_data_[col_name]):
    OS_array[i] = OS_dict[elem]
    
    
# 处理 OpenPortNum 按照小于等于5 <=5  分两类   
# 字典编码
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
    

# 处理 hopNum 分成5个箱
col_name = "hopNum"
N = 1
elem_number = modbus_data_[col_name].value_counts()
name_list = list(elem_number.index)
less_than_list = []
# 使用sklearn分箱
kbd = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
# kbd.bin_edges_  查看离散分界
col_data = modbus_data_[col_name]
discrete_hopNum_data = kbd.fit_transform(col_data.values.reshape(-1,1))
        

# 组合成numpy数组
modbus_label_np = modbus_label.to_numpy()
com_data = np.hstack((discrete_modbusErrorRequestTime_array, modbusReadRegister_array))
com_data = np.hstack((com_data, IPwhoisNetsDescription_array))
com_data = np.hstack((com_data, OS_array))
com_data = np.hstack((com_data, OpenPortNum_array))
com_data = np.hstack((com_data, discrete_hopNum_data))


# 保存数据
np.savetxt("numpy_array_data_modbus", com_data)
np.savetxt("numpy_label_data_modbus", modbus_label_np)









