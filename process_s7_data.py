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

# 读取所有数据
data = pd.read_csv("./2000.csv")

"""
    提取S7数据
"""
s7_row_index = data[data["IsS7"]==True].index.tolist()
s7_data_ = data.loc[s7_row_index, :]
# s7标签数据
s7_label = s7_data_["IsHoneypot"]
# 填补空缺值
s7_data_ = s7_data_.fillna(value="-1")

# 处理s7nameofplc 将数量少于N=5的设为一个标签
col_name = "s7NameOfThePLC"
N = 5
elem_number = s7_data_[col_name].value_counts()
name_list = list(elem_number.index)
less_than_list = []
# 统计出有哪些名字数量小于N
for i, elem in enumerate(elem_number):
    if elem <= N:
        less_than_list.append(name_list[i])
# 统计less_than_list中各元素的对应设备数的和
less_device_sum = 0
for i in less_than_list:
    less_device_sum += elem_number.loc[i]
# 将名字 - 数量变成字典
plc_name_dict = {}
for elem in less_than_list:
    plc_name_dict[elem] = 0   # less_than_list中元素编号为0
k = 1
for elem in (set(name_list) - set(less_than_list)):
    plc_name_dict[elem] = k
    k = k + 1
# 将s7nameofplc列都转换成数字
s7nameofplc_array = np.zeros((555, 1))
for i, elem in enumerate(s7_data_[col_name]):
    s7nameofplc_array[i] = plc_name_dict[elem]

# 处理s7plantidentification  一共5个 就不用看数量小的了
col_name = "s7PlantIdentification"
N = 5
elem_number = s7_data_[col_name].value_counts()
name_list = list(elem_number.index)
# 同样转换成字典形式
s7PlantIdentification_dict = {}
for i, elem in enumerate(name_list):
    s7PlantIdentification_dict[elem] = i
# 将s7PlantIdentification列都转化成数字
s7PlantIdentification_array = np.zeros((555, 1))
for i, elem in enumerate(s7_data_[col_name]):
    s7PlantIdentification_array[i] = s7PlantIdentification_dict[elem]
    
# 处理s7SerialNumberOfModule 将数量少于N=1的设为一个标签
col_name = "s7SerialNumberOfModule"  
N = 1
elem_number = s7_data_[col_name].value_counts()
name_list = list(elem_number.index)
less_than_list = []
# 统计出有哪些名字数量小于N
for i, elem in enumerate(elem_number):
    if elem <= N:
        less_than_list.append(name_list[i])
# 将名字 - 数量变成字典
s7SerialNumberOfModule_dict = {}
for elem in less_than_list:
    s7SerialNumberOfModule_dict[elem] = 0   # less_than_list中元素编号为0
k = 1
for elem in (set(name_list) - set(less_than_list)):
    s7SerialNumberOfModule_dict[elem] = k
    k = k + 1
# 将s7SerialNumberOfModule列都转化成数字
s7SerialNumberOfModule_array = np.zeros((555, 1))
for i, elem in enumerate(s7_data_[col_name]):
    s7SerialNumberOfModule_array[i] = s7SerialNumberOfModule_dict[elem]
    
# 处理IPwhoisNetsDescription 将数量少于N=1的设为一个标签
col_name = "IPwhoisNetsDescription"
N = 5
elem_number = s7_data_[col_name].value_counts()
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
IPwhoisNetsDescription_array = np.zeros((555, 1))
for i, elem in enumerate(s7_data_[col_name]):
    IPwhoisNetsDescription_array[i] = IPwhoisNetsDescription_dict[elem]
    
# 处理 s7time5After False 0 True 1 
col_name = "s7time5After"
col_data = s7_data_[col_name]
s7time5After_array = np.zeros((555, 1))
for i, elem in enumerate(s7_data_[col_name]):
    if elem == True:
        s7time5After_array[i] = 1
# s7time5After_array[col_data[col_data==True].index] = 1

# 处理s7ResponseTime  -1表示没有，即无回复  可以取2    （一个大数字）  分箱
col_name = "s7ResponseTime"
col_data = s7_data_[col_name]
col_data.replace("-1", 2, inplace=True)
# 使用sklearn分箱
kbd = KBinsDiscretizer(n_bins=20, encode='ordinal', strategy='uniform')
# kbd.bin_edges_  查看离散分界
discrete_s7ResponseTime_data = kbd.fit_transform(col_data.values.reshape(-1,1))

# 处理OS 将数量少于N=1的设为一个标签
col_name = "OS"
N = 1
elem_number = s7_data_[col_name].value_counts()
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
OS_array = np.zeros((555, 1))
for i, elem in enumerate(s7_data_[col_name]):
    OS_array[i] = OS_dict[elem]


# 处理 OpenPortNum 按照小于等于5 <=5  分两类   
# 字典编码
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
    

# 处理 hopNum 分成5个箱
col_name = "hopNum"
N = 1
elem_number = s7_data_[col_name].value_counts()
name_list = list(elem_number.index)
less_than_list = []
# 使用sklearn分箱
kbd = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
# kbd.bin_edges_  查看离散分界
col_data = s7_data_[col_name]
discrete_hopNum_data = kbd.fit_transform(col_data.values.reshape(-1,1))


# 组合成numpy数组
s7_label_np = s7_label.to_numpy()
com_data = np.hstack((s7nameofplc_array, s7PlantIdentification_array))
com_data = np.hstack((com_data, s7SerialNumberOfModule_array))
com_data = np.hstack((com_data, IPwhoisNetsDescription_array))
com_data = np.hstack((com_data, s7time5After_array))
com_data = np.hstack((com_data, discrete_s7ResponseTime_data))
com_data = np.hstack((com_data, OS_array))
com_data = np.hstack((com_data, OpenPortNum_array))
com_data = np.hstack((com_data, discrete_hopNum_data))

# 保存数据
np.savetxt("numpy_array_data_s7", com_data)
np.savetxt("numpy_label_data_s7", s7_label_np)



