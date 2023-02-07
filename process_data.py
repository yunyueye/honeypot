# -*- coding: utf-8 -*-
"""
Created on Mon May 17 15:20:41 2021

@author: ztong
"""

import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import KBinsDiscretizer
from collections import Counter

data = pd.read_csv("./2000.csv")

# 获取列名
columns_name_array = data.columns
# 获取蜜罐标签
honeypot_label = data.loc[:,"IsHoneypot"]
# 去掉id ip ishoneypot iscloud标签
columns_name_list = list(columns_name_array)
columns_name_list.remove("id")
columns_name_list.remove("ip")
columns_name_list.remove("IsHoneypot")

# 统计各列的元素个数 s7NameOfThePLC s7PlantIdentification s7SerialNumberOfModule IPwhoisNetsDescription OS
# 填充nan值为"-1"
dat = data.fillna(value="-1")
col_name_list = ["s7NameOfThePLC", "s7PlantIdentification", "s7SerialNumberOfModule", "IPwhoisNetsDescription", "OS"]
elem_number_list = []
for col_name in col_name_list:
    elem_number_list.append(dat[col_name].value_counts())
    
# 处理s7nameofplc 将数量少于N=5的设为一个标签
col_name = "s7NameOfThePLC"
N = 5
elem_number = dat[col_name].value_counts()
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
s7nameofplc_array = np.zeros((2000, 1))
for i, elem in enumerate(dat[col_name]):
    s7nameofplc_array[i] = plc_name_dict[elem]

# 处理s7plantidentification  一共5个 就不用看数量小的了
col_name = "s7PlantIdentification"
N = 5
elem_number = dat[col_name].value_counts()
name_list = list(elem_number.index)
# 同样转换成字典形式
s7PlantIdentification_dict = {}
for i, elem in enumerate(name_list):
    s7PlantIdentification_dict[elem] = i
# 将s7PlantIdentification列都转化成数字
s7PlantIdentification_array = np.zeros((2000, 1))
for i, elem in enumerate(dat[col_name]):
    s7PlantIdentification_array[i] = s7PlantIdentification_dict[elem]
    
# 处理s7SerialNumberOfModule 将数量少于N=1的设为一个标签
col_name = "s7SerialNumberOfModule"  
N = 1
elem_number = dat[col_name].value_counts()
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
s7SerialNumberOfModule_array = np.zeros((2000, 1))
for i, elem in enumerate(dat[col_name]):
    s7SerialNumberOfModule_array[i] = s7SerialNumberOfModule_dict[elem]

# 处理IPwhoisNetsDescription 将数量少于N=1的设为一个标签
col_name = "IPwhoisNetsDescription"
N = 5
elem_number = dat[col_name].value_counts()
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
IPwhoisNetsDescription_array = np.zeros((2000, 1))
for i, elem in enumerate(dat[col_name]):
    IPwhoisNetsDescription_array[i] = IPwhoisNetsDescription_dict[elem]

# 处理OS 将数量少于N=1的设为一个标签
col_name = "OS"
N = 5
elem_number = dat[col_name].value_counts()
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
OS_array = np.zeros((2000, 1))
for i, elem in enumerate(dat[col_name]):
    OS_array[i] = OS_dict[elem]

# 处理s7ResponseTime  -1表示没有，即无回复  可以取2    （一个大数字）  分箱
col_name = "s7ResponseTime"
col_data = dat[col_name]
col_data.replace("-1", 2, inplace=True)
# 使用sklearn分箱
kbd = KBinsDiscretizer(n_bins=20, encode='ordinal', strategy='uniform')
# kbd.bin_edges_  查看离散分界
discrete_s7ResponseTime_data = kbd.fit_transform(col_data.values.reshape(-1,1))
# discrete_col_data = KBinsDiscretizer(n_bins=20, encode='ordinal',
#                  strategy='uniform').fit_transform(col_data.values.reshape(-1,1))
# 使用pd.cut分箱 0-0.1 0.1-0.2这样
# a = pd.cut(col_data, bins=[i/10 for i in range(21)], labels=False)

# 处理modbusErrorRequestTime 这一列没有空值  分箱
col_name = "modbusErrorRequestTime"
col_data = dat[col_name].values
# 大于20的替换成20
col_data[col_data>20]=20
# 使用sklearn分箱
kbd = KBinsDiscretizer(n_bins=20, encode='ordinal', strategy='uniform')
# kbd.bin_edges_  查看离散分界
discrete_modbusErrorRequestTime_array = kbd.fit_transform(col_data.reshape(-1,1))

# 处理modbusReadRegister 本身就是离散型的 按照字典编码
col_name = "modbusReadRegister"
col_data = dat[col_name]
elem_number = dat[col_name].value_counts()
# 分成4类 1.connection failed 2.全0 3.全65535 4.其他
modbusReadRegister_dict = {}
modbusReadRegister_dict["connection failed"] = 0
modbusReadRegister_dict["0\t0\t0"] = 1
modbusReadRegister_dict["65535\t65535\t65535"] = 2
# 其他不在上面三个中的编号为3
modbusReadRegister_array = np.zeros((2000, 1))
for i, elem in enumerate(col_data):
    if elem in modbusReadRegister_dict:
        modbusReadRegister_array[i] = modbusReadRegister_dict[elem]
    else:
        modbusReadRegister_array[i] = 3

# 处理 OSaccuracy 同样分箱，将-1替换成0
col_name = "OSaccuracy"
col_data = dat[col_name]
col_data.replace("-1", 80, inplace=True)
# 使用sklearn分箱
kbd = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
# kbd.bin_edges_  查看离散分界
discrete_OSaccuracy_data = kbd.fit_transform(col_data.values.reshape(-1,1))

# 处理各蜜罐协议标签  看是否有同时是两个协议的 三个协议两两相与  再相或
is_s7 = dat["IsS7"]
is_modbus = dat["IsModbus"]
is_atg = dat["IsAtg"]
s7_and_modbus = np.logical_and(is_s7, is_modbus)   # 就这个有一个
s7_and_atg = np.logical_and(is_s7, is_atg)
modbus_and_atg = np.logical_and(is_modbus, is_atg)
all_protocol_or = np.logical_or(modbus_and_atg, np.logical_or(s7_and_atg, s7_and_modbus)) # 或运算 有一个相交的即为多协议并存
# 转成数字 False 0 True 1
all_protocol_or[all_protocol_or == True] = 1
protocol_cross = all_protocol_or

# 处理 s7time5After False 0 True 1 
col_name = "s7time5After"
col_data = dat[col_name]
s7time5After_array = np.zeros((2000, 1))
s7time5After_array[col_data[col_data==True].index] = 1

# 处理 OpenPortNum 按照小于等于5 <=5  分两类
col_name = "OpenPortNum"
col_data = dat[col_name]
N = 1
elem_number = dat[col_name].value_counts()
OpenPortNum_array = np.zeros((2000, 1))
OpenPortNum_array[col_data[col_data<=5].index] = 0
OpenPortNum_array[col_data[col_data>5].index] = 1

# 处理 hopNum 分成5个箱
col_name = "hopNum"
N = 1
elem_number = dat[col_name].value_counts()
name_list = list(elem_number.index)
less_than_list = []
# 使用sklearn分箱
kbd = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
# kbd.bin_edges_  查看离散分界
col_data = dat[col_name]
discrete_hopNum_data = kbd.fit_transform(col_data.values.reshape(-1,1))

# 处理atg atgSUPER atgUNLEAD  atgDIESEL  atgPREMIUM
col_name = ["atgSUPER", "atgUNLEAD", "atgDIESEL", "atgPREMIUM"]
col_data = dat[col_name]
# 转成字符串 进行编码
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
atg_product_array = np.zeros((2000, 1))
for i, elem in enumerate(str_col_data):
    atg_product_array[i] = atg_product_dict[elem]

# 处理容量信息数据
col_name = ["atgProduct1FirstTimeVolumeTC", "atgProduct1SecondTimeVolumeTC", 
            "atgProduct1FirstTimeULLAGE", "atgProduct1SecondTimeULLAGE",
            "atgProduct2FirstTimeVolumeTC", "atgProduct2SecondTimeVolumeTC",
            "atgProduct2FirstTimeULLAGE", "atgProduct2SecondTimeULLAGE",
            "atgProduct3FirstTimeVolumeTC", "atgProduct3SecondTimeVolumeTC",
            "atgProduct3FirstTimeULLAGE", "atgProduct3SecondTimeULLAGE", 
            "atgProduct4FirstTimeVolumeTC", "atgProduct4SecondTimeVolumeTC",
            "atgProduct4FirstTimeULLAGE", "atgProduct4SecondTimeULLAGE"]
col_name_compare = "atgTwoTimesCompare"
col_data = dat[col_name]
col_data_compare = dat[col_name_compare]
# 处理数据类型 v11 v12
v11tc, v12tc = dat[col_name[0]], dat[col_name[1]]
v11tc_list, v12tc_list = [], []
for i,j in zip(v11tc, v12tc):
    v11tc_list.append(float(i))
    v12tc_list.append(float(j))
v11tc_array = np.array(v11tc_list)
v12tc_array = np.array(v12tc_list)
# 处理数据类型 u11 u12
u11, u12 = dat[col_name[2]], dat[col_name[3]]
u11_list, u12_list = [], []
for i, j in zip(u12, u12):
    u11_list.append(float(i))
    u12_list.append(float(j))
u11_array = np.array(u11_list)
u12_array = np.array(u12_list)    
# 处理数据类型 v21 v22
v21tc, v22tc = dat[col_name[4]], dat[col_name[5]]
v21tc_list, v22tc_list = [], []
for i,j in zip(v21tc, v22tc):
    v21tc_list.append(float(i))
    v22tc_list.append(float(j))
v21tc_array = np.array(v21tc_list)
v22tc_array = np.array(v22tc_list)
# 处理数据类型 u21 u22
u21, u22 = dat[col_name[6]], dat[col_name[7]]
u21_list, u22_list = [], []
for i, j in zip(u21, u22):
    u21_list.append(float(i))
    u22_list.append(float(j))
u21_array = np.array(u21_list)
u22_array = np.array(u22_list)
# 处理数据类型 v31 v32
v31tc, v32tc = dat[col_name[8]], dat[col_name[9]]
v31tc_list, v32tc_list = [], []
for i,j in zip(v31tc, v32tc):
    v31tc_list.append(float(i))
    v32tc_list.append(float(j))
v31tc_array = np.array(v31tc_list)
v32tc_array = np.array(v32tc_list)
# 处理数据类型 u31 u32
u31, u32 = dat[col_name[10]], dat[col_name[11]]
u31_list, u32_list = [], []
for i, j in zip(u31, u32):
    u31_list.append(float(i))
    u32_list.append(float(j))
u31_array = np.array(u31_list)
u32_array = np.array(u32_list)
# 处理数据类型 v41 v42
v41tc, v42tc = dat[col_name[12]], dat[col_name[13]]
v41tc_list, v42tc_list = [], []
for i,j in zip(v41tc, v42tc):
    v41tc_list.append(float(i))
    v42tc_list.append(float(j))
v41tc_array = np.array(v41tc_list)
v42tc_array = np.array(v42tc_list)
# 处理数据类型 u41 u42
u41, u42 = dat[col_name[14]], dat[col_name[15]]
u41_list, u42_list = [], []
for i, j in zip(u41, u42):
    u41_list.append(float(i))
    u42_list.append(float(j))
u41_array = np.array(u41_list)
u42_array = np.array(u42_list)
# 分成4类  差相等  差不等   有0的   全0的
atg_vu_array = np.zeros((2000,1))
threshold = 1000
col_data = col_data.replace("-1", -1)
for i in range(2000):
    # 先将全零的分成第0类
    if col_data.loc[i].values.sum() == 0:
        atg_vu_array[i] = 0
    # 含有0的
    elif 0 in col_data.loc[i].values:
        atg_vu_array[i] = 1
    else:
        # 算差值
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
        # 算差结果
        com_array = np.array([com_1<threshold, com_2<threshold, com_3<threshold, com_4<threshold]) + 0
        # 和大于2表示有两个在阈值内
        if com_array.sum() >= 2:
            atg_vu_array[i] = 2
        else:
            atg_vu_array[i] = 3

# 组合成一张二维向量表   各向量维度 (2000, 1)
com_data = np.hstack((s7nameofplc_array, s7PlantIdentification_array))
com_data = np.hstack((com_data, s7SerialNumberOfModule_array))
com_data = np.hstack((com_data, IPwhoisNetsDescription_array))
com_data = np.hstack((com_data, OS_array))
com_data = np.hstack((com_data, discrete_s7ResponseTime_data))
com_data = np.hstack((com_data, discrete_modbusErrorRequestTime_array))
com_data = np.hstack((com_data, modbusReadRegister_array))
# com_data = np.hstack((com_data, discrete_OSaccuracy_data))
com_data = np.hstack((com_data, protocol_cross.values.reshape(2000,1)))
com_data = np.hstack((com_data, s7time5After_array))  
com_data = np.hstack((com_data, OpenPortNum_array))   
com_data = np.hstack((com_data, discrete_hopNum_data))
com_data = np.hstack((com_data, atg_product_array))
com_data = np.hstack((com_data, atg_vu_array))

# 标签数据
label_data = dat.loc[:,"IsHoneypot"].values.reshape(2000,1)
# 保存数据
np.savetxt("numpy_array_data_no_OSaccuracy", com_data)
np.savetxt("numpy_label_data", label_data)




"""
# 处理tc 与 u的差值  设定阈值  小于阈值认定相等  大于阈值认定不相等
threshold = 10
# first 1
dis_1_tc = np.abs(v12tc_array - v11tc_array)
dis_1_u = np.abs(u12_array - u11_array)
com_1_array = np.abs(dis_1_tc - dis_1_u)
com_1_result = np.zeros((2000,1)) 
com_1_result[com_1_array < threshold] = 1
# second 2
dis_2_tc = np.abs(v22tc_array - v21tc_array)
dis_2_u = np.abs(u22_array - u21_array)
com_2_array = np.abs(dis_2_u - dis_2_tc)
com_2_result = np.zeros((2000,1))
com_2_result[com_2_array < threshold] = 1
# third 3
dis_3_tc = np.abs(v31tc_array - v32tc_array)
dis_3_u = np.abs(u31_array - u32_array)
com_3_array = np.abs(dis_3_tc - dis_3_u)
com_3_result = np.zeros((2000,1))
com_3_result[com_3_array < threshold] = 1
# fourth 4
dis_4_tc = np.abs(v41tc_array- v42tc_array)
dis_4_u = np.abs(u41_array - u42_array)
com_4_array = np.abs(dis_4_tc - dis_4_u)
com_4_result = np.zeros((2000,1))
com_4_result[com_4_array < threshold] = 1
# 对这四个取或运算
finally_com_array = np.logical_or(
                                np.logical_or(com_1_result, com_2_result),
                                np.logical_or(com_3_result, com_4_result))
# finally_com_array = finally_com_array + 0
# finally_com_array = finally_com_array.reshape((2000,1))
"""












