#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 08:07:42 2022

@author: bbq
"""
import numpy as np
import sys

#读取风电场给的测量信息
#设定将风向角分为num_direction份，自动计算出每一份的风向区间
#设定风速的划分段，比如0到U_max，分为num_zone段
#自动计算每个风向角，每一段出现的概率
#读取文件
#YYYYMMDD HHMM  110-M(m/s) 110-D(deg) 110-SD(m/s) 110-DSD(deg)  110-Gust3s(m/s)   110-T(C)    110-PRE(hPa)      110-RiNumber  110-VertM(m/s)
#年月日   小时分钟 风速  风向 风速标准差 风向标准差 3s阵风风速 温度 气压 Ri数 竖向风速
def obtain_wind_probability(loc_file,height_mast,num_direction,num_speed,threshold_probability=0.1) :


    if height_mast > 350:
        wind_speed_in_CFD_at_height_mast = 10*(350/10)**0.16
    else:
        wind_speed_in_CFD_at_height_mast = 10*(height_mast/10)**0.16
    pass
    #读取测风塔离地110m处数据
    data = np.loadtxt(loc_file,skiprows=1)
    data_speed = data[:,2]
    data_direction = data[:,3]
    del data
    speed_min = np.min(data_speed)
    speed_max = np.max(data_speed)
    num_data = len(data_speed)
    #split wind direction
    direction_piecies = np.linspace(0,360,num_direction+1)
    direction_labels = direction_piecies[:-1]
    #split wind speed
    speed_pieces = np.linspace(speed_min,speed_max,num_speed+1)
    speed_labels = speed_pieces[:-1]+0.5*(speed_pieces[1]-speed_pieces[0])
    speed_ratios = speed_labels/wind_speed_in_CFD_at_height_mast
    #grouping data into different wind directions
    index = np.argmin(np.abs(np.tile(data_direction.reshape((-1,1)),(1,num_direction+1))-direction_piecies),axis=1)
    index[index==num_direction]=0#To handle 0=360
    #grouping data into different wind directions and different wind speed
    record_count = np.zeros(shape=(num_direction,num_speed))
    for ci in range(num_direction):
        index_selected = index==ci
        data_selected = np.array(data_speed[index_selected])
        speed_pieces = np.array(speed_pieces)
        record_count[ci,:],_ =  np.histogram(data_selected,bins=speed_pieces)
    pass
    probability_wind_direction = (np.sum(record_count,1)/num_data)
    probability_wind_speed_in_each_direction = np.matmul(np.diag(1./np.sum(record_count,1)),record_count)
    #adjust probability_wind_direction
    probability_wind_direction[probability_wind_direction<=threshold_probability]=0
    probability_wind_direction = probability_wind_direction/np.sum(probability_wind_direction)
    #adjust probability_wind_speed_in_each_direction
    probability_wind_speed_in_each_direction[probability_wind_speed_in_each_direction<=threshold_probability]=0
    probability_wind_speed_in_each_direction = np.matmul(np.diag(1./np.sum(probability_wind_speed_in_each_direction,1)),probability_wind_speed_in_each_direction)


    print('labels of wind directions:')
    print(direction_labels)
    print('probability_each_direction:')
    print(probability_wind_direction)
    print('labels of wind speeds:')
    print(speed_labels)
    print('Ratios of wind speeds:')
    print(speed_ratios)
    print('probability_wind_speed_in_each_direction:')
    print(probability_wind_speed_in_each_direction)


    return direction_labels,probability_wind_direction,speed_labels,speed_ratios,probability_wind_speed_in_each_direction




if __name__ == '__main__':


    loc_file = '/home/bbq/PythonLocal/WindFarmOptimization/settings/vortex110.txt'
    height_mast = 110
    num_direction = 8
    num_speed = 5
    threshold_probability = 0.1
    direction_labels,probability_wind_direction,speed_labels,speed_ratios,probability_wind_speed_in_each_direction = obtain_wind_probability(loc_file,height_mast,num_direction,num_speed,threshold_probability)























