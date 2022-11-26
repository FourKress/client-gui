#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2022-05-27

@author: like
"""

import numpy as np
import matplotlib.pyplot as plt


class Wind_turbine():

    # Define a wind turbine's location and characteristics
    # Dalian: self,coordinate = np.array([0,0]),height =0, r_fan=88,h_hub=115,CT=0.8888, z0=0.3, CP = 0.4
    # 2D:     self,coordinate = np.array([0,0]),height =0, r_fan=20,h_hub=78,CT=0.8888, z0=0.3, CP = 0.4
    def __init__(self, coordinate=np.array([0, 0]), height=0, r_fan=20, h_hub=78, CT=0.8888, z0=0.3, CP=0.4):
        assert len(coordinate) == 2
        self.coordinate = np.array(coordinate)  # [2,]
        self.height = height  # defalut 0, means the terrain is not considered
        self.r_fan = r_fan
        self.h_hub = h_hub  # hub hight
        self.CT = CT
        self.CP = CP
        # Set information from wind farm
        self.z0 = z0  # roughness lenth
        # Calculat other key parameters
        self.alpha = 0.5 / np.log(h_hub / z0)  # diffusion coefficient
        self.a = (1 - (1 - CT) ** 0.5) / 2  # actial induction factor
        self.Ia = 0.077  # self.alpha/0.4/2 #Turbulence intensity

        self.ki = 0.11 * self.CT ** 1.07 * self.Ia ** 0.20  # Wake model coefficient
        self.ai = 0.93 * self.CT ** -0.75 * self.Ia ** 0.17  # Wake model coefficient
        self.b = 0.42 * self.CT ** 0.6 * self.Ia ** 0.2  # Wake model coefficient
        self.c = 0.15 * self.CT ** -0.25 * self.Ia ** -0.7  # Wake model coefficient
        self.rd = self.r_fan * ((1 - self.a) / (1 - 2 * self.a)) ** 0.5;
        self.epsilon = 0.23 * self.CT ** -0.25 * self.Ia ** 0.17
        self.factor_reduce = None
        self.wind_org = np.array([None, None, None])
        self.wind_hub = None
        self.index = None
        self.falg_in_boundary = None
        self.belonging_zone = None
        self.min_dist_to_boundary = None

    # =============================================================================
    #         self.r_wake_initial = self.r_fan*np.sqrt((1-self.a)/(1-2*self.a))
    #         self.area_fan = np.pi*r_fan**2
    # =============================================================================

    # Calculate windward distance from turbineA to turbineB
    # direction is the angle in rad
    # =============================================================================
    #     def get_windward_distance_wrong(direction,coordinate_a, coordinate_b):
    #         vector_direction = np.array([np.cos(direction),np.sin(direction)])
    #         vector_distance = np.array(coordinate_b-coordinate_a)
    #         #windward_distance = np.dot(vector_direction,vector_distance)
    #         #vector_direction = np.array([1.,1.])
    #         #vector_distance = np.array([1.,0.])
    #         cos_ = np.dot(vector_distance,vector_direction)/(np.linalg.norm(vector_distance)*np.linalg.norm(vector_direction))
    #         sin_ = np.cross(vector_distance,vector_direction)/(np.linalg.norm(vector_distance)*np.linalg.norm(vector_direction))
    #         angle = np.arctan2(sin_, cos_)
    #         #print('Angle = {0} degree'.format(angle/np.pi*180))
    #         if 0.5*np.pi <= angle and angle <= np.pi:
    #             windward_distance = 0
    #         elif -np.pi <= angle and angle <= -0.5*np.pi:
    #             windward_distance = 0
    #         else:
    #             windward_distance = np.linalg.norm(vector_distance)/np.cos(angle)
    #         #print('windward_distance = {0}'.format(windward_distance))
    #         return windward_distance
    # =============================================================================

    def update_wind_hub(self):
        u_mag = np.linalg.norm(self.wind_org, ord=2)
        self.wind_hub = u_mag * (1 - self.factor_reduce)

    def update_para(self):
        u = self.wind_hub
        if u < 3:
            self.CT = -0.141 * 3 + 1.339
        elif 3 <= u and u < 4:
            self.CT = -0.141 * u + 1.339
        elif 4 <= u and u < 6:
            self.CT = 0.781
        elif 6 <= u and u < 14:
            self.CT = 0.003004 * u ** 3 - 0.09354 * u ** 2 + 0.8484 * u - 1.6
        elif 14 <= u and u < 25:
            self.CT = -0.0001281 * u ** 3 + 0.0087 * u ** 2 - 0.2029 * u + 1.666
        elif 25 <= u:
            self.CT = -0.0001281 * 25 ** 3 + 0.0087 * 25 ** 2 - 0.2029 * 25 + 1.666

        self.a = (1 - (1 - self.CT) ** 0.5) / 2  # actial induction factor
        self.ki = 0.11 * self.CT ** 1.07 * self.Ia ** 0.20  # Wake model coefficient
        self.ai = 0.93 * self.CT ** -0.75 * self.Ia ** 0.17  # Wake model coefficient
        self.b = 0.42 * self.CT ** 0.6 * self.Ia ** 0.2  # Wake model coefficient
        self.c = 0.15 * self.CT ** -0.25 * self.Ia ** -0.7  # Wake model coefficient
        self.rd = self.r_fan * ((1 - self.a) / (1 - 2 * self.a)) ** 0.5
        self.epsilon = 0.23 * self.CT ** -0.25 * self.Ia ** 0.17

    pass

    def get_CT(speeds):
        num_speeds = len(speeds)
        CTs = np.zeros((num_speeds,))
        for i in range(num_speeds):
            u = speeds[i]
            if u < 3:
                CTs[i] = -0.141 * 3 + 1.339
            elif 3 <= u and u < 4:
                CTs[i] = -0.141 * u + 1.339
            elif 4 <= u and u < 6:
                CTs[i] = 0.781
            elif 6 <= u and u < 14:
                CTs[i] = 0.003004 * u ** 3 - 0.09354 * u ** 2 + 0.8484 * u - 1.6
            elif 14 <= u and u < 25:
                CTs[i] = -0.0001281 * u ** 3 + 0.0087 * u ** 2 - 0.2029 * u + 1.666
            elif 25 <= u:
                CTs[i] = -0.0001281 * 25 ** 3 + 0.0087 * 25 ** 2 - 0.2029 * 25 + 1.666
        pass
        return CTs

    pass

    def get_power(speeds):
        num_speeds = len(speeds)
        powers = np.zeros((num_speeds,))
        for i in range(num_speeds):
            u = speeds[i]
            if u <= 3 or u > 25:
                continue
            elif u > 3 and u <= 12:
                powers[i] = (-23.95 * u ** 3
                             + 545.4 * u ** 2
                             - 2994 * u
                             + 5072)
            elif u > 12 and u <= 19:
                powers[i] = 6250
            elif u > 19 and u <= 25:
                powers[i] = (-116.9 * u ** 2
                             + 4552 * u
                             - 38020)
        pass
        return powers

    pass

    #   y
    #   ^
    #   |
    #   |
    #   |
    #   ----------->x
    # <direction>: wind angle (rad). anti-clockwise started from x axis
    # downwind_distance: If the returned distance is negative, it means coordinate_a is a windward point
    # crosswind_distance: always positive
    def get_distance(direction, coordinate_a, coordinate_b):
        vector_direction = np.array([np.cos(direction), np.sin(direction)])
        vector_distance = np.array(coordinate_b - coordinate_a)
        downwind_distance = np.dot(vector_direction, vector_distance)
        crosswind_distance = np.sqrt((np.linalg.norm(vector_distance)) ** 2 - downwind_distance ** 2)

        return (downwind_distance, crosswind_distance)

    # Calculate area_overlap for turbine_b by turbin_a
    def get_area_overlap(direction, turbine_a, turbine_b):
        coordinate_a = turbine_a.coordinate
        coordinate_b = turbine_b.coordinate
        (x, d) = Wind_turbine.get_distance(direction, coordinate_a, coordinate_b)
        if x < 0:
            raise Exception(print("error: downwind_distance < 0"))
        pass
        R = turbine_a.r_wake_initial + turbine_a.alpha * x
        Rr = turbine_b.r_fan
        if R <= Rr:
            raise Exception(print("Check: R <= Rr"))
        if d < R - Rr:
            area_overlap = np.pi * Rr ** 2
        elif R - Rr < d and d < R:
            angle_a = np.arccos((R ** 2 + d ** 2 - Rr ** 2) / (2 * R * d))
            angle_b = np.pi - np.arccos((Rr ** 2 + d ** 2 - R ** 2) / (2 * Rr * d))
            area_overlap = angle_a * R ** 2 + angle_b * Rr ** 2 - R * d * np.sin(angle_a)
        elif R < d and d < R + Rr:
            angle_a = np.arccos((R ** 2 + d ** 2 - Rr ** 2) / (2 * R * d))
            angle_b = np.arccos((Rr ** 2 + d ** 2 - R ** 2) / (2 * Rr * d))
            area_overlap = angle_a * R ** 2 + angle_b * Rr ** 2 - R * d * np.sin(angle_a)
        elif R + Rr < d:
            area_overlap = 0
        pass
        return area_overlap

    # Calculate speed_loss_factor for turbine_b by turbin_a
    def get_speed_loss_factor_b_from_a(direction, turbine_a, turbine_b):
        if Wind_turbine.is_too_closed(turbine_a, turbine_b) == True:
            speed_loss_factor = 1
            return speed_loss_factor
        pass
        (downwind_distance, crosswind_distance) = Wind_turbine.get_distance(direction, turbine_a.coordinate,
                                                                            turbine_b.coordinate)
        if downwind_distance <= 0:
            speed_loss_factor = 0
            return speed_loss_factor
        pass
        # turbine_b is in the downstream direction of turbine_a
        CT = turbine_a.CT
        alpha = turbine_a.alpha
        RD = turbine_a.r_wake_initial
        A_overlap = Wind_turbine.get_area_overlap(direction, turbine_a, turbine_b)
        A = turbine_b.area_fan
        speed_loss_factor = np.sqrt(((1 - np.sqrt(1 - CT)) / (1 + alpha * downwind_distance / RD)) ** 2 *
                                    np.sqrt(A_overlap / A))

        return speed_loss_factor

    def get_speed_loss_factor(direction, list_turbines):
        num_turbines = len(list_turbines)
        speed_loss_factors = np.zeros(shape=(num_turbines,))
        for i in range(num_turbines):
            for j in range(num_turbines):
                if j == i:
                    continue
                pass
                # Calculate speed_loss_factor for turbine_b by turbin_a
                # get_speed_loss_factor_b_from_a(direction,turbine_a, turbine_b)
                speed_loss_i_from_j = Wind_turbine.get_speed_loss_factor_b_from_a(direction, list_turbines[j],
                                                                                  list_turbines[i])
                print('speed_loss_{0}({1})_from_{2}({3}={4})'.format(i, list_turbines[i].coordinate, j,
                                                                     list_turbines[j].coordinate, speed_loss_i_from_j))
                speed_loss_factors[i] += speed_loss_i_from_j ** 2
            pass
            if speed_loss_factors[i] >= 1.0:
                speed_loss_factors[i] = 1
            pass
            speed_loss_factors[i] = np.sqrt(speed_loss_factors[i])
        pass
        return speed_loss_factors

    def cal_wind_reduction_for_all(self):
        pass

    pass

    def cal_wind_reduction(self):
        # =============================================================================
        #             U0=12;%初始来流风速
        #     if i==1 %第一个风机无风速亏损
        #         u=(U0*log(z/z0)/log(H/z0));%剖面曲线z高度处风速
        #     else
        #         umiddle=0;
        #         for j=1:i-1%对该风机之前的每台风机
        #             x=(XY(i,2)-XY(j,2));%计算前后风机距离
        #             yi=(XY(i,1)-XY(j,1));%计算前后风机横向距离
        #             yia=abs(XY(i,1)-XY(j,1));%计算前后风机横向距离的绝对值
        #             theta=(ki*x/D+epsilon)*D;%尾流模型系数
        #             RR=rd+x*alpha;%尾流影响半径
        #             if (XY(j,1)+RR+R > XY(i,1) && XY(j,1)-RR-R < XY(i,1))%如果风轮扫掠面积和尾流影响面积有重叠部分，就计算尾流亏损
        #                 umiddle=umiddle+(1/(ai+b*x/D+c*(1+x/D)^-2)^2*exp(-(((y+yi).^2+(z-H).^2).^0.5).^2/(2*theta^2))).^2;
        #             else
        #                 umiddle=umiddle;
        #             end
        #         end
        #         u=(U0*log(z/z0)/log(H/z0)).*(1-umiddle.^0.5);%计算风速
        # =============================================================================

        pass

    pass

    def is_too_closed(turbine_a, turbine_b):
        vector = turbine_a.coordinate - turbine_b.coordinate
        distance = np.linalg.norm(vector)
        if distance <= turbine_a.r_fan + turbine_b.r_fan:
            return True
        else:
            return False

    def get_speed_loss_factor_b(direction, list_turbine_a, turbine_b):
        pass


if __name__ == '__main__':

    turbine_a = Wind_turbine(np.array([0, 0]))
    turbine_b = Wind_turbine(np.array([-40, 1]))

    num = 6
    x = np.linspace(0, 10000, num=num)
    y = np.linspace(0, 10000, num=num)
    list_turbines = []
    for i in range(num):
        for j in range(num):
            list_turbines.append(Wind_turbine(np.array([x[i], y[j]])))
        pass
    pass
    direction = 90 / 180 * np.pi

    speed_loss_factors = Wind_turbine.get_speed_loss_factor(direction, list_turbines)

    fig, ax = plt.subplots()
    for i in range(num):
        for j in range(num):
            ax.scatter(x[i], y[j], c='none', s=(1 - speed_loss_factors[i * num + j]) * 500, edgecolors='black')
        pass
    pass
    legend = ax.legend()

# =============================================================================
# num = 101        
# x = np.linspace(0, 100, num=num)
# theta = np.linspace(90, 180, num=5)
# theta = [0]
# turbine_a = Wind_turbine(np.array([0,0]))
# list_speed_loss_factor = np.zeros(shape=(len(theta),num))
# downwind_distance = np.zeros(shape=(len(theta),num))
# crosswind_distance = np.zeros(shape=(len(theta),num))
# for j in range(len(theta)):
#     direction = theta[j]/180*np.pi
#     for i in range(num):
#         turbine_b = Wind_turbine(np.array([x[i],0]))
#         (downwind_distance[j,i],crosswind_distance[j,i]) = Wind_turbine.get_distance(direction, turbine_a.coordinate, turbine_b.coordinate)
#         list_speed_loss_factor[j,i] = Wind_turbine.get_speed_loss_factor_b_from_a(direction,turbine_a, turbine_b)
#     pass
# pass
# 
# fig,ax = plt.subplots()
# for j in range(len(theta)):
#     ax.plot(x, list_speed_loss_factor[j,:],label=theta[j])
# pass
# legend = ax.legend()
# =============================================================================


# turbine_a = Wind_turbine(np.array([0,0]))
# turbine_b = Wind_turbine(np.array([40.1,0]))
# direction = 90/180*np.pi
# (downwind_distance,crosswind_distance) = Wind_turbine.get_distance(direction, turbine_a.coordinate, turbine_b.coordinate)
# print('downwind_distance = {0}, crosswind_distance = {1}'.format(downwind_distance,crosswind_distance))
# area_overlap = Wind_turbine.get_area_overlap(direction,turbine_a, turbine_b)
# print('Overlapped area = {0}'.format(area_overlap))
# speed_loss_factor = Wind_turbine.get_speed_loss_factor_b_from_a(direction,turbine_a, turbine_b)
# print('speed_loss_factor = {0}'.format(speed_loss_factor))


# =============================================================================
#     turbine_a = Wind_turbine(np.array([0,0]))
#     turbine_b = Wind_turbine(np.array([-40,1]))
#     direction = 90/180*np.pi
#     (downwind_distance,crosswind_distance) = Wind_turbine.get_distance(direction, turbine_a.coordinate, turbine_b.coordinate)
#     print(downwind_distance,crosswind_distance)
# =============================================================================
