#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 11 01:47:59 2022

@author: bbq
"""

# =============================================================================
# __init__：初始化类，你可以在此配置一些网络层需要的参数，并且也可以在此实例化tf.keras提供的一些基础算子比如DepthwiseConv2D，方便在call方法中应用；你可以在其中执行所有与输入无关的初始化。
# build：可以获得输入张量的形状，并可以进行其余的初始化。该方法可以获取输入的shape维度，方便动态构建某些需要的算子比如Pool或基于input shape构建权重；
# call: 网络层进行前向推理的实现方法；构建网络结构，进行前向传播。
# 一般常见的自定义网络层如下，其中build方法不是必需的，大部分情况下都可以省略：
# =============================================================================

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import time
import os

class Generator(keras.Model):
    def __init__(self,input_shape=(1),output_shape=(4)):
        super(Generator,self).__init__()
        self.input_layer = tf.keras.layers.Input(input_shape)
        #layer 1
        self.fc1 = layers.Dense(10,activation='relu',kernel_regularizer=tf.keras.regularizers.l2(),kernel_initializer='random_uniform',bias_initializer='random_uniform')
        self.bn1 = layers.BatchNormalization()
        #layer 2
        self.fc2 = layers.Dense(10,activation='relu',kernel_regularizer=tf.keras.regularizers.l2(),kernel_initializer='random_uniform',bias_initializer='random_uniform')
        self.bn2 = layers.BatchNormalization()
        #layer 3
        self.fc3 = layers.Dense(10,activation='relu',kernel_regularizer=tf.keras.regularizers.l2(),kernel_initializer='random_uniform',bias_initializer='random_uniform')
        self.bn3 = layers.BatchNormalization()
        #layer 4
        self.fc4 = layers.Dense(10,activation='relu',kernel_regularizer=tf.keras.regularizers.l2(),kernel_initializer='random_uniform',bias_initializer='random_uniform')
        self.bn4 = layers.BatchNormalization()
        #layer 5
        self.fc5 = layers.Dense(10,activation='relu',kernel_regularizer=tf.keras.regularizers.l2(),kernel_initializer='glorot_uniform',bias_initializer='glorot_uniform')
        self.bn5 = layers.BatchNormalization()
        #layer 6
        self.fc6 = layers.Dense(output_shape, activation='sigmoid',kernel_regularizer=tf.keras.regularizers.l2(),kernel_initializer='glorot_uniform',bias_initializer='glorot_uniform')
        self.out = self.call(self.input_layer)



    def call(self, inputs, training=None):
        #x = layers.Flatten()(inputs)
        #Reshape as 4D: (b,1,1,100)
        x = inputs
        x = self.bn1(self.fc1(x),training=training)
        #x = tf.nn.dropout(x,0.1)
        x = self.bn2(self.fc2(x),training=training)
        x = self.bn3(self.fc3(x),training=training)
        x = self.bn4(self.fc4(x),training=training)
        x = self.bn5(self.fc5(x),training=training)
        x = self.fc6(x)
        return x

    def loss(self,batch_x, is_traning):
        coordinate_ref = tf.reshape(self(batch_x,is_traning),shape=(-1,2))
        coordinate = -6+(12)*coordinate_ref
        v = coordinate[1:,:]-coordinate[0:-1,:]
        #v = v+0.1*tf.random.normal(tf.shape(v))
        length = tf.norm(v,ord=2)
        length_min = tf.reduce_min(length)
        print('length_min={0}'.format((length_min)))
        length_threshold = 1.1
        factor = 1.0 if length_min>length_threshold else -1.0*(length_threshold-length_min)
        bias = 0.0
        print('factor={0}'.format((factor)))
        print('bias={0}'.format((bias)))
        print('coordinate=\n{0}'.format((coordinate)))

        value = himmelblauXY(coordinate)
        print('value:')
        print(value)
        reward = tf.norm(value,ord=1)#,axis=0
        loss = reward*factor+bias
        print('loss={0}'.format((loss)))
        time.sleep(0.1)
        print(loss)
        return loss
    pass

    def get_loss_value(self,batch_x, is_traning):
        coordinate_ref = tf.reshape(self(batch_x,is_traning),shape=(-1,2))
        coordinate = -6+(12)*coordinate_ref
        v = coordinate[1:,:]-coordinate[0:-1,:]
        #v = v+0.1*tf.random.normal(tf.shape(v))
        length = tf.norm(v,ord=2)
        length_min = tf.reduce_min(length)
        length_threshold = 1.1
        factor = 1.0 if length_min>length_threshold else -1.0*(length_threshold-length_min)
        bias = 0.0

        value = himmelblauXY(coordinate)
        reward = tf.norm(value,ord=1)#,axis=0
        loss = reward*factor+bias
        time.sleep(0.1)
        print('loss={0}'.format((loss)))
        return loss.numpy()
    pass
# =============================================================================
# def himmelblau(coordinate):
#     x = coordinate[:,0]
#     y = coordinate[:,1]
#     return (x**2+y-11)**2+(x+y**2-7)**2
# =============================================================================

def himmelblauXY(coordinate):
    x = coordinate[:,0]
    y = coordinate[:,1]
    z = (x**2+y-11)**2+(x+y**2-7)**2
    return z


###########################################
###########################################
###########################################
###########################################
###########################################
###########################################
class ParticleSwarm():
    def __init__(self,flag_status,loc_particles=None,name=None,is_set_new_vel=False,num_points=3,num_particles=100,num_iter=100,
                 range_x=[-6,6],range_y=[-6,6],seed=np.random.randint(100),fitness_initial=10000,dist_threshold=1e5,boundaries = None,is_new_flow_field=False):

        if flag_status == 'initialize':
            self.init_first(loc_particles,num_points,num_particles,num_iter,range_x,range_y,seed,fitness_initial,dist_threshold,boundaries)
        elif flag_status == 'continue':
            self.init_continue(name,num_iter,is_set_new_vel,boundaries,is_new_flow_field)
        else:
            raise('Non-supported status.')
        pass

    pass

    def init_continue(self,dir_working,name,num_iter_new,is_set_new_vel,boundaries,is_new_flow_field):

        #name = 'Optimizer'

        # path_dir = os.getcwd()+'/logs'
        path_dir = dir_working
        path_file = path_dir+'/'+name+'_info.txt'
        path_data = path_dir+'/'+name+'_data.npz'

        with open(path_file,'r') as file:
            file.readline()
            lines =file.readlines()
            dict_info = {}
            for l in lines:
                s = l.replace('\n','')
                s0 = s.split(sep=':')
                dict_info[s0[0]]=s0[1]
            pass
            #file.close()
        pass

        #assign nessesary info
        self.seed = int(dict_info['seed'])
        self.num_points = int(dict_info['seed'])
        self.num_particles = int(dict_info['num_particles'])
        self.num_iter = int(dict_info['num_iter'])
        self.num_arvs = int(dict_info['num_arvs'])
        self.fitness_initial = float(dict_info['fitness_initial'])
        self.dist_threshold = float(dict_info['dist_threshold'])
        self.vel_threshold = float(dict_info['vel_threshold'])
        self.mu = float(dict_info['mu'])
        self.c1 = float(dict_info['c1'])
        self.c1s = float(dict_info['c1s'])
        self.c1e = float(dict_info['c1e'])
        self.c2 = float(dict_info['c2'])
        self.c2s = float(dict_info['c2s'])
        self.c2e = float(dict_info['c2e'])
        self.w = float(dict_info['w'])
        self.ws = float(dict_info['ws'])
        self.we = float(dict_info['we'])
        self.fitness_gbest = float(dict_info['fitness_gbest'])
        self.index = int(dict_info['index'])
        self.num_pbest_changed = int(dict_info['num_pbest_changed'])
        self.num_gbest_changed = int(dict_info['num_gbest_changed'])
        self.num_solutions_found = int(dict_info['num_solutions_found'])
        self.num_entirely_updated = int(dict_info['num_entirely_updated'])
        self.boundaries = boundaries



        dict_data = np.load(path_data)
        num_iter_old = self.num_iter
        if num_iter_new < num_iter_old:
            num_iter_new = num_iter_old
        pass
        self.num_iter = num_iter_new
        #assign data
        np.random.seed(self.seed)
        self.random_values = np.concatenate((dict_data['random_values'],np.random.random((num_iter_new-num_iter_old, self.num_particles,2))),axis=0)
        self.random_initial = dict_data['random_initial']
        self.seed_random = np.concatenate((dict_data['seed_random'],np.random.choice(num_iter_new-num_iter_old, size=num_iter_new-num_iter_old, replace=False)),axis=0)+self.num_iter
        self.z_initial = dict_data['z_initial']
        self.z = dict_data['z']
        self.range_x = dict_data['range_x']
        self.range_y = dict_data['range_y']
        self.loc_particles = dict_data['loc_particles']
        if is_set_new_vel == False:
            self.vel_particles = dict_data['vel_particles']
        else:
            self.set_new_vel()
        pass
        self.fitness = dict_data['fitness']
        self.fitness_pbest = dict_data['fitness_pbest']
        self.pbest = dict_data['pbest']
        self.gbest = dict_data['gbest']
        self.gbest_coord = dict_data['gbest_coord']
        self.record_fitnessbest = np.concatenate((dict_data['record_fitnessbest'],np.zeros(shape=(num_iter_new-num_iter_old,))+np.inf),axis=0)
        self.record_fitness_all = np.concatenate((dict_data['record_fitness_all'],np.zeros(shape=(num_iter_new-num_iter_old,self.num_particles))+np.inf),axis=0)
        self.record_num_pbest_changed = np.concatenate((dict_data['record_num_pbest_changed'],np.zeros(shape=(num_iter_new-num_iter_old,))+np.inf),axis=0)
        self.record_num_gbest_changed = np.concatenate((dict_data['record_num_gbest_changed'],np.zeros(shape=(num_iter_new-num_iter_old,))+np.inf),axis=0)
        self.record_num_solutions_found = np.concatenate((dict_data['record_num_solutions_found'],np.zeros(shape=(num_iter_new-num_iter_old,))+np.inf),axis=0)
        self.record_num_entirely_updated = np.concatenate((dict_data['record_num_entirely_updated'],np.zeros(shape=(num_iter_new-num_iter_old,))+np.inf),axis=0)

        if is_new_flow_field == True:
            self.reset_fitness(value=self.fitness_initial)
            self.reset_fitness_pbest(value=self.fitness_initial)
            self.reset_fitness_gbest(value=self.fitness_initial)
            self.pbest = self.loc_particles.copy()
            self.gbest = self.loc_particles[0,:].copy()
        pass


    pass

    def set_new_vel(self):
        np.random.seed(self.seed)
        self.vel_particles = 1.0*(1.0-2*self.random_initial[1,:,:]) #[-1,1]
    pass



    def init_first(self,loc_particles,num_points=3,num_particles=100,num_iter=100,range_x=[-6,6],range_y=[-6,6],seed=np.random.randint(100),fitness_initial=10000,dist_threshold=1e5,boundaries=None):
        # int values
        self.seed = seed
        self.num_points = num_points
        self.num_particles = num_particles
        self.num_iter = num_iter
        self.num_arvs=2*num_points
        self.fitness_initial = fitness_initial
        self.dist_threshold = dist_threshold
        self.vel_threshold = 1.0
        self.mu = 4
        self.c1 = None
        self.c1s=2.5
        self.c1e=0.5
        self.c2 = None
        self.c2s=0.5
        self.c2e=2.5
        self.w=None
        self.ws = 0.9
        self.we = 0.4
        self.fitness_gbest = None
        self.index = 0
        self.num_pbest_changed = 0
        self.num_gbest_changed = 0
        self.num_solutions_found = 0
        self.num_entirely_updated = 0
        #########
        np.random.seed(self.seed)
        self.random_values = np.random.random((self.num_iter, self.num_particles,2))
        self.random_initial = np.random.random((2,self.num_particles,self.num_arvs))
        self.seed_random = np.random.choice(num_iter, size=num_iter, replace=False)
        self.z_initial = np.random.random((self.num_particles,1))
        self.z = np.zeros(shape=(self.num_particles,self.num_arvs))
        self.range_x = np.array(range_x)
        self.range_y = np.array(range_y)
        self.boundaries = boundaries

        #initialize particles
        self.loc_particles = loc_particles #[-0.5,0.5] #self.loc_particles = 0.5-self.random_initial[0,:,:] #[-0.5,0.5]
        self.vel_particles = 1.0*(1.0-2*self.random_initial[1,:,:]) #[-1,1]
        self.sort_particles()
        self.fitness = None
        self.fitness_pbest = None
        # initialzing
        self.initialize_z()
        self.reset_fitness(value=fitness_initial)
        self.reset_fitness_pbest(value=fitness_initial)
        self.reset_fitness_gbest(value=fitness_initial)
        self.pbest = self.loc_particles.copy()
        self.gbest = self.loc_particles[0,:].copy()
        # recording
        #self.record_fitnessbest = np.zeros(shape=(self.num_iter,))
        self.record_fitnessbest = np.zeros(shape=(self.num_iter,))+np.inf
        self.gbest_coord = ParticleSwarm.particle_to_coordinate(self.gbest,self.range_x,self.range_y)
        self.record_fitness_all = np.zeros(shape=(self.num_iter,self.num_particles))+np.inf
        # flag of iteration


        #debugging
        self.record_num_pbest_changed = np.zeros(shape=(self.num_iter,))+np.inf
        self.record_num_gbest_changed = np.zeros(shape=(self.num_iter,))+np.inf
        self.record_num_solutions_found = np.zeros(shape=(self.num_iter,))+np.inf
        self.record_num_entirely_updated = np.zeros(shape=(self.num_iter,))+np.inf

# =============================================================================
#         self.pbest = self.loc_particles
#         self.fitness = self.fitness_all()
#         index = np.argmin(self.fitness)
#         self.gbest = self.loc_particles[index,:]
# =============================================================================
    pass




    def variable_assign(self):
        pass
    pass

    def variable_default_assign(self):
        pass
    pass

    def get_points_in_polygon():
        pass
    pass

    def initialize_z(self):
        for i in range(self.num_arvs-1):
            self.z[:,i+1] = self.mu*self.z[:,i]*(1-self.z[:,i])
        pass
    pass

    def sort_particles(self):
        for i in range(self.num_particles):
            x = self.loc_particles[i,0::2]
            y = self.loc_particles[i,1::2]
            u = self.vel_particles[i,0::2]
            v = self.vel_particles[i,1::2]
            index = np.argsort(x)
            self.loc_particles[i,0::2] = x[index]
            self.loc_particles[i,1::2] = y[index]
            self.vel_particles[i,0::2] = u[index]
            self.vel_particles[i,1::2] = v[index]
        pass
    pass

    def update_fitness(self,fitness,index_particle):
        self.fitness[index_particle] = fitness
    pass

    def update_add_fitness(self,fitness,index_particle):
        self.fitness[index_particle] += fitness
    pass

    def reset_fitness(self,value=10000):
        self.fitness = np.zeros(shape=(self.num_particles,))
        self.fitness += value
    pass

    def reset_fitness_pbest(self,value=10000):
        self.fitness_pbest = np.zeros(shape=(self.num_particles,))
        self.fitness_pbest += value
    pass

    def reset_fitness_gbest(self,value=10000):
        self.fitness_gbest = 0.0
        self.fitness_gbest += value
    pass

    def calc_vel(self,index_particle,index_iter,w_matrix):
        i = index_particle
        j = index_iter
        self.vel_particles[i,:]=(w_matrix[i,:]*self.vel_particles[i,:]
                                 +self.c1*self.random_values[j,i,0]*(self.pbest[i,:]-self.loc_particles[i,:])
                                 +self.c2*self.random_values[j,i,1]*(self.gbest-self.loc_particles[i,:]))
        self.vel_particles[i,self.vel_particles[i,:]>self.vel_threshold]=self.vel_threshold
        self.vel_particles[i,self.vel_particles[i,:]<-self.vel_threshold]=-self.vel_threshold
        #
        #self.loc_particles[i,:] += self.vel_particles[i,:]
    pass


    def is_point_in_polygon(pointTp, verts):
        verts = verts.tolist()
        #is_contains_edge=True
        x,y=pointTp[0],pointTp[1]
        try:
            x, y = float(x), float(y)
        except:
            return False
        vertx = [xyvert[0] for xyvert in verts]
        verty = [xyvert[1] for xyvert in verts]

        # N个点中，横坐标和纵坐标的最大值和最小值，判断目标坐标点是否在这个外包四边形之内
        if not verts or not min(vertx) <= x <= max(vertx) or not min(verty) <= y <= max(verty):
            return False

        # 上一步通过后，核心算法部分
        nvert = len(verts)
        is_in = False
        for i in range(nvert):
            j = nvert - 1 if i == 0 else i - 1
            if ParticleSwarm.is_in_line((x,y), verts[j], verts[i]):
                    return True
            if ((verty[i] > y) != (verty[j] > y)) and (
                        x < (vertx[j] - vertx[i]) * (y - verty[i]) / (verty[j] - verty[i]) + vertx[i]):
                is_in = not is_in

        return is_in

    def is_in_line(pointTp, vert_1,vert_2,err_lim=1e-5):

        x = pointTp[0]
        y = pointTp[1]
        x0 = vert_1[0]
        y0 = vert_1[1]
        x1 = vert_2[0]
        y1 = vert_2[1]
        err = np.sqrt(((x-x0)*(y1-y0)+(y-y0)*(x0-x1))**2/((y1-y0)**2+(x0-x1)**2))
        check_line_x = x <= max([x0,x1])+err_lim and x >= min([x0,x1])-err_lim
        Check_line_y = y <= max([y0,y1])+err_lim and y >= min([y0,y1])-err_lim
        if err < err_lim and check_line_x and Check_line_y:
            return True
        else:
            return False
        pass

    def is_point_in_boundaries(self,loc_particle_i_point_j,boundaries):
        flag = True
        num_boundaries = len(boundaries)
        for k in range(num_boundaries):
            verts = boundaries[k][:,0:2]
            x_mapped,y_mapped = ParticleSwarm.particle_to_coordinate(loc_particle_i_point_j,self.range_x,self.range_y)
            pointTp = np.array([x_mapped,y_mapped])
            flag = ParticleSwarm.is_point_in_polygon(pointTp, verts)
            if flag == False:
                break
            pass
        pass
        return flag
    pass

    def update_loc_and_adjust_vel(self,index_particle,index_iter,boundaries):
        i = index_particle


# =============================================================================
#         temp_loc_particles = self.loc_particles[i,:] + self.vel_particles[i,:]
#         index_over_max = temp_loc_particles> 0.5
#         index_over_min = temp_loc_particles<-0.5
#         value_over_max = temp_loc_particles[index_over_max]
#         value_over_min = temp_loc_particles[index_over_min]
#         temp_loc_particles[index_over_max] = ( 0.5-(value_over_max-0.5)/(value_over_max+0.5))
#         temp_loc_particles[index_over_min] = (-0.5+(value_over_min+0.5)/(value_over_min-0.5))
#
#         min_dist = ParticleSwarm.calc_min_dist_between_points_by_loc(temp_loc_particles,self.range_x,self.range_y)
#
#         if min_dist >= self.dist_threshold:
#             #vel: no change
#             #loc: change
#             self.loc_particles[i,:] = temp_loc_particles
#             self.num_solutions_found += 1
#             self.num_entirely_updated += 1
#             return
#         pass
# =============================================================================



        flag_find_new_solution = False
        np.random.seed(self.seed_random[index_iter])
        index = (np.random.choice(self.num_points, size=self.num_points, replace=False)).astype(int)

        num_changed = 0
        for k in range(self.num_points):
            j = index[k]
            loc_particle_i_point_j_new = self.loc_particles[i,2*j:2*(j+1)]+self.vel_particles[i,2*j:2*(j+1)]


            index_over_max = loc_particle_i_point_j_new> 0.5
            index_over_min = loc_particle_i_point_j_new<-0.5
            value_over_max = loc_particle_i_point_j_new[index_over_max]
            value_over_min = loc_particle_i_point_j_new[index_over_min]
            loc_particle_i_point_j_new[index_over_max] = ( 0.5-(value_over_max-0.5)/(value_over_max+0.5))
            loc_particle_i_point_j_new[index_over_min] = (-0.5+(value_over_min+0.5)/(value_over_min-0.5))

            is_point_changable_dist = self.check_is_point_changable_dist(i,j,loc_particle_i_point_j_new)

            is_point_changable_zone = self.is_point_in_boundaries(loc_particle_i_point_j_new,boundaries)

            is_point_changable = is_point_changable_dist and is_point_changable_zone

# =============================================================================
#             print('is_point_changable_dist:{}'.format(is_point_changable_dist))
#             print('is_point_changable_zone:{}'.format(is_point_changable_zone))
# =============================================================================


            if is_point_changable == True:
                #vel: change
                self.vel_particles[i,2*j:2*(j+1)] = loc_particle_i_point_j_new-self.loc_particles[i,2*j:2*(j+1)]
                #loc: change
                self.loc_particles[i,2*j:2*(j+1)] = loc_particle_i_point_j_new
                flag_find_new_solution = True
                num_changed += 1
            else:
                #vel:apply zero
                self.vel_particles[i,2*j:2*(j+1)] = 0.0
                #loc:do nothing
                #self.loc_particles[i,2*j:2*(j+1)] = self.loc_particles[i,2*j:2*(j+1)]
            pass
        pass

        if flag_find_new_solution == True:
            self.num_solutions_found += 1
        pass
        if num_changed == self.num_points:
            self.num_entirely_updated += 1
        pass

        #self.loc_particles[i,:] += self.vel_particles[i,:]
    pass

    def check_is_point_changable_dist(self,index_particle,index_point,loc_particle_i_point_j_new):
        is_point_changable = True
        min_dist_to_other_points = self.calc_min_dist_to_other_points(index_particle,index_point,loc_particle_i_point_j_new)
        if min_dist_to_other_points < self.dist_threshold:
            is_point_changable = False
        pass
        return is_point_changable
    pass




    def calc_min_dist_to_other_points(self,index_particle,index_point,loc_particle_i_point_j_new):

        num_points = self.num_points
        min_dist = 1e5

        x_0_mapped,y_0_mapped = ParticleSwarm.particle_to_coordinate(loc_particle_i_point_j_new,self.range_x,self.range_y)

        for j in range(num_points):
            if j != index_point:
                x_1_mapped,y_1_mapped = ParticleSwarm.particle_to_coordinate(self.loc_particles[index_particle,2*j:2*(j+1)],self.range_x,self.range_y)
                v = np.array([x_1_mapped-x_0_mapped,y_1_mapped-y_0_mapped])
                dist = np.linalg.norm(v,ord=2)
                if dist < min_dist:
                    min_dist = dist
                pass
            pass
        pass
        return min_dist
    pass

    def calc_min_dist_to_other_points_by_loc(loc,index_point,range_x,range_y):

        num_points = (int)(len(loc)/2)
        min_dist = 1e5

        loc_particle_i_point_j = loc[index_point*2:(index_point+1)*2]

        x_0_mapped,y_0_mapped = ParticleSwarm.particle_to_coordinate(loc_particle_i_point_j,range_x,range_y)

        for j in range(num_points):
            if j != index_point:
                x_1_mapped,y_1_mapped = ParticleSwarm.particle_to_coordinate(loc[2*j:2*(j+1)],range_x,range_y)
                v = np.array([x_1_mapped-x_0_mapped,y_1_mapped-y_0_mapped])
                dist = np.linalg.norm(v,ord=2)
                if dist < min_dist:
                    min_dist = dist
                pass
            pass
        pass
        return min_dist
    pass


    def calc_min_dist_between_points_by_loc(loc,range_x,range_y):
        num_points = (int)(len(loc)/2)
        min_dist = 1e5
        for i in range(num_points):
            dist = ParticleSwarm.calc_min_dist_to_other_points_by_loc(loc,i,range_x,range_y)
            if dist < min_dist:
                min_dist = dist
            pass
        pass
        return min_dist
    pass

    def update_particles(self,index_iter):
        j = index_iter
        self.w=self.ws-(self.ws-self.we)*(j/self.num_iter)
        self.c1 = self.c1s-(self.c1s-self.c1e)*(j/self.num_iter)
        self.c2 = self.c2s-(self.c2s-self.c2e)*(j/self.num_iter)
        w_matrix = (self.we-self.w)*self.z+self.w
        for i in range(self.num_particles):
            self.calc_vel(index_particle=i,index_iter=index_iter,w_matrix=w_matrix)
            self.update_loc_and_adjust_vel(index_particle=i,index_iter=j,boundaries = self.boundaries)
        pass
    pass

    def update_pbest_gbest(self):
        for i in range(self.num_particles):
            if self.fitness[i] < self.fitness_pbest[i]:
                self.fitness_pbest[i] = self.fitness[i]
                self.pbest[i,:]=self.loc_particles[i,:]

                self.num_pbest_changed += 1
            pass

            if self.fitness_pbest[i] < self.fitness_gbest:
                self.fitness_gbest = self.fitness_pbest[i]
                self.gbest=self.pbest[i,:]

                self.num_gbest_changed += 1
            pass
        pass
    pass

    def update_iter_index(self):
        self.index += 1
    pass

    def update_records(self):
        self.record_fitnessbest[self.index]=self.fitness_gbest
        self.gbest_coord = ParticleSwarm.particle_to_coordinate(self.gbest,self.range_x,self.range_y)
        self.record_fitness_all[self.index,:] = self.fitness
        self.record_num_pbest_changed[self.index] = self.num_pbest_changed
        self.record_num_gbest_changed[self.index] = self.num_gbest_changed
        self.record_num_solutions_found[self.index] = self.num_solutions_found
        self.record_num_entirely_updated[self.index] = self.num_entirely_updated

    pass

    def particle_to_coordinate(loc_particle,range_x,range_y):
        dx = range_x[1]-range_x[0]
        dy = range_y[1]-range_y[0]
        x = range_x[0]+dx*(loc_particle[0::2]+0.5)
        y = range_y[0]+dy*(loc_particle[1::2]+0.5)
        return x,y
    pass

    def coordinate_to_particle(x_abs,y_abs,range_x,range_y):
        dx = range_x[1]-range_x[0]
        dy = range_y[1]-range_y[0]
        loc_particle = np.zeros(shape=(len(x_abs)*2,))
        loc_particle[0::2] = (x_abs - range_x[0])/dx - 0.5
        loc_particle[1::2] = (y_abs - range_y[0])/dy - 0.5
        return loc_particle
    pass


    def himmelblau(x,y):
        z = (x**2+y-11)**2+(x+y**2-7)**2
        return z
    pass

    def fitness_himmelblau(loc_particle,range_x,range_y):

        dx = range_x[1]-range_x[0]
        dy = range_y[1]-range_y[0]
        x = range_x[0]+dx*(loc_particle[0::2]+0.5)
        y = range_y[0]+dy*(loc_particle[1::2]+0.5)
        fitness_1 = np.sum(ParticleSwarm.himmelblau(x,y))
        dist = 100
        num_points = len(x)
        for i in range(num_points):
            p1 = np.array([x[i],y[i]])
            for j in range(num_points):
                if i!=j:
                    p2 = np.array([x[j],y[j]])
                    v = p1-p2
                    dist_new = np.linalg.norm(v,ord=2)
                    dist = dist_new if dist_new<dist else dist
                pass
            pass
        pass
        if dist<1:
            fitness = fitness_1+100000*dist
        else:
            fitness = fitness_1

        return fitness
    pass

    def fitness_all(self):
        range_x = self.range_x
        range_y = self.range_y
        fitness_all = np.zeros(shape=(self.num_particles,))
        dx = range_x[1]-range_x[0]
        dy = range_y[1]-range_y[0]
        for i in range(self.num_particles):
            x = range_x[0]+dx*(self.loc_particles[i,0::2]+0.5)
            y = range_y[0]+dy*(self.loc_particles[i,1::2]+0.5)
            fitness_all[i] = np.sum(ParticleSwarm.himmelblau(x,y))
        pass
        return fitness_all
    pass

    def optimize_particles(self):
        for j in range(self.num_iter):
            for i in range(self.num_particles):
                self.w=self.ws-(self.ws-self.we)*(i/self.num_iter)
                self.vel_particles[i,:]=(self.w*self.vel_particles[i,:]
                                         +self.c1*np.random.random()*(self.pbest[i,:]-self.loc_particles[i,:])
                                         +self.c2*np.random.random()*(self.gbest-self.loc_particles[i,:]))
                self.vel_particles[i,self.vel_particles[i,:]>1]=1
                self.vel_particles[i,self.vel_particles[i,:]<-1]=-1
                #
                self.loc_particles[i,:] += self.vel_particles[i,:]
                #
                self.loc_particles[i,self.loc_particles[i,:]>0.5]=0.5
                self.loc_particles[i,self.loc_particles[i,:]<-0.5]=-0.5

                fitness_particle_i = ParticleSwarm.fitness(self.loc_particles[i,:],self.range_x,self.range_y)
                fitness_pbest = ParticleSwarm.fitness(self.pbest[i,:],self.range_x,self.range_y)
                fitness_gbest = ParticleSwarm.fitness(self.gbest,self.range_x,self.range_y)

                if fitness_particle_i < fitness_pbest:
                    self.pbest[i,:]=self.loc_particles[i,:]
                pass

                if fitness_pbest < fitness_gbest:
                    self.gbest=self.pbest[i,:]
                pass
            pass
            self.fitnessbest[j]=ParticleSwarm.fitness(self.gbest,self.range_x,self.range_y)
            self.gbest_coord = ParticleSwarm.get_coordinates(self.gbest,self.range_x,self.range_y)
        pass
    pass

    def optimize_per_iter(self):
        for i in range(self.num_particles):
            #self.w=self.ws-(self.ws-self.we)*(i/self.num_iter)
            #print(self.w)
            self.vel_particles[i,:]=(self.w*self.vel_particles[i,:]
                                     +self.c1*np.random.random()*(self.pbest[i,:]-self.loc_particles[i,:])
                                     +self.c2*np.random.random()*(self.gbest-self.loc_particles[i,:]))
            self.vel_particles[i,self.vel_particles[i,:]>1]=1
            self.vel_particles[i,self.vel_particles[i,:]<-1]=-1
            #
            self.loc_particles[i,:] += self.vel_particles[i,:]
            #
            self.loc_particles[i,self.loc_particles[i,:]>0.5]=0.5
            self.loc_particles[i,self.loc_particles[i,:]<-0.5]=-0.5

            #fitness_particle_i = ParticleSwarm.fitness(self.loc_particles[i,:],self.range_x,self.range_y)
            fitness_particle_i = self.fitness[i]
            fitness_pbest = ParticleSwarm.fitness(self.pbest[i,:],self.range_x,self.range_y)
            #fitness_gbest = ParticleSwarm.fitness(self.gbest,self.range_x,self.range_y)

            if fitness_particle_i < fitness_pbest:
                self.pbest[i,:]=self.loc_particles[i,:]
            pass
        pass

# =============================================================================
#             if fitness_pbest < fitness_gbest:
#                 self.gbest=self.pbest[i,:]
#             pass
#         pass
#         self.record_fitnessbest[self.index]=ParticleSwarm.fitness(self.gbest,self.range_x,self.range_y)
#         self.gbest_coord = ParticleSwarm.particle_to_coordinate(self.gbest,self.range_x,self.range_y)
#         self.index += 1
# =============================================================================
        temp_index = np.argmin(my_optimizer.fitness)
        my_optimizer.gbest = my_optimizer.pbest[temp_index,:]
        my_optimizer.record_fitnessbest[my_optimizer.index]=ParticleSwarm.fitness(my_optimizer.gbest,my_optimizer.range_x,my_optimizer.range_y)
        my_optimizer.gbest_coord = ParticleSwarm.particle_to_coordinate(my_optimizer.gbest,my_optimizer.range_x,my_optimizer.range_y)
        my_optimizer.index += 1

    pass

    def save_optimizer(self,dir_working,name='Optimizer'):

        #path_dir = os.getcwd()+'/logs'
        path_dir = dir_working
        if os.path.exists(path_dir) == False:
            os.makedirs(path_dir)
        pass
        path_file_info = path_dir+'/'+name+'_info.txt'

        with open(path_file_info,'w') as file:
        #---key info--
            date = time.strftime('%Y-%m-%d %H:%M:%S')
            file.write('date:'+date+'\n')
            file.write('seed:'+str(self.seed)+'\n')
            file.write('num_points:'+str(self.num_points)+'\n')
            file.write('num_particles:'+str(self.num_particles)+'\n')
            file.write('num_iter:'+str(self.num_iter)+'\n')
            file.write('num_arvs:'+str(self.num_arvs)+'\n')
            file.write('fitness_initial:'+str(self.fitness_initial)+'\n')
            file.write('dist_threshold:'+str(self.dist_threshold)+'\n')
            file.write('vel_threshold:'+str(self.vel_threshold)+'\n')
            file.write('mu:'+str(self.mu)+'\n')
            file.write('c1:'+str(self.c1)+'\n')
            file.write('c1s:'+str(self.c1s)+'\n')
            file.write('c1e:'+str(self.c1e)+'\n')
            file.write('c2:'+str(self.c2)+'\n')
            file.write('c2s:'+str(self.c2s)+'\n')
            file.write('c2e:'+str(self.c2e)+'\n')
            file.write('w:'+str(self.w)+'\n')
            file.write('ws:'+str(self.ws)+'\n')
            file.write('we:'+str(self.we)+'\n')
            file.write('fitness_gbest:'+str(self.fitness_gbest)+'\n')
            file.write('index:'+str(self.index)+'\n')
            file.write('num_pbest_changed:'+str(self.num_pbest_changed)+'\n')
            file.write('num_gbest_changed:'+str(self.num_gbest_changed)+'\n')
            file.write('num_solutions_found:'+str(self.num_solutions_found)+'\n')
            file.write('num_entirely_updated:'+str(self.num_entirely_updated)+'\n')
            file.flush()
        pass


        #---data--
        path_file_data = path_dir+'/'+name+'_data.npz'
        np.savez(path_file_data,
                 random_values = self.random_values,
                 random_initial = self.random_initial,
                 range_x = self.range_x,
                 range_y = self.range_y,
                 loc_particles = self.loc_particles,
                 vel_particles = self.vel_particles,
                 seed_random = self.seed_random,
                 z_initial = self.z_initial,
                 z = self.z,
                 fitness = self.fitness,
                 fitness_pbest = self.fitness_pbest,
                 pbest = self.pbest,
                 gbest = self.gbest,
                 record_fitnessbest = self.record_fitnessbest,
                 gbest_coord = self.gbest_coord,
                 record_fitness_all = self.record_fitness_all,
                 record_num_pbest_changed = self.record_num_pbest_changed,
                 record_num_gbest_changed = self.record_num_gbest_changed,
                 record_num_solutions_found = self.record_num_solutions_found,
                 record_num_entirely_updated = self.record_num_entirely_updated,
                 )

        # data = np.load(path_data)
        # print(data['random_values'])
    pass

    def save_optimizer_target(target,name='Optimizer'):

        path_dir = os.getcwd()+'/logs'
        if os.path.exists(path_dir) == False:
            os.makedirs(path_dir)
        pass
        path_file_info = path_dir+'/'+name+'_info.txt'

        with open(path_file_info,'w') as file:
        #---key info--
            date = time.strftime('%Y-%m-%d %H:%M:%S')
            file.write('date:'+date+'\n')
            file.write('seed:'+str(target.seed)+'\n')
            file.write('num_points:'+str(target.num_points)+'\n')
            file.write('num_particles:'+str(target.num_particles)+'\n')
            file.write('num_iter:'+str(target.num_iter)+'\n')
            file.write('num_arvs:'+str(target.num_arvs)+'\n')
            file.write('fitness_initial:'+str(target.fitness_initial)+'\n')
            file.write('dist_threshold:'+str(target.dist_threshold)+'\n')
            file.write('vel_threshold:'+str(target.vel_threshold)+'\n')
            file.write('mu:'+str(target.mu)+'\n')
            file.write('c1:'+str(target.c1)+'\n')
            file.write('c1s:'+str(target.c1s)+'\n')
            file.write('c1e:'+str(target.c1e)+'\n')
            file.write('c2:'+str(target.c2)+'\n')
            file.write('c2s:'+str(target.c2s)+'\n')
            file.write('c2e:'+str(target.c2e)+'\n')
            file.write('w:'+str(target.w)+'\n')
            file.write('ws:'+str(target.ws)+'\n')
            file.write('we:'+str(target.we)+'\n')
            file.write('fitness_gbest:'+str(target.fitness_gbest)+'\n')
            file.write('index:'+str(target.index)+'\n')
            file.write('num_pbest_changed:'+str(target.num_pbest_changed)+'\n')
            file.write('num_gbest_changed:'+str(target.num_gbest_changed)+'\n')
            file.write('num_solutions_found:'+str(target.num_solutions_found)+'\n')
            file.write('num_entirely_updated:'+str(target.num_entirely_updated)+'\n')
            file.flush()
        pass


        #---data--
        path_file_data = path_dir+'/'+name+'_data.npz'
        np.savez(path_file_data,
                 random_values = target.random_values,
                 random_initial = target.random_initial,
                 range_x = target.range_x,
                 range_y = target.range_y,
                 loc_particles = target.loc_particles,
                 vel_particles = target.vel_particles,
                 seed_random = target.seed_random,
                 z_initial = target.z_initial,
                 z = target.z,
                 fitness = target.fitness,
                 fitness_pbest = target.fitness_pbest,
                 pbest = target.pbest,
                 gbest = target.gbest,
                 record_fitnessbest = target.record_fitnessbest,
                 gbest_coord = target.gbest_coord,
                 record_fitness_all = target.record_fitness_all,
                 record_num_pbest_changed = target.record_num_pbest_changed,
                 record_num_gbest_changed = target.record_num_gbest_changed,
                 record_num_solutions_found = target.record_num_solutions_found,
                 record_num_entirely_updated = target.record_num_entirely_updated,
                 )

        # data = np.load(path_data)
        # print(data['random_values'])
    pass


    def draw_iteration_status(self):
        figure = plt.figure()
        axes = figure.add_subplot(1,1,1)
        num_particles = self.num_particles
        index = self.index
        for i_particle in range(num_particles):
            axes.scatter(np.arange(self.index),
                         (self.record_fitness_all[0:index,i_particle]).flatten(),
                         marker ='o',
                         s = 1,
                         color='gray')
        pass
        axes.plot(np.arange(index),
                  self.record_fitnessbest[0:index],
                  linewidth =2.0,
                  color='black')

        figure = plt.figure()
        axes = figure.add_subplot(1,4,1)
        axes.plot(np.arange(index),self.record_num_pbest_changed[0:index],linewidth =2.0)
        axes.set_title('pbest_changed')
        axes = figure.add_subplot(1,4,2)
        axes.plot(np.arange(index),self.record_num_gbest_changed[0:index],linewidth =2.0)
        axes.set_title('gbest_changed')
        axes = figure.add_subplot(1,4,3)
        axes.plot(np.arange(index),self.record_num_solutions_found[0:index],linewidth =2.0)
        axes.set_title('solutions_found')
        axes = figure.add_subplot(1,4,4)
        axes.plot(np.arange(index),self.record_num_entirely_updated[0:index],linewidth =2.0)
        axes.set_title('entirely_updated')
        return figure
    pass


    def assemble_data_from_queues(self,index_range,list_queues):
        num_cores = len(list_queues)
        #int
        fitness_gbest_cores = np.zeros((num_cores,))
        num_pbest_changed_cores = np.zeros((num_cores,))
        num_gbest_changed_cores = np.zeros((num_cores,))
        num_solutions_found_cores = np.zeros((num_cores,))
        num_entirely_updated_cores = np.zeros((num_cores,))

        record_num_pbest_changed_cores = np.zeros((num_cores,))
        record_num_gbest_changed_cores = np.zeros((num_cores,))
        record_num_solutions_found_cores = np.zeros((num_cores,))
        record_num_entirely_updated_cores = np.zeros((num_cores,))
        #array
        list_gbest = [[] for i in range(num_cores)]
        list_gbest_coord = [[] for i in range(num_cores)]
        #record
        list_record_fitnessbest = [[] for i in range(num_cores)]#shape=(self.num_iter,)
        list_record_fitness_all = [[] for i in range(num_cores)]#shape=(self.num_iter,self.num_particles)
        for i_core in range(num_cores):
            indexs_row = index_range[i_core]
            #int values
            fitness_gbest_cores[i_core] = list_queues[i_core].get()
            num_pbest_changed_cores[i_core] = list_queues[i_core].get()
            num_gbest_changed_cores[i_core] = list_queues[i_core].get()
            num_solutions_found_cores[i_core] = list_queues[i_core].get()
            num_entirely_updated_cores[i_core] = list_queues[i_core].get()

            record_num_pbest_changed_cores[i_core] = list_queues[i_core].get()
            record_num_gbest_changed_cores[i_core] = list_queues[i_core].get()
            record_num_solutions_found_cores[i_core] = list_queues[i_core].get()
            record_num_entirely_updated_cores[i_core] = list_queues[i_core].get()


            #array values
            self.loc_particles[indexs_row[0]:indexs_row[1],:] = list_queues[i_core].get()
            self.vel_particles[indexs_row[0]:indexs_row[1],:] = list_queues[i_core].get()
            self.fitness_pbest[indexs_row[0]:indexs_row[1],:] = list_queues[i_core].get()
            self.pbest[indexs_row[0]:indexs_row[1],:] = list_queues[i_core].get()

            #To be compared
            list_gbest[i_core] = list_queues[i_core].get()
            list_gbest_coord[i_core] = list_queues[i_core].get()
            list_record_fitnessbest[i_core] = list_queues[i_core].get()
            list_record_fitness_all[i_core] = list_queues[i_core].get()




            # flag of iteration


            #debugging
            self.record_num_pbest_changed = np.zeros(shape=(self.num_iter,))+np.inf
            self.record_num_gbest_changed = np.zeros(shape=(self.num_iter,))+np.inf
            self.record_num_solutions_found = np.zeros(shape=(self.num_iter,))+np.inf
            self.record_num_entirely_updated = np.zeros(shape=(self.num_iter,))+np.inf

    # =============================================================================
    #         self.pbest = self.loc_particles
    #         self.fitness = self.fitness_all()
    #         index = np.argmin(self.fitness)
    #         self.gbest = self.loc_particles[index,:]
    # =============================================================================












        pass
    pass



if __name__ == '__main__':
    #print(ParticleSwarm.fitness(np.array([-0.5,0.5]),[-6,-6],[-6,6]))
    num_points = 20
    num_iter = 200
    num_particles=800
    my_optimizer = ParticleSwarm(num_points=num_points,
                                 num_particles=num_particles,
                                 num_iter=num_iter)
    for i in range(num_iter):
        #reset fitness befor iteration
        my_optimizer.reset_fitness(value = 0)
        # uupdate fitness
        for j in range(num_particles):
            fitness_particle_j = ParticleSwarm.fitness_himmelblau(my_optimizer.loc_particles[j,:],my_optimizer.range_x,my_optimizer.range_y)
            my_optimizer.update_add_fitness(1.0*fitness_particle_j,index_particle=j)
        pass
        my_optimizer.update_pbest_gbest()
        my_optimizer.update_particles(index_iter=i)

        my_optimizer.update_records()
        my_optimizer.update_iter_index()
        print('iter={0}'.format(i))
    pass

    gbest_x,gbest_y = my_optimizer.gbest_coord
    print(gbest_x)
    print(gbest_y)


    figure = plt.figure()
    axes = figure.add_subplot(1,1,1)
    axes.plot(np.arange(my_optimizer.num_iter),my_optimizer.record_fitnessbest)


    x = np.arange(-6,6,0.1)
    y = np.arange(-6,6,0.1)
    #print('x,y range:',x.shape,y.shape)
    #X,Y = np.meshgrid(x,y)
    X,Y = np.mgrid[-6:6:0.1,-6:6:0.1]
    print('X,Y maps:', X.shape, Y.shape)
    XY = tf.Variable([X.flatten(),Y.flatten()])
    XY = tf.transpose(XY,[1,0])
    Z = himmelblauXY(XY)
    Z = tf.reshape(Z,X.shape)

    fig = plt.figure()#constrained_layout=True
    ax = fig.add_subplot(1,1,1)
    ax.xaxis.set_label_text('X')
    ax.yaxis.set_label_text('Y')
    ax.set_xlim(xmin=np.min(x), xmax=np.max(x))
    ax.set_ylim(ymin=np.min(y), ymax=np.max(y))
    num = 10
    ax.contourf(X,Y,Z,num,alpha=1,cmap='hsv')
    C = plt.contour(X,Y,Z,num,colors='black',linewidth=.4)
    plt.clabel(C,inline=True,fontsize=10)
    ax.scatter(3,2,color='g')
    ax.scatter(-2.805,3.131,color='g')
    ax.scatter(-3.779,-3.283,color='g')
    ax.scatter(3.584,-1.848,color='g')
    ax.scatter(gbest_x,gbest_y,color='w',s=5)
# =============================================================================
#     x = np.arange(-6,6,0.1)
#     y = np.arange(-6,6,0.1)
#     #print('x,y range:',x.shape,y.shape)
#     #X,Y = np.meshgrid(x,y)
#     X,Y = np.mgrid[-6:6:0.1,-6:6:0.1]
#     print('X,Y maps:', X.shape, Y.shape)
#     XY = tf.Variable([X.flatten(),Y.flatten()])
#     XY = tf.transpose(XY,[1,0])
#     Z = himmelblauXY(XY)
#     Z = tf.reshape(Z,X.shape)
#
#     fig = plt.figure()#constrained_layout=True
#     ax = fig.add_subplot(1,1,1, projection='3d')
#     ax.xaxis.set_label_text('X')
#     ax.yaxis.set_label_text('Y')
#     ax.zaxis.set_label_text('Z')
#     ax.set_xlim(xmin=np.min(x), xmax=np.max(x))
#     ax.set_ylim(ymin=np.min(y), ymax=np.max(y))
#     ax.set_zlim(zmin=np.min(Z), zmax=np.max(Z))
#     ax.plot_surface(X,Y,Z)
#     ax.view_init(elev=60, azim=-30, vertical_axis='z')
#
#     input_shape = 1
#     epochs = 1000
#     batch_size = 1
#     learning_rate = 0.1
#     is_traning = True
#     #input_init = tf.Variable([[0.0]], dtype=tf.float32)
#     input_init = tf.Variable([[0]])
#     my_generator = Generator()
#     my_generator.build(input_shape = (batch_size,1))
#     my_generator.summary()
#     optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate, beta_1 = 0.5)
#     #output_init = my_generator(input_init)
#
#     # aaaa = my_generator(input_init,False)
#     # tf.reshape(my_generator(input_init,is_traning),shape=(-1,2))
#
#     # x = np.array([[1,2]])
#     for step in range (epochs):
#         with tf.GradientTape() as tape:
#             loss = my_generator.loss(input_init,is_traning)
#         grads = tape.gradient(loss, my_generator.trainable_variables)
#         optimizer.apply_gradients(zip(grads,my_generator.trainable_variables))
#         if step %20 == 0:
#             print('step{0}:x={1},loss={2}'.format(step,my_generator(input_init)[0].numpy(),loss))
#         pass
#     pass
#
#     #print(himmelblau(my_generator(input_init)[0]))
# =============================================================================

# =============================================================================
#     for epoch in range(epochs):
#         for _ in range(5):
#             batch_z = tf.random.normal([batch_size,z_dim])
#             batch_x = next(db_iter)
# =============================================================================






















