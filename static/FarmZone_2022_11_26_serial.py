# multiple processor
# 2022-11-03
# Probability of wind and its ratio to the CFD results is considered
# start offictial optimization
# 2022-11-09
# change the contour drawing functions to consider probability
# 2022-11-12
# Adjust the program to make a GUI

import sys

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas
import os
import time


import scipy.interpolate as interpolate


from WindTurbine_2022_11_26 import Wind_turbine
from FlowField_2022_11_26 import FlowField
from Optimizer_2022_11_26 import ParticleSwarm
import Obtain_wind_probability_2022_11_26 as PreProcessing


import multiprocessing as mp

import json

matplotlib.use('Agg')

params = json.loads(sys.argv[1])

class Farm_zone():

    def __init__(self):
        #<boundaries>: Contain several <boundary>
        #each <boundary>: Contain start points and an end points
        #shape=(N,4), each row represents a boundary line
        #Each line contains 4 numbers, namely X0,Y0,X1,Y1
        self.boundaries = []
        self.list_turbines = [] # It contains several Turbines (class: WindTurbine)
        self.list_turbine_index = [] # It contains several Turbines (class: WindTurbine)
        #flow field information
        self.flowfield = FlowField()
        self.dict_power = {}
        self.num_wind_directions = 0
        self.num_turbines = 0


    def initialize_list_power(self):
        self.num_wind_directions = len(self.flowfield.dict_wind_info)
        list_directions = list(self.flowfield.dict_wind_info.keys())
        for i in range(self.num_wind_directions):
            direction = list_directions[i]
            self.dict_power.update({direction:0.0})
        pass
    pass


    def update_list_power(self,direction):
        self.dict_power[direction] = 0.0
        for i in range(self.num_turbines):
            self.dict_power[direction] += self.flowfield.dict_wind_info[direction]['probability'] * self.list_turbines[i].power
        pass
    pass



    def get_power_direction(self,direction):
        return self.dict_power[direction]
    pass

    def clear_turbines(self):
        self.list_turbines = [] # It contains several Turbines (class: WindTurbine)
        self.list_turbine_index = [] # It contains several Turbines (class: WindTurbine)
        self.num_turbines = 0
        self.flowfield.locations = []
        self.flowfield.l_local_wind_direction = []
    pass

    def add_turbines(self, turbine: Wind_turbine):
        #if index not existed
        self.list_turbines.append(turbine)
        index = len(self.list_turbine_index)
        self.list_turbines[index].index = index
        self.list_turbine_index.append(index)
        self.flowfield.locations.append(np.array([turbine.coordinate[0],turbine.coordinate[1],turbine.height]))
        self.flowfield.l_local_wind_direction.append(np.array([None,None,None]))
        self.num_turbines += 1
    pass



    def update_turbines_coordinate(self,index,coordinate):
        self.list_turbines[index].coordinate = np.array(coordinate)
        self.flowfield.locations[index][0:2] = np.array([coordinate[0],coordinate[1]])
    pass

    #def assign_perturbation(self,)

    def update_turbine_heights(self):
        #calculate and assigne the height of the turbines
        X_ground = self.flowfield.X_ground
        Y_ground = self.flowfield.Y_ground
        Z_ground = self.flowfield.Z_ground
        dx = X_ground[1,0]-X_ground[0,0]
        dy = Y_ground[0,1]-Y_ground[0,0]
        x = np.arange(X_ground[0,0], X_ground[-1,0]+dx, dx)
        y = np.arange(Y_ground[0,0], Y_ground[0,-1]+dy, dy)
        z = Z_ground.transpose()
        f = interpolate.interp2d(x, y, z, kind='linear')
        num_points = len(self.list_turbines)
        for i in range(num_points):
            self.list_turbines[i].height = f(self.list_turbines[i].coordinate[0],self.list_turbines[i].coordinate[1])[0]
            self.flowfield.locations[i][2] = self.list_turbines[i].height+self.list_turbines[i].h_hub
        pass
    pass

    def update_turbine_loc_wind_for_all(self,direction):
        #calculate and assigne the height of the turbines
        num_turbines = len(self.list_turbines)
        for i in range(num_turbines):
            self.list_turbines[i].wind_org = self.flowfield.get_3D_interporlation(direction,
                                                                                  [self.list_turbines[i].coordinate[0],
                                                                                   self.list_turbines[i].coordinate[1],
                                                                                   self.list_turbines[i].height])
            self.flowfield.l_local_wind_direction[i] = self.list_turbines[i].wind_org/np.linalg.norm(self.list_turbines[i].wind_org,ord=2)
        pass
    pass

    def update_point_loc_wind_for_all(self,direction,coords_point):
        #calculate and assigne the height of the turbines
        num_points = np.shape(coords_point)[0]
        l_local_wind_direction = []
        for i in range(num_points):
            l_local_wind_direction.append(np.array([None,None,None]))
        pass
        for i in range(num_points):
            wind_org = self.flowfield.get_3D_interporlation(direction,coords_point[i,:])
            l_local_wind_direction[i] = wind_org/np.linalg.norm(wind_org,ord=2)
        pass
        return l_local_wind_direction
    pass

    def update_turbine_hub_wind_for_all(self):
        #calculate and assigne the height of the turbines
        num_turbines = len(self.list_turbines)
        for i in range(num_turbines):
            self.list_turbines[i].update_wind_hub()
        pass
    pass

    def update_turbine_para_for_all(self):
        #calculate and assigne the height of the turbines
        num_turbines = len(self.list_turbines)
        for i in range(num_turbines):
            self.list_turbines[i].update_para()
        pass
    pass

    # To be updated. To define wind environment
    def get_wind_info(self,poisition):
        #
        return self.list_wind_info

    # To define or add a zone, which is described by a <boundary>
    # shape(points): [N,M]
    # N: The number of the anti-clockwise points
    # M: 2, namely each point: [xcoordinate, y coordinate]
    def add_boundary(self, points):
        assert points.shape[1] == 2
        assert points.shape[0] >= 3
        num_lines = points.shape[0]
        new_boundary = np.zeros(shape=(num_lines,4),dtype=float)
        #Form boundary lines
        for i in range(num_lines-1):
            new_boundary[i,0:2] = points[i]
            new_boundary[i,2:4] = points[i+1]
        new_boundary[num_lines-1,0:2] = points[num_lines-1]
        new_boundary[num_lines-1,2:4] = points[0]
        #Add new boundary into self
        self.boundaries.append(new_boundary)

    # To check if a point is on a <boundary>
    # len(point): [2,], namely [xcoordinate, y coordinate]
    def is_point_on_a_boundary(boundary,point,err_lim=1e-5):
        assert len(point) == 2
        assert boundary.shape[1] == 4
        num_lines = boundary.shape[0]
        flag_on_segment = np.ones((num_lines,), dtype = bool)
        x = point[0]
        y = point[1]
        for i in range(num_lines):
            segment = boundary[i]
            x0 = segment[0]
            y0 = segment[1]
            x1 = segment[2]
            y1 = segment[3]
            err = np.sqrt(((x-x0)*(y1-y0)+(y-y0)*(x0-x1))**2/((y1-y0)**2+(x0-x1)**2));
            check_line_x = x <= max([x0,x1])+err_lim and x >= min([x0,x1])-err_lim;
            Check_line_y = y <= max([y0,y1])+err_lim and y >= min([y0,y1])-err_lim;
            if err < err_lim and check_line_x and Check_line_y:
                flag_on_segment[i] = True
            else:
                flag_on_segment[i] = False
            pass
        pass
        if True in flag_on_segment:
            return True
        else:
            return False

    # To check if two segments are crossed
    # shape(line): [4,], namely [X0,Y0,X1,Y1]
    def is_segments_crossed(line_a,line_b):
        #A1
        x_a_0 = line_a[0]
        y_a_0 = line_a[1]
        #A2
        x_a_1 = line_a[2]
        y_a_1 = line_a[3]
        #B1
        x_b_0 = line_b[0]
        y_b_0 = line_b[1]
        #B2
        x_b_1 = line_b[2]
        y_b_1 = line_b[3]
        flag_crossed = True
        flag_x_crossed = True
        flag_y_crossed = True
        #A is left, B is right
        if max([x_a_0,x_a_1]) < min([x_b_0,x_b_1]) or max([x_b_0,x_b_1]) < min([x_a_0,x_a_1]):
            #A and B does not have crossed x coordinates
            flag_x_crossed = False
        if max([y_a_0,y_a_1]) < min([y_b_0,y_b_1]) or max([y_b_0,y_b_1]) < min([y_a_0,y_a_1]):
            #A and B does not have crossed y coordinates
            flag_y_crossed = False
        if flag_x_crossed == False and flag_y_crossed == False:
            flag_crossed = False

        if flag_crossed != False:
            flag_crossed_1 = (
                np.cross([x_a_0-x_b_0,y_a_0-y_b_0],[x_b_1-x_b_0,y_b_1-y_b_0])*
                np.cross([x_b_1-x_b_0,y_b_1-y_b_0],[x_a_1-x_b_0,y_a_1-y_b_0]) > 0)
            flag_crossed_2 = (
                np.cross([x_b_0-x_a_0,y_b_0-y_a_0],[x_a_1-x_a_0,y_a_1-y_a_0])*
                np.cross([x_a_1-x_a_0,y_a_1-y_a_0],[x_b_1-x_a_0,y_b_1-y_a_0]) > 0)
            flag_crossed = flag_crossed_1 == True and flag_crossed_2 == True
        return flag_crossed

    # To get the minimum and maximum coordinates of the <boundaries>
    def get_all_zone_coordinates_range(boundaries):
        num_boundaries = len(boundaries)
        min_x = []
        max_x = []
        min_y = []
        max_y = []
        for i in range(num_boundaries):
            min_x_i,max_x_i,min_y_i,max_y_i = Farm_zone.get_a_zone_coordinates_range(boundaries[i])
            min_x.append(min_x_i)
            max_x.append(max_x_i)
            min_y.append(min_y_i)
            max_y.append(max_y_i)
        pass

        min_x = np.min(min_x)
        max_x = np.max(max_x)
        min_y = np.min(min_y)
        max_y = np.max(max_y)

        return np.array([min_x,max_x,min_y,max_y])

    def get_a_zone_coordinates_range(zone_boundary):
        min_x = np.min(zone_boundary[:,[0,2]])
        max_x = np.max(zone_boundary[:,[0,2]])
        min_y = np.min(zone_boundary[:,[1,3]])
        max_y = np.max(zone_boundary[:,[1,3]])
        return min_x,max_x,min_y,max_y
    pass

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
            if Farm_zone.is_in_line((x,y), verts[j], verts[i]):
                    return True
            if ((verty[i] > y) != (verty[j] > y)) and (
                        x < (vertx[j] - vertx[i]) * (y - verty[i]) / (verty[j] - verty[i]) + vertx[i]):
                is_in = not is_in

        return is_in

    def get_info_turbines_to_boundaries(self):
        #<boundaries>: Contain several <boundary>
        #each <boundary>: Contain start points and end points
        #shape=(N,4), each row represents a boundary line
        #Each line contains 4 numbers, namely X0,Y0,X1,Y1
        num_zones = len(self.boundaries)
        num_turbines = len(self.list_turbines)
        is_in_boundary = [False for i in range(num_turbines)]
        in_which_zone = [None for i in range(num_turbines)]
        min_dist = [1e10 for i in range(num_turbines)]
        for i in range(num_turbines):
            pointTp = self.list_turbines[i].coordinate #[2,]
            for j in range(num_zones):
                temp_dist = Farm_zone.min_dist_to_zone_boundary(pointTp, self.boundaries[j])
                if temp_dist < min_dist[i]:
                    min_dist[i] = temp_dist
                pass
                verts = self.boundaries[j][:,0:2]
                temp = Farm_zone.is_point_in_polygon(pointTp, verts)
                if temp == True:
                    is_in_boundary[i] = True
                    in_which_zone[i] = j
                    #min_dist[i] = Farm_zone.min_dist_to_zone_boundary(pointTp, self.boundaries[j])
                    break
                pass
            pass
        pass


        return is_in_boundary,in_which_zone,min_dist
    pass

    def update_boundary_info_of_turbines(self):
        is_in_boundary,in_which_zone,min_dist = self.get_info_turbines_to_boundaries()
        num_turbines = len(self.list_turbines)
        for i in range(num_turbines):
            self.list_turbines[i].falg_in_boundary = is_in_boundary[i]
            self.list_turbines[i].belonging_zone = in_which_zone[i]
            self.list_turbines[i].min_dist_to_boundary = min_dist[i]
        pass
    pass

    def min_dist_to_zone_boundary(pointTp, boundary):
        min_dist = 0
        num_lines = np.shape(boundary)[0]
        xp,yp = pointTp
        #print(xp,yp)
        dists = np.zeros(shape=(num_lines,))
        for i in range(num_lines):
            x0,y0,x1,y1=boundary[i,:]
            A = -(y0-y1)
            B = x0-x1
            C = -x0*y1+y0*x1
            dists[i] = np.abs(A*xp+B*yp+C)/np.sqrt(A**2+B**2)
        pass
        min_dist = np.min(dists)
        return min_dist
    pass


    def check_are_turbines_in_boundary(self):
        #<boundaries>: Contain several <boundary>
        #each <boundary>: Contain start points and an end points
        #shape=(N,4), each row represents a boundary line
        #Each line contains 4 numbers, namely X0,Y0,X1,Y1
        num_zones = len(self.boundaries)
        num_turbines = len(self.list_turbines)
        results = [False for i in range(num_turbines)]
        for i in range(num_turbines):
            pointTp = self.list_turbines[i].coordinate
            for j in range(num_zones):
                verts = self.boundaries[j][:,0:2]
                temp = Farm_zone.is_point_in_polygon(pointTp, verts)
                if temp == True:
                    results[i] = True
                    break
            pass
        pass
        return results
    pass



    # To draw boundaries
    def draw_boundaries(self):
        num_boundaries = len(self.boundaries)
        # plot the data
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        for i in range(num_boundaries):
            x = self.boundaries[i][:,0]
            x = np.append(x,[self.boundaries[i][-1,2]],axis=0)
            y = self.boundaries[i][:,1]
            y = np.append(y,[self.boundaries[i][-1,3]],axis=0)
            ax.plot(x, y, color='tab:blue')
        pass
        plt.show()
        return (fig,ax)



    # To add points on a boundary map
    def draw_points(self,points):
        num_boundaries = len(self.boundaries)
        # plot the data
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        for i in range(num_boundaries):
            x = self.boundaries[i][:,0]
            x = np.append(x,[self.boundaries[i][-1,2]],axis=0)
            y = self.boundaries[i][:,1]
            y = np.append(y,[self.boundaries[i][-1,3]],axis=0)
            ax.plot(x, y, color='tab:blue')
        pass

        results = self.check_are_turbines_in_boundary(self)
        for i in range(results):
            if results[i] == True:
                color = 'g'
            else:
                color ='r'
            pass
            ax.scatter(points[i][0], points[i][1],c=color)
            pass
        pass
        plt.show()

    # To draw wind turbines
    def draw_turbines_boundary(self):
        num_boundaries = len(self.boundaries)
        # plot the data
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        for i in range(num_boundaries):
            x = self.boundaries[i][:,0]
            x = np.append(x,[self.boundaries[i][-1,2]],axis=0)
            y = self.boundaries[i][:,1]
            y = np.append(y,[self.boundaries[i][-1,3]],axis=0)
            ax.plot(x, y, color='k')
        pass

        results = self.check_are_turbines_in_boundary()
        num_points = len(results)
        for i in range(num_points):
            point = self.list_turbines[i].coordinate
            if results[i] == True:
                color = 'g'
            else:
                color ='r'
            pass
            ax.scatter(point[0], point[1],c=color,zorder=2000)
            pass
        pass
        return fig, ax

    def draw_turbines_boundary_with_fig(self,fig,ax):
        num_boundaries = len(self.boundaries)
        # plot the data
        for i in range(num_boundaries):
            x = self.boundaries[i][:,0]
            x = np.append(x,[self.boundaries[i][-1,2]],axis=0)
            y = self.boundaries[i][:,1]
            y = np.append(y,[self.boundaries[i][-1,3]],axis=0)
            ax.plot(x, y, color='k',zorder=2000)
        pass

        results = self.check_are_turbines_in_boundary()
        num_points = len(results)
        for i in range(num_points):
            point = self.list_turbines[i].coordinate
            if results[i] == True:
                color = 'g'
            else:
                color ='r'
            pass
            ax.scatter(point[0], point[1],c=color,s = 30, edgecolors = 'w', linewidth = 1, zorder=3000)
            pass
        pass
        return fig, ax

    def draw_turbines_boundary_ground(self):
        fig,ax = self.draw_turbines_boundary()
        #--------------------------see:  Contour X-Y------------------------------------------#
        ax.xaxis.set_label_text('X')
        ax.yaxis.set_label_text('Y')

        ax.set_xlim(xmin=self.flowfield.x_range[0], xmax=self.flowfield.x_range[1])
        ax.set_ylim(ymin=self.flowfield.y_range[0], ymax=self.flowfield.y_range[1])
        ax.contourf(self.flowfield.X_ground.transpose(), self.flowfield.Y_ground.transpose(), self.flowfield.Z_ground.transpose(),zorder=0)
        ax.set_aspect(1)
    pass

    def draw_all_particles(self,optimizer):

        #Draw boundaries
        num_boundaries = len(self.boundaries)
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        for i in range(num_boundaries):
            x = self.boundaries[i][:,0]
            x = np.append(x,[self.boundaries[i][-1,2]],axis=0)
            y = self.boundaries[i][:,1]
            y = np.append(y,[self.boundaries[i][-1,3]],axis=0)
            ax.plot(x, y, color='k')
        pass



        ax.xaxis.set_label_text('X')
        ax.yaxis.set_label_text('Y')
        ax.set_xlim(xmin=self.flowfield.x_range[0], xmax=self.flowfield.x_range[1])
        ax.set_ylim(ymin=self.flowfield.y_range[0], ymax=self.flowfield.y_range[1])
        ax.contourf(self.flowfield.X_ground.transpose(), self.flowfield.Y_ground.transpose(), self.flowfield.Z_ground.transpose(),zorder=0)
        ax.set_aspect(1)
        # load turbine locations of particle k
        num_particles = optimizer.num_particles
        num_turbines = self.num_turbines
        for i in range(num_particles):
            print('Draw all particles: {}/{}'.format(i,num_particles))
            sys.stdout.flush()
            coords_x,coords_y = ParticleSwarm.particle_to_coordinate(optimizer.loc_particles[i,:],
                                                                     optimizer.range_x,
                                                                     optimizer.range_y)
            for j in range(num_turbines):
                my_zone.update_turbines_coordinate(j,[coords_x[j],coords_y[j]])
            pass
            # plot the data
            for j in range(num_turbines):
                point = self.list_turbines[j].coordinate
                color = 'k'
                ax.scatter(point[0], point[1],c=color,zorder=2000,s = 1)
                pass
            pass
        pass
    pass

    def draw_turbines_boundary_flowfield(self,direction,z_target):
        self.flowfield.update_wind_mesh(direction=direction)
        _,_,_,Speed = self.flowfield.get_isoheight_wind(direction=direction,z_target=z_target,is_draw=False)
        fig,ax = self.draw_turbines_boundary()
        #--------------------------see:  Contour X-Y------------------------------------------#
        ax.xaxis.set_label_text('X')
        ax.yaxis.set_label_text('Y')

        ax.set_xlim(xmin=self.flowfield.x_range[0], xmax=self.flowfield.x_range[1])
        ax.set_ylim(ymin=self.flowfield.y_range[0], ymax=self.flowfield.y_range[1])

        #pcolormesh,or,contourf
        color_bar_fraction = 0.05
        #Speed-Field
        im = ax.contourf(self.flowfield.X_ground, self.flowfield.Y_ground, Speed, 10)
        ax.set_title('Speed-Field: at isoheight={0}'.format(z_target),fontsize=10)
        fig.colorbar(im, ax=ax,fraction=color_bar_fraction)
        ax.set_aspect(1)


    pass

    def draw_turbines_boundary_flowfield_all_direction(self,z_target,list_wind_directions,n_row,n_col):
        num_direction = self.num_wind_directions
        min_vels = np.zeros((num_direction,))
        max_vels = np.zeros((num_direction,))
        Us = [None for i in range(num_direction)]
        Vs = [None for i in range(num_direction)]
        Ws = [None for i in range(num_direction)]
        Speeds = [None for i in range(num_direction)]
        for i in range(num_direction):
            self.flowfield.update_wind_mesh(direction=list_wind_directions[i])
            Us[i],Vs[i],Ws[i],Speeds[i] = self.flowfield.get_isoheight_wind(direction=list_wind_directions[i],z_target=z_target,is_draw=False)
            min_vels[i] = np.min(Speeds[i])
            max_vels[i] = np.max(Speeds[i])
        pass
        vel_min = np.floor(np.min(min_vels))
        vel_max = np.ceil(np.max(max_vels))
        fig = plt.figure(figsize=(n_col*5, n_row*5))
        levels = np.linspace(vel_min, vel_max, 11)
        if vel_min == vel_max:
            levels = np.linspace(vel_min*0.5, vel_max*1.5, 3)
        pass
        for i in range(num_direction):
            ax = fig.add_subplot(n_row,n_col,i+1)
            self.draw_turbines_boundary_with_fig(fig,ax)
            #--------------------------see:  Contour X-Y------------------------------------------#
            # ax.xaxis.set_label_text('X')
            # ax.yaxis.set_label_text('Y')

            ax.set_xlim(xmin=self.flowfield.x_range[0], xmax=self.flowfield.x_range[1])
            ax.set_ylim(ymin=self.flowfield.y_range[0], ymax=self.flowfield.y_range[1])
            #pcolormesh,or,contourf

            #Speed-Field
            im = ax.contourf(self.flowfield.X_ground, self.flowfield.Y_ground, Speeds[i], 10,levels=levels)#levels=levels
            plt.xticks(())
            plt.yticks(())
            color_bar_fraction = 0.04
            cbar = fig.colorbar(im, ax=ax,fraction=color_bar_fraction)
            cbar.mappable.set_clim(vel_min,vel_max) #this works


            #ax.set_title('speed: direction={:.2f} at isoheight={:}'.format(list_wind_directions[i]/np.pi*180,z_target),fontsize=10)

            ax.set_aspect(1)
        pass
        return fig

    pass

    def draw_isoheight_wind_with_reduce(self,direction,z_target,factor_grid,num_err,step_err):

        dx = self.flowfield.delta_x
        dy = self.flowfield.delta_y

        X,Y = np.mgrid[self.flowfield.x_range[0]:my_zone.flowfield.x_range[1]+dx/factor_grid:dx/factor_grid,
                       self.flowfield.y_range[0]:my_zone.flowfield.y_range[1]+dy/factor_grid:dy/factor_grid]

        x = X.flatten()
        y = Y.flatten()

        points_interp_2D = np.array([x,y]).transpose()
        z_ground = interpolate.interpn((self.flowfield.xx,self.flowfield.yy), self.flowfield.Z_ground, points_interp_2D)
        z = z_target+z_ground.flatten()
        coords = np.array([x,y,z]).transpose()

        list_factor_reduce = self.calc_factor_reduce_all_with_perturbation_for_points(direction=direction,coords_point=coords,num_err=num_err,step_err=step_err)

        points_interp_3D = np.array([x,y,z]).transpose()
        u = interpolate.interpn((self.flowfield.xx,self.flowfield.yy,self.flowfield.zz), self.flowfield.dict_wind_info[direction]['U'], points_interp_3D)
        v = interpolate.interpn((self.flowfield.xx,self.flowfield.yy,self.flowfield.zz), self.flowfield.dict_wind_info[direction]['V'], points_interp_3D)
        w = interpolate.interpn((self.flowfield.xx,self.flowfield.yy,self.flowfield.zz), self.flowfield.dict_wind_info[direction]['W'], points_interp_3D)
        u = u*(1-list_factor_reduce)
        v = v*(1-list_factor_reduce)
        w = w*(1-list_factor_reduce)
        U = np.reshape(u, np.shape(X))
        V = np.reshape(v, np.shape(X))
        W = np.reshape(w, np.shape(X))
        Speed = np.sqrt(U**2+V**2+W**2)

        #pcolormesh,or,contourf
        n_figure_row = 1
        n_figure_col = 4
        color_bar_fraction = 0.05
        factor = 15
        fig, ax = plt.subplots(n_figure_row,n_figure_col,constrained_layout=True,figsize=(factor*n_figure_row, factor*n_figure_col))#dpi=200
        #norm = matplotlib.colors.Normalize(vmin=-20, vmax=20)#模型一结果绘图
        #U-Field
        im0 = ax[0].contourf(X, Y, U, 10)
        ax[0].xaxis.set_label_text('X')
        ax[0].yaxis.set_label_text('Y')
        ax[0].set_title('U-Field: at isoheight={0}'.format(z_target),fontsize=10)
        ax[0].set_aspect(1)
        fig.colorbar(im0, ax=ax[0],fraction=color_bar_fraction)
        self.draw_turbines_boundary_with_fig(fig,ax[0])
        #V-Field

        im1 = ax[1].contourf(X, Y, V, 10)
        ax[1].xaxis.set_label_text('X')
        ax[1].yaxis.set_label_text('Y')
        ax[1].set_title('V-Field: at isoheight={0}'.format(z_target),fontsize=10)
        ax[1].set_aspect(1)
        fig.colorbar(im1, ax=ax[1],fraction=color_bar_fraction)
        self.draw_turbines_boundary_with_fig(fig,ax[1])
        #W-Field

        im2 = ax[2].contourf(X, Y, W, 10)
        ax[2].xaxis.set_label_text('X')
        ax[2].yaxis.set_label_text('Y')
        ax[2].set_title('W-Field: at isoheight={0}'.format(z_target),fontsize=10)
        ax[2].set_aspect(1)
        fig.colorbar(im2, ax=ax[2],fraction=color_bar_fraction)
        self.draw_turbines_boundary_with_fig(fig,ax[2])
        #Speed-Field

        im2 = ax[3].contourf(X, Y, Speed, 10)
        ax[3].xaxis.set_label_text('X')
        ax[3].yaxis.set_label_text('Y')
        ax[3].set_title('Speed-Field: at isoheight={0}'.format(z_target),fontsize=10)
        ax[3].set_aspect(1)
        fig.colorbar(im2, ax=ax[3],fraction=color_bar_fraction)
        self.draw_turbines_boundary_with_fig(fig,ax[3])



    pass



    def calc_isoheight_wind_with_reduce(self,direction,z_target,factor_grid,num_err,step_err):

        dx = self.flowfield.delta_x
        dy = self.flowfield.delta_y

        X,Y = np.mgrid[self.flowfield.x_range[0]:my_zone.flowfield.x_range[1]+dx/factor_grid:dx/factor_grid,
                       self.flowfield.y_range[0]:my_zone.flowfield.y_range[1]+dy/factor_grid:dy/factor_grid]

        x = X.flatten()
        y = Y.flatten()

        points_interp_2D = np.array([x,y]).transpose()
        z_ground = interpolate.interpn((self.flowfield.xx,self.flowfield.yy), self.flowfield.Z_ground, points_interp_2D)
        z = z_target+z_ground.flatten()
        coords = np.array([x,y,z]).transpose()

        list_factor_reduce = self.calc_factor_reduce_all_with_perturbation_for_points(direction=direction,coords_point=coords,num_err=num_err,step_err=step_err)


        points_interp_3D = np.array([x,y,z]).transpose()
        u = interpolate.interpn((self.flowfield.xx,self.flowfield.yy,self.flowfield.zz), self.flowfield.dict_wind_info[direction]['U'], points_interp_3D)
        v = interpolate.interpn((self.flowfield.xx,self.flowfield.yy,self.flowfield.zz), self.flowfield.dict_wind_info[direction]['V'], points_interp_3D)
        w = interpolate.interpn((self.flowfield.xx,self.flowfield.yy,self.flowfield.zz), self.flowfield.dict_wind_info[direction]['W'], points_interp_3D)


        u = u*(1-list_factor_reduce)
        v = v*(1-list_factor_reduce)
        w = w*(1-list_factor_reduce)
        U = np.reshape(u, np.shape(X))
        V = np.reshape(v, np.shape(X))
        W = np.reshape(w, np.shape(X))
        Speed = np.sqrt(U**2+V**2+W**2)
        factor_reduce = np.reshape(list_factor_reduce, np.shape(X))

        return X,Y,U,V,W,Speed,factor_reduce

    pass

    def draw_isoheight_wind_with_reduce_in_fig_data(self,X,Y,data,title,fig,index_row,index_col,direction,z_target,factor_grid,num_err,step_err,n_figure_row,n_figure_col,figsize):



        #pcolormesh,or,contourf
        # n_figure_col = 4
        color_bar_fraction = 0.05
        if fig==None:
            fig, ax = plt.subplots(n_figure_row,n_figure_col,constrained_layout=True,figsize=figsize)#dpi=200
        pass
        #norm = matplotlib.colors.Normalize(vmin=-20, vmax=20)#模型一结果绘图
        #U-Field
        ax = fig.add_subplot(n_figure_row,n_figure_col,index_row*n_figure_col+index_col+1)
        im0 = ax.contourf(X, Y, data, 10)
        ax.xaxis.set_label_text('X')
        ax.yaxis.set_label_text('Y')
        ax.set_title(title+': at isoheight={0}'.format(z_target),fontsize=10)
        ax.set_aspect(1)
        fig.colorbar(im0, ax=ax,fraction=color_bar_fraction)
        self.draw_turbines_boundary_with_fig(fig,ax)


        return fig

    pass

    def draw_isoheight_wind_with_reduce_in_fig(self,X,Y,U,V,W,Speed,factor_reduce,fig,row,direction,z_target,factor_grid,num_err,step_err,n_figure_row=1,n_figure_col=1,figsize=(20,5*8)):



        #pcolormesh,or,contourf
        # n_figure_col = 4
        color_bar_fraction = 0.05
        if fig==None:
            fig, ax = plt.subplots(n_figure_row,n_figure_col,constrained_layout=True,figsize=figsize)#dpi=200
        pass
        #norm = matplotlib.colors.Normalize(vmin=-20, vmax=20)#模型一结果绘图
        #U-Field
        ax = fig.add_subplot(n_figure_row,n_figure_col,row*n_figure_col+1)
        im0 = ax.contourf(X, Y, U, 10)
        ax.xaxis.set_label_text('X')
        ax.yaxis.set_label_text('Y')
        ax.set_title('U-Field: at isoheight={0}'.format(z_target),fontsize=10)
        ax.set_aspect(1)
        fig.colorbar(im0, ax=ax,fraction=color_bar_fraction)
        self.draw_turbines_boundary_with_fig(fig,ax)
        #V-Field
        ax = fig.add_subplot(n_figure_row,n_figure_col,row*n_figure_col+2)
        im1 = ax.contourf(X, Y, V, 10)
        ax.xaxis.set_label_text('X')
        ax.yaxis.set_label_text('Y')
        ax.set_title('V-Field: at isoheight={0}'.format(z_target),fontsize=10)
        ax.set_aspect(1)
        fig.colorbar(im1, ax=ax,fraction=color_bar_fraction)
        self.draw_turbines_boundary_with_fig(fig,ax)
        #W-Field
        ax = fig.add_subplot(n_figure_row,n_figure_col,row*n_figure_col+3)
        im2 = ax.contourf(X, Y, W, 10)
        ax.xaxis.set_label_text('X')
        ax.yaxis.set_label_text('Y')
        ax.set_title('W-Field: at isoheight={0}'.format(z_target),fontsize=10)
        ax.set_aspect(1)
        fig.colorbar(im2, ax=ax,fraction=color_bar_fraction)
        self.draw_turbines_boundary_with_fig(fig,ax)
        #Speed-Field
        ax = fig.add_subplot(n_figure_row,n_figure_col,row*n_figure_col+4)
        im2 = ax.contourf(X, Y, Speed, 10)
        ax.xaxis.set_label_text('X')
        ax.yaxis.set_label_text('Y')
        ax.set_title('Speed-Field: at isoheight={0}'.format(z_target),fontsize=10)
        ax.set_aspect(1)
        fig.colorbar(im2, ax=ax,fraction=color_bar_fraction)
        self.draw_turbines_boundary_with_fig(fig,ax)
        #factor_reduce-Field
        ax = fig.add_subplot(n_figure_row,n_figure_col,row*n_figure_col+5)
        im2 = ax.contourf(X, Y, factor_reduce, 10)
        ax.xaxis.set_label_text('X')
        ax.yaxis.set_label_text('Y')
        ax.set_title('factor_reduce-Field: at isoheight={0}'.format(z_target),fontsize=10)
        ax.set_aspect(1)
        fig.colorbar(im2, ax=ax,fraction=color_bar_fraction)
        self.draw_turbines_boundary_with_fig(fig,ax)

        return fig

    pass

    def draw_isoheight_wind_with_reduce_all_direction(self,list_wind_directions,z_target,factor_grid,num_err,step_err):
        fig = None
        for i in range(self.num_wind_directions):
            self.flowfield.update_wind_mesh(direction=list_wind_directions[i])
            self.update_turbine_loc_wind_for_all(direction=list_wind_directions[i])
            self.flowfield.get_streamlines_3D_for_all()
            fig = self.draw_isoheight_wind_with_reduce_in_fig(fig,i,list_wind_directions[i],z_target,factor_grid,num_err,step_err,n_figure_row=self.num_wind_directions,figsize=(5*4,5*self.num_wind_directions))
        pass
        plt.show()


    pass

    def draw_isoheight_wind_with_reduce_all_direction_probability(self,list_wind_directions,z_target,factor_grid,num_err,step_err,probability_direction,probability_wind_in_each_direction,ratios_wind,threshold_probability):
        fig_speed = None
        fig_factor_reduce = None
        num_wind_directions = np.shape(probability_wind_in_each_direction)[0]
        num_wind_ranges = np.shape(probability_wind_in_each_direction)[1]

        for i_direction in range(num_wind_directions):
            X = []
            Y = []
            Us = []
            Vs = []
            Ws = []
            Speeds = []
            factor_reduces = []
            for i_range in range(num_wind_ranges):
                if probability_direction[i_direction] <= threshold_probability or probability_wind_in_each_direction[i_direction,i_range]<=threshold_probability:
                    continue
                else:
                    # load wind mesh of a specified direction
                    self.flowfield.update_current_wind_info(direction=list_wind_directions[i_direction],
                                                            factor=ratios_wind[i_range],
                                                            probability=probability_direction[i_direction]*probability_wind_in_each_direction[i_direction,i_range])
                    self.flowfield.update_wind_mesh(direction=list_wind_directions[i_direction])
                    self.update_turbine_loc_wind_for_all(direction=list_wind_directions[i_direction])
                    self.flowfield.get_streamlines_3D_for_all()
                    X,Y,U,V,W,Speed,factor_reduce = self.calc_isoheight_wind_with_reduce(list_wind_directions[i_direction],z_target,factor_grid,num_err,step_err)
                    Us.append(U*probability_wind_in_each_direction[i_direction,i_range])
                    Vs.append(V*probability_wind_in_each_direction[i_direction,i_range])
                    Ws.append(W*probability_wind_in_each_direction[i_direction,i_range])
                    Speeds.append(Speed*probability_wind_in_each_direction[i_direction,i_range])
                    factor_reduces.append(factor_reduce*probability_wind_in_each_direction[i_direction,i_range])
                pass
            pass
            U = np.sum(Us,0)
            V = np.sum(Vs,0)
            W = np.sum(Ws,0)
            Speed = np.sum(Speeds,0)
            factor_reduce = np.sum(factor_reduces,0)
            if probability_direction[i_direction] <= threshold_probability:
                continue
            else:
                n_figure_row=2
                n_figure_col=4
                figsize=(5*n_figure_col,5*n_figure_row)
                index_row = i_direction//n_figure_col
                index_col = i_direction%n_figure_col
                fig_speed = self.draw_isoheight_wind_with_reduce_in_fig_data(X,Y,Speed,'speed',fig_speed,index_row,index_col,list_wind_directions[i_direction],z_target,factor_grid,num_err,step_err,n_figure_row,n_figure_col,figsize)
                fig_factor_reduce = self.draw_isoheight_wind_with_reduce_in_fig_data(X,Y,factor_reduce,'factor_reduce',fig_factor_reduce,index_row,index_col,list_wind_directions[i_direction],z_target,factor_grid,num_err,step_err,n_figure_row,n_figure_col,figsize)



            pass
        pass
        plt.show()
        return fig

    pass

    def generate_rand_points_in_a_zone(zone_boundary,num_points,seed=np.random.randint(1000)):

        min_x,max_x,min_y,max_y = Farm_zone.get_a_zone_coordinates_range(zone_boundary)
        num_div = num_points*3
        dx = (max_x-min_x)/num_div
        dy = (max_y-min_y)/num_div
        for index_iter in range(10):
            list_points_in_zone = []
            verts = zone_boundary[:,0:2]
            for i in range(num_div):
                for j in range(num_div):
                    pointTp = np.array([min_x+dx*(i+0.5),min_y+dy*(j+0.5)])
                    is_point_in_zone = Farm_zone.is_point_in_polygon(pointTp, verts)
                    if is_point_in_zone == True:
                        list_points_in_zone.append(pointTp)
                    pass
                pass
            pass
            if len(list_points_in_zone) > num_points*2:
                break
            pass
        pass
        np.random.seed(seed)
        rand_indexes = np.random.choice(len(list_points_in_zone), size=num_points, replace=False)
        rand_indexes = np.array(rand_indexes,dtype=int)
        points = np.array(list_points_in_zone)[rand_indexes]

        return points
    pass

    def generate_rand_points_in_a_zone_consider_fitness_and_dict_threshold(self,zone_boundary,num_points,dist_threshold,direction,height_hub,seed=np.random.randint(1000)):

        min_x,max_x,min_y,max_y = Farm_zone.get_a_zone_coordinates_range(zone_boundary)

        dx = dist_threshold*1.0
        dy = dist_threshold*1.0
        num_div_x = (np.floor((max_x-min_x)/dx)).astype(int)
        num_div_y = (np.floor((max_y-min_y)/dy)).astype(int)
        x_residual = (max_x-min_x)-dx*num_div_x
        y_residual = (max_y-min_y)-dy*num_div_y
        np.random.seed(seed)
        x_start = x_residual*np.random.random()
        y_start = y_residual*np.random.random()


        list_points_in_zone = []
        verts = zone_boundary[:,0:2]
        for i in range(num_div_x+1):
            for j in range(num_div_y+1):
                pointTp = np.array([min_x+x_start+dx*i,
                                    min_y+y_start+dy*j])
                is_point_in_zone = Farm_zone.is_point_in_polygon(pointTp, verts)

                # print(pointTp)
                # print(is_point_in_zone)

                if is_point_in_zone == True:
                    list_points_in_zone.append(pointTp)
                pass
            pass
        pass
        list_points_in_zone = np.array(list_points_in_zone)


        #######################
        num_points_in_zone = len(list_points_in_zone)
        _,_,_,Speed = self.flowfield.get_isoheight_wind(direction=direction,z_target=height_hub,is_draw=False)
        speeds = np.zeros(shape=(num_points_in_zone,))
        for i in range(num_points_in_zone):
            coordinate_2D = list_points_in_zone[i]
            speeds[i] = self.flowfield.get_2D_interporlation(Speed,coordinate_2D)
        pass


        factor = 2
        num_selected = num_points*factor
        if num_selected > len(list_points_in_zone):
            num_selected = len(list_points_in_zone)
        pass

        temp_index = np.argsort(-speeds)
        index_selected = temp_index[0:num_selected]
        speeds = speeds[index_selected]
        list_points_in_zone = list_points_in_zone[index_selected]

        if len(list_points_in_zone) < num_points:
            print('error:\n')
            print('len(list_points_in_zone) < num_points')
            print(list_points_in_zone)
            sys.stdout.flush()
            raise
        pass

        np.random.seed(seed)
        rand_indexes = np.random.choice(num_selected, size=num_points, replace=False)
        rand_indexes = np.array(rand_indexes,dtype=int)

        # print('list_points_in_zone=\n{0}'.format(list_points_in_zone))
        # print('rand_indexes=\n{0}'.format(rand_indexes))

        points = list_points_in_zone[rand_indexes]

        return points
    pass

    def generate_rand_particles_consider_fitness_and_dict_threshold(self,num_particles,num_turbines,dist_threshold,list_wind_directions,height_hub,seed=np.random.randint(1000)):
        np.random.seed(seed)
        num_zones = len(self.boundaries)
        areas_zone = np.zeros(shape=(num_zones,))
        for i in range(num_zones):
            areas_zone[i] = Farm_zone.cal_area_of_a_zone(self.boundaries[i])
        pass

        num_points_in_each_zone = np.round(num_turbines*areas_zone/np.sum(areas_zone))

        if np.sum(num_points_in_each_zone) > num_turbines:
            temp_index = np.argmax(areas_zone)
            num_points_in_each_zone[temp_index] -= np.sum(num_points_in_each_zone) - num_turbines
        pass

        if np.sum(num_points_in_each_zone) < num_turbines or np.min(num_points_in_each_zone) < 0:
            temp_index = np.argmin(areas_zone)
            num_points_in_each_zone[temp_index] += num_turbines-np.sum(num_points_in_each_zone)
        pass
        num_points_in_each_zone = num_points_in_each_zone.astype(int)

        #indexes_zone = np.random.choice(len(self.boundaries), size=num_particles, replace=True)
        seed_random = np.random.choice(num_particles, size=num_particles, replace=False)

        direction_choice = (np.random.choice(self.num_wind_directions, size=num_particles, replace=True)).astype(int)

        particles = np.zeros(shape=(num_particles,num_turbines*2))
        dx = self.flowfield.x_range[1]-self.flowfield.x_range[0]
        dy = self.flowfield.y_range[1]-self.flowfield.y_range[0]
        print('num_particles={0}'.format(num_particles))
        print('num_zones={0}'.format(num_zones))
        sys.stdout.flush()
        for i in range(num_particles):
            print('initializing: particle_{0}/{1}'.format(i,num_particles))
            sys.stdout.flush()
            index_start = 0
            for j in range(num_zones):
# =============================================================================
#                 coords_particle_i_in_zone_j = Farm_zone.generate_rand_points_in_a_zone(self.boundaries[j],
#                                                                                        num_points=num_points_in_each_zone[j],
#                                                                                        seed=seed_random[i])
# =============================================================================

                coords_particle_i_in_zone_j = self.generate_rand_points_in_a_zone_consider_fitness_and_dict_threshold(self.boundaries[j],
                                                                                                                      num_points=num_points_in_each_zone[j],
                                                                                                                      dist_threshold=dist_threshold,
                                                                                                                      direction=list_wind_directions[direction_choice[i]],
                                                                                                                      height_hub=height_hub,
                                                                                                                      seed=seed_random[i])


                coords_particle_i_in_zone_j = coords_particle_i_in_zone_j.flatten()
                coords_particle_i_in_zone_j[0::2] = (coords_particle_i_in_zone_j[0::2]-self.flowfield.x_range[0])/dx-0.5
                coords_particle_i_in_zone_j[1::2] = (coords_particle_i_in_zone_j[1::2]-self.flowfield.y_range[0])/dy-0.5
                index_add = num_points_in_each_zone[j]*2
                particles[i,index_start:index_start+index_add] = coords_particle_i_in_zone_j
                index_start = index_start+index_add
            pass
        pass
        return particles
    pass

    def generate_rand_particles(self,num_particles,num_turbines,seed=np.random.randint(1000)):
        np.random.seed(seed)
        num_zones = len(self.boundaries)
        areas_zone = np.zeros(shape=(num_zones,))
        for i in range(num_zones):
            areas_zone[i] = Farm_zone.cal_area_of_a_zone(self.boundaries[i])
        pass

        num_points_in_each_zone = np.round(num_turbines*areas_zone/np.sum(areas_zone))

        if np.sum(num_points_in_each_zone) > num_turbines:
            temp_index = np.argmax(areas_zone)
            num_points_in_each_zone[temp_index] -= np.sum(num_points_in_each_zone) - num_turbines
        pass

        if np.sum(num_points_in_each_zone) < num_turbines or np.min(num_points_in_each_zone) < 0:
            temp_index = np.argmin(areas_zone)
            num_points_in_each_zone[temp_index] += num_turbines-np.sum(num_points_in_each_zone)
        pass
        num_points_in_each_zone = num_points_in_each_zone.astype(int)

        #indexes_zone = np.random.choice(len(self.boundaries), size=num_particles, replace=True)
        seed_random = np.random.choice(num_particles, size=num_particles, replace=False)

        particles = np.zeros(shape=(num_particles,num_turbines*2))
        dx = self.flowfield.x_range[1]-self.flowfield.x_range[0]
        dy = self.flowfield.y_range[1]-self.flowfield.y_range[0]
        for i in range(num_particles):
            index_start = 0
            for j in range(num_zones):
                coords_particle_i_in_zone_j = Farm_zone.generate_rand_points_in_a_zone(self.boundaries[j],
                                                                                       num_points=num_points_in_each_zone[j],
                                                                                       seed=seed_random[i])
                coords_particle_i_in_zone_j = coords_particle_i_in_zone_j.flatten()
                coords_particle_i_in_zone_j[0::2] = (coords_particle_i_in_zone_j[0::2]-self.flowfield.x_range[0])/dx-0.5
                coords_particle_i_in_zone_j[1::2] = (coords_particle_i_in_zone_j[1::2]-self.flowfield.y_range[0])/dy-0.5
                index_add = num_points_in_each_zone[j]*2
                particles[i,index_start:index_start+index_add] = coords_particle_i_in_zone_j
                index_start = index_start+index_add
            pass
        pass
        return particles
    pass


    def cal_area_of_a_zone(zone_boundary):
        area = 0
        num_verts = np.shape(zone_boundary)[0]
        for i in range(num_verts):
            x_i = zone_boundary[i,0]
            y_i = zone_boundary[i,1]
            x_i_next = zone_boundary[(i+1)%num_verts,0]
            y_i_next = zone_boundary[(i+1)%num_verts,1]
            area += 0.5*(x_i*y_i_next-x_i_next*y_i)
        pass
        area = np.abs(area)
        return area
    pass



    #this fuyction will not change the storage order
    def sort_turbines_by_wind_direction(self,angle):
        #   y
        #   ^
        #   |
        #   |
        #   |
        #   ----------->x
        # <angle>: wind angle (rad). anti-clockwise started from x axis
        vector_wind_direction = np.array([np.cos(angle),np.sin(angle)])
        num_turbines = len(self.list_turbines)
        dists = np.zeros((num_turbines,))
        for i in range(num_turbines):
            dists[i] = np.dot(self.list_turbines[i].coordinate,vector_wind_direction)
        pass
        index_searching = np.argsort(dists)
        return index_searching
    pass


    def calc_factor_reduce(self,index_windward,index_leeward):
        dist,length_accumulated,p_cross = self.flowfield.search_along_streamlines(index_windward,index_leeward)
        if dist==None or length_accumulated==None:
            return 0
        else:
            x=length_accumulated
            yia=dist
            ai = self.list_turbines[index_windward].ai
            b = self.list_turbines[index_windward].b
            ki = self.list_turbines[index_windward].ki
            D = self.list_turbines[index_windward].r_fan*2
            c = self.list_turbines[index_windward].c*2
            epsilon = self.list_turbines[index_windward].epsilon
            theta=(ki*x/D+epsilon)*D
            factor_reduce=1/(ai+b*x/D+c*(1+x/D)**-2)**2*np.exp(-(yia)**2/(2*theta**2))
            return factor_reduce**2
        pass
    pass

    def calc_factor_reduce_with_perturbation(self,index_windward,index_leeward,num_err,step_err):
        dist,length_accumulated = self.flowfield.search_along_streamlines_with_perturbation(index_windward,index_leeward,num_err,step_err)

        if dist==None or length_accumulated==None:
            return 0
        else:
            x=length_accumulated

            yia=dist
            ai = self.list_turbines[index_windward].ai
            b = self.list_turbines[index_windward].b
            ki = self.list_turbines[index_windward].ki
            D = self.list_turbines[index_windward].r_fan*2
            c = self.list_turbines[index_windward].c
            epsilon = self.list_turbines[index_windward].epsilon
            theta=(ki*x/D+epsilon)*D
            factor_reduce=1/(ai+b*x/D+c*(1+x/D)**-2)**2*np.exp(-(yia)**2/(2*theta**2))
            return factor_reduce
        pass
    pass

    def calc_factor_reduce_with_perturbation_for_a_point(self,index_windward,coord_leeward,normal_leeward,num_err,step_err):
        dist,length_accumulated = self.flowfield.search_along_streamlines_with_perturbation_for_a_point(index_windward,coord_leeward,normal_leeward,num_err,step_err)

        if dist==None or length_accumulated==None:
            return 0
        else:
            x=length_accumulated

            yia=dist
            ai = self.list_turbines[index_windward].ai
            b = self.list_turbines[index_windward].b
            ki = self.list_turbines[index_windward].ki
            D = self.list_turbines[index_windward].r_fan*2
            c = self.list_turbines[index_windward].c
            epsilon = self.list_turbines[index_windward].epsilon
            theta=(ki*x/D+epsilon)*D
            factor_reduce=1/(ai+b*x/D+c*(1+x/D)**-2)**2*np.exp(-(yia)**2/(2*theta**2))
            return factor_reduce
        pass
    pass

    def calc_factor_reduce_all(self,direction):
        index_searching = self.sort_turbines_by_wind_direction(direction)
        num_turbines = len(index_searching)
        list_factor_reduce = np.zeros([num_turbines,])
        for i in range(num_turbines):
            n = 1
            index_leeward = index_searching[i]
            #print('calc:{0}'.format(index_leeward))
            for j in range(i):
                index_windward = index_searching[j]
                #print('calc:from_{0}_to_{1}'.format(index_windward,index_leeward))
                list_factor_reduce[i] = list_factor_reduce[i]+self.calc_factor_reduce(index_windward,index_leeward)
            pass
            list_factor_reduce[i] = list_factor_reduce[i]/n
        pass
        return list_factor_reduce
    pass

    def calc_factor_reduce_all_with_perturbation(self,direction,num_err,step_err):
        index_searching = self.sort_turbines_by_wind_direction(direction)
        #print(index_searching)
        num_turbines = len(index_searching)
        list_factor_reduce = np.zeros([num_turbines,])
        for i in range(num_turbines):
            index_leeward = index_searching[i]
            for j in range(i):
                index_windward = index_searching[j]
                #print('{0}to{1}'.format(index_windward,index_leeward))
                list_factor_reduce[index_leeward] = (list_factor_reduce[index_leeward]+
                                                     self.calc_factor_reduce_with_perturbation(index_windward,index_leeward,num_err,step_err)**2)
                #print(self.calc_factor_reduce_with_perturbation(index_windward,index_leeward,num_err,step_err))
            pass
            list_factor_reduce[index_leeward] = np.sqrt(list_factor_reduce[index_leeward])
        pass
        for i in range(num_turbines):
            self.list_turbines[i].factor_reduce = list_factor_reduce[i]
        pass
        return list_factor_reduce

    pass

    def calc_factor_reduce_all_with_perturbation_consider_boundary(self,direction,num_err,step_err):
        index_searching = self.sort_turbines_by_wind_direction(direction)
        #print(index_searching)
        num_turbines = len(index_searching)
        list_factor_reduce = np.zeros([num_turbines,])

        for i in range(num_turbines):
            index_leeward = index_searching[i]
            if self.list_turbines[index_leeward].falg_in_boundary == False:
                #list_factor_reduce[i] = 1.0
                list_factor_reduce[index_leeward] = 1.0
                if list_factor_reduce[index_leeward] > 1.0:
                    list_factor_reduce[index_leeward] = 1.0
                pass
                continue
            pass

            for j in range(i):

                index_windward = index_searching[j]
                #print('{0}to{1}'.format(index_windward,index_leeward))
                list_factor_reduce[index_leeward] = (list_factor_reduce[index_leeward]+
                                                     self.calc_factor_reduce_with_perturbation(index_windward,index_leeward,num_err,step_err)**2)
                #print(self.calc_factor_reduce_with_perturbation(index_windward,index_leeward,num_err,step_err))
                if list_factor_reduce[index_leeward] > 1.0:
                    list_factor_reduce[index_leeward] = 1.0
                pass
            pass
            list_factor_reduce[index_leeward] = np.sqrt(list_factor_reduce[index_leeward])
        pass
        for i in range(num_turbines):
            self.list_turbines[i].factor_reduce = list_factor_reduce[i]
        pass
        return list_factor_reduce

    pass

    def calc_power_for_a_wind_direction(self):
        pass
    pass

    def calc_factor_reduce_all_with_perturbation_consider_boundary_strict(self,direction,num_err,step_err):
        num_turbines = len(self.list_turbines)
        index_searching = self.sort_turbines_by_wind_direction(direction)

        list_factor_reduce = np.zeros([num_turbines,])

        for i in range(num_turbines):
            index_leeward = index_searching[i]
            if self.list_turbines[index_leeward].falg_in_boundary == False:
                #list_factor_reduce[i] = 1.0
                list_factor_reduce = np.ones([num_turbines,])
                break
            pass

            for j in range(i):

                index_windward = index_searching[j]
                #print('{0}to{1}'.format(index_windward,index_leeward))
                list_factor_reduce[index_leeward] = (list_factor_reduce[index_leeward]+
                                                     self.calc_factor_reduce_with_perturbation(index_windward,index_leeward,num_err,step_err)**2)
                #print(self.calc_factor_reduce_with_perturbation(index_windward,index_leeward,num_err,step_err))
            pass
            if list_factor_reduce[index_leeward] > 1.0:
                list_factor_reduce[index_leeward] = 1.0
            pass
            list_factor_reduce[index_leeward] = np.sqrt(list_factor_reduce[index_leeward])
        pass


        for i in range(num_turbines):
            if list_factor_reduce[i] < 0:
                print(list_factor_reduce)
                sys.stdout.flush()
                raise
            pass
            if list_factor_reduce[i] > 1.0:
                list_factor_reduce[i] = 1.0
            pass
            self.list_turbines[i].factor_reduce = list_factor_reduce[i]
        pass
        return list_factor_reduce

    pass

    def calc_factor_reduce_all_with_perturbation_for_points(self,direction,coords_point,num_err,step_err):
        num_turbines = len(self.list_turbines)
        num_points = np.shape(coords_point)[0]
        index_searching = self.sort_turbines_by_wind_direction(direction)
        l_norm_points = self.update_point_loc_wind_for_all(direction,coords_point)

        list_factor_reduce = np.zeros([num_points,])

        for i in range(num_points):
            coord_leeward = coords_point[i,:]
            normal_leeward = l_norm_points[i]
            for j in range(num_turbines):
                index_windward = index_searching[j]
                #print('{0}to{1}'.format(index_windward,index_leeward))
                list_factor_reduce[i] = (list_factor_reduce[i]+
                                         self.calc_factor_reduce_with_perturbation_for_a_point(index_windward,coord_leeward,normal_leeward,num_err,step_err)**2)
                #print(self.calc_factor_reduce_with_perturbation(index_windward,index_leeward,num_err,step_err))
            pass
            if list_factor_reduce[i] > 1.0:
                list_factor_reduce[i] = 1.0
            pass
            list_factor_reduce[i] = np.sqrt(list_factor_reduce[i])
        pass


        for i in range(num_points):
            if list_factor_reduce[i] < 0:
                print(list_factor_reduce)
                sys.stdout.flush()
                raise
            pass
        pass
        return list_factor_reduce

    pass



    def calc_power_for_a_wind_direction(self):
        pass
    pass

    def calc_min_dist_to_other_turbines(self,index_turbine):
        num_turbines = len(self.list_turbines)
        min_dist = 1e5
        for i in range(num_turbines):
            if i != index_turbine:
                v = self.list_turbines[i].coordinate - self.list_turbines[index_turbine].coordinate
                dist = np.linalg.norm(v,ord=2)
                if dist < min_dist:
                    min_dist = dist
                pass
            pass
        pass
        return min_dist
    pass

    def calc_min_dist_between_turbines(self):
        num_turbines = len(self.list_turbines)
        min_dist = 1e5
        for i in range(num_turbines):
            dist = self.calc_min_dist_to_other_turbines(i)
            if dist < min_dist:
                min_dist = dist
            pass
        pass
        return min_dist
    pass

    def draw_streamlines_3D_all_directions(self,list_wind_directions):
        fig = None
        for i in range(self.num_wind_directions):
            self.flowfield.update_wind_mesh(direction=list_wind_directions[i])
            self.update_turbine_loc_wind_for_all(direction=list_wind_directions[i])
            self.flowfield.get_streamlines_3D_for_all()
            fig = self.flowfield.draw_3D_streamlines_in_fig(fig,row=i,n_figure_row = self.num_wind_directions,figsize = (5*4,5*self.num_wind_directions))
        pass
        plt.show()
        return fig
    pass

    def save_setting(dir_working,name_to_save,setting_info):
        #---data--
        if os.path.exists(dir_working) == False:
            os.makedirs(dir_working)
        pass
        path_file_data = dir_working+'/'+name_to_save+'_setting.npy'
        print(path_file_data)
        #date = time.strftime('%Y-%m-%d %H:%M:%S')
        np.save(path_file_data,setting_info)
    pass


if __name__ == '__main__':


    click_load = False
    setting_info = {}
    #============================User defined parts===========================#

    #-------------------------if click load setting-----------------------------#

    if click_load == False:
        #-------------Label-0:Overall setting-------------------#
        #Button
        setting_info['flag_optimizer_status'] = params['flag_optimizer_status'] # 'initialize'or'continue'

        #select working directory
        setting_info['dir_working'] = params['dir_working']
        #TextBox-must specify
        setting_info['num_opt_iteration'] = int(params['num_opt_iteration']) #unsigned int
        setting_info['step_check'] = int(params['step_check'])  #unsigned int
        setting_info['name_to_save'] = params['name_to_save']
        setting_info['name_to_load'] = params['name_to_load']
        #TextBox-advanced options
        setting_info['seed_numpy'] = int(params['seed_numpy']) #default:1, unsigned int
        setting_info['num_particles'] = int(params['num_particles']) #default: 100, unsigned int
        setting_info['fitness_initial'] = float(params['fitness_initial']) #default 0.0, int or float (converted to float)
        #checkBox
        setting_info['is_set_new_vel'] = bool(params['is_set_new_vel']) #default False
        setting_info['is_new_flow_field'] = bool(params['is_new_flow_field']) #default False

        #-------------------Label-1:turbines-------------------#
        #TextBox-must specify
        setting_info['num_turbines'] = int(params['num_turbines'])        #unsigned int
        setting_info['dist_threshold'] = float(params['dist_threshold'])      #int or float (converted to float)
        # select file
        setting_info['turbine_setting'] = params['turbine_setting']
        #checkBox
        setting_info['is_specify_loc_turbines_initial'] = bool(params['is_specify_loc_turbines_initial']) #default False
        #select file when is_specify_loc_turbines==True
        setting_info['dir_turbine_loc'] = params['dir_turbine_loc']


        #-------------------Label-2:boundary setting-------------------#
        #TextBox-must specify
        setting_info['n_row_to_draw'] = 2 #unsigned int
        setting_info['n_col_to_draw'] = 4 #unsigned int
        #group: click to add
        setting_info['list_boundary_files'] = [] #add directories to this list: list_boundary_files.append()
            #for each new adding,select file:
        for boundary_file in params['boundary_files']:
            setting_info['list_boundary_files'].append(boundary_file)



        #-------------------Label-3:flow fields-------------------#

        #select file
        setting_info['dir_measured_wind'] = params['dir_measured_wind']
        setting_info['dir_ground_file'] = params['dir_ground_file']
        setting_info['dir_mesh_file'] = params['dir_mesh_file']
        #TextBox-must specify
        setting_info['height_mast'] = float(params['height_mast']) #int or float (converted to float)
        setting_info['num_direction'] = int(params['num_direction']) #unsigned int
        setting_info['num_speed'] = int(params['num_speed']) #unsigned int
        setting_info['threshold_probability'] = float(params['threshold_probability']) #int or float (converted to float), must between 0.0 and 1.0
        # select file: the selection number is num_direction
        setting_info['list_wind_files'] = []
        setting_info['list_wind_directions'] = []
            #for each new adding,select file:

        for wind_file in params['wind_files']:
            setting_info['list_wind_files'].append(wind_file)

        for wind_direction in params['wind_directions']:
            setting_info['list_wind_directions'].append(float(wind_direction))

    else:
        #--------------user specify the file-------------#
        dir_file = ('/Users/wudong/Works/python/GUI_Version/Optimizer_2022_11_26_setting.npy')
        setting_info_loaded = np.load(dir_file,allow_pickle=True)
        setting_info = setting_info_loaded.item()
    pass


    #============================Automatic calculation parts===========================#

    Farm_zone.save_setting(setting_info['dir_working'],
                           setting_info['name_to_save'],
                           setting_info)


    direction_labels,probability_direction,labels_wind,ratios_wind,probability_wind_in_each_direction = PreProcessing.obtain_wind_probability(setting_info['dir_measured_wind'],
                                                                                                                                              setting_info['height_mast'],
                                                                                                                                              setting_info['num_direction'],
                                                                                                                                              setting_info['num_speed'],
                                                                                                                                              setting_info['threshold_probability'])


    #default setting
    num_err = 0
    step_err = 5
    fitness_best = setting_info['fitness_initial']
    dist_threshold_initial = setting_info['dist_threshold']*1.0

    #1. set boundary information
    my_zone = Farm_zone()
    for boundary_file in setting_info['list_boundary_files']:
        data = pandas.read_excel(boundary_file)
        boundary_points = np.array(data)
        my_zone.add_boundary(boundary_points)
    pass

    #2. set wind information

    my_zone.flowfield.add_ground_info(setting_info['dir_ground_file'],'X_ground','Y_ground','Z_ground')# ground_zone_big
    my_zone.flowfield.add_mesh_info(setting_info['dir_mesh_file'],'X','Y','Z')#mesh_zone_big
    my_zone.flowfield.dict_wind_info.clear()
    list_wind_directions = np.array(setting_info['list_wind_directions'])/180*np.pi
    for i in range(setting_info['num_direction']):
        my_zone.flowfield.add_wind_info(setting_info['list_wind_files'][i],'Wind_U','Wind_V','Wind_W',direction=list_wind_directions[i])
    pass
    my_zone.initialize_list_power() #initialize the power at different wind directions



    #3. initialize the optimizer
    # loc_particles = my_zone.generate_rand_particles(num_particles,num_turbines,seed=seed_numpy)
    temp_turbine = Wind_turbine(setting_info['turbine_setting'])
    setting_info['height_hub'] = temp_turbine.height
    if setting_info['flag_optimizer_status'] == 'initialize':
        loc_particles = my_zone.generate_rand_particles_consider_fitness_and_dict_threshold(num_particles = setting_info['num_particles'],
                                                                                            num_turbines = setting_info['num_turbines'],
                                                                                            dist_threshold = dist_threshold_initial,
                                                                                            list_wind_directions = list_wind_directions,
                                                                                            height_hub = setting_info['height_hub'],
                                                                                            seed=setting_info['seed_numpy'])

        my_optimizer = ParticleSwarm(flag_status='initialize',
                                     num_points=setting_info['num_turbines'],
                                     loc_particles = loc_particles,
                                     num_particles=setting_info['num_particles'],
                                     num_iter=setting_info['num_opt_iteration'],
                                     range_x=my_zone.flowfield.x_range,
                                     range_y=my_zone.flowfield.y_range,
                                     seed=setting_info['seed_numpy'],
                                     fitness_initial=setting_info['fitness_initial'],
                                     dist_threshold = setting_info['dist_threshold'],
                                     boundaries = my_zone.boundaries)
    elif setting_info['flag_optimizer_status'] == 'continue':
        name_loc = setting_info['dir_working']+'/'+setting_info['name_to_load']
        my_optimizer = ParticleSwarm(flag_status='continue',
                                     name=name_loc,
                                     num_iter=setting_info['num_opt_iteration'],
                                     is_set_new_vel=setting_info['is_set_new_vel'],
                                     boundaries = my_zone.boundaries,
                                     is_new_flow_field=setting_info['is_new_flow_field'])
    pass



    #4. Register turbines
    gbest_x,gbest_y = my_optimizer.gbest_coord
    for i in range(setting_info['num_turbines']):
        my_zone.add_turbines(Wind_turbine(setting_info['turbine_setting'],[gbest_x[i],gbest_y[i]]))
    pass
    my_zone.update_turbine_heights()
    #------
    if setting_info['is_specify_loc_turbines_initial'] == True:
        loc_original = np.array(pandas.read_excel(setting_info['dir_turbine_loc']))
        # For Dalian
        for i in range(setting_info['num_turbines']):
            my_zone.update_turbines_coordinate(i,[loc_original[i,0],loc_original[i,1]])
        pass
        my_zone.update_turbine_heights()
        loc_particle_original = ParticleSwarm.coordinate_to_particle(loc_original[:,0],
                                                                     loc_original[:,1],
                                                                     my_zone.flowfield.x_range,
                                                                     my_zone.flowfield.y_range)
        my_optimizer.loc_particles[0,:] = loc_particle_original.copy()
    pass
    #------


    #5. draw turbines and check distances
    #my_zone.draw_turbines_boundary_ground()#for check
    #is_in_boundary,in_which_zone,min_dist = my_zone.get_info_turbines_to_boundaries()

    my_zone.draw_turbines_boundary_flowfield_all_direction(z_target=setting_info['height_hub'],
                                                           list_wind_directions=list_wind_directions,
                                                           n_row=setting_info['n_row_to_draw'],
                                                           n_col=setting_info['n_col_to_draw'])
    #my_zone.draw_all_particles(my_optimizer)


    speeds = np.linspace(start=0, stop=30,num=31)
    CTs = my_zone.list_turbines[0].get_CT(speeds)
    powers = my_zone.list_turbines[0].get_power(speeds)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(speeds,CTs)
    ax.set_title('CT vs. Speeds')
    fig.show()
    fig.savefig(setting_info['dir_working']+'/CT_vs_Speeds.jpg')

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(speeds,powers)
    ax.set_title('Power vs. Speeds')
    fig.show()
    fig.savefig(setting_info['dir_working']+'/Power_vs_Speeds.jpg')

    #7. initialzing multiprocessing
# =============================================================================
#     num_cores = int(mp.cpu_count())
#     pool = mp.Pool(processes=num_cores)
#     print('num_cores={0}'.format(num_cores))
#     print(pool)
# =============================================================================


    #8. Start optimization
    threshold_probability = 1e-3
    num_wind_directions = my_zone.num_wind_directions
    num_wind_pieces = np.shape(probability_wind_in_each_direction)[1]
    is_draw_check = False
    i = my_optimizer.index
    while i < setting_info['num_opt_iteration']:
        # flag of debug
        if i % setting_info['step_check'] == 0:
            is_draw_check = True
        else:
            is_draw_check = False
        pass


        # reset the fitness before each iteration
        my_optimizer.reset_fitness(value = setting_info['fitness_initial']) #Each particle's fitness is set to 'value'
        my_optimizer.update_particles(index_iter=i)


        #--------------------------------start calculating fitness-------------------------------------#
        for j in range(num_wind_directions):
            for jj in range(num_wind_pieces):
                if probability_direction[j] <= threshold_probability or probability_wind_in_each_direction[j,jj]<=threshold_probability:
                    print('skip: direction={},wind={}'.format(list_wind_directions[j]/np.pi*180,labels_wind[jj]))
                    sys.stdout.flush()
                    continue
                else:
                    # load wind mesh of a specified direction
                    print('updating mesh: direction={},wind={}'.format(list_wind_directions[j]/np.pi*180,labels_wind[jj]))
                    sys.stdout.flush()
                    my_zone.flowfield.update_current_wind_info(direction=list_wind_directions[j],
                                                               factor=ratios_wind[jj],
                                                               probability=probability_direction[j]*probability_wind_in_each_direction[j,jj])
                    my_zone.flowfield.update_wind_mesh(direction=list_wind_directions[j])
                pass
                for k in range(setting_info['num_particles']):
                    # load turbine locations of particle k
                    coords_x,coords_y = ParticleSwarm.particle_to_coordinate(my_optimizer.loc_particles[k,:],
                                                                             my_optimizer.range_x,
                                                                             my_optimizer.range_y)

                    # update turbine locations from particle k
                    for m in range(setting_info['num_turbines']):
                        my_zone.update_turbines_coordinate(m,[coords_x[m],coords_y[m]])
                    pass#m
                    my_zone.update_turbine_heights()
                    my_zone.update_boundary_info_of_turbines()
                    # calc local wind by interporlation
                    my_zone.update_turbine_loc_wind_for_all(direction=list_wind_directions[j])
                    my_zone.flowfield.get_streamlines_3D_for_all()


                    #my_zone.calc_factor_reduce_all_with_perturbation(direction=list_wind_directions[j],num_err=num_err,step_err=step_err)
                    list_factor_reduce = my_zone.calc_factor_reduce_all_with_perturbation_consider_boundary_strict(direction=list_wind_directions[j],num_err=num_err,step_err=step_err)
                    for inner_iter in range(3):
                        my_zone.update_turbine_hub_wind_for_all()
                        my_zone.update_turbine_para_for_all()
                        list_factor_reduce_adjusted = my_zone.calc_factor_reduce_all_with_perturbation_consider_boundary_strict(direction=list_wind_directions[j],num_err=num_err,step_err=step_err)
                        max_err_factor_reduce = np.max(np.abs(list_factor_reduce-list_factor_reduce_adjusted))
                        list_factor_reduce = list_factor_reduce_adjusted
                        #print('max_err_factor_reduce={0}'.format(max_err_factor_reduce))
                        if max_err_factor_reduce < 0.1:
                            break
                        pass
                    pass#inner_iter

                    my_zone.update_list_power(direction=list_wind_directions[j]) #print(15*(1-list_factor_reduce))
                    # update firness of the optimizer
                    fitness_direction_j_particle_k = -1.0 * my_zone.get_power_direction(direction=list_wind_directions[j])
                    if fitness_direction_j_particle_k < fitness_best:
                        fitness_best = fitness_direction_j_particle_k
                    pass
                    # update fitness!
                    my_optimizer.update_add_fitness(fitness_direction_j_particle_k,k)
                    print('iter={:}, dir={:}, particle={:}, fit={:.2f}'.format(i,j,k,fitness_direction_j_particle_k))
                    sys.stdout.flush()
                pass#k in range(num_particles)
            pass#jj in range(num_wind_pieces)
        pass#j in range(num_wind_directions)
        #----------------------------------end calculating fitness-------------------------------------#
        my_optimizer.update_pbest_gbest()
        my_optimizer.update_particles(index_iter=i)
        my_optimizer.update_records()
        print('Solu_num:{:}, pbest:{:},gbest:{:},entire_updated:{:},fit_best={:.2f}'.format(my_optimizer.num_solutions_found,
                                                                                            my_optimizer.num_pbest_changed,
                                                                                            my_optimizer.num_gbest_changed,
                                                                                            my_optimizer.num_entirely_updated,
                                                                                            fitness_best))
        sys.stdout.flush()
        if is_draw_check == True:
            # using the gbest locations
            coords_x,coords_y = my_optimizer.gbest_coord
            for m in range(setting_info['num_turbines']):
                my_zone.update_turbines_coordinate(m,[coords_x[m],coords_y[m]])
            pass
            my_zone.update_turbine_heights()
            my_zone.update_boundary_info_of_turbines()
            #my_zone.draw_turbines_boundary_ground()#for check
            fig = my_zone.draw_turbines_boundary_flowfield_all_direction(z_target=setting_info['height_hub'],
                                                                         list_wind_directions=list_wind_directions,
                                                                         n_row=setting_info['n_row_to_draw'],
                                                                         n_col=setting_info['n_col_to_draw'])
            fig.savefig(setting_info['dir_working']+'/turbines_boundary_flowfield_all_direction.jpg')

            #draw iteration status
            fig = my_optimizer.draw_iteration_status()
            fig.savefig(setting_info['dir_working']+'/iteration_status.jpg')
            #draw 3D Streams
            fig = my_zone.draw_streamlines_3D_all_directions(list_wind_directions)
            fig.savefig(setting_info['dir_working']+'/streamlines_3D_all_directions.jpg')
            #draw wind reduce
            fig = my_zone.draw_isoheight_wind_with_reduce_all_direction_probability(list_wind_directions,
                                                                                    setting_info['height_hub'],
                                                                                    1,
                                                                                    num_err,
                                                                                    step_err,
                                                                                    probability_direction,
                                                                                    probability_wind_in_each_direction,
                                                                                    ratios_wind,
                                                                                    threshold_probability)
            fig.savefig(setting_info['dir_working']+'/isoheight_wind_with_reduce_all_direction_probability.jpg')
            my_optimizer.save_optimizer(setting_info['dir_working'],name=setting_info['name_to_save'])
        pass

        my_optimizer.update_iter_index()
        i += 1

    pass





































