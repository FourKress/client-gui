import sys
import os
print(os.getcwd())
sys.stdout.flush()
sys.path.append(os.getcwd() + '/resources/static/package')

import numpy as np
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.interpolate as interpolate
import scipy.io as scio

import pyvista as pv

class FlowField():

    def __init__(self):
        # wind information
        self.dict_wind_info_org = {}  # wind fields
        self.dict_wind_info = {}  # current wind fields
        # ground information
        self.X_ground = []
        self.Y_ground = []
        self.Z_ground = []
        # mesh information
        self.X = []
        self.Y = []
        self.Z = []
        #
        self.xx = []
        self.yy = []
        self.zz = []
        #
        self.x_range = []
        self.y_range = []
        self.z_range = []
        #
        self.delta_x = 0
        self.delta_y = 0
        self.delta_z = 0
        #
        self.num_x = 0
        self.num_y = 0
        self.num_z = 0
        # mesh: including wind field information
        self.wind_mesh = []
        self.l_points_forming_line = []
        self.l_list_lines = []
        self.locations = []  # record the hub's coordinates
        self.l_local_wind_direction = []

    # To define ground information
    def add_ground_info(self, path, key_ground_X, key_ground_Y, key_ground_Z):
        data = scio.loadmat(path)
        self.X_ground = data[key_ground_X]
        self.Y_ground = data[key_ground_Y]
        self.Z_ground = data[key_ground_Z]

    pass

    # To define mesh information
    def add_mesh_info(self, path, key_X, key_Y, key_Z):
        data = scio.loadmat(path)
        self.X = data[key_X]
        self.Y = data[key_Y]
        self.Z = data[key_Z]
        self.x_range = np.array([np.min(self.X), np.max(self.X)])
        self.y_range = np.array([np.min(self.Y), np.max(self.Y)])
        self.z_range = np.array([np.min(self.Z), np.max(self.Z)])
        self.delta_x = self.X[1, 0, 0] - self.X[0, 0, 0]
        self.delta_y = self.Y[0, 1, 0] - self.Y[0, 0, 0]
        self.delta_z = self.Z[0, 0, 1] - self.Z[0, 0, 0]

        self.num_x = np.shape(self.X)[0]
        self.num_y = np.shape(self.Y)[1]
        self.num_z = np.shape(self.Z)[2]

        self.xx = np.linspace(self.x_range[0], self.x_range[1], self.num_x)
        self.yy = np.linspace(self.y_range[0], self.y_range[1], self.num_y)
        self.zz = np.linspace(self.z_range[0], self.z_range[1], self.num_z)

        self.wind_mesh = pv.UniformGrid(dims=(self.num_x, self.num_y, self.num_z),
                                        spacing=(self.delta_x, self.delta_y, self.delta_z))

    pass

    def get_3D_interporlation(self, direction, coordinate):
        u = interpolate.interpn((self.xx, self.yy, self.zz), self.dict_wind_info[direction]['U'], coordinate)[0]
        v = interpolate.interpn((self.xx, self.yy, self.zz), self.dict_wind_info[direction]['V'], coordinate)[0]
        w = interpolate.interpn((self.xx, self.yy, self.zz), self.dict_wind_info[direction]['W'], coordinate)[0]

        return np.array([u, v, w])

    pass

    def get_2D_interporlation(self, map_speed, coordinate):

        u = interpolate.interpn((self.xx, self.yy), map_speed, coordinate)[0]

        return np.array(u)

    pass

    # To define wind information
    def add_wind_info(self, path, key_U, key_V, key_W, direction):
        data = scio.loadmat(path)

        U = data[key_U]
        V = data[key_V]
        W = data[key_W]

        U[np.isnan(U)] = 0.0
        V[np.isnan(V)] = 0.0
        W[np.isnan(W)] = 0.0

        self.dict_wind_info_org.update({
            direction: {
                'U': data[key_U],
                'V': data[key_V],
                'W': data[key_W]}})

        self.dict_wind_info.update({
            direction: {
                'U': data[key_U],
                'V': data[key_V],
                'W': data[key_W],
                'probability': 1.0}})

    pass

    # To define wind information
    def update_current_wind_info(self, direction, factor=1.0, probability=1.0):
        self.dict_wind_info.update({
            direction: {
                'U': self.dict_wind_info_org[direction]['U'] * factor,
                'V': self.dict_wind_info_org[direction]['V'] * factor,
                'W': self.dict_wind_info_org[direction]['W'] * factor,
                'probability': probability}})

    pass

    def draw_2D_contour_nearest(self, direction, z_target, points):
        X = self.X[:, :, 0]
        Y = self.Y[:, :, 0]
        z_temp = np.arange(self.z_range[0], self.z_range[1] + self.delta_z, self.delta_z)
        index = np.argmin(np.abs(z_temp - z_target))
        U = self.dict_wind_info[direction]['U'][:, :, index]
        V = self.dict_wind_info[direction]['V'][:, :, index]
        W = self.dict_wind_info[direction]['W'][:, :, index]

        # pcolormesh,or,contourf
        n_figure_row = 1
        n_figure_col = 3
        color_bar_fraction = 0.05
        factor = 15
        fig, ax = plt.subplots(n_figure_row, n_figure_col, constrained_layout=True,
                               figsize=(factor * n_figure_row, factor * n_figure_col))  # dpi=200
        # norm = matplotlib.colors.Normalize(vmin=-20, vmax=20)#模型一结果绘图
        # U-Field
        im0 = ax[0].contourf(X, Y, U, 10)
        ax[0].xaxis.set_label_text('X')
        ax[0].yaxis.set_label_text('Y')
        ax[0].set_title('U-Field: at {0} and z_target={1}'.format(z_temp[index], z_target), fontsize=10)
        ax[0].streamplot(np.transpose(X), np.transpose(Y), np.transpose(U), np.transpose(V),
                         start_points=points[:, 0:2], color='k')
        ax[0].scatter(points[:, 0], points[:, 1], color='r')
        ax[0].set_aspect(1)
        fig.colorbar(im0, ax=ax[0], fraction=color_bar_fraction)
        # V-Field
        im1 = ax[1].contourf(X, Y, V, 10)
        ax[1].xaxis.set_label_text('X')
        ax[1].yaxis.set_label_text('Y')
        ax[1].set_title('V-Field: at {0} and z_target={1}'.format(z_temp[index], z_target), fontsize=10)
        ax[1].streamplot(np.transpose(X), np.transpose(Y), np.transpose(U), np.transpose(V),
                         start_points=points[:, 0:2], color='k')
        ax[1].scatter(points[:, 0], points[:, 1], color='r')
        ax[1].set_aspect(1)
        fig.colorbar(im1, ax=ax[1], fraction=color_bar_fraction)
        # W-Field
        im2 = ax[2].contourf(X, Y, W, 10)
        ax[2].xaxis.set_label_text('X')
        ax[2].yaxis.set_label_text('Y')
        ax[2].set_title('W-Field: at {0} and z_target={1}'.format(z_temp[index], z_target), fontsize=10)
        ax[2].streamplot(np.transpose(X), np.transpose(Y), np.transpose(U), np.transpose(V),
                         start_points=points[:, 0:2], color='k')
        ax[2].scatter(points[:, 0], points[:, 1], color='r')
        ax[2].set_aspect(1)
        fig.colorbar(im2, ax=ax[2], fraction=color_bar_fraction)

    pass

    def update_wind_mesh(self, direction):

        vectors = np.empty((self.wind_mesh.n_points, 3))
        Wind_U = self.dict_wind_info[direction]['U']
        Wind_V = self.dict_wind_info[direction]['V']
        Wind_W = self.dict_wind_info[direction]['W']
        U_reshape = Wind_U.reshape((self.num_x * self.num_y * self.num_z, 1, 1), order='F').flatten()
        V_reshape = Wind_V.reshape((self.num_x * self.num_y * self.num_z, 1, 1), order='F').flatten()
        W_reshape = Wind_W.reshape((self.num_x * self.num_y * self.num_z, 1, 1), order='F').flatten()
        vectors[:, 0] = U_reshape
        vectors[:, 1] = V_reshape
        vectors[:, 2] = W_reshape
        self.wind_mesh['vectors'] = vectors

    pass

    # BE CAREFUL: should use update_wind_mesh() first
    def get_streamlines_3D(self, start_point):

        # streamlines_3D
        stream, src = self.wind_mesh.streamlines(
            'vectors',
            return_source=True,
            terminal_speed=1e-12,
            start_position=start_point[0],
            min_step_length=0.5, max_step_length=1,
            max_steps=20000,
            max_error=1e-03, max_time=5000,
            integration_direction='forward'  # both
        )

        points_forming_line = np.array(stream.points)
        l = stream.lines
        list_lines = np.array(FlowField.seperate_streamlines(l))

        return points_forming_line, list_lines

    # BE CAREFUL: should use update_wind_mesh() first
    def get_streamlines_3D_with_perturbation(self, start_point, num_err, step_err):
        # streamlines_3D
        num = (2 * num_err + 1) ** 3
        list_streams = [[] for i in range(num)]
        index = 0
        for i in range(-num_err, num_err + 1):
            for j in range(-num_err, num_err + 1):
                for k in range(-num_err, num_err + 1):
                    p = start_point[0]
                    p[0] = p[0] + i * step_err
                    p[1] = p[1] + j * step_err
                    p[2] = p[2] + k * step_err
                    stream, src = self.wind_mesh.streamlines(
                        'vectors',
                        return_source=True,
                        terminal_speed=1e-12,
                        start_position=p,
                        min_step_length=0.5, max_step_length=1,
                        max_steps=20000,
                        max_error=1e-03, max_time=5000,
                        integration_direction='forward'  # both
                    )
                    list_streams[index] = stream
                    index = index + 1
                    pass
                pass
            pass
        pass

        num_min_array = np.zeros([num, ])
        for i in range(num):
            num_min_array[i] = np.shape(np.array(list_streams[i].points))[0]
        pass
        num_min = np.min(num_min_array).astype(int)
        index_num_min = np.argmin(num_min_array)

        points_forming_line = np.zeros([num_min, 3])
        for i in range(index):
            stream = list_streams[i]
            points_forming_line = np.array(stream.points[0:num_min, :]) + points_forming_line
        pass
        points_forming_line = points_forming_line / index
        l = list_streams[index_num_min].lines
        list_lines = np.array(FlowField.seperate_streamlines(l))
        # get local wind directions
        p = points_forming_line[list_lines[0, 0:2]]
        v = p[1, :] - p[0, :]
        local_wind_direction = v / np.linalg.norm(v, ord=2)

        return points_forming_line, list_lines, local_wind_direction

    def assign_locations(self, locations):
        self.locations = np.array(locations)

    pass

    def get_streamlines_3D_for_all_with_perturbation(self, num_err, step_err):
        locations = self.locations
        num_locations = np.shape(locations)[0]
        self.l_points_forming_line = [[] for i in range(num_locations)]
        self.l_list_lines = [[] for i in range(num_locations)]
        for i in range(num_locations):
            self.l_points_forming_line[i], self.l_list_lines[i] = self.get_streamlines_3D_with_perturbation(
                [locations[i]], num_err, step_err)
        pass

    pass

    def get_streamlines_3D_for_all(self):
        locations = self.locations
        num_locations = np.shape(locations)[0]
        self.l_points_forming_line = [[] for i in range(num_locations)]
        self.l_list_lines = [[] for i in range(num_locations)]
        for i in range(num_locations):
            self.l_points_forming_line[i], self.l_list_lines[i] = self.get_streamlines_3D([locations[i]])
        pass

    pass

    def draw_3D_streamlines(self):
        locations = self.locations
        l_points_forming_line = self.l_points_forming_line
        l_list_lines = self.l_list_lines
        num_locations = len(locations)
        n_figure_row = 1
        n_figure_col = 4
        factor_resolving = 20
        fig = plt.figure(
            figsize=(n_figure_row * factor_resolving, n_figure_col * factor_resolving))  # constrained_layout=True
        ax_return = [[] for i in range(n_figure_row * n_figure_col)]
        # --------------------------see:  Contour X-Y------------------------------------------#
        ax = fig.add_subplot(n_figure_row, n_figure_col, 1)
        ax.xaxis.set_label_text('X')
        ax.yaxis.set_label_text('Y')
        ax.set_xlim(xmin=self.x_range[0], xmax=self.x_range[1])
        ax.set_ylim(ymin=self.y_range[0], ymax=self.y_range[1])
        ax.contourf(self.X_ground.transpose(), self.Y_ground.transpose(), self.Z_ground.transpose(), zorder=0)
        for j in range(num_locations):
            # plot lines for locations[j]
            points = l_points_forming_line[j]
            list_lines = l_list_lines[j]
            num_lines = len(l_list_lines[j])
            for i in range(num_lines):
                ax.plot(points[list_lines[i], 0], points[list_lines[i], 1], lw=5, c='k', zorder=1000 * (j + 1) + i)
            pass
            are_points_in_ground = self.check_if_points_in_gorund(points)
            num_points = len(points)
            # plot points
            for i in range(num_points):
                if are_points_in_ground[i] == False:
                    ax.scatter(points[i, 0], points[i, 1], c='g', s=2, zorder=2000 * (j + 1) + i)
                else:
                    ax.scatter(points[i, 0], points[i, 1], c='r', s=2, zorder=2000 * (j + 1) + i)
                pass
            pass
        pass
        for i in range(len(locations)):
            ax.scatter(locations[i][0], locations[i][1], c='w', s=5, zorder=30000)
        pass
        ax.set_aspect(1)
        ax_return[0] = ax
        # --------------------------see:  3D------------------------------------------#
        ax = fig.add_subplot(n_figure_row, n_figure_col, 2, projection='3d')
        ax.xaxis.set_label_text('X')
        ax.yaxis.set_label_text('Y')
        ax.zaxis.set_label_text('Z')
        ax.set_xlim(xmin=self.x_range[0], xmax=self.x_range[1])
        ax.set_ylim(ymin=self.y_range[0], ymax=self.y_range[1])
        ax.set_zlim(zmin=self.z_range[0], zmax=self.z_range[1])
        ax.plot_wireframe(self.X_ground.transpose(), self.Y_ground.transpose(), self.Z_ground.transpose(), linewidths=1,
                          zorder=0)
        for j in range(num_locations):
            # plot lines for locations[j]
            points = l_points_forming_line[j]
            list_lines = l_list_lines[j]
            num_lines = len(l_list_lines[j])
            for i in range(len(list_lines)):
                ax.plot3D(points[list_lines[i], 0], points[list_lines[i], 1], points[list_lines[i], 2], lw=2, c='k',
                          zorder=10000 * (j + 1) + i)
            pass
        pass
        for i in range(len(locations)):
            ax.scatter3D(locations[i][0], locations[i][1], locations[i][2], c='y', s=5, zorder=30000)
        pass
        ax.view_init(elev=45, azim=45, vertical_axis='z')
        ax_return[1] = ax
        # --------------------------see: Surf X-Z------------------------------------------#
        ax = fig.add_subplot(n_figure_row, n_figure_col, 3, projection='3d')
        ax.xaxis.set_label_text('X')
        ax.yaxis.set_label_text('Y')
        ax.zaxis.set_label_text('Z')
        ax.set_xlim(xmin=self.x_range[0], xmax=self.x_range[1])
        ax.set_ylim(ymin=self.y_range[0], ymax=self.y_range[1])
        ax.set_zlim(zmin=self.z_range[0], zmax=self.z_range[1])
        ax.plot_wireframe(self.X_ground.transpose(), self.Y_ground.transpose(), self.Z_ground.transpose(), linewidths=1,
                          zorder=0)
        for j in range(num_locations):
            # plot lines for locations[j]
            points = l_points_forming_line[j]
            list_lines = l_list_lines[j]
            num_lines = len(l_list_lines[j])
            for i in range(len(list_lines)):
                ax.plot3D(points[list_lines[i], 0], points[list_lines[i], 1], points[list_lines[i], 2], lw=2, c='k',
                          zorder=10000 * (j + 1) + i)
            pass
        pass
        for i in range(len(locations)):
            ax.scatter3D(locations[i][0], locations[i][1], locations[i][2], c='y', s=5, zorder=30000)
        pass
        ax.view_init(elev=0, azim=-90, vertical_axis='z')
        ax_return[2] = ax
        # --------------------------see: Surf Y-Z------------------------------------------#
        ax = fig.add_subplot(n_figure_row, n_figure_col, 4, projection='3d')
        ax.xaxis.set_label_text('X')
        ax.yaxis.set_label_text('Y')
        ax.zaxis.set_label_text('Z')
        ax.set_xlim(xmin=self.x_range[0], xmax=self.x_range[1])
        ax.set_ylim(ymin=self.y_range[0], ymax=self.y_range[1])
        ax.set_zlim(zmin=self.z_range[0], zmax=self.z_range[1])
        ax.plot_wireframe(self.X_ground.transpose(), self.Y_ground.transpose(), self.Z_ground.transpose(), linewidths=1,
                          zorder=0)
        for j in range(num_locations):
            # plot lines for locations[j]
            points = l_points_forming_line[j]
            list_lines = l_list_lines[j]
            num_lines = len(l_list_lines[j])
            for i in range(len(list_lines)):
                ax.plot3D(points[list_lines[i], 0], points[list_lines[i], 1], points[list_lines[i], 2], lw=2, c='k',
                          zorder=10000 * (j + 1) + i)
            pass
        pass
        for i in range(len(locations)):
            ax.scatter3D(locations[i][0], locations[i][1], locations[i][2], c='y', s=5, zorder=30000)
        pass
        ax.view_init(elev=0, azim=180, vertical_axis='z')
        ax_return[3] = ax

        return fig, ax_return

    def draw_3D_streamlines_in_fig(self, fig, row, n_figure_row=1, n_figure_col=4, figsize=(20, 32)):
        locations = self.locations
        l_points_forming_line = self.l_points_forming_line
        l_list_lines = self.l_list_lines
        num_locations = len(locations)
        if fig == None:
            fig = plt.figure(figsize=figsize)  # constrained_layout=True
        pass
        # --------------------------see:  Contour X-Y------------------------------------------#
        ax = fig.add_subplot(n_figure_row, n_figure_col, row * n_figure_col + 1)
        ax.xaxis.set_label_text('X')
        ax.yaxis.set_label_text('Y')
        ax.set_xlim(xmin=self.x_range[0], xmax=self.x_range[1])
        ax.set_ylim(ymin=self.y_range[0], ymax=self.y_range[1])
        ax.contourf(self.X_ground.transpose(), self.Y_ground.transpose(), self.Z_ground.transpose(), zorder=0)
        for j in range(num_locations):
            # plot lines for locations[j]
            points = l_points_forming_line[j]
            list_lines = l_list_lines[j]
            num_lines = len(l_list_lines[j])
            for i in range(num_lines):
                ax.plot(points[list_lines[i], 0], points[list_lines[i], 1], lw=5, c='k', zorder=1000 * (j + 1) + i)
            pass
            are_points_in_ground = self.check_if_points_in_gorund(points)
            num_points = len(points)
            # plot points
            for i in range(num_points):
                if are_points_in_ground[i] == False:
                    ax.scatter(points[i, 0], points[i, 1], c='g', s=2, zorder=2000 * (j + 1) + i)
                else:
                    ax.scatter(points[i, 0], points[i, 1], c='r', s=2, zorder=2000 * (j + 1) + i)
                pass
            pass
        pass
        for i in range(len(locations)):
            ax.scatter(locations[i][0], locations[i][1], c='w', s=5, zorder=30000)
        pass
        ax.set_aspect(1)
        # --------------------------see:  3D------------------------------------------#
        ax = fig.add_subplot(n_figure_row, n_figure_col, row * n_figure_col + 2, projection='3d')
        ax.xaxis.set_label_text('X')
        ax.yaxis.set_label_text('Y')
        ax.zaxis.set_label_text('Z')
        ax.set_xlim(xmin=self.x_range[0], xmax=self.x_range[1])
        ax.set_ylim(ymin=self.y_range[0], ymax=self.y_range[1])
        ax.set_zlim(zmin=self.z_range[0], zmax=self.z_range[1])
        ax.plot_wireframe(self.X_ground.transpose(), self.Y_ground.transpose(), self.Z_ground.transpose(), linewidths=1,
                          zorder=0)
        for j in range(num_locations):
            # plot lines for locations[j]
            points = l_points_forming_line[j]
            list_lines = l_list_lines[j]
            num_lines = len(l_list_lines[j])
            for i in range(len(list_lines)):
                ax.plot3D(points[list_lines[i], 0], points[list_lines[i], 1], points[list_lines[i], 2], lw=2, c='k',
                          zorder=10000 * (j + 1) + i)
            pass
        pass
        for i in range(len(locations)):
            ax.scatter3D(locations[i][0], locations[i][1], locations[i][2], c='y', s=5, zorder=30000)
        pass
        ax.view_init(elev=45, azim=45, vertical_axis='z')
        # --------------------------see: Surf X-Z------------------------------------------#
        ax = fig.add_subplot(n_figure_row, n_figure_col, row * n_figure_col + 3, projection='3d')
        ax.xaxis.set_label_text('X')
        ax.yaxis.set_label_text('Y')
        ax.zaxis.set_label_text('Z')
        ax.set_xlim(xmin=self.x_range[0], xmax=self.x_range[1])
        ax.set_ylim(ymin=self.y_range[0], ymax=self.y_range[1])
        ax.set_zlim(zmin=self.z_range[0], zmax=self.z_range[1])
        ax.plot_wireframe(self.X_ground.transpose(), self.Y_ground.transpose(), self.Z_ground.transpose(), linewidths=1,
                          zorder=0)
        for j in range(num_locations):
            # plot lines for locations[j]
            points = l_points_forming_line[j]
            list_lines = l_list_lines[j]
            num_lines = len(l_list_lines[j])
            for i in range(len(list_lines)):
                ax.plot3D(points[list_lines[i], 0], points[list_lines[i], 1], points[list_lines[i], 2], lw=2, c='k',
                          zorder=10000 * (j + 1) + i)
            pass
        pass
        for i in range(len(locations)):
            ax.scatter3D(locations[i][0], locations[i][1], locations[i][2], c='y', s=5, zorder=30000)
        pass
        ax.view_init(elev=0, azim=-90, vertical_axis='z')
        # --------------------------see: Surf Y-Z------------------------------------------#
        ax = fig.add_subplot(n_figure_row, n_figure_col, row * n_figure_col + 4, projection='3d')
        ax.xaxis.set_label_text('X')
        ax.yaxis.set_label_text('Y')
        ax.zaxis.set_label_text('Z')
        ax.set_xlim(xmin=self.x_range[0], xmax=self.x_range[1])
        ax.set_ylim(ymin=self.y_range[0], ymax=self.y_range[1])
        ax.set_zlim(zmin=self.z_range[0], zmax=self.z_range[1])
        ax.plot_wireframe(self.X_ground.transpose(), self.Y_ground.transpose(), self.Z_ground.transpose(), linewidths=1,
                          zorder=0)
        for j in range(num_locations):
            # plot lines for locations[j]
            points = l_points_forming_line[j]
            list_lines = l_list_lines[j]
            num_lines = len(l_list_lines[j])
            for i in range(len(list_lines)):
                ax.plot3D(points[list_lines[i], 0], points[list_lines[i], 1], points[list_lines[i], 2], lw=2, c='k',
                          zorder=10000 * (j + 1) + i)
            pass
        pass
        for i in range(len(locations)):
            ax.scatter3D(locations[i][0], locations[i][1], locations[i][2], c='y', s=5, zorder=30000)
        pass
        ax.view_init(elev=0, azim=180, vertical_axis='z')

        return fig

    # check if the specified points are under the ground
    # points: numpy array
    # shape(points): (N,3), N points and for each row [x,y,z]
    # shape(are_points_in_ground): (N,)
    def check_if_points_in_gorund(self, points):
        X_ground = self.X_ground
        Y_ground = self.Y_ground
        Z_ground = self.Z_ground
        dx = X_ground[1, 0] - X_ground[0, 0]
        dy = Y_ground[0, 1] - Y_ground[0, 0]
        x = np.arange(X_ground[0, 0], X_ground[-1, 0] + dx, dx)
        y = np.arange(Y_ground[0, 0], Y_ground[0, -1] + dy, dy)
        z = Z_ground.transpose()
        f = interpolate.interp2d(x, y, z, kind='linear')
        num_points = points.shape[0]
        points_on_gorund_z = np.zeros((num_points,))
        for i in range(num_points):
            points_on_gorund_z[i] = f(points[i, 0], points[i, 1])
        pass
        are_points_in_ground = [points[i, 2] < points_on_gorund_z[i] for i in range(num_points)]
        return are_points_in_ground

    pass

    # The lines defined in pyVista is line number following points' index
    # This is to seperate and abstarct the point index
    # shape(lines): (N,), namely [line0_num_points, index_p0, index_p1, line1_num_points, index_p2, index_p3, index_p4]
    # shape(list_lines), (M,), contents: [[index_p0,index_p1],[index_p2, index_p3, index_p4]]
    def seperate_streamlines(lines):
        list_lines = []
        num = len(lines)
        num_lines = 0
        current_line = -1
        for i in range(num):
            if num_lines == 0:
                list_lines.append([])
                current_line = current_line + 1
                num_lines = lines[i]
            else:
                list_lines[current_line].append(lines[i])
                num_lines = num_lines - 1
            pass
        pass
        for i in range(len(list_lines)):
            list_lines[i] = np.array(list_lines[i], int)
        pass
        return list_lines

    pass

    # Defined by line: [[x0,y0,z0],[x1,y1,z1]], plane: point and normal
    # shape(normal): (3,)
    def cal_coordinate_from_2Points_and_plane(segment, point, normal):
        P1 = segment[0, :]
        P2 = segment[1, :]
        a = normal[0]
        b = normal[1]
        c = normal[2]
        d = -1.0 * (a * point[0] + b * point[1] + c * point[2])

        LineVector = np.array(P1 - P2)
        m = LineVector[0]
        n = LineVector[1]
        p = LineVector[2]
        x1 = np.array(P1[0])
        y1 = np.array(P1[1])
        z1 = np.array(P1[2])
        tt = a * m + b * n + c * p
        if np.abs(tt) < 1e-10:
            return None, None, None
        else:
            t = (-a * x1 - b * y1 - c * z1 - d) / tt
            x = m * t + x1
            y = n * t + y1
            z = p * t + z1
            return x, y, z

    # Defined by line: P1 and P2, plane: point and normal
    def get_dist_point_to_streamlines(segment, point, normal):
        x, y, z = FlowField.cal_coordinate_from_2Points_and_plane(segment, point, normal)

        if x == None or y == None or z == None:
            return None

        may_p_affected = True
        if x < np.min(segment[:, 0]) or x > np.max(segment[:, 0]):
            may_p_affected = False
        elif y < np.min(segment[:, 1]) or y > np.max(segment[:, 1]):
            may_p_affected = False
        elif z < np.min(segment[:, 2]) or z > np.max(segment[:, 2]):
            may_p_affected = False
        pass
        if may_p_affected == False:
            return None, None, None
        else:
            dist_cross = np.linalg.norm(np.array([x - point[0], y - point[1], z - point[2]]), ord=2)
            p_start = segment[0, :]
            dist_downstream = np.linalg.norm(np.array([x - p_start[0], y - p_start[1], z - p_start[2]]), ord=2)
            p_cross = np.array([x, y, z])
            return dist_cross, dist_downstream, p_cross
        pass

    pass

    # 1. mesh['vectors'] should have been updated
    # 2. streamlines should have been calculated
    def search_along_streamlines(self, index_windward_p, index_leeward_p):
        p = self.locations[index_leeward_p]
        length_accumulated = 0
        # only search the forward part
        points = self.l_points_forming_line[index_windward_p]
        index_lines = self.l_list_lines[index_windward_p][0]
        num = len(index_lines) - 1
        dist_cross = None
        normal = self.l_local_wind_direction[index_leeward_p]
        for i in range(num):
            segment = points[index_lines[i:i + 2], :]
            v = segment[1, :] - segment[0, :]
            dist_cross, dist_downstream, p_cross = FlowField.get_dist_point_to_streamlines(segment, p, normal)
            if dist_cross != None:
                length_accumulated = length_accumulated + dist_downstream
                # print('length_accumulated(middle)={0}+{1}'.format(length_accumulated,dist_downstream))
                break
            else:
                length_accumulated = length_accumulated + np.linalg.norm(v, ord=2)
                # print('length_accumulated(whole)={0}+{1}'.format(length_accumulated,np.linalg.norm(v,ord=2)))
        pass
        if dist_cross == None:
            length_accumulated = None
        return dist_cross, length_accumulated, p_cross

    pass

    def search_along_streamlines_with_perturbation(self, index_windward_p, index_leeward_p, num_err, step_err):
        p = self.locations[index_leeward_p]
        num_perturbation = (2 * num_err + 1) ** 3
        points = self.l_points_forming_line[index_windward_p]

        if len(self.l_list_lines[index_windward_p]) == 0:
            dist_cross = None
            length_accumulated = None
            return dist_cross, length_accumulated
        pass

        index_lines = self.l_list_lines[index_windward_p][0]

        if len(index_lines) >= 2:
            dist_extended = 1000
            i_a = index_lines[-2]
            i_b = index_lines[-1]
            v_extended = points[i_b, :] - points[i_a, :]
            p_extended = points[i_b, :] + dist_extended * v_extended / np.linalg.norm(v_extended, ord=2)
            points = np.row_stack((points, p_extended))
            index_lines = np.append(index_lines, i_b + 1)
        pass

        num = len(index_lines) - 1
        normal = self.l_local_wind_direction[index_leeward_p]

        list_dist_cross = [[] for i in range(num_perturbation)]
        list_length_accumulated = [[] for i in range(num_perturbation)]
        index = 0
        for ii in range(-num_err, num_err + 1):
            for jj in range(-num_err, num_err + 1):
                for kk in range(-num_err, num_err + 1):
                    p_with_perturbation = np.array([p[0] + ii * step_err, p[1] + jj * step_err, p[2] + kk * step_err])
                    length_accumulated = 0
                    for i in range(num):
                        segment = points[index_lines[i:i + 2], :]
                        v = segment[1, :] - segment[0, :]
                        dist_cross, dist_downstream, __ = FlowField.get_dist_point_to_streamlines(segment,
                                                                                                  p_with_perturbation,
                                                                                                  normal)
                        # print('dist_cross_1={0}'.format(dist_cross))
                        # print('dist_cross'.format(dist_cross))

                        if dist_cross != None:
                            length_accumulated = length_accumulated + dist_downstream
                            # print('length_accumulated(middle)={0}+{1}'.format(length_accumulated,dist_downstream))
                            break
                        else:
                            length_accumulated = length_accumulated + np.linalg.norm(v, ord=2)
                            # print('length_accumulated(whole)={0}+{1}'.format(length_accumulated,np.linalg.norm(v,ord=2)))
                        pass
                    pass
                    if dist_cross == None:
                        length_accumulated = None
                    if dist_cross == None or length_accumulated == None:
                        list_dist_cross[index] = None
                        list_length_accumulated[index] = None
                    else:
                        list_dist_cross[index] = dist_cross
                        list_length_accumulated[index] = length_accumulated
                    pass
                    index = index + 1
                pass
            pass
        pass
        list_dist_cross = np.array(list_dist_cross)
        list_length_accumulated = np.array(list_length_accumulated)

        if len(list_dist_cross[list_dist_cross != None]) == 0:
            dist_cross = None
        else:
            dist_cross = np.mean(list_dist_cross[list_dist_cross != None])
        pass

        if len(list_length_accumulated[list_length_accumulated != None]) == 0:
            length_accumulated = None
        else:
            length_accumulated = np.mean(list_length_accumulated[list_length_accumulated != None])
        pass

        # print('dist_cross_final={0}'.format(dist_cross))
        return dist_cross, length_accumulated

    pass

    def search_along_streamlines_with_perturbation_for_a_point(self, index_windward, coord_leeward, normal_leeward,
                                                               num_err, step_err):
        # self.flowfield.locations[index][0:2]
        p = coord_leeward
        num_perturbation = (2 * num_err + 1) ** 3
        points = self.l_points_forming_line[index_windward]

        if len(self.l_list_lines[index_windward]) == 0:
            dist_cross = None
            length_accumulated = None
            return dist_cross, length_accumulated
        pass

        index_lines = self.l_list_lines[index_windward][0]

        if len(index_lines) >= 2:
            dist_extended = 1000
            i_a = index_lines[-2]
            i_b = index_lines[-1]
            v_extended = points[i_b, :] - points[i_a, :]
            p_extended = points[i_b, :] + dist_extended * v_extended / np.linalg.norm(v_extended, ord=2)
            points = np.row_stack((points, p_extended))
            index_lines = np.append(index_lines, i_b + 1)
        pass

        num = len(index_lines) - 1
        normal = normal_leeward

        list_dist_cross = [[] for i in range(num_perturbation)]
        list_length_accumulated = [[] for i in range(num_perturbation)]
        index = 0
        for ii in range(-num_err, num_err + 1):
            for jj in range(-num_err, num_err + 1):
                for kk in range(-num_err, num_err + 1):
                    p_with_perturbation = np.array([p[0] + ii * step_err, p[1] + jj * step_err, p[2] + kk * step_err])
                    length_accumulated = 0
                    for i in range(num):
                        segment = points[index_lines[i:i + 2], :]
                        v = segment[1, :] - segment[0, :]
                        dist_cross, dist_downstream, __ = FlowField.get_dist_point_to_streamlines(segment,
                                                                                                  p_with_perturbation,
                                                                                                  normal)
                        # print('dist_cross_1={0}'.format(dist_cross))
                        # print('dist_cross'.format(dist_cross))

                        if dist_cross != None:
                            length_accumulated = length_accumulated + dist_downstream
                            # print('length_accumulated(middle)={0}+{1}'.format(length_accumulated,dist_downstream))
                            break
                        else:
                            length_accumulated = length_accumulated + np.linalg.norm(v, ord=2)
                            # print('length_accumulated(whole)={0}+{1}'.format(length_accumulated,np.linalg.norm(v,ord=2)))
                        pass
                    pass
                    if dist_cross == None:
                        length_accumulated = None
                    if dist_cross == None or length_accumulated == None:
                        list_dist_cross[index] = None
                        list_length_accumulated[index] = None
                    else:
                        list_dist_cross[index] = dist_cross
                        list_length_accumulated[index] = length_accumulated
                    pass
                    index = index + 1
                pass
            pass
        pass
        list_dist_cross = np.array(list_dist_cross)
        list_length_accumulated = np.array(list_length_accumulated)

        if len(list_dist_cross[list_dist_cross != None]) == 0:
            dist_cross = None
        else:
            dist_cross = np.mean(list_dist_cross[list_dist_cross != None])
        pass

        if len(list_length_accumulated[list_length_accumulated != None]) == 0:
            length_accumulated = None
        else:
            length_accumulated = np.mean(list_length_accumulated[list_length_accumulated != None])
        pass

        # print('dist_cross_final={0}'.format(dist_cross))
        return dist_cross, length_accumulated

    pass

    def get_isoheight_wind(self, direction, z_target, is_draw):
        # calculate and assigne the height of the turbines

        x_interp = self.X_ground.flatten()
        y_interp = self.Y_ground.flatten()
        z_interp = self.Z_ground.flatten() + z_target
        points_interp = np.array([x_interp, y_interp, z_interp]).transpose()
        num_points = np.shape(points_interp)[0]
        results = np.zeros(shape=(num_points, 3))
        for i in range(num_points):
            results[i, :] = self.get_3D_interporlation(coordinate=points_interp[i], direction=direction)
        pass

        U = np.reshape(results[:, 0], np.shape(self.X_ground))
        V = np.reshape(results[:, 1], np.shape(self.X_ground))
        W = np.reshape(results[:, 2], np.shape(self.X_ground))
        Speed = np.sqrt(U ** 2 + V ** 2 + W ** 2)

        if is_draw == True:
            # pcolormesh,or,contourf
            n_figure_row = 1
            n_figure_col = 4
            color_bar_fraction = 0.05
            factor = 15
            fig, ax = plt.subplots(n_figure_row, n_figure_col, constrained_layout=True,
                                   figsize=(factor * n_figure_row, factor * n_figure_col))  # dpi=200
            # norm = matplotlib.colors.Normalize(vmin=-20, vmax=20)#模型一结果绘图
            # U-Field
            im0 = ax[0].contourf(self.X_ground, self.Y_ground, U, 10)
            ax[0].xaxis.set_label_text('X')
            ax[0].yaxis.set_label_text('Y')
            ax[0].set_title('U-Field: at isoheight={0}'.format(z_target), fontsize=10)
            ax[0].set_aspect(1)
            fig.colorbar(im0, ax=ax[0], fraction=color_bar_fraction)
            # V-Field
            im1 = ax[1].contourf(self.X_ground, self.Y_ground, V, 10)
            ax[1].xaxis.set_label_text('X')
            ax[1].yaxis.set_label_text('Y')
            ax[1].set_title('V-Field: at isoheight={0}'.format(z_target), fontsize=10)
            ax[1].set_aspect(1)
            fig.colorbar(im1, ax=ax[1], fraction=color_bar_fraction)
            # W-Field
            im2 = ax[2].contourf(self.X_ground, self.Y_ground, W, 10)
            ax[2].xaxis.set_label_text('X')
            ax[2].yaxis.set_label_text('Y')
            ax[2].set_title('W-Field: at isoheight={0}'.format(z_target), fontsize=10)
            ax[2].set_aspect(1)
            fig.colorbar(im2, ax=ax[2], fraction=color_bar_fraction)
            # Speed-Field
            im2 = ax[3].contourf(self.X_ground, self.Y_ground, Speed, 10)
            ax[3].xaxis.set_label_text('X')
            ax[3].yaxis.set_label_text('Y')
            ax[3].set_title('Speed-Field: at isoheight={0}'.format(z_target), fontsize=10)
            ax[3].set_aspect(1)
            fig.colorbar(im2, ax=ax[3], fraction=color_bar_fraction)
        pass
        return U, V, W, Speed

    pass


if __name__ == '__main__':
    my_flowfield = FlowField()
    my_flowfield.add_ground_info('ground_2D.mat', 'X_ground', 'Y_ground', 'Z_ground')
    my_flowfield.add_mesh_info('mesh_2D.mat', 'X', 'Y', 'Z')
    my_flowfield.add_wind_info('Wind_2D.mat', 'Wind_U', 'Wind_V', 'Wind_W', direction=0)

    U = my_flowfield.dict_wind_info[0]['U']

    check_points_3D = np.array([[200, 200, 78], [220, 220, 78]])
    my_flowfield.get_3D_interporlation(coordinate=check_points_3D[0], direction=0)

    x = np.array([2.14950138e-02, 6.77185690e-01, 2.31408769e+02, 3.66919547e+01,
                  6.67398550e+01, 9.07624758e+01, 4.41044931e+02, 1.29303619e+02,
                  1.45609995e+02, 1.74671919e+02, 1.99305605e+02, 2.05987985e+02,
                  2.65956192e+02, 2.95916788e+02, 2.43267742e+02, 3.33262726e+02,
                  3.72698194e+02, 3.33114830e+02, 3.78299560e+02, 4.00001072e+02,
                  4.35075610e+02, 4.72039472e+02, 5.00000000e+02, 4.99979465e+02,
                  4.34382077e+02])
    y = np.array([2.46316182e+02, 4.88492412e+02, 4.60962428e+02, 1.45479926e+02,
                  3.26428265e+01, 3.19224668e+02, 2.54639689e-01, 4.17266197e+02,
                  1.26677058e+02, 3.02897232e+01, 4.99999959e+02, 2.18031632e+02,
                  3.98044688e+02, 2.90306531e+02, 1.03504538e+02, 4.77496282e+02,
                  3.55573190e+02, 2.23658770e+01, 1.35262354e+02, 2.57602528e+02,
                  6.31543745e-03, 3.27071214e+02, 7.60659310e+01, 4.23088702e+02,
                  4.99450809e+02])

    num_points = len(x)
    check_points_2D = np.array([[x[i], y[i]] for i in range(num_points)])

    z_target = 78
    my_flowfield.draw_2D_contour_nearest(0, z_target, check_points_2D)

    my_flowfield.assign_locations(check_points_3D)

    my_flowfield.update_wind_mesh(direction=0)
    my_flowfield.get_streamlines_3D_for_all()

    fig, ax_return = my_flowfield.draw_3D_streamlines()

    U, V, W, Speed = my_flowfield.get_isoheight_wind(direction=0, z_target=z_target, is_draw=True)

    coordinate_2D = np.array([220, 220])
    results = my_flowfield.get_2D_interporlation(Speed, coordinate_2D)
    print(results)

    # dist,length_accumulated,p_cross = my_flowfield.search_along_streamlines(index_windward_p=0,index_leeward_p=1)
    # print(dist,length_accumulated)
    # print(p_cross)
    # ax_return[0].scatter(p_cross[0],p_cross[1],c='w',s=5,zorder=40000)
    # #v = check_points_2D[]

    # temp_segment = np.array([[1500.   ,  1500.    ,  850. ,   ],[1457.8529, 1498.9017,  840.1319]])
    # v = temp_segment[1,:]-temp_segment[0,:]

# =============================================================================
# read the abstracted CFD results
# add_ground_info()
# add_mesh_info()
# for i in range(num_wind_directions):
#     add_wind_info()
# #when optimizing the locations of the turbines
# for i in range(num_locations):
#     assign_locations() # of the turbines
#     for j in range(num_wind_directions):
#         update_wind_mesh()
#         get_streamlines_3D_for_all()
#         calculate_wind_reduction_for_all()
#     pass
# pass
# =============================================================================

# =============================================================================
#     segment = np.array([[0, 0, 0],[2, 2, 2]])
#     point = np.array([2.5,1.5,0])
#     normal = np.array([1,1,0])
#     dist = FlowField.check_if_p_in_affected_zone(segment,point,normal)
#     print(dist)
# =============================================================================
