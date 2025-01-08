# -*- coding: utf-8 -*-
import logging
import os
import numpy as np

from matplotlib import pyplot as plt

try:
    from mayavi import mlab
    from tvtk.api import tvtk
except ImportError:
    pass

from loadskernel import plotting_standard
from modelviewer import plotting as plotting_modelviewer
from loadskernel.io_functions.data_handling import load_hdf5_dict

plt.rcParams.update({'font.size': 16,
                     'svg.fonttype': 'none'})


class DetailedPlots(plotting_standard.LoadPlots, plotting_modelviewer.Plotting):

    def __init__(self, jcl, model):
        self.jcl = jcl
        self.model = model
        # load data from HDF5
        self.aerogrid = load_hdf5_dict(self.model['aerogrid'])
        self.strcgrid = load_hdf5_dict(self.model['strcgrid'])
        self.splinegrid = load_hdf5_dict(self.model['splinegrid'])
        self.calc_parameters_from_model_size()
        self.calc_focalpoint()

    def plot_pressure_distribution(self):
        for response in self.responses:
            trimcase = self.jcl.trimcase[response['i'][()]]
            logging.info('interactive plotting of resulting pressure distributions for trim {:s}'.format(trimcase['desc']))
            Pk = response['Pk_aero']  # response['Pk_rbm'] + response['Pk_cam']
            rho = self.model['atmo'][trimcase['altitude']]['rho'][()]
            Vtas = trimcase['Ma'] * self.model['atmo'][trimcase['altitude']]['a'][()]
            F = Pk[0, self.aerogrid['set_k'][:, 2]]  # * -1.0
            cp = F / (rho / 2.0 * Vtas ** 2) / self.aerogrid['A']
            cp_minmax = [cp.min(), cp.max()]
            self.add_figure(mlab.figure())
            self.setup_aero_display(scalars=cp, colormap='plasma', vminmax=cp_minmax)
            self.set_view_left_above()
            mlab.show()

    def plot_time_data(self):
        # Create all plots
        _, (ax11, ax12) = plt.subplots(nrows=2, ncols=1, sharex=True,)
        _, (ax21, ax22, ax23) = plt.subplots(nrows=3, ncols=1, sharex=True,)
        _, (ax31, ax32) = plt.subplots(nrows=2, ncols=1, sharex=True,)
        _, (ax41, ax42) = plt.subplots(nrows=2, ncols=1, sharex=True,)
        _, (ax51, ax52, ax53) = plt.subplots(nrows=3, ncols=1, sharex=True,)
        _, (ax61, ax62) = plt.subplots(nrows=2, ncols=1, sharex=True,)
        if hasattr(self.jcl, 'landinggear'):
            _, (ax71, ax72) = plt.subplots(nrows=2, ncols=1, sharex=True,)
        Dkx1 = self.model['Dkx1'][()]
        # Loop over responses and fill plots with data
        for response in self.responses:
            trimcase = self.jcl.trimcase[response['i'][()]]
            logging.info('plotting for simulation {:s}'.format(trimcase['desc']))

            self.n_modes = self.model['mass'][trimcase['mass']]['n_modes'][()]

            if self.jcl.aero['method'] in ['mona_steady', 'mona_unsteady']:
                Cl = response['Pmac'][:, 2] / response['q_dyn'][:].T / self.jcl.general['A_ref']
                ax11.plot(response['t'], response['Pmac'][:, 2], 'b-')
                ax12.plot(response['t'], Cl.T, 'b-')

                ax21.plot(response['t'], response['q_dyn'], 'k-')
                ax22.plot(response['t'], response['alpha'][:] / np.pi * 180.0, 'r-')
                ax22.plot(response['t'], response['beta'][:] / np.pi * 180.0, 'c-')
                ax23.plot(response['t'], response['Nxyz'][:, 1], 'g-')
                ax23.plot(response['t'], response['Nxyz'][:, 2], 'b-')

            if self.jcl.aero['method'] in ['mona_unsteady']:
                Pb_gust = []
                Pb_unsteady = []
                for i_step in range(len(response['t'])):
                    Pb_gust.append(np.dot(Dkx1.T, response['Pk_gust'][i_step, :])[2])
                    Pb_unsteady.append(np.dot(Dkx1.T, response['Pk_unsteady'][i_step, :])[2])
                ax11.plot(response['t'], Pb_gust, 'k-')
                ax11.plot(response['t'], Pb_unsteady, 'r-')

            ax32.plot(response['t'], response['X'][:, 3] / np.pi * 180.0, 'b-')
            ax32.plot(response['t'], response['X'][:, 4] / np.pi * 180.0, 'g-')
            ax32.plot(response['t'], response['X'][:, 5] / np.pi * 180.0, 'r-')

            ax41.plot(response['t'], response['X'][:, 6], 'b-')
            ax41.plot(response['t'], response['X'][:, 7], 'g-')
            ax41.plot(response['t'], response['X'][:, 8], 'r-')

            ax41.plot(response['t'], response['X'][:, 6], 'b-')
            ax41.plot(response['t'], response['X'][:, 7], 'g-')
            ax41.plot(response['t'], response['X'][:, 8], 'r-')

            ax42.plot(response['t'], response['X'][:, 9] / np.pi * 180.0, 'b-')
            ax42.plot(response['t'], response['X'][:, 10] / np.pi * 180.0, 'g-')
            ax42.plot(response['t'], response['X'][:, 11] / np.pi * 180.0, 'r-')

            ax51.plot(response['t'], response['X'][:, 12 + 2 * self.n_modes + 0] / np.pi * 180.0, 'b-')
            ax51.plot(response['t'], response['X'][:, 12 + 2 * self.n_modes + 1] / np.pi * 180.0, 'g-')
            ax51.plot(response['t'], response['X'][:, 12 + 2 * self.n_modes + 2] / np.pi * 180.0, 'r-')

            ax52.plot(response['t'], response['X'][:, 12 + 2 * self.n_modes + 3], 'k-')

            ax53.plot(response['t'], response['X'][:, 12 + 2 * self.n_modes + 4], 'b-')
            ax53.plot(response['t'], response['X'][:, 12 + 2 * self.n_modes + 5], 'g-')

            ax61.plot(response['t'], response['Uf'], 'b-')

            ax62.plot(response['t'], response['d2Ucg_dt2'][:, 0], 'b-')
            ax62.plot(response['t'], response['d2Ucg_dt2'][:, 1], 'g-')
            ax62.plot(response['t'], response['d2Ucg_dt2'][:, 2], 'r-')

            if hasattr(self.jcl, 'landinggear'):
                ax71.plot(response['t'], response['p1'])
                ax72.plot(response['t'], response['F1'])

        # Make plots nice
        ax11.set_ylabel('Fz [N]')
        ax11.grid(True)
        if self.jcl.aero['method'] in ['mona_unsteady']:
            ax11.legend(['aero', 'gust', 'unsteady'])
        ax12.set_xlabel('t [sec]')
        ax12.set_ylabel('Cz [-]')
        ax12.grid(True)
        ax12.legend(['Cz'])

        ax21.set_ylabel('[Pa]')
        ax21.grid(True)
        ax21.legend(['q_dyn'])
        ax22.legend(['alpha', 'beta'])
        ax22.grid(True)
        ax22.set_ylabel('[deg]')
        ax23.set_xlabel('t [sec]')
        ax23.legend(['Ny', 'Nz'])
        ax23.grid(True)
        ax23.set_ylabel('[-]')

        ax31.set_ylabel('[m]')
        ax31.grid(True)
        ax31.legend(['x', 'y', 'z'])
        ax32.set_xlabel('t [sec]')
        ax32.set_ylabel('[deg]')
        ax32.grid(True)
        ax32.legend(['phi', 'theta', 'psi'])

        ax41.set_ylabel('[m/s]')
        ax41.grid(True)
        ax41.legend(['u', 'v', 'w'])
        ax42.set_xlabel('t [sec]')
        ax42.set_ylabel('[deg/s]')
        ax42.grid(True)
        ax42.legend(['p', 'q', 'r'])

        ax51.set_ylabel('Inputs [deg]')
        ax51.grid(True)
        ax51.legend(['Xi', 'Eta', 'Zeta'])
        ax52.set_ylabel('Inputs [N]')
        ax52.grid(True)
        ax52.legend(['Thrust'])
        ax53.set_xlabel('t [sec]')
        ax53.set_ylabel('Inputs [deg]')
        ax53.grid(True)
        ax53.legend(['stabilizer', 'flap setting'])

        ax61.set_ylabel('Uf')
        ax61.grid(True)
        ax62.set_xlabel('t [sec]')
        ax62.set_ylabel('d2Ucg_dt2 [m/s^2]')
        ax62.legend(['du', 'dv', 'dw'])
        ax62.grid(True)

        if hasattr(self.jcl, 'landinggear'):
            ax71.legend(self.jcl.landinggear['key'], loc='best')
            ax71.set_ylabel('p1 [m]')
            ax71.grid(True)
            ax72.legend(self.jcl.landinggear['key'], loc='best')
            ax72.set_xlabel('t [s]')
            ax72.set_ylabel('F1 [N]')
            ax72.grid(True)

        # Show plots
        plt.show()

    def plot_forces_deformation_interactive(self):

        # loop over all responses
        for response in self.responses:
            trimcase = self.jcl.trimcase[response['i'][()]]
            logging.info('interactive plotting of forces and deformations for trim {:s}'.format(trimcase['desc']))

            # plot aerodynamic forces
            x = self.aerogrid['offset_k'][:, 0]
            y = self.aerogrid['offset_k'][:, 1]
            z = self.aerogrid['offset_k'][:, 2]
            fscale = 0.5 * self.model_size / np.max(np.abs(response['Pk_aero'][0][self.aerogrid['set_k'][:, (0, 1, 2)]]))
            for name in ['Pk_aero', 'Pk_rbm', 'Pk_cam', 'Pk_cs', 'Pk_f', 'Pk_idrag']:
                if response[name][0].sum() != 0.0:
                    fx = response[name][0][self.aerogrid['set_k'][:, 0]]
                    fy = response[name][0][self.aerogrid['set_k'][:, 1]]
                    fz = response[name][0][self.aerogrid['set_k'][:, 2]]
                    mlab.figure()
                    mlab.points3d(x, y, z, scale_factor=self.pscale)
                    mlab.quiver3d(x, y, z, fx, fy, fz, color=(0, 1, 0), scale_factor=fscale)
                    mlab.title(name, size=0.5, height=0.9)
                else:
                    logging.info('Forces {} are zero, skip plotting'.format(name))

            # plot structural deformations
            x = self.strcgrid['offset'][:, 0]
            y = self.strcgrid['offset'][:, 1]
            z = self.strcgrid['offset'][:, 2]
            x_r = self.strcgrid['offset'][:, 0] + response['Ug_r'][0][self.strcgrid['set'][:, 0]]
            y_r = self.strcgrid['offset'][:, 1] + response['Ug_r'][0][self.strcgrid['set'][:, 1]]
            z_r = self.strcgrid['offset'][:, 2] + response['Ug_r'][0][self.strcgrid['set'][:, 2]]
            x_f = self.strcgrid['offset'][:, 0] + response['Ug'][0][self.strcgrid['set'][:, 0]]
            y_f = self.strcgrid['offset'][:, 1] + response['Ug'][0][self.strcgrid['set'][:, 1]]
            z_f = self.strcgrid['offset'][:, 2] + response['Ug'][0][self.strcgrid['set'][:, 2]]

            mlab.figure()
            mlab.points3d(x_r, y_r, z_r, color=(0, 1, 0), scale_factor=self.pscale)
            mlab.points3d(x_f, y_f, z_f, color=(0, 0, 1), scale_factor=self.pscale)
            mlab.title('rbm (green) and flexible deformation (blue, true scale) in 9300 coord', size=0.5, height=0.9)

            # plot structural forces
            mlab.figure()
            mlab.points3d(x, y, z, scale_factor=self.pscale)
            fscale = 0.5 * self.model_size / np.max(np.abs(response['Pg'][0][self.strcgrid['set'][:, (0, 1, 2)]]))
            fx = response['Pg'][0][self.strcgrid['set'][:, 0]]
            fy = response['Pg'][0][self.strcgrid['set'][:, 1]]
            fz = response['Pg'][0][self.strcgrid['set'][:, 2]]
            mlab.quiver3d(x, y, z, fx * fscale, fy * fscale, fz * fscale, color=(1, 1, 0), mode='2ddash', opacity=0.4,
                          scale_mode='vector', scale_factor=1.0)
            mlab.quiver3d(x + fx * fscale, y + fy * fscale, z + fz * fscale, fx * fscale, fy * fscale, fz * fscale,
                          color=(1, 1, 0), mode='cone', scale_mode='vector', scale_factor=0.1, resolution=16)
            mlab.points3d(self.splinegrid['offset'][:, 0], self.splinegrid['offset'][:, 1], self.splinegrid['offset'][:, 2],
                          color=(1, 1, 0), scale_factor=self.pscale * 1.5)
            mlab.title('Pg', size=0.5, height=0.9)

            if response['Pg_cfd'][0].sum() != 0.0:
                mlab.figure()
                mlab.points3d(x, y, z, scale_factor=self.pscale)
                fscale = 0.5 * self.model_size / np.max(np.abs(response['Pg_cfd'][0][self.strcgrid['set'][:, (0, 1, 2)]]))
                fx = response['Pg_cfd'][0][self.strcgrid['set'][:, 0]]
                fy = response['Pg_cfd'][0][self.strcgrid['set'][:, 1]]
                fz = response['Pg_cfd'][0][self.strcgrid['set'][:, 2]]
                mlab.quiver3d(x, y, z, fx * fscale, fy * fscale, fz * fscale, color=(1, 1, 0), mode='2ddash',
                              opacity=0.4, scale_mode='vector', scale_factor=1.0)
                mlab.quiver3d(x + fx * fscale, y + fy * fscale, z + fz * fscale, fx * fscale, fy * fscale, fz * fscale,
                              color=(1, 1, 0), mode='cone', scale_mode='vector', scale_factor=0.1, resolution=16)
                mlab.points3d(self.splinegrid['offset'][:, 0],
                              self.splinegrid['offset'][:, 1],
                              self.splinegrid['offset'][:, 2],
                              color=(1, 1, 0), scale_factor=self.pscale * 1.5)
                mlab.title('Pg_cfd', size=0.5, height=0.9)
            else:
                logging.info('Forces Pg_cfd are zero, skip plotting')

            # Render all plots
            mlab.show()


class Animations(DetailedPlots):

    def make_movie(self, path_output, speedup_factor=1.0):
        for response in self.responses:
            self.plot_time_animation_3d(response, path_output, speedup_factor=speedup_factor, make_movie=True)

    def make_animation(self, speedup_factor=1.0):
        for response in self.responses:
            self.plot_time_animation_3d(response, speedup_factor=speedup_factor)

    def plot_time_animation_3d(self, response, path_output='./', speedup_factor=1.0, make_movie=False):
        trimcase = self.jcl.trimcase[response['i'][()]]
        simcase = self.jcl.simcase[response['i'][()]]

        def update_timestep(self, i):
            self.fig.scene.disable_render = True
            points_i = np.array([self.x[i], self.y[i], self.z[i]]).T
            scalars_i = self.color_scalar[i, :]
            update_strc_display(self, points_i, scalars_i)
            update_text_display(self, response['t'][i][0])
            for ug_vector, ug_cone, data in zip(self.ug_vectors, self.ug_cones, self.vector_data):
                vector_data_i = np.vstack((data['u'][i, :], data['v'][i, :], data['w'][i, :])).T
                update_vector_display(self, ug_vector, ug_cone, points_i, vector_data_i)
            # get current view and set new focal point
            v = mlab.view()
            r = mlab.roll()
            # view from right and above
            mlab.view(azimuth=v[0], elevation=v[1], roll=r,
                      distance=v[2], focalpoint=points_i.mean(axis=0))
            self.fig.scene.disable_render = False

        @mlab.animate(delay=int(simcase['dt'] * 1000.0 / speedup_factor), ui=True)
        def anim(self):
            # internal function that actually updates the animation
            while True:
                for i in range(len(response['t'])):
                    update_timestep(self, i)
                    yield

        def movie(self):
            # internal function that actually updates the animation
            for i in range(len(response['t'])):
                update_timestep(self, i)
                self.fig.scene.render()
                self.fig.scene.save_png(
                    '{}anim/subcase_{}_frame_{:06d}.png'.format(path_output, trimcase['subcase'], i))

        self.vector_data = []

        def calc_vector_data(self, grid, set='', name='Pg_aero_global', exponent=0.33):
            Pg = response[name][:]
            # scaling to enhance small vectors
            uvw_t0 = np.linalg.norm(Pg[:, grid['set' + set][:, (0, 1, 2)]], axis=2)
            f_e = uvw_t0 ** exponent
            # apply scaling to Pg
            u = Pg[:, grid['set' + set][:, 0]] / uvw_t0 * f_e
            v = Pg[:, grid['set' + set][:, 1]] / uvw_t0 * f_e
            w = Pg[:, grid['set' + set][:, 2]] / uvw_t0 * f_e
            # guard for NaNs due to pervious division by uvw
            u[np.isnan(u)] = 0.0
            v[np.isnan(v)] = 0.0
            w[np.isnan(w)] = 0.0
            # maximale Ist-Laenge eines Vektors
            r_max = np.max((u ** 2.0 + v ** 2.0 + w ** 2.0) ** 0.5)
            # maximale Soll-Laenge eines Vektors, abgeleitet von der Ausdehnung des Modells
            r_scale = 0.5 * np.max([grid['offset' + set][:, 0].max() - grid['offset' + set][:, 0].min(),
                                    grid['offset' + set][:, 1].max() - grid['offset' + set][:, 1].min(),
                                    grid['offset' + set][:, 2].max() - grid['offset' + set][:, 2].min()])
            # skalieren der Vektoren
            u = u / r_max * r_scale
            v = v / r_max * r_scale
            w = w / r_max * r_scale
            # store
            self.vector_data.append({'u': u, 'v': v, 'w': w})

        self.ug_vectors = []
        self.ug_cones = []

        def setup_vector_display(self, vector_data, color=(1, 0, 0), opacity=0.4):
            # vectors
            ug_vector = tvtk.UnstructuredGrid(points=np.vstack((self.x[0, :], self.y[0, :], self.z[0, :])).T)
            ug_vector.point_data.vectors = np.vstack((vector_data['u'][0, :],
                                                      vector_data['v'][0, :],
                                                      vector_data['w'][0, :])).T
            src_vector = mlab.pipeline.add_dataset(ug_vector)
            vector = mlab.pipeline.vectors(src_vector, color=color, mode='2ddash', opacity=opacity,
                                           scale_mode='vector', scale_factor=1.0)
            vector.glyph.glyph.clamping = False
            self.ug_vectors.append(ug_vector)
            # cones for vectors
            ug_cone = tvtk.UnstructuredGrid(points=np.vstack((self.x[0, :] + vector_data['u'][0, :],
                                                              self.y[0, :] + vector_data['v'][0, :],
                                                              self.z[0, :] + vector_data['w'][0, :])).T)
            ug_cone.point_data.vectors = np.vstack(
                (vector_data['u'][0, :], vector_data['v'][0, :], vector_data['w'][0, :])).T
            src_cone = mlab.pipeline.add_dataset(ug_cone)
            cone = mlab.pipeline.vectors(src_cone, color=color, mode='cone', opacity=opacity, scale_mode='vector',
                                         scale_factor=0.1, resolution=16)
            cone.glyph.glyph.clamping = False
            self.ug_cones.append(ug_cone)

        def update_vector_display(self, ug_vector, ug_cone, points, vector):
            ug_vector.points.from_array(points)
            ug_vector.point_data.vectors.from_array(vector)
            ug_vector.modified()
            ug_cone.points.from_array(points + vector)
            ug_cone.point_data.vectors.from_array(vector)
            ug_cone.modified()

        def setup_strc_display(self, color=(1, 1, 1)):
            points = np.vstack((self.x[0, :], self.y[0, :], self.z[0, :])).T
            scalars = self.color_scalar[0, :]
            self.strc_ug = tvtk.UnstructuredGrid(points=points)
            self.strc_ug.point_data.scalars = scalars
            if hasattr(self.model, 'strcshell'):
                # plot shell as surface
                shells = []
                for shell in self.strcshell['cornerpoints']:
                    shells.append([np.where(self.strcgrid['ID'] == id)[0][0]
                                  for id in shell(np.isfinite(shell))])
                shell_type = tvtk.Polygon().cell_type
                self.strc_ug.set_cells(shell_type, shells)
                src_points = mlab.pipeline.add_dataset(self.strc_ug)
                mlab.pipeline.glyph(src_points, colormap='viridis', scale_factor=self.pscale, scale_mode='none')
                mlab.pipeline.surface(src_points, colormap='viridis')
            else:
                # plot points as glyphs
                src_points = mlab.pipeline.add_dataset(self.strc_ug)
                mlab.pipeline.glyph(src_points, colormap='viridis', scale_factor=self.pscale, scale_mode='none')

        def update_strc_display(self, points, scalars):
            self.strc_ug.points.from_array(points)
            self.strc_ug.point_data.scalars.from_array(scalars)
            self.strc_ug.modified()

        def setup_text_display(self):
            self.scr_text = mlab.text(x=0.1, y=0.8, text='Time', line_width=0.5, width=0.1)
            self.scr_text.property.background_color = (1, 1, 1)
            self.scr_text.property.color = (0, 0, 0)

        def update_text_display(self, t):
            self.scr_text.text = 't = {:>5.3f}s'.format(t)

        def setup_runway(self, length, width, elevation):
            x, y = np.mgrid[0:length, -width / 2.0:width / 2.0 + 1]
            elev = np.ones(x.shape) * elevation
            mlab.surf(x, y, elev, warp_scale=1.0, color=(0.9, 0.9, 0.9))

        def setup_grid(self, altitude):
            spacing = 100.0
            x, y = np.mgrid[0:1000 + spacing:spacing, -500:500 + spacing:spacing]
            z = np.ones(x.shape) * altitude
            mlab.surf(x, y, z, representation='wireframe', line_width=1.0, color=(0.9, 0.9, 0.9), opacity=0.4)
            mlab.quiver3d(0.0, 0.0, altitude, 1000.0, 0.0, 0.0, color=(0, 0, 0), mode='axes', opacity=0.4,
                          scale_mode='vector', scale_factor=1.0)

        # --------------
        # configure plot
        # ---------------
        grid = self.strcgrid
        set = ''

        # get deformations
        self.x = grid['offset' + set][:, 0] + response['Ug'][:][:, grid['set' + set][:, 0]]
        self.y = grid['offset' + set][:, 1] + response['Ug'][:][:, grid['set' + set][:, 1]]
        self.z = grid['offset' + set][:, 2] + response['Ug'][:][:, grid['set' + set][:, 2]]
        self.color_scalar = np.linalg.norm(response['Ug_f'][:][:, grid['set' + set][:, (0, 1, 2)]], axis=2)

        # get forces
        # 'Pg_idrag_global', 'Pg_cs_global']
        names = ['Pg_aero_global', 'Pg_iner_global', 'Pg_ext_global', ]
        colors = [(1, 0, 0), (0, 1, 1), (0, 0, 0), (0, 0, 1)]  # red, cyan, black, blue
        for name in names:
            calc_vector_data(self, grid=grid, set=set, name=name)

        # get figure
        if make_movie:
            logging.info('rendering offscreen simulation {:s} ...'.format(trimcase['desc']))
            mlab.options.offscreen = True
            self.fig = mlab.figure(size=(1920, 1080), bgcolor=(1, 1, 1))
        else:
            logging.info('interactive plotting of forces and deformations for simulation {:s}'.format(trimcase['desc']))
            self.fig = mlab.figure(bgcolor=(1, 1, 1))

        # plot initial position
        setup_strc_display(self, color=(0.9, 0.9, 0.9))  # light grey
        setup_text_display(self)

        # plot initial forces
        opacity = 0.4
        for data, color in zip(self.vector_data, colors):
            setup_vector_display(self, data, color, opacity)

        # plot coordinate system
        mlab.orientation_axes()

        # --- optional ---
        # setup_runway(self, length=1000.0, width=30.0, elevation=0.0)
        setup_grid(self, 0.0)

        # view from left and above
        mlab.view(azimuth=-120.0, elevation=100.0, roll=-75.0, distance=self.distance, focalpoint=self.focalpoint)

        if make_movie:
            if not os.path.exists('{}anim/'.format(path_output)):
                os.makedirs('{}anim/'.format(path_output))
            movie(self)  # launch animation
            mlab.close()
            # h.246
            cmd = 'ffmpeg -framerate {} -i {}anim/subcase_{}_frame_%06d.png  -r 30 -y {}anim/subcase_{}.mov'.format(
                speedup_factor / simcase['dt'], path_output, trimcase['subcase'], path_output, trimcase['subcase'])
            logging.info(cmd)
            os.system(cmd)
            # MPEG-4 - besser geeignet fuer PowerPoint & Co.
            cmd = 'ffmpeg -framerate {} -i {}anim/subcase_{}_frame_%06d.png -c:v mpeg4 -q:v 3 -r 30 -y \
                {}anim/subcase_{}.avi'.format(speedup_factor / simcase['dt'], path_output, trimcase['subcase'],
                                              path_output, trimcase['subcase'])
            logging.info(cmd)
            os.system(cmd)
            # GIF als Notloesung.
            cmd1 = 'ffmpeg -i {}anim/subcase_{}_frame_000001.png -filter_complex palettegen -y \
                /tmp/palette.png'.format(path_output, trimcase['subcase'])
            cmd2 = 'ffmpeg -framerate {} -i {}anim/subcase_{}_frame_%06d.png -i /tmp/palette.png -r 15 \
                -filter_complex paletteuse -y {}anim/subcase_{}.gif'.format(
                speedup_factor / simcase['dt'], path_output, trimcase['subcase'], path_output, trimcase['subcase'])
            logging.info(cmd1)
            os.system(cmd1)
            logging.info(cmd2)
            os.system(cmd2)

        else:
            # launch animation
            anim(self)
            mlab.show()
