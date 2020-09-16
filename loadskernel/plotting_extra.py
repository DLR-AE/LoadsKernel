# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 11:26:27 2015

@author: voss_ar
"""
import numpy as np
from  matplotlib import pyplot as plt
plt.rcParams.update({'font.size': 16,
                     'svg.fonttype':'none'})
from mpl_toolkits.mplot3d import axes3d
from matplotlib import animation
import os, logging, pickle
from PIL.ImageColor import colormap

from loadskernel import plotting_standard
import loadskernel.io_functions as io_functions

class DetailedPlots(plotting_standard.StandardPlots):
    
    def plot_aerogrid(self, aerogrid, cp = '', colormap = 'jet', value_min = '', value_max = ''):
        # This function plots aerogrids as used in the Loads Kernel
        # - By default, the panales are plotted as a wireframe.
        # - If a pressure distribution (or any numpy array with n values) is given, 
        #   the panels are colored according to this value.
        # - It is possible to give a min and max value for the color distirbution, 
        #   which is useful to compare severeal plots.  
        
        if len(cp) == aerogrid['n']:
            colors = plt.cm.get_cmap(name=colormap)  
            if value_min == '':
                value_min = cp.min()
            if value_max == '':
                value_max = cp.max()   
    
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # plot evry panel seperatly
        # (plotting all at once is much more complicated!)
        for i_panel in range(aerogrid['n']):
            # construct matrices xx, yy and zz from cornerpoints for each panale
            point0 =  aerogrid['cornerpoint_grids'][aerogrid['cornerpoint_grids'][:,0] == aerogrid['cornerpoint_panels'][i_panel][0],1:]
            point1 =  aerogrid['cornerpoint_grids'][aerogrid['cornerpoint_grids'][:,0] == aerogrid['cornerpoint_panels'][i_panel][1],1:]
            point2 =  aerogrid['cornerpoint_grids'][aerogrid['cornerpoint_grids'][:,0] == aerogrid['cornerpoint_panels'][i_panel][2],1:]
            point3 =  aerogrid['cornerpoint_grids'][aerogrid['cornerpoint_grids'][:,0] == aerogrid['cornerpoint_panels'][i_panel][3],1:]
            xx = np.array(([point0[0,0], point1[0,0]], [point3[0,0], point2[0,0]]))
            yy = np.array(([point0[0,1], point1[0,1]], [point3[0,1], point2[0,1]]))
            zz = np.array(([point0[0,2], point1[0,2]], [point3[0,2], point2[0,2]]))
            # determine the color of the panel according to pressure coefficient
            # (by default, panels are colored according to its z-component)
            if len(cp) == aerogrid['n']:
                color_i = colors(np.int(np.round( colors.N / (value_max - value_min ) * (cp[i_panel] - value_min ) )))
                ax.plot_surface(xx, yy, zz, rstride=1, cstride=1, linewidth=0.5, edgecolor='black', color=color_i, shade=False )
            else:
                ax.plot_wireframe(xx, yy, zz, rstride=1, cstride=1, color='black')
        
        X,Y,Z = aerogrid['cornerpoint_grids'][:,1], aerogrid['cornerpoint_grids'][:,2], aerogrid['cornerpoint_grids'][:,3]
        # Create cubic bounding box to simulate equal aspect ratio
        # see http://stackoverflow.com/questions/13685386/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to
        max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0
        mid_x = (X.max()+X.min()) * 0.5
        mid_y = (Y.max()+Y.min()) * 0.5
        mid_z = (Z.max()+Z.min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
       
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.view_init(elev=30.0, azim=-120.0) 
        fig.colorbar(plt.cm.ScalarMappable(cmap=colors, norm=plt.cm.colors.Normalize(value_min, value_max)), ax=ax)
        fig.tight_layout()
        
        return ax

    def plot_pressure_distribution(self):
        for response in self.responses:
            trimcase   = self.jcl.trimcase[response['i'][()]]
            logging.info('interactive plotting of resulting pressure distributions for trim {:s}'.format(trimcase['desc']))
            Pk = response['Pk_aero'] #response['Pk_rbm'] + response['Pk_cam']
            i_atmo = self.model.atmo['key'].index(trimcase['altitude'])
            rho = self.model.atmo['rho'][i_atmo]
            Vtas = trimcase['Ma'] * self.model.atmo['a'][i_atmo]
            F = Pk[self.model.aerogrid['set_k'][:,2]] # * -1.0
            cp = F / (rho/2.0*Vtas**2) / self.model.aerogrid['A']
            ax = self.plot_aerogrid(self.model.aerogrid, cp, 'viridis_r',)# -0.5, 0.5)
            ax.set_title('Cp for {:s}'.format(trimcase['desc']))
            plt.show()
            
    def plot_time_data(self):
        # Create all plots
        fig1, (ax11, ax12) = plt.subplots(nrows=2, ncols=1, sharex=True,)
        fig2, (ax21, ax22) = plt.subplots(nrows=2, ncols=1, sharex=True,)
        fig3, (ax31, ax32) = plt.subplots(nrows=2, ncols=1, sharex=True,)
        fig4, (ax41, ax42) = plt.subplots(nrows=2, ncols=1, sharex=True,)
        fig5, (ax51, ax52, ax53) = plt.subplots(nrows=3, ncols=1, sharex=True,)
        fig6, (ax61, ax62) = plt.subplots(nrows=2, ncols=1, sharex=True,)
        if hasattr(self.jcl, 'landinggear'):
            fig7, (ax71, ax72) = plt.subplots(nrows=2, ncols=1, sharex=True,)
        
        # Loop over responses and fill plots with data
        for response in self.responses:
            trimcase   = self.jcl.trimcase[response['i'][()]]
            logging.info('plotting for simulation {:s}'.format(trimcase['desc']))
            
            i_mass     = self.model.mass['key'].index(trimcase['mass'])
            n_modes    = self.model.mass['n_modes'][i_mass] 
            
            Cl = response['Pmac'][:,2] / response['q_dyn'][:].T / self.jcl.general['A_ref']
            ax11.plot(response['t'], response['Pmac'][:,2], 'b-')
            ax12.plot(response['t'], Cl.T, 'b-')
            
            if self.jcl.aero['method'] in ['mona_unsteady']:
                Pb_gust = []
                Pb_unsteady = []
                for i_step in range(len(response['t'])):        
                    Pb_gust.append(np.dot(self.model.Dkx1.T, response['Pk_gust'][i_step,:])[2])
                    Pb_unsteady.append(np.dot(self.model.Dkx1.T, response['Pk_unsteady'][i_step,:])[2])
                ax11.plot(response['t'], Pb_gust, 'k-')
                ax11.plot(response['t'], Pb_unsteady, 'r-')

            ax21.plot(response['t'], response['q_dyn'], 'k-')
            ax22.plot(response['t'], response['Nxyz'][:,2], 'b-')
            ax22.plot(response['t'], response['alpha'][:]/np.pi*180.0, 'r-')
            ax22.plot(response['t'], response['beta'][:]/np.pi*180.0, 'c-')

            ax31.plot(response['t'], response['X'][:,0], 'b-')
            ax31.plot(response['t'], response['X'][:,1], 'g-')
            ax31.plot(response['t'], response['X'][:,2], 'r-')

            ax32.plot(response['t'], response['X'][:,3]/np.pi*180.0, 'b-')
            ax32.plot(response['t'], response['X'][:,4]/np.pi*180.0, 'g-')
            ax32.plot(response['t'], response['X'][:,5]/np.pi*180.0, 'r-')

            ax41.plot(response['t'], response['X'][:,6], 'b-')
            ax41.plot(response['t'], response['X'][:,7], 'g-')
            ax41.plot(response['t'], response['X'][:,8], 'r-')

            ax42.plot(response['t'], response['X'][:,9]/np.pi*180.0, 'b-')
            ax42.plot(response['t'], response['X'][:,10]/np.pi*180.0, 'g-')
            ax42.plot(response['t'], response['X'][:,11]/np.pi*180.0, 'r-')

            ax51.plot(response['t'], response['X'][:,12+2*n_modes+0]/np.pi*180.0, 'b-')
            ax51.plot(response['t'], response['X'][:,12+2*n_modes+1]/np.pi*180.0, 'g-')
            ax51.plot(response['t'], response['X'][:,12+2*n_modes+2]/np.pi*180.0, 'r-')

            ax52.plot(response['t'], response['X'][:,12+2*n_modes+3], 'k-')
            
            ax53.plot(response['t'], response['X'][:,12+2*n_modes+4], 'b-')
            ax53.plot(response['t'], response['X'][:,12+2*n_modes+5], 'g-')

            ax61.plot(response['t'], response['Uf'], 'b-')

            ax62.plot(response['t'], response['d2Ucg_dt2'][:,0], 'b-')
            ax62.plot(response['t'], response['d2Ucg_dt2'][:,1], 'g-')
            ax62.plot(response['t'], response['d2Ucg_dt2'][:,2], 'r-')
            
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
        ax22.set_xlabel('t [sec]')
        ax22.legend(['Nz', 'alpha', 'beta'])
        ax22.grid(True)
        ax22.set_ylabel('[-]/[deg]')
        
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

class Animations(plotting_standard.StandardPlots):   
                    
    def make_movie(self, path_output, speedup_factor=1.0):
        for response in self.responses:
            self.plot_time_animation_3d(response, path_output, speedup_factor=speedup_factor, make_movie=True)
    
    def make_animation(self, speedup_factor=1.0):
        for response in self.responses:
            self.plot_time_animation_3d(response, speedup_factor=speedup_factor)
                  
    def plot_time_animation_3d(self, response, path_output='./', speedup_factor=1.0, make_movie=False):
        from mayavi import mlab
        from tvtk.api import tvtk
        trimcase   = self.jcl.trimcase[response['i'][()]]
        simcase    = self.jcl.simcase[response['i'][()]] 
        
        def update_timestep(self, i):
            self.fig.scene.disable_render = True
            points_i = np.array([self.x[i], self.y[i], self.z[i]]).T
            scalars_i = self.color_scalar[i,:]
            update_strc_display(self, points_i, scalars_i)
            update_text_display(self, response['t'][i][0])
            for ug_vector, ug_cone, data in zip(self.ug_vectors, self.ug_cones, self.vector_data):
                vector_data_i = np.vstack((data['u'][i,:], data['v'][i,:], data['w'][i,:])).T
                update_vector_display(self, ug_vector, ug_cone, points_i, vector_data_i)
            # get current view and set new focal point
            v = mlab.view()
            r = mlab.roll()
            mlab.view(azimuth=v[0], elevation=v[1], roll=r, distance=v[2], focalpoint=points_i.mean(axis=0)) # view from right and above
            self.fig.scene.disable_render = False
        
        @mlab.animate(delay=int(simcase['dt']*1000.0/speedup_factor), ui=True)
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
                self.fig.scene.save_png('{}anim/subcase_{}_frame_{:06d}.png'.format(path_output, trimcase['subcase'], i))

        self.vector_data = []
        def calc_vector_data(self, grid, set='', name='Pg_aero_global', exponent=0.33):
            Pg = response[name][:]
            # scaling to enhance small vectors
            uvw_t0 = np.linalg.norm(Pg[:,grid['set'+set][:,(0,1,2)]], axis=2)
            f_e = uvw_t0**exponent
            # apply scaling to Pg
            u = Pg[:,grid['set'+set][:,0]] / uvw_t0 * f_e
            v = Pg[:,grid['set'+set][:,1]] / uvw_t0 * f_e
            w = Pg[:,grid['set'+set][:,2]] / uvw_t0 * f_e
            # guard for NaNs due to pervious division by uvw
            u[np.isnan(u)] = 0.0
            v[np.isnan(v)] = 0.0
            w[np.isnan(w)] = 0.0
            # maximale Ist-Laenge eines Vektors
            r_max = np.max((u**2.0 + v**2.0 + w**2.0)**0.5)
            # maximale Soll-Laenge eines Vektors, abgeleitet von der Ausdehnung des Modells
            r_scale = 0.5*np.max([grid['offset'+set][:,0].max() - grid['offset'+set][:,0].min(), grid['offset'+set][:,1].max() - grid['offset'+set][:,1].min(), grid['offset'+set][:,2].max() - grid['offset'+set][:,2].min()])
            # skalieren der Vektoren
            u = u / r_max * r_scale
            v = v / r_max * r_scale
            w = w / r_max * r_scale
            # store
            self.vector_data.append({'u':u, 'v':v, 'w':w  })
            
        self.ug_vectors = []
        self.ug_cones   = []
        def setup_vector_display(self, vector_data, color=(1,0,0), opacity=0.4):
             # vectors
            ug_vector = tvtk.UnstructuredGrid(points=np.vstack((self.x[0,:], self.y[0,:], self.z[0,:])).T)
            ug_vector.point_data.vectors = np.vstack((vector_data['u'][0,:], vector_data['v'][0,:], vector_data['w'][0,:])).T
            src_vector = mlab.pipeline.add_dataset(ug_vector)
            vector = mlab.pipeline.vectors(src_vector, color=color, mode='2ddash', opacity=opacity,  scale_mode='vector', scale_factor=1.0)
            vector.glyph.glyph.clamping=False
            self.ug_vectors.append(ug_vector)
            # cones for vectors
            ug_cone = tvtk.UnstructuredGrid(points=np.vstack((self.x[0,:]+vector_data['u'][0,:], self.y[0,:]+vector_data['v'][0,:], self.z[0,:]+vector_data['w'][0,:])).T)
            ug_cone.point_data.vectors = np.vstack((vector_data['u'][0,:], vector_data['v'][0,:], vector_data['w'][0,:])).T
            src_cone = mlab.pipeline.add_dataset(ug_cone)
            cone = mlab.pipeline.vectors(src_cone, color=color, mode='cone', opacity=opacity, scale_mode='vector', scale_factor=0.1, resolution=16)
            cone.glyph.glyph.clamping=False
            self.ug_cones.append(ug_cone)
        
        def update_vector_display(self, ug_vector, ug_cone, points, vector):
            ug_vector.points.from_array(points)
            ug_vector.point_data.vectors.from_array(vector)
            ug_vector.modified()
            ug_cone.points.from_array(points+vector)
            ug_cone.point_data.vectors.from_array(vector)
            ug_cone.modified()
            
        def setup_strc_display(self, color=(1,1,1)):
            points = np.vstack((self.x[0,:], self.y[0,:], self.z[0,:])).T
            scalars = self.color_scalar[0,:]
            self.strc_ug = tvtk.UnstructuredGrid(points=points)
            self.strc_ug.point_data.scalars = scalars
            if hasattr(self.model, 'strcshell'):
                # plot shell as surface
                shells = []
                for shell in self.model.strcshell['cornerpoints']: 
                    shells.append([np.where(self.model.strcgrid['ID']==id)[0][0] for id in shell])
                shell_type = tvtk.Polygon().cell_type
                self.strc_ug.set_cells(shell_type, shells)
                src_points = mlab.pipeline.add_dataset(self.strc_ug)
                points  = mlab.pipeline.glyph(src_points, colormap='viridis', scale_factor=self.p_scale, scale_mode = 'none')
                surface = mlab.pipeline.surface(src_points, colormap='viridis')
            else: 
                # plot points as glyphs
                src_points = mlab.pipeline.add_dataset(self.strc_ug)
                points = mlab.pipeline.glyph(src_points, colormap='viridis', scale_factor=self.p_scale, scale_mode = 'none')
        
        def update_strc_display(self, points, scalars):
            self.strc_ug.points.from_array(points)
            self.strc_ug.point_data.scalars.from_array(scalars)
            self.strc_ug.modified()
            
        def setup_text_display(self):
            self.scr_text = mlab.text(x=0.1, y=0.8, text='Time', line_width=0.5, width=0.1)
            self.scr_text.property.background_color=(1,1,1)
            self.scr_text.property.color=(0,0,0)
            
        def update_text_display(self, t):
            self.scr_text.text = 't = {:>5.3f}s'.format(t)
        
        def setup_runway(self, length, width, elevation):
            x, y = np.mgrid[0:length,-width/2.0:width/2.0+1]
            elev = np.ones(x.shape)*elevation
            surf = mlab.surf(x, y, elev, warp_scale=1.0, color=(0.9,0.9,0.9))
        
        def setup_grid(self, altitude):
            spacing=100.0
            x, y = np.mgrid[0:1000+spacing:spacing,-500:500+spacing:spacing]
            z = np.ones(x.shape)*altitude
            xy = mlab.surf(x, y, z, representation='wireframe', line_width=1.0, color=(0.9,0.9,0.9), opacity=0.4)
            #x, z = np.mgrid[0:1000+spacing:spacing,altitude-500:altitude+500+spacing:spacing]
            #y = np.zeros(x.shape)
            #xz = mlab.surf(x, y, z, representation='wireframe', line_width=1.0, color=(0.9,0.9,0.9), opacity=0.4)
            mlab.quiver3d(0.0, 0.0, altitude, 1000.0, 0.0, 0.0, color=(0,0,0),  mode='axes', opacity=0.4,  scale_mode='vector', scale_factor=1.0)

        # --------------
        # configure plot 
        #---------------
        grid = self.model.strcgrid
        set = ''
        
        # get deformations
        self.x = grid['offset'+set][:,0] + response['Ug'][:,grid['set'+set][:,0]]
        self.y = grid['offset'+set][:,1] + response['Ug'][:,grid['set'+set][:,1]]
        self.z = grid['offset'+set][:,2] + response['Ug'][:,grid['set'+set][:,2]]
        self.color_scalar = np.linalg.norm(response['Ug_f'][:][:,grid['set'+set][:,(0,1,2)]], axis=2)
        
        # get forces
        names = ['Pg_aero_global', 'Pg_iner_global', 'Pg_ext_global', ]# 'Pg_idrag_global', 'Pg_cs_global']
        colors = [(1,0,0), (0,1,1), (0,0,0), (0,0,1)] # red, cyan, black, blue
        for name in names:
            calc_vector_data(self, grid=grid, set=set, name=name)

        # get figure
        if make_movie:
            logging.info('rendering offscreen simulation {:s} ...'.format(trimcase['desc']))
            mlab.options.offscreen = True
            self.fig = mlab.figure(size=(1920, 1080), bgcolor=(1,1,1))
        else: 
            logging.info('interactive plotting of forces and deformations for simulation {:s}'.format(trimcase['desc']))
            self.fig = mlab.figure(bgcolor=(1,1,1))
        
        # plot initial position
        setup_strc_display(self, color=(0.9,0.9,0.9)) # light grey
        setup_text_display(self)
        
        # plot initial forces     
        opacity=0.4  
        for data, color in zip(self.vector_data, colors):
            setup_vector_display(self, data, color, opacity)
        
        # plot coordinate system
        mlab.orientation_axes()
        
        # --- optional ---
        # get earth
#         with open('harz.pickle', 'r') as f:  
#             (x,y,elev) = pickle.load(f)
        # plot earth, scale colormap
#         surf = mlab.surf(x,y,elev, colormap='terrain', warp_scale=-1.0, vmin = -500.0, vmax=1500.0) #gist_earth terrain summer
        #setup_runway(self, length=1000.0, width=30.0, elevation=0.0)
        setup_grid(self, 0.0)
        
        distance = 2.5*((self.x[0,:].max()-self.x[0,:].min())**2 + (self.y[0,:].max()-self.y[0,:].min())**2 + (self.z[0,:].max()-self.z[0,:].min())**2)**0.5
        mlab.view(azimuth=-120.0, elevation=100.0, roll=-75.0,  distance=distance, focalpoint=np.array([self.x[0,:].mean(),self.y[0,:].mean(),self.z[0,:].mean()])) # view from left and above

        if make_movie:
            if not os.path.exists('{}anim/'.format(path_output)):
                os.makedirs('{}anim/'.format(path_output))
            movie(self) # launch animation
            mlab.close()
            # h.246
            cmd = 'ffmpeg -framerate {} -i {}anim/subcase_{}_frame_%06d.png  -r 30 -y {}anim/subcase_{}.mov'.format( speedup_factor/simcase['dt'], path_output, trimcase['subcase'], path_output, trimcase['subcase']) 
            logging.info(cmd)
            os.system(cmd)
            # MPEG-4 - besser geeignet fuer PowerPoint & Co.
            cmd = 'ffmpeg -framerate {} -i {}anim/subcase_{}_frame_%06d.png -c:v mpeg4 -q:v 3 -r 30 -y {}anim/subcase_{}.avi'.format( speedup_factor/simcase['dt'], path_output, trimcase['subcase'], path_output, trimcase['subcase'])
            logging.info(cmd)
            os.system(cmd)
            # GIF als Notloesung.
            cmd1 = 'ffmpeg -i {}anim/subcase_{}_frame_000001.png -filter_complex palettegen -y /tmp/palette.png'.format( path_output, trimcase['subcase'])
            cmd2 = 'ffmpeg -framerate {} -i {}anim/subcase_{}_frame_%06d.png -i /tmp/palette.png -r 15 -filter_complex paletteuse -y {}anim/subcase_{}.gif'.format( speedup_factor/simcase['dt'], path_output, trimcase['subcase'], path_output, trimcase['subcase'])
            logging.info(cmd1)
            os.system(cmd1)
            logging.info(cmd2)
            os.system(cmd2)
            
        else:
            anim(self) # launch animation
            mlab.show()
            
