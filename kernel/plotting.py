# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 11:26:27 2015

@author: voss_ar
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
#from mayavi import mlab
from matplotlib.backends.backend_pdf import PdfPages
from scipy.spatial import ConvexHull
import time, os, copy, logging, cPickle
import build_aero
from PIL.ImageColor import colormap


class plotting:
    def __init__(self, jcl, model, response=None):
        self.jcl = jcl
        self.model = model
        self.response = response
        plt.rcParams['svg.fonttype'] = 'none'
        
        # Allegra
        if self.jcl.general['aircraft'] == 'ALLEGRA':
            self.potatos_Fz_Mx = ['ZWCUT01', 'ZWCUT04', 'ZWCUT08', 'ZWCUT12', 'ZWCUT16', 'ZWCUT20', 'ZWCUT24', 'ZHCUT01' ]
            self.potatos_Mx_My = ['ZWCUT01', 'ZWCUT04', 'ZWCUT08', 'ZWCUT12', 'ZWCUT16', 'ZWCUT20', 'ZWCUT24', 'ZHCUT01' ]
            self.potatos_Fz_My = ['ZWCUT01', 'ZWCUT04', 'ZWCUT08', 'ZWCUT12', 'ZWCUT16', 'ZWCUT20', 'ZWCUT24', 'ZHCUT01', 'ZFCUT27', 'ZFCUT28']
            self.cuttingforces_wing = ['ZWCUT01', 'ZWCUT04', 'ZWCUT08', 'ZWCUT12', 'ZWCUT16', 'ZWCUT20', 'ZWCUT24', ]
            self.f_scale = 0.002 # vectors
            self.p_scale = 0.4 # points
        # DLR-F19
        elif self.jcl.general['aircraft'] == 'DLR F-19-S':
            self.potatos_Fz_Mx = ['MON1', 'MON2', 'MON3', 'MON33', 'MON8', 'MON6', 'MON7', 'MON9']
            self.potatos_Mx_My = ['MON1', 'MON2', 'MON3', 'MON33', 'MON8', 'MON6', 'MON7', 'MON9']
            self.potatos_Fz_My = ['MON1', 'MON2', 'MON3', 'MON33', 'MON4', 'MON5']
            self.cuttingforces_wing = ['MON1', 'MON2', 'MON3', 'MON33', 'MON8']
            self.f_scale = 0.002 # vectors
            self.p_scale = 0.04 # points
        # MULDICON
        elif self.jcl.general['aircraft'] == 'MULDICON':
            self.potatos_Fz_Mx = ['MON1', 'MON2', 'MON3', 'MON33', 'MON8', 'MON9']
            self.potatos_Mx_My = ['MON1', 'MON2', 'MON3', 'MON33', 'MON8', 'MON9']
            self.potatos_Fz_My = ['MON4', 'MON5']
#             self.potatos_Fz_Mx = []
#             self.potatos_Mx_My = []
#             self.potatos_Fz_My = ['MON81', 'MON82', 'MON83']
            self.cuttingforces_wing = ['MON10', 'MON1', 'MON2', 'MON3', 'MON33', 'MON8']
            self.f_scale = 0.002 # vectors
            self.p_scale = 0.03 # points
        # Discus2c
        elif self.jcl.general['aircraft'] == 'Discus2c':
            self.potatos_Fz_Mx = ['MON646', 'MON644', 'MON641', 'MON546', 'MON544', 'MON541', 'MON348', 'MON346', 'MON102']
            self.potatos_Mx_My = ['MON646', 'MON644', 'MON641', 'MON546', 'MON544', 'MON541', 'MON348', 'MON346', 'MON102']
            self.potatos_Fz_My = ['MON102']
            self.cuttingforces_wing = ['MON646', 'MON644', 'MON641', 'MON541', 'MON544', 'MON546']
            self.f_scale = 0.1 # vectors
            self.p_scale = 0.05 # points
        # FLEXOP
        elif self.jcl.general['aircraft'] == 'FLEXOP':
            self.potatos_Fz_Mx = ['MON1']
            self.potatos_Mx_My = ['MON1']
            self.potatos_Fz_My = ['MON1']
            self.cuttingforces_wing = ['MON1']
            self.f_scale = 0.1 # vectors
            self.p_scale = 0.05 # points
        # HALO
        elif self.jcl.general['aircraft'] == 'HALO':
            self.potatos_Fz_Mx = ['MON1']
            self.potatos_Mx_My = ['MON1']
            self.potatos_Fz_My = ['MON1']
            self.cuttingforces_wing = ['MON1']
            self.f_scale = 0.02 # vectors
            self.p_scale = 0.4 # points
        else:
            logging.error('Unknown aircraft: ' + str(self.jcl.general['aircraft']))
            return
        
        
        
    def plot_pressure_distribution(self):
        for i_trimcase in range(len(self.jcl.trimcase)):
            response   = self.response[i_trimcase]
            trimcase   = self.jcl.trimcase[i_trimcase]
            logging.info('interactive plotting of resulting pressure distributions for trim {:s}'.format(trimcase['desc']))
            Pk = response['Pk_aero'] #response['Pk_rbm'] + response['Pk_cam']
            i_atmo = self.model.atmo['key'].index(trimcase['altitude'])
            rho = self.model.atmo['rho'][i_atmo]
            Vtas = trimcase['Ma'] * self.model.atmo['a'][i_atmo]
#             F = np.linalg.norm(Pk[self.model.aerogrid['set_k'][:,:3]], axis=1)
            F = Pk[self.model.aerogrid['set_k'][:,2]] # * -1.0
            cp = F / (rho/2.0*Vtas**2) / self.model.aerogrid['A']
            ax = build_aero.plot_aerogrid(self.model.aerogrid, cp, 'viridis_r',)# -0.5, 0.5)
            ax.set_title('Cp for {:s}'.format(trimcase['desc']))
#             ax.set_xlim(0, 16)
#             ax.set_ylim(-8, 8)
            F = response['Pk_idrag'][self.model.aerogrid['set_k'][:,0]]
            cp = F / (rho/2.0*Vtas**2) / self.model.aerogrid['A']
            ax = build_aero.plot_aerogrid(self.model.aerogrid, cp, 'viridis_r',)# -0.01, 0.03)
            ax.set_title('Cd_ind for {:s}'.format(trimcase['desc']))
            plt.show()
      
    def plot_forces_deformation_interactive(self):
        from mayavi import mlab
                
        for i_trimcase in range(len(self.jcl.trimcase)):
            response   = self.response[i_trimcase]
            trimcase   = self.jcl.trimcase[i_trimcase]
            logging.info('interactive plotting of forces and deformations for trim {:s}'.format(trimcase['desc']))

            x = self.model.aerogrid['offset_k'][:,0]
            y = self.model.aerogrid['offset_k'][:,1]
            z = self.model.aerogrid['offset_k'][:,2]
            fx, fy, fz = response['Pk_rbm'][self.model.aerogrid['set_k'][:,0]],response['Pk_rbm'][self.model.aerogrid['set_k'][:,1]], response['Pk_rbm'][self.model.aerogrid['set_k'][:,2]]

            mlab.figure()
            mlab.points3d(x, y, z, scale_factor=self.p_scale)
            #mlab.quiver3d(x, y, z, response['Pk_rbm'][self.model.aerogrid['set_k'][:,0]], response['Pk_rbm'][self.model.aerogrid['set_k'][:,1]], response['Pk_rbm'][self.model.aerogrid['set_k'][:,2]], color=(0,1,0), scale_factor=self.f_scale)            
            mlab.quiver3d(x, y, z, fx*self.f_scale, fy*self.f_scale, fz*self.f_scale , color=(0,1,0),  mode='2ddash', opacity=0.4,  scale_mode='vector', scale_factor=1.0)
            mlab.quiver3d(x+fx*self.f_scale, y+fy*self.f_scale, z+fz*self.f_scale,fx*self.f_scale, fy*self.f_scale, fz*self.f_scale , color=(0,1,0),  mode='cone', scale_mode='vector', scale_factor=0.2, resolution=16)
            mlab.title('Pk_rbm', size=0.2, height=0.95)
            
            mlab.figure() 
            mlab.points3d(x, y, z, scale_factor=self.p_scale)
            mlab.quiver3d(x, y, z, response['Pk_cam'][self.model.aerogrid['set_k'][:,0]], response['Pk_cam'][self.model.aerogrid['set_k'][:,1]], response['Pk_cam'][self.model.aerogrid['set_k'][:,2]], color=(0,1,1), scale_factor=self.f_scale)            
            mlab.title('Pk_camber_twist', size=0.2, height=0.95)
            
            mlab.figure()        
            mlab.points3d(x, y, z, scale_factor=self.p_scale)
            mlab.quiver3d(x, y, z, response['Pk_cs'][self.model.aerogrid['set_k'][:,0]], response['Pk_cs'][self.model.aerogrid['set_k'][:,1]], response['Pk_cs'][self.model.aerogrid['set_k'][:,2]], color=(1,0,0), scale_factor=self.f_scale)
            mlab.title('Pk_cs', size=0.2, height=0.95)
            
            mlab.figure()   
            mlab.points3d(x, y, z, scale_factor=self.p_scale)
            mlab.quiver3d(x, y, z, response['Pk_f'][self.model.aerogrid['set_k'][:,0]], response['Pk_f'][self.model.aerogrid['set_k'][:,1]], response['Pk_f'][self.model.aerogrid['set_k'][:,2]], color=(1,0,1), scale_factor=self.f_scale)
            mlab.title('Pk_flex', size=0.2, height=0.95)
            
            mlab.figure()   
            mlab.points3d(x, y, z, scale_factor=self.p_scale)
            mlab.quiver3d(x, y, z, response['Pk_cfd'][self.model.aerogrid['set_k'][:,0]], response['Pk_cfd'][self.model.aerogrid['set_k'][:,1]], response['Pk_cfd'][self.model.aerogrid['set_k'][:,2]], color=(0,1,1), scale_factor=self.f_scale)
            mlab.title('Pk_cfd', size=0.2, height=0.95)
            
            mlab.figure()
            mlab.points3d(x, y, z, scale_factor=self.p_scale)
            fx, fy, fz = response['Pk_idrag'][self.model.aerogrid['set_k'][:,0]],response['Pk_idrag'][self.model.aerogrid['set_k'][:,1]], response['Pk_idrag'][self.model.aerogrid['set_k'][:,2]]
            mlab.quiver3d(x, y, z, fx*self.f_scale, fy*self.f_scale, fz*self.f_scale , color=(0,1,0),  mode='2ddash', opacity=0.4,  scale_mode='vector', scale_factor=1.0)
            mlab.quiver3d(x+fx*self.f_scale, y+fy*self.f_scale, z+fz*self.f_scale,fx*self.f_scale, fy*self.f_scale, fz*self.f_scale , color=(0,1,0),  mode='cone', scale_mode='vector', scale_factor=0.2, resolution=16)
            mlab.title('Pk_idrag', size=0.2, height=0.95)
            
            mlab.figure()   
            mlab.points3d(x, y, z, scale_factor=self.p_scale)
            mlab.quiver3d(x, y, z, response['Pk_aero'][self.model.aerogrid['set_k'][:,0]], response['Pk_aero'][self.model.aerogrid['set_k'][:,1]], response['Pk_aero'][self.model.aerogrid['set_k'][:,2]], color=(0,1,0), scale_factor=self.f_scale)
            mlab.title('Pk_aero', size=0.2, height=0.95)
            
            x = self.model.strcgrid['offset'][:,0]
            y = self.model.strcgrid['offset'][:,1]
            z = self.model.strcgrid['offset'][:,2]
            x_r = self.model.strcgrid['offset'][:,0] + response['Ug_r'][self.model.strcgrid['set'][:,0]]
            y_r = self.model.strcgrid['offset'][:,1] + response['Ug_r'][self.model.strcgrid['set'][:,1]]
            z_r = self.model.strcgrid['offset'][:,2] + response['Ug_r'][self.model.strcgrid['set'][:,2]]
            x_f = self.model.strcgrid['offset'][:,0] + response['Ug'][self.model.strcgrid['set'][:,0]]
            y_f = self.model.strcgrid['offset'][:,1] + response['Ug'][self.model.strcgrid['set'][:,1]]
            z_f = self.model.strcgrid['offset'][:,2] + response['Ug'][self.model.strcgrid['set'][:,2]]
            
            mlab.figure()
            #mlab.points3d(x, y, z, scale_factor=self.p_scale)
            mlab.points3d(x_r, y_r, z_r, color=(0,1,0), scale_factor=self.p_scale)
            mlab.points3d(x_f, y_f, z_f, color=(0,0,1), scale_factor=self.p_scale)
            mlab.title('rbm (green) and flexible deformation (blue, true scale) in 9300 coord', size=0.2, height=0.95)

            mlab.figure()   
            mlab.points3d(x, y, z, scale_factor=self.p_scale)
            #mlab.quiver3d(x, y, z, response['Pg'][self.model.strcgrid['set'][:,0]], response['Pg'][self.model.strcgrid['set'][:,1]], response['Pg'][self.model.strcgrid['set'][:,2]], color=(1,1,0), scale_factor=self.f_scale)
            fx, fy, fz = response['Pg'][self.model.strcgrid['set'][:,0]], response['Pg'][self.model.strcgrid['set'][:,1]], response['Pg'][self.model.strcgrid['set'][:,2]]
            mlab.quiver3d(x, y, z, fx*self.f_scale, fy*self.f_scale, fz*self.f_scale , color=(1,1,0),  mode='2ddash', opacity=0.4,  scale_mode='vector', scale_factor=1.0)
            mlab.quiver3d(x+fx*self.f_scale, y+fy*self.f_scale, z+fz*self.f_scale,fx*self.f_scale, fy*self.f_scale, fz*self.f_scale , color=(1,1,0),  mode='cone', scale_mode='vector', scale_factor=0.2, resolution=16)
            mlab.points3d(self.model.splinegrid['offset'][:,0], self.model.splinegrid['offset'][:,1], self.model.splinegrid['offset'][:,2], color=(1,1,0), scale_factor=self.p_scale*1.5)
            mlab.title('Pg', size=0.2, height=0.95)
            
            mlab.show()

    def plot_monstations(self, monstations, filename_pdf, dyn2stat=False):

        # launch plotting
        pp = PdfPages(filename_pdf)
        self.potato_plots(monstations, pp, self.potatos_Fz_Mx, self.potatos_Mx_My, self.potatos_Fz_My, dyn2stat)
        self.cuttingforces_along_wing_plots(monstations, pp, self.cuttingforces_wing, dyn2stat)
        pp.close()
        logging.info('plots saved as ' + filename_pdf)
        #print 'opening '+ filename_pdf
        #os.system('evince ' + filename_pdf + ' &')
        
    def potato_plots(self, monstations, pp, potatos_Fz_Mx, potatos_Mx_My, potatos_Fz_My, dyn2stat=False):
        logging.info('start potato-plotting...')
        # get data needed for plotting from monstations
        if dyn2stat:
            loads_string = 'loads_dyn2stat'
            subcase_string = 'subcases_dyn2stat'
        else:
            loads_string = 'loads'
            subcase_string = 'subcase'
        loads = []
        offsets = []
        potato = np.unique(potatos_Fz_Mx + potatos_Mx_My + potatos_Fz_My)
        for station in potato:
            loads.append(monstations[station][loads_string])
            offsets.append(monstations[station]['offset'])
        loads = np.array(loads)
        offsets = np.array(offsets)
        
        self.crit_trimcases = []
        for i_station in range(len(potato)):
            
            if potato[i_station] in potatos_Fz_Mx:
                points = np.vstack((loads[i_station][:,2], loads[i_station][:,3])).T
                labels = ['Fz [N]', 'Mx [Nm]' ]
                # --- plot ---
                plt.figure()
                plt.scatter(points[:,0], points[:,1], color='cornflowerblue') # plot points
                if points.shape[0] >= 3:
                    hull = ConvexHull(points) # calculated convex hull from scattered points
                    for simplex in hull.simplices:                   # plot convex hull
                        plt.plot(points[simplex,0], points[simplex,1], color='cornflowerblue', linewidth=2.0, linestyle='--')
                    for i_case in range(hull.nsimplex):              # plot text   
                        plt.text(points[hull.vertices[i_case],0], points[hull.vertices[i_case],1], str(monstations[potato[i_station]][subcase_string][hull.vertices[i_case]]), fontsize=8)
                        self.crit_trimcases.append(monstations[potato[i_station]][subcase_string][hull.vertices[i_case]])
                else:
                    self.crit_trimcases += monstations[potato[i_station]][subcase_string][:]
                plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
                plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
                plt.title(potato[i_station])
                plt.grid('on')
                plt.xlabel(labels[0])
                plt.ylabel(labels[1])
                pp.savefig()
                plt.close()
            if potato[i_station] in potatos_Mx_My:
                points = np.vstack((loads[i_station][:,3], loads[i_station][:,4])).T
                labels = ['Mx [N]', 'My [Nm]' ]
                # --- plot ---
                plt.figure()
                plt.scatter(points[:,0], points[:,1], color='cornflowerblue') # plot points
                if points.shape[0] >= 3:
                    hull = ConvexHull(points) # calculated convex hull from scattered points
                    for simplex in hull.simplices:                   # plot convex hull
                        plt.plot(points[simplex,0], points[simplex,1], color='cornflowerblue', linewidth=2.0, linestyle='--')
                    for i_case in range(hull.nsimplex):              # plot text   
                        plt.text(points[hull.vertices[i_case],0], points[hull.vertices[i_case],1], str(monstations[potato[i_station]][subcase_string][hull.vertices[i_case]]), fontsize=8)
                        self.crit_trimcases.append(monstations[potato[i_station]][subcase_string][hull.vertices[i_case]])
                else:
                    self.crit_trimcases += monstations[potato[i_station]][subcase_string][:]
                plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
                plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
                plt.title(potato[i_station])
                plt.grid('on')
                plt.xlabel(labels[0])
                plt.ylabel(labels[1])
                pp.savefig()
                plt.close()   
                
            if potato[i_station] in potatos_Fz_My:
                points = np.vstack((loads[i_station][:,2], loads[i_station][:,4])).T
                labels = ['Fz [N]', 'My [Nm]' ]
                # --- plot ---
                plt.figure()
                plt.scatter(points[:,0], points[:,1], color='cornflowerblue') # plot points
                if points.shape[0] >= 3:
                    hull = ConvexHull(points) # calculated convex hull from scattered points
                    for simplex in hull.simplices:                   # plot convex hull
                        plt.plot(points[simplex,0], points[simplex,1], color='cornflowerblue', linewidth=2.0, linestyle='--')
                    for i_case in range(hull.nsimplex):              # plot text   
                        plt.text(points[hull.vertices[i_case],0], points[hull.vertices[i_case],1], str(monstations[potato[i_station]][subcase_string][hull.vertices[i_case]]), fontsize=8)
                        self.crit_trimcases.append(monstations[potato[i_station]][subcase_string][hull.vertices[i_case]])
                else:
                    self.crit_trimcases += monstations[potato[i_station]][subcase_string][:]
                plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
                plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
                plt.title(potato[i_station])
                plt.grid('on')
                plt.xlabel(labels[0])
                plt.ylabel(labels[1])
                pp.savefig()
                plt.close()  
          
    def cuttingforces_along_wing_plots(self, monstations, pp, cuttingforces_wing, dyn2stat=False):
        logging.info('start plotting cutting forces along wing...')
        cuttingforces = ['Fx [N]', 'Fy [N]', 'Fz [N]', 'Mx [Nm]', 'My [Nm]', 'Mz [Nm]']
        if dyn2stat:
            loads_string = 'loads_dyn2stat'
            subcase_string = 'subcases_dyn2stat'
        else:
            loads_string = 'loads'
            subcase_string = 'subcase'
        loads = []
        offsets = []
        for station in cuttingforces_wing:
            loads.append(monstations[station][loads_string])
            offsets.append(monstations[station]['offset'])
        loads = np.array(loads)
        offsets = np.array(offsets)
        
        for i_cuttingforce in range(len(cuttingforces)):
            i_max = np.argmax(loads[:,:,i_cuttingforce], 1)
            i_min = np.argmin(loads[:,:,i_cuttingforce], 1)
            plt.figure()
            plt.plot(offsets[:,1],loads[:,:,i_cuttingforce], color='cornflowerblue', linestyle='-', marker='.')
            for i_station in range(len(cuttingforces_wing)):
                # verticalalignment or va 	[ 'center' | 'top' | 'bottom' | 'baseline' ]
                # max
                plt.scatter(offsets[i_station,1],loads[i_station,i_max[i_station],i_cuttingforce], color='r')
                plt.text(   offsets[i_station,1],loads[i_station,i_max[i_station],i_cuttingforce], str(monstations[monstations.keys()[0]][subcase_string][i_max[i_station]]), fontsize=8, verticalalignment='bottom' )
                # min
                plt.scatter(offsets[i_station,1],loads[i_station,i_min[i_station],i_cuttingforce], color='r')
                plt.text(   offsets[i_station,1],loads[i_station,i_min[i_station],i_cuttingforce], str(monstations[monstations.keys()[0]][subcase_string][i_min[i_station]]), fontsize=8, verticalalignment='top' )
            plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            plt.title('Wing')        
            plt.grid('on')
            plt.xlabel('y [m]')
            plt.ylabel(cuttingforces[i_cuttingforce]) 
            #plt.show()
            pp.savefig()
            plt.close()
              
    def plot_monstations_time(self, monstations, filename_pdf):
        logging.info('start plotting cutting forces over time ...')
        pp = PdfPages(filename_pdf)
        for key in monstations.keys():
            monstation = monstations[key]
            plt.figure()
            for i_simcase in range(len(self.jcl.simcase)):
                loads = np.array(monstation['loads'][i_simcase])
                t = monstation['t'][i_simcase]
                
                plt.subplot(3,1,1)
                plt.title(key)
                plt.plot(t, loads[:,2], 'b')
                plt.xlabel('t [sec]')
                plt.ylabel('Fz [N]')
                plt.grid('on')
                plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
                plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
                plt.subplot(3,1,2)
                plt.plot(t, loads[:,3], 'g')
                plt.xlabel('t [sec]')
                plt.ylabel('Mx [N]')
                plt.grid('on')
                plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
                plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
                plt.subplot(3,1,3)
                plt.plot(t, loads[:,4], 'r')
                plt.xlabel('t [sec]')
                plt.ylabel('My [N]')
                plt.grid('on')
                plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
                plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            
            pp.savefig()
            plt.close()
        pp.close()
        logging.info('plots saved as ' + filename_pdf)
        
    def plot_time_data(self):
        for i_simcase in range(len(self.jcl.simcase)):
            trimcase = self.jcl.trimcase[i_simcase]
            logging.info('plotting for simulation {:s}'.format(trimcase['desc']))
            Pb_gust = []
            Pb_unsteady = []
            Pb_aero = []
            for i_step in range(len(self.response[i_simcase]['t'])):        
                Pb_gust.append(np.dot(self.model.Dkx1.T, self.response[i_simcase]['Pk_gust'][i_step,:]))
                Pb_unsteady.append(np.dot(self.model.Dkx1.T, self.response[i_simcase]['Pk_unsteady'][i_step,:]))
                Pb_aero.append(np.dot(self.model.Dkx1.T, self.response[i_simcase]['Pk_aero'][i_step,:]))
            Pb_gust = np.array(Pb_gust)
            Pb_unsteady = np.array(Pb_unsteady)
            Pb_aero = np.array(Pb_aero)
            plt.figure(1)
            plt.plot(self.response[i_simcase]['t'], Pb_gust[:,2], 'b-')
            plt.plot(self.response[i_simcase]['t'], Pb_unsteady[:,2], 'r-')
            plt.plot(self.response[i_simcase]['t'], Pb_gust[:,2] + Pb_unsteady[:,2], 'b--')
            plt.plot(self.response[i_simcase]['t'], Pb_aero[:,2], 'g-')
            plt.plot(self.response[i_simcase]['t'], Pb_aero[:,2] - Pb_unsteady[:,2], 'k--')
            plt.xlabel('t [sec]')
            plt.ylabel('Pb [N]')
            plt.grid('on')
            plt.legend(['Pb_gust', 'Pb_unsteady', 'Pb_gust+unsteady', 'Pb_aero', 'Pb_aero-unsteady'])
            
#             plt.figure()
#             plt.subplot(3,1,1)
#             plt.plot(self.response[i_simcase]['t'], self.response[i_simcase]['p1'])
#             plt.plot(self.response[i_simcase]['t'], self.response[i_simcase]['X'][:,95:98], '--')
#             plt.legend(('p1 MLG1', 'p1 MLG2', 'p1 NLG', 'p2 MLG1', 'p2 MLG2', 'p2 NLG'), loc='best')
#             plt.xlabel('t [s]')
#             plt.ylabel('p1,2 [m]')
#             plt.grid('on')
#             plt.subplot(3,1,2)
#             plt.plot(self.response[i_simcase]['t'], self.response[i_simcase]['F1'])
#             plt.plot(self.response[i_simcase]['t'], self.response[i_simcase]['F2'], '--')
#             plt.legend(('F1 MLG1', 'F1 MLG2', 'F1 NLG', 'F2 MLG1', 'F2 MLG2', 'F2 NLG'), loc='best')
#             plt.xlabel('t [s]')
#             plt.ylabel('F1,2 [N]')
#             plt.grid('on')
#              
#             plt.subplot(3,1,3)
#             plt.plot(self.response[i_simcase]['t'], self.response[i_simcase]['dp1'])
#             plt.plot(self.response[i_simcase]['t'], self.response[i_simcase]['X'][:,98:101], '--')
#             plt.legend(('dp1 MLG1', 'dp1 MLG2', 'dp1 NLG', 'dp2 MLG1', 'dp2 MLG2', 'dp2 NLG'), loc='best')
#             plt.xlabel('t [s]')
#             plt.ylabel('dp1,2 [m/s]')
#             plt.grid('on')
        
            plt.figure(2)
            plt.subplot(2,1,1)
            plt.plot(self.response[i_simcase]['t'], self.response[i_simcase]['q_dyn'], 'k-')
            plt.xlabel('t [sec]')
            plt.ylabel('[Pa]')
            plt.grid('on')
            plt.legend(['q_dyn'])
            plt.subplot(2,1,2)
            plt.plot(self.response[i_simcase]['t'], self.response[i_simcase]['Nxyz'][:,2], 'b-')
            plt.plot(self.response[i_simcase]['t'], self.response[i_simcase]['alpha']/np.pi*180.0, 'r-')
            #plt.plot(self.response[i_simcase]['t'], self.response[i_simcase]['X'][:,4]/np.pi*180.0, 'g-') # 'alpha/pitch'
            #plt.plot(self.response[i_simcase]['t'], np.arctan(self.response[i_simcase]['X'][:,8]/self.response[i_simcase]['X'][:,6])/np.pi*180.0, 'k-') # 'alpha/heave'
            plt.plot(self.response[i_simcase]['t'], self.response[i_simcase]['beta']/np.pi*180.0, 'c-')
            plt.xlabel('t [sec]')
            plt.legend(['Nz', 'alpha', 'beta']) 
            plt.grid('on')
            plt.ylabel('[-]/[deg]')
            
            
            plt.figure(3)
            plt.subplot(2,1,1)
            plt.plot(self.response[i_simcase]['t'], self.response[i_simcase]['X'][:,0], 'b-')
            plt.plot(self.response[i_simcase]['t'], self.response[i_simcase]['X'][:,1], 'g-')
            plt.plot(self.response[i_simcase]['t'], self.response[i_simcase]['X'][:,2], 'r-')
            plt.xlabel('t [sec]')
            plt.ylabel('[m]')
            plt.grid('on')
            plt.legend(['x', 'y', 'z'])
            plt.subplot(2,1,2)
            plt.plot(self.response[i_simcase]['t'], self.response[i_simcase]['X'][:,3]/np.pi*180.0, 'b-')
            plt.plot(self.response[i_simcase]['t'], self.response[i_simcase]['X'][:,4]/np.pi*180.0, 'g-')
            plt.plot(self.response[i_simcase]['t'], self.response[i_simcase]['X'][:,5]/np.pi*180.0, 'r-')
            plt.xlabel('t [sec]')
            plt.ylabel('[deg]')
            plt.grid('on')
            plt.legend(['phi', 'theta', 'psi'])
            
            plt.figure(4)
            plt.subplot(2,1,1)
            plt.plot(self.response[i_simcase]['t'], self.response[i_simcase]['X'][:,6], 'b-')
            plt.plot(self.response[i_simcase]['t'], self.response[i_simcase]['X'][:,7], 'g-')
            plt.plot(self.response[i_simcase]['t'], self.response[i_simcase]['X'][:,8], 'r-')
            plt.xlabel('t [sec]')
            plt.ylabel('[m/s]')
            plt.grid('on')
            plt.legend(['u', 'v', 'w'])
            plt.subplot(2,1,2)
            plt.plot(self.response[i_simcase]['t'], self.response[i_simcase]['X'][:,9]/np.pi*180.0, 'b-')
            plt.plot(self.response[i_simcase]['t'], self.response[i_simcase]['X'][:,10]/np.pi*180.0, 'g-')
            plt.plot(self.response[i_simcase]['t'], self.response[i_simcase]['X'][:,11]/np.pi*180.0, 'r-')
            plt.xlabel('t [sec]')
            plt.ylabel('[deg/s]')
            plt.grid('on')
            plt.legend(['p', 'q', 'r'])
            
            plt.figure(5)
            plt.subplot(2,1,1)
            plt.plot(self.response[i_simcase]['t'], self.response[i_simcase]['Y'][:,6], 'b-')
            plt.plot(self.response[i_simcase]['t'], self.response[i_simcase]['Y'][:,7], 'g-')
            plt.plot(self.response[i_simcase]['t'], self.response[i_simcase]['Y'][:,8], 'r-')
            plt.xlabel('t [sec]')
            plt.ylabel('[m/s^2]')
            plt.grid('on')
            plt.legend(['du', 'dv', 'dw'])
            plt.subplot(2,1,2)
            plt.plot(self.response[i_simcase]['t'], self.response[i_simcase]['Y'][:,9]/np.pi*180.0, 'b-')
            plt.plot(self.response[i_simcase]['t'], self.response[i_simcase]['Y'][:,10]/np.pi*180.0, 'g-')
            plt.plot(self.response[i_simcase]['t'], self.response[i_simcase]['Y'][:,11]/np.pi*180.0, 'r-')
            plt.xlabel('t [sec]')
            plt.ylabel('[deg/s^2]')
            plt.grid('on')
            plt.legend(['dp', 'dq', 'dr'])
            
            
        # show time plots
        plt.show()
   
    def plot_cs_signal(self):
        from efcs import discus2c
        discus2c = discus2c()
        discus2c.cs_signal_init(self.jcl.trimcase[0]['desc'])
        line0 = np.argmin(np.abs(discus2c.data[:,0] - discus2c.tstart))
        line1 = np.argmin(np.abs(discus2c.data[:,0] - discus2c.tstart - self.jcl.simcase[0]['t_final']))
        
        i_mass     = self.model.mass['key'].index(self.jcl.trimcase[0]['mass'])
        n_modes    = self.model.mass['n_modes'][i_mass] 
        
        cs_states = []
        for i_step in range(len(self.response[0]['t'])):
            cs_states.append(self.response[0]['X'][i_step, 12+n_modes*2:12+n_modes*2+3])
        cs_states = np.array(cs_states)/np.pi*180.0
        
        plt.figure()
        plt.subplot(2,1,1)
        plt.plot(discus2c.data[line0:line1,0], discus2c.data[line0:line1,(1,2)]/np.pi*180.0, 'r')
        plt.plot(discus2c.data[line0:line1,0], discus2c.data[line0:line1,3]/np.pi*180.0, 'c')
        plt.plot(discus2c.data[line0:line1,0], discus2c.data[line0:line1,4]/np.pi*180.0, 'b')
        plt.xlabel('t [sec]')
        plt.ylabel('[deg]')
        plt.grid('on')
        plt.title('CS Measurement Signals')
        plt.legend(['xi_l_corr', 'xi_r_corr', 'eta_corr', 'zeta_corr'])
        plt.subplot(2,1,2)
        plt.plot(self.response[0]['t'], cs_states[:,0], 'r')
        plt.plot(self.response[0]['t'], cs_states[:,1], 'c')
        plt.plot(self.response[0]['t'], cs_states[:,2], 'b')
        plt.xlabel('t [sec]')
        plt.ylabel('[deg]')
        plt.grid('on')
        plt.title('CS Commands in Loads Kernel')
        plt.legend(['Xi', 'Eta', 'Zeta'])
                    
    def make_movie(self, path_output, speedup_factor=1.0):
        for i_simcase in range(len(self.jcl.simcase)):
            self.plot_time_animation_3d(i_simcase, path_output, speedup_factor=speedup_factor, make_movie=True)
    
    def make_animation(self, speedup_factor=1.0):
        for i_simcase in range(len(self.jcl.simcase)):
            self.plot_time_animation_3d(i_simcase, speedup_factor=speedup_factor)
                  
    def plot_time_animation_3d(self, i_trimcase, path_output='./', speedup_factor=1.0, make_movie=False):
        # To Do: show simulation time in animation
        from mayavi import mlab
        from tvtk.api import tvtk
        response   = self.response[i_trimcase]
        trimcase   = self.jcl.trimcase[i_trimcase]
        simcase    = self.jcl.simcase[i_trimcase] 
        
        @mlab.animate(delay=int(simcase['dt']*1000.0/speedup_factor), ui=True)
        def anim(self):
            # internal function that actually updates the animation
            while True:
                for i in range(len(response['t'])):
                    self.fig.scene.disable_render = True
                    points_i = np.array([self.x[i], self.y[i], self.z[i]]).T
                    scalars_i = self.color_scalar[i,:]
                    update_strc_display(self, points_i, scalars_i)
                    #update_aerogrid_display(self, scalars_i)
                    update_text_display(self, response['t'][i][0])
                    for src_vector, src_cone, data in zip(self.src_vectors, self.src_cones, self.vector_data):
                        vector_data_i = np.vstack((data['u'][i,:], data['v'][i,:], data['w'][i,:])).T
                        update_vector_display(self, src_vector, src_cone, points_i, vector_data_i)
                    
                    # get current view and set new focal point
                    v = mlab.view()
                    r = mlab.roll()
                    mlab.view(azimuth=v[0], elevation=v[1], roll=r, distance=v[2], focalpoint=points_i.mean(axis=0)) # view from right and above
                    self.fig.scene.disable_render = False
                    yield
                
        def movie(self):
            # internal function that actually updates the animation
            self.fig.scene.disable_render = True
            for i in range(len(response['t'])):
                self.fig.scene.disable_render = True
                points_i = np.array([self.x[i], self.y[i], self.z[i]]).T
                scalars_i = self.color_scalar[i,:]
                update_strc_display(self, points_i, scalars_i)
                #update_aerogrid_display(self, scalars_i)
                update_text_display(self, response['t'][i][0])
                for src_vector, src_cone, data in zip(self.src_vectors, self.src_cones, self.vector_data):
                    vector_data_i = np.vstack((data['u'][i,:], data['v'][i,:], data['w'][i,:])).T
                    update_vector_display(self, src_vector, src_cone, points_i, vector_data_i)
                # get current view and set new focal point
                v = mlab.view()
                r = mlab.roll()
                mlab.view(azimuth=v[0], elevation=v[1], roll=r, distance=v[2], focalpoint=points_i.mean(axis=0)) # view from right and above
                self.fig.scene.render()
                self.fig.scene.save_png('{}anim/subcase_{}_frame_{:06d}.png'.format(path_output, trimcase['subcase'], i))

        self.vector_data = []
        def calc_vector_data(self, grid, set='', name='Pg_aero_global', exponent=0.33):
            Pg = response[name]
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
            
        self.src_vectors = []
        self.src_cones   = []
        def setup_vector_display(self, vector_data, color=(1,0,0), opacity=0.4):
             # vectors
            ug_vector = tvtk.UnstructuredGrid(points=np.vstack((self.x[0,:], self.y[0,:], self.z[0,:])).T)
            ug_vector.point_data.vectors = np.vstack((vector_data['u'][0,:], vector_data['v'][0,:], vector_data['w'][0,:])).T
            src_vector = mlab.pipeline.add_dataset(ug_vector)
            vector = mlab.pipeline.vectors(src_vector, color=color, mode='2ddash', opacity=opacity,  scale_mode='vector', scale_factor=1.0)
            vector.glyph.glyph.clamping=False
            self.src_vectors.append(src_vector)
            # cones for vectors
            ug_cone = tvtk.UnstructuredGrid(points=np.vstack((self.x[0,:]+vector_data['u'][0,:], self.y[0,:]+vector_data['v'][0,:], self.z[0,:]+vector_data['w'][0,:])).T)
            ug_cone.point_data.vectors = np.vstack((vector_data['u'][0,:], vector_data['v'][0,:], vector_data['w'][0,:])).T
            src_cone = mlab.pipeline.add_dataset(ug_cone)
            cone = mlab.pipeline.vectors(src_cone, color=color, mode='cone', opacity=opacity, scale_mode='vector', scale_factor=0.1, resolution=16)
            cone.glyph.glyph.clamping=False
            self.src_cones.append(src_cone)
        
        def update_vector_display(self, src_vector, src_cone, points, vector):
            src_vector.outputs[0].points.from_array(points)
            src_vector.outputs[0].point_data.vectors.from_array(vector)
            src_cone.outputs[0].points.from_array(points+vector)
            src_cone.outputs[0].point_data.vectors.from_array(vector)
            
        def setup_strc_display(self, color=(1,1,1)):
            points = np.vstack((self.x[0,:], self.y[0,:], self.z[0,:])).T
            scalars = self.color_scalar[0,:]
            ug = tvtk.UnstructuredGrid(points=points)
            ug.point_data.scalars = scalars
            if hasattr(self.model, 'strcshell'):
                # plot shell as surface
                shells = []
                for shell in self.model.strcshell['cornerpoints']: 
                    shells.append([np.where(self.model.strcgrid['ID']==id)[0][0] for id in shell])
                shell_type = tvtk.Polygon().cell_type
                ug.set_cells(shell_type, shells)
                self.src_points = mlab.pipeline.add_dataset(ug)
                points  = mlab.pipeline.glyph(self.src_points, colormap='viridis', scale_factor=self.p_scale) #color=color
                surface = mlab.pipeline.surface(self.src_points, colormap='viridis') #color=color)
            else: 
                # plot points as glyphs
                self.src_points = mlab.pipeline.add_dataset(ug)
                points = mlab.pipeline.glyph(self.src_points, colormap='viridis', scale_factor=self.p_scale)
            points.glyph.glyph.scale_mode = 'data_scaling_off'
        
        def update_strc_display(self, points, scalars):
            self.src_points.outputs[0].points.from_array(points)
            self.src_points.outputs[0].point_data.scalars.from_array(scalars)
            
        def setup_aerogrid_display(self, color):
            points = self.model.aerogrid['cornerpoint_grids'][:,(1,2,3)]
            scalars = self.color_scalar[0,:]
            ug = tvtk.UnstructuredGrid(points=points)
            shells = []
            for shell in self.model.aerogrid['cornerpoint_panels']: 
                shells.append([np.where(self.model.aerogrid['cornerpoint_grids'][:,0]==id)[0][0] for id in shell])
            shell_type = tvtk.Polygon().cell_type
            ug.set_cells(shell_type, shells)
            ug.cell_data.scalars = scalars
            self.src_aerogrid = mlab.pipeline.add_dataset(ug)
            
            points = mlab.pipeline.glyph(self.src_aerogrid, color=color, scale_factor=self.p_scale)
            points.glyph.glyph.scale_mode = 'data_scaling_off'
            
            surface = mlab.pipeline.surface(self.src_aerogrid, colormap='viridis')
            surface.actor.mapper.scalar_visibility=True
            surface.actor.property.edge_visibility=False
            surface.actor.property.edge_color=(0.9,0.9,0.9)
            surface.actor.property.line_width=0.5
        
        def update_aerogrid_display(self, scalars):
            self.src_aerogrid.outputs[0].cell_data.scalars.from_array(scalars)
            self.src_aerogrid.update()
            
        def setup_text_display(self):
            self.scr_text = mlab.text(x=0.1, y=0.8, text='Time', line_width=0.5, width=0.05)
            self.scr_text.property.background_color=(1,1,1)
            self.scr_text.property.color=(0,0,0)
            
        def update_text_display(self, t):
            self.scr_text.text = 't = ' + str(t) + 's'

        # --------------
        # configure plot 
        #---------------
        grid = self.model.strcgrid
        set = ''
        
        # get deformations
        self.x = grid['offset'+set][:,0] + response['Ug'][:,grid['set'+set][:,0]]# - response['X'][:,0].repeat(grid['n']).reshape(-1,grid['n'])
        self.y = grid['offset'+set][:,1] + response['Ug'][:,grid['set'+set][:,1]]
        self.z = grid['offset'+set][:,2] + response['Ug'][:,grid['set'+set][:,2]]# - response['X'][:,2].repeat(grid['n']).reshape(-1,grid['n'])
        self.color_scalar = np.linalg.norm(response['Ug_f'][:,grid['set'+set][:,(0,1,2)]], axis=2)
        #self.color_scalar = -np.sum(response['Ug_f'][:,grid['set'+set][:,(0,1,2)]], axis=2)
#         self.x = np.tile(grid['offset'+set][:,0],(len(response['t']),1))
#         self.y = np.tile(grid['offset'+set][:,1],(len(response['t']),1))
#         self.z = np.tile(grid['offset'+set][:,2],(len(response['t']),1))
#         self.color_scalar = response['Pk_aero'][:,grid['set'+set][:,2]]
        
        # get forces
        names = ['Pg_aero_global', 'Pg_iner_global',]# 'Pg_idrag_global', 'Pg_cs_global']
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
            #self.fig = mlab.figure()
        
        # plot initial position
        #setup_aerogrid_display(self, color=(0.9,0.9,0.9))
        #if hasattr(self.model, 'strcshell'): del self.model.strcshell 
        setup_strc_display(self, color=(0.9,0.9,0.9)) # light grey
        setup_text_display(self)
        
        # plot initial forces     
        opacity=0.4  
        for data, color in zip(self.vector_data, colors):
            setup_vector_display(self, data, color, opacity)
        
        # plot coordinate system
        mlab.orientation_axes()
        
        # get earth
        with open('harz.pickle', 'r') as f:  
            (x,y,elev) = cPickle.load(f)
        # plot earth, scale colormap
        surf = mlab.surf(x,y,elev, colormap='terrain', warp_scale=-1.0, vmin = -500.0, vmax=1500.0) #gist_earth terrain summer
        
        #mlab.view(azimuth=180.0, elevation=90.0, roll=-90.0, distance=70.0, focalpoint=np.array([self.x.mean(),self.y.mean(),self.z.mean()])) # back view
        distance = 3.5*((self.x[0,:].max()-self.x[0,:].min())**2 + (self.y[0,:].max()-self.y[0,:].min())**2 + (self.z[0,:].max()-self.z[0,:].min())**2)**0.5
        #mlab.view(azimuth=135.0, elevation=100.0, roll=-100.0, distance=distance, focalpoint=np.array([self.x[0,:].mean(),self.y[0,:].mean(),self.z[0,:].mean()])) # view from right and above
        mlab.view(azimuth=-120.0, elevation=100.0, roll=-75.0,  distance=distance, focalpoint=np.array([self.x[0,:].mean(),self.y[0,:].mean(),self.z[0,:].mean()])) # view from left and above
        #mlab.view(azimuth=-100.0, elevation=65.0, roll=25.0, distance=distance, focalpoint=np.array([self.x[0,:].mean(),self.y[0,:].mean(),self.z[0,:].mean()])) # view from right and above

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
        else:
            anim(self) # launch animation
            mlab.show()
            
