# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 11:26:27 2015

@author: voss_ar
"""
import numpy as np
from  matplotlib import pyplot as plt
plt.rcParams.update({'font.size': 16,
                     'svg.fonttype':'none'})
from matplotlib.backends.backend_pdf import PdfPages
from scipy.spatial import ConvexHull
import logging

class StandardPlots():
    def __init__(self, jcl, model):
        self.jcl = jcl
        self.model = model
        self.responses = None
        self.monstations = None
                
        # Allegra
        if self.jcl.general['aircraft'] == 'ALLEGRA':
            self.potatos_fz_mx = ['ZWCUT01', 'ZWCUT04', 'ZWCUT08', 'ZWCUT12', 'ZWCUT16', 'ZWCUT20', 'ZWCUT24', 'ZHCUT01' ]
            self.potatos_mx_my = ['ZWCUT01', 'ZWCUT04', 'ZWCUT08', 'ZWCUT12', 'ZWCUT16', 'ZWCUT20', 'ZWCUT24', 'ZHCUT01' ]
            self.potatos_fz_my = ['ZWCUT01', 'ZWCUT04', 'ZWCUT08', 'ZWCUT12', 'ZWCUT16', 'ZWCUT20', 'ZWCUT24', 'ZHCUT01', 'ZFCUT27', 'ZFCUT28']
            self.cuttingforces_wing = ['ZWCUT01', 'ZWCUT04', 'ZWCUT08', 'ZWCUT12', 'ZWCUT16', 'ZWCUT20', 'ZWCUT24', ]
            self.f_scale = 0.002 # vectors
            self.p_scale = 0.4 # points
        # DLR-F19
        elif self.jcl.general['aircraft'] == 'DLR F-19-S':
            self.potatos_fz_mx = ['MON1', 'MON2', 'MON3', 'MON33', 'MON8', 'MON6', 'MON7', 'MON9']
            self.potatos_mx_my = ['MON1', 'MON2', 'MON3', 'MON33', 'MON8', 'MON6', 'MON7', 'MON9']
            self.potatos_fz_my = ['MON1', 'MON2', 'MON3', 'MON33', 'MON4', 'MON5']
            self.cuttingforces_wing = ['MON1', 'MON2', 'MON3', 'MON33', 'MON8']
            self.f_scale = 0.002 # vectors
            self.p_scale = 0.04 # points
        # MULDICON
        elif self.jcl.general['aircraft'] == 'MULDICON':
            self.potatos_fz_mx = ['MON1', 'MON2', 'MON3', 'MON33', 'MON8', 'MON9']
            self.potatos_mx_my = ['MON1', 'MON2', 'MON3', 'MON33', 'MON8', 'MON9']
            self.potatos_fz_my = ['MON4', 'MON5', 'MON81', 'MON82', 'MON83']
            self.cuttingforces_wing = ['MON10', 'MON1', 'MON2', 'MON3', 'MON33', 'MON8']
            self.f_scale = 0.002 # vectors
            self.p_scale = 0.03 # points
        # Discus2c
        elif self.jcl.general['aircraft'] == 'Discus2c':
            self.potatos_fz_mx = ['MON646', 'MON644', 'MON641', 'MON546', 'MON544', 'MON541', 'MON348', 'MON346', 'MON102']
            self.potatos_mx_my = ['MON646', 'MON644', 'MON641', 'MON546', 'MON544', 'MON541', 'MON348', 'MON346', 'MON102']
            self.potatos_fz_my = ['MON102']
            self.cuttingforces_wing = ['MON646', 'MON644', 'MON641', 'MON541', 'MON544', 'MON546']
            self.f_scale = 0.1 # vectors
            self.p_scale = 0.05 # points
        # FLEXOP
        elif self.jcl.general['aircraft'] in ['FLEXOP', 'fs35']:
            self.potatos_fz_mx = ['MON1']
            self.potatos_mx_my = ['MON1']
            self.potatos_fz_my = ['MON1']
            self.cuttingforces_wing = ['MON1']
            self.f_scale = 0.1 # vectors
            self.p_scale = 0.05 # points
        # HALO
        elif self.jcl.general['aircraft'] == 'HALO':
            self.potatos_fz_mx = ['PMS_L', 'PMS_R']
            self.potatos_mx_my = ['PMS_L', 'PMS_R']
            self.potatos_fz_my = ['PMS_L', 'PMS_R']
            self.cuttingforces_wing = ['PMS_L', 'PMS_R']
            self.f_scale = 0.02 # vectors
            self.p_scale = 0.4 # points
        elif self.jcl.general['aircraft'] in ['ACFA']:
            self.potatos_fz_mx = ['MON1']
            self.potatos_mx_my = ['MON1']
            self.potatos_fz_my = ['MON1']
            self.cuttingforces_wing = ['MON1']
            self.f_scale = 0.002 # vectors
            self.p_scale = 0.4 # points
        elif self.jcl.general['aircraft'] in ['XRF1']:
            self.potatos_fz_mx = ['MON4', 'MON10', 'MON16', 'MON22', 'MON28', 'MON34']
            self.potatos_mx_my = ['MON4', 'MON10', 'MON16', 'MON22', 'MON28', 'MON34']
            self.potatos_fz_my = ['MON4', 'MON10', 'MON16', 'MON22', 'MON28', 'MON34']
            self.cuttingforces_wing = ['MON4', 'MON10', 'MON16', 'MON22', 'MON28', 'MON34']
            self.f_scale = 0.002 # vectors
            self.p_scale = 0.4 # points
        elif self.jcl.general['aircraft'] in ['HAP-C0']:
            self.potatos_fz_mx = ['64001']
            self.potatos_mx_my = ['64001']
            self.potatos_fz_my = ['64001']
            self.cuttingforces_wing = ['54033', '54032', '54031', '54030', '54029', '54028', '54027', '54026', '54025', '54024', 
                                       '54023', '54022', '54021', '54020', '54019', '54018', '54017', '54016', '54015', '54014', 
                                       '54013', '54012', '54011', '54010', '54009', '54008', '54007', '54006', '54005', '54004', 
                                       '54003', '54002', '54001',
                                       '64001', '64002', '64003', '64004', '64005', '64006', '64007', '64008', '64009', '64010', 
                                       '64011', '64012', '64013', '64014', '64015', '64016', '64017', '64018', '64019', '64020', 
                                       '64021', '64022', '64023', '64024', '64025', '64026', '64027', '64028', '64029', '64030', 
                                       '64032', '64023', '64033',]
            self.f_scale = 0.1 # vectors
            self.p_scale = 0.05 # points
        else:
            logging.error('Unknown aircraft: ' + str(self.jcl.general['aircraft']))
            return
    
    def add_responses(self, responses):
        self.responses = responses
    
    def add_monstations(self, monstations):
        self.monstations = monstations
        
    def get_loads_strings(self, station):
        if np.size(self.monstations[station]['t'][0]) == 1:
            # Scenario 1: There are only static loads.
            loads_string = 'loads'
            subcase_string = 'subcase'
        elif (np.size(self.monstations[station]['t'][0]) > 1) and ('loads_dyn2stat' in self.monstations[station].keys()) and (self.monstations[station]['loads_dyn2stat'] != []):
            # Scenario 2: Dynamic loads have been converted to quasi-static time slices / snapshots.
            loads_string = 'loads_dyn2stat'
            subcase_string = 'subcases_dyn2stat'
        else:
            # Scenario 3: There are only dynamic loads. 
            logging.error('Dynamic loads need to be converted to static loads (dyn2stat).')
        return loads_string, subcase_string
        
    def plot_forces_deformation_interactive(self):
        from mayavi import mlab
                
        for i_trimcase in range(len(self.jcl.trimcase)):
            response   = self.responses[i_trimcase]
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
            
            mlab.figure()   
            mlab.points3d(x, y, z, scale_factor=self.p_scale)
            #mlab.quiver3d(x, y, z, response['Pg'][self.model.strcgrid['set'][:,0]], response['Pg'][self.model.strcgrid['set'][:,1]], response['Pg'][self.model.strcgrid['set'][:,2]], color=(1,1,0), scale_factor=self.f_scale)
            fx, fy, fz = response['Pg_cfd'][self.model.strcgrid['set'][:,0]], response['Pg_cfd'][self.model.strcgrid['set'][:,1]], response['Pg_cfd'][self.model.strcgrid['set'][:,2]]
            mlab.quiver3d(x, y, z, fx*self.f_scale, fy*self.f_scale, fz*self.f_scale , color=(1,1,0),  mode='2ddash', opacity=0.4,  scale_mode='vector', scale_factor=1.0)
            mlab.quiver3d(x+fx*self.f_scale, y+fy*self.f_scale, z+fz*self.f_scale,fx*self.f_scale, fy*self.f_scale, fz*self.f_scale , color=(1,1,0),  mode='cone', scale_mode='vector', scale_factor=0.2, resolution=16)
            mlab.points3d(self.model.splinegrid['offset'][:,0], self.model.splinegrid['offset'][:,1], self.model.splinegrid['offset'][:,2], color=(1,1,0), scale_factor=self.p_scale*1.5)
            mlab.title('Pg_cfd', size=0.2, height=0.95)
            
            mlab.show()

    def plot_monstations(self, filename_pdf):

        # launch plotting
        self.pp = PdfPages(filename_pdf)
        self.potato_plots()
        self.cuttingforces_along_wing_plots()
        self.pp.close()
        logging.info('plots saved as ' + filename_pdf)
    
    def potato_plot(self, station, desc, color, dof_xaxis, dof_yaxis, show_hull, show_labels):
        loads_string, subcase_string = self.get_loads_strings(station)
        loads   = np.array(self.monstations[station][loads_string])
        points = np.vstack((loads[:,dof_xaxis], loads[:,dof_yaxis])).T
        crit_trimcases = []
        self.subplot.scatter(points[:,0], points[:,1], color=color, label=desc) # plot points
        
        if show_hull and points.shape[0] >= 3:
            try:
                hull = ConvexHull(points) # calculated convex hull from scattered points
                for simplex in hull.simplices:                   # plot convex hull
                    self.subplot.plot(points[simplex,0], points[simplex,1], color=color, linewidth=2.0, linestyle='--')
                for i_case in range(hull.nsimplex):
                    crit_trimcases.append(self.monstations[station][subcase_string][hull.vertices[i_case]])
                    if show_labels:                    
                        self.subplot.text(points[hull.vertices[i_case],0], points[hull.vertices[i_case],1], str(self.monstations[station][subcase_string][hull.vertices[i_case]]), fontsize=8)
            except:
                pass
        else:
            crit_trimcases = self.monstations[station][subcase_string][:]
        self.crit_trimcases += crit_trimcases
    
    def potato_plot_nicely(self, station, desc, dof_xaxis, dof_yaxis, var_xaxis, var_yaxis):
            self.subplot.cla()
            self.potato_plot(station, 
                             desc=station, 
                             color='cornflowerblue', 
                             dof_xaxis=dof_xaxis, 
                             dof_yaxis=dof_yaxis, 
                             show_hull=True, 
                             show_labels=True)
            
            self.subplot.legend(loc='best')
            self.subplot.ticklabel_format(style='sci', axis='x', scilimits=(-2,2))
            self.subplot.ticklabel_format(style='sci', axis='y', scilimits=(-2,2))
            self.subplot.grid('on')
            yax = self.subplot.get_yaxis()
            yax.set_label_coords(x=-0.18, y=0.5)
            self.subplot.set_xlabel(var_xaxis)
            self.subplot.set_ylabel(var_yaxis)
            self.pp.savefig()
            
    def potato_plots(self):
        logging.info('start potato-plotting...')
        fig = plt.figure()
        self.subplot = fig.add_axes([0.2, 0.15, 0.7, 0.75]) # List is [left, bottom, width, height]
        
        potato = np.unique(self.potatos_fz_mx + self.potatos_mx_my + self.potatos_fz_my)
        self.crit_trimcases = []
        for station in potato:            
            if station in self.potatos_fz_mx:
                var_xaxis='Fz [N]'
                var_yaxis='Mx [Nm]'
                dof_xaxis=2
                dof_yaxis=3
                self.potato_plot_nicely(station, station, dof_xaxis, dof_yaxis, var_xaxis, var_yaxis)
            if station in self.potatos_mx_my:
                var_xaxis='Mx [N]'
                var_yaxis='My [Nm]'
                dof_xaxis=3
                dof_yaxis=4
                self.potato_plot_nicely(station, station, dof_xaxis, dof_yaxis, var_xaxis, var_yaxis)
            if station in self.potatos_fz_my:
                var_xaxis='Fz [N]'
                var_yaxis='My [Nm]'
                dof_xaxis=2
                dof_yaxis=4
                self.potato_plot_nicely(station, station, dof_xaxis, dof_yaxis, var_xaxis, var_yaxis)
        plt.close(fig)
          
    def cuttingforces_along_wing_plots(self):
        logging.info('start plotting cutting forces along wing...')
        fig = plt.figure()
        self.subplot = fig.add_axes([0.2, 0.15, 0.7, 0.75]) # List is [left, bottom, width, height]
        cuttingforces = ['Fx [N]', 'Fy [N]', 'Fz [N]', 'Mx [Nm]', 'My [Nm]', 'Mz [Nm]']
        loads = []
        offsets = []
        for station in self.cuttingforces_wing:
            loads_string, subcase_string = self.get_loads_strings(station)
            loads.append(self.monstations[station][loads_string])
            offsets.append(self.monstations[station]['offset'])
        loads = np.array(loads)
        offsets = np.array(offsets)
        
        for i_cuttingforce in range(len(cuttingforces)):
            i_max = np.argmax(loads[:,:,i_cuttingforce], 1)
            i_min = np.argmin(loads[:,:,i_cuttingforce], 1)
            self.subplot.cla()
            self.subplot.plot(offsets[:,1],loads[:,:,i_cuttingforce], color='cornflowerblue', linestyle='-', marker='.')
            for i_station in range(len(self.cuttingforces_wing)):
                # verticalalignment or va 	[ 'center' | 'top' | 'bottom' | 'baseline' ]
                # max
                plt.scatter(offsets[i_station,1],loads[i_station,i_max[i_station],i_cuttingforce], color='r')
                plt.text(   offsets[i_station,1],loads[i_station,i_max[i_station],i_cuttingforce], str(self.monstations[self.monstations.keys()[0]][subcase_string][i_max[i_station]]), fontsize=8, verticalalignment='bottom' )
                # min
                plt.scatter(offsets[i_station,1],loads[i_station,i_min[i_station],i_cuttingforce], color='r')
                plt.text(   offsets[i_station,1],loads[i_station,i_min[i_station],i_cuttingforce], str(self.monstations[self.monstations.keys()[0]][subcase_string][i_min[i_station]]), fontsize=8, verticalalignment='top' )

            self.subplot.set_title('Wing')        
            self.subplot.ticklabel_format(style='sci', axis='y', scilimits=(-2,2))
            self.subplot.grid('on')
            yax = self.subplot.get_yaxis()
            yax.set_label_coords(x=-0.18, y=0.5)
            self.subplot.set_xlabel('y [m]')
            self.subplot.set_ylabel(cuttingforces[i_cuttingforce])
            self.pp.savefig()
        plt.close(fig)
              
    def plot_monstations_time(self, filename_pdf):
        logging.info('start plotting cutting forces over time ...')
        pp = PdfPages(filename_pdf)
        for key in self.monstations.keys():
            monstation = self.monstations[key]
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
