# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 11:26:27 2015

@author: voss_ar
"""
import numpy as np
from  matplotlib import pyplot as plt
plt.rcParams.update({'font.size': 16,
                     'svg.fonttype':'none',
                     'savefig.dpi': 300,})
from matplotlib.backends.backend_pdf import PdfPages
from scipy.spatial import ConvexHull
import logging, itertools
from loadskernel.units import tas2eas
from loadskernel.units import eas2tas


class StandardPlots():
    def __init__(self, jcl, model):
        self.jcl = jcl
        self.model = model
        self.responses = None
        self.monstations = None
        self.potatos_fz_mx = [] # Wing, HTP
        self.potatos_mx_my = [] # Wing, HTP, VTP
        self.potatos_fz_my = []
        self.potatos_fy_mx = [] # VTP
        self.potatos_mx_mz = [] # VTP
        self.potatos_my_mz = [] # FUS
        self.cuttingforces_wing = []
        
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
        elif self.jcl.general['aircraft'] in ['FLEXOP', 'fs35', 'Openclass']:
            self.potatos_fz_mx = ['MON1']
            self.potatos_mx_my = ['MON1']
            self.potatos_fz_my = ['MON1']
            self.cuttingforces_wing = ['MON1']
            self.f_scale = 0.1 # vectors
            self.p_scale = 0.2 # points
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
        elif self.jcl.general['aircraft'] in ['HAP-C1', 'HAP-C2', 'HAP']:
            self.potatos_fz_mx =      ['64001', '64002', '64003', '64004', '64005', '64006', '64007', '64008', '64009', '64010', 
                                       '64011', '64012', '64013', '64014', '64015', '64016', '64017', '64018', '64019', '64020', 
                                       '64021', '64022', '64023', '64024', '64025', '64026', '64027', '64028', '64029', '64030', 
                                       '64031', '64032', '64033',
                                       '33401', '33402', '33403', '33404', '33405', '33406', '33407', '33408', '33409', '33410', 
                                       '33411'] # Wing, HTP
            self.potatos_mx_my =      ['64001', '64002', '64003', '64004', '64005', '64006', '64007', '64008', '64009', '64010', 
                                       '64011', '64012', '64013', '64014', '64015', '64016', '64017', '64018', '64019', '64020', 
                                       '64021', '64022', '64023', '64024', '64025', '64026', '64027', '64028', '64029', '64030', 
                                       '64031', '64032', '64033',
                                       '33401', '33402', '33403', '33404', '33405', '33406', '33407', '33408', '33409', '33410', 
                                       '33411',
                                       '100001', '100002', '100003', '100004', '100005', '100006',
                                       '64100001', '64100002', '64100003', '54100001', '54100002', '54100003'] # Wing, HTP, Prop
            self.potatos_fz_my =      []
            self.potatos_fy_mx =      ['33201', '33202', '33203', '33204', '33205', '33206', '33207'] # VTP
            self.potatos_mx_mz =      ['33201', '33202', '33203', '33204', '33205', '33206', '33207'] # VTP 
            self.potatos_my_mz =      ['100001', '100002', '100003', '100004', '100005', '100006',
                                       '64100001', '64100002', '64100003', '54100001', '54100002', '54100003'] # FUS, Prop
             
            self.cuttingforces_wing = ['54033', '54032', '54031', '54030', '54029', '54028', '54027', '54026', 
                                       '54025', '54024', '54023', '54022', '54021', '54020', '54019', '54018', '54017', '54016', 
                                       '54015', '54014', '54013', '54012', '54011', '54010', '54009', '54008', '54007', '54006', 
                                       '54005', '54004', '54003', '54002', '54001',
                                       '64001', '64002', '64003', '64004', '64005', '64006', '64007', '64008', '64009', '64010', 
                                       '64011', '64012', '64013', '64014', '64015', '64016', '64017', '64018', '64019', '64020', 
                                       '64021', '64022', '64023', '64024', '64025', '64026', '64027', '64028', '64029', '64030', 
                                       '64031', '64032', '64033']
            self.f_scale = 0.1 # vectors
            self.p_scale = 0.3 # points
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
    
    def potato_plot(self, station, desc, color, dof_xaxis, dof_yaxis, show_hull=True, show_labels=False, show_minmax=False):
        loads_string, subcase_string = self.get_loads_strings(station)
        loads   = np.array(self.monstations[station][loads_string])
        points = np.vstack((loads[:,dof_xaxis], loads[:,dof_yaxis])).T
        self.subplot.scatter(points[:,0], points[:,1], color=color, label=desc, zorder=-2) # plot points
        
        if show_hull and points.shape[0] >= 3:
            try:
                hull = ConvexHull(points) # calculated convex hull from scattered points
                for simplex in hull.simplices: # plot convex hull
                    self.subplot.plot(points[simplex,0], points[simplex,1], color=color, linewidth=2.0, linestyle='--')
                crit_trimcases = [self.monstations[station][subcase_string][i] for i in hull.vertices]
                if show_labels: 
                    for i_case in range(crit_trimcases.__len__()): 
                        self.subplot.text(points[hull.vertices[i_case],0], points[hull.vertices[i_case],1], str(self.monstations[station][subcase_string][hull.vertices[i_case]]), fontsize=8)
            except:
                crit_trimcases = []
        
        elif show_minmax:
            pos_max_loads = np.argmax(points, 0)
            pos_min_loads = np.argmin(points, 0)
            pos_minmax_loads = np.concatenate((pos_min_loads, pos_max_loads))
            self.subplot.scatter(points[pos_minmax_loads,0], points[pos_minmax_loads,1], color=(1,0,0), zorder=-2) # plot points
            crit_trimcases = [self.monstations[station][subcase_string][i] for i in pos_minmax_loads]

        else:
            crit_trimcases = self.monstations[station][subcase_string][:]
  
        if show_labels: 
            for crit_trimcase in crit_trimcases:
                pos = self.monstations[station][subcase_string].index(crit_trimcase)
                self.subplot.text(points[pos,0], points[pos,1], str(self.monstations[station][subcase_string][pos]), fontsize=8)
                        
        self.crit_trimcases += crit_trimcases
    
    def potato_plot_nicely(self, station, desc, dof_xaxis, dof_yaxis, var_xaxis, var_yaxis):
            self.subplot.cla()
            self.potato_plot(station, 
                             desc=station, 
                             color='cornflowerblue', 
                             dof_xaxis=dof_xaxis, 
                             dof_yaxis=dof_yaxis, 
                             show_hull=True,
                             show_labels=True,
                             show_minmax=False)
            
            self.subplot.legend(loc='best')
            self.subplot.ticklabel_format(style='sci', axis='x', scilimits=(-2,2))
            self.subplot.ticklabel_format(style='sci', axis='y', scilimits=(-2,2))
            self.subplot.grid(b=True, which='major', axis='both')
            self.subplot.minorticks_on()
            yax = self.subplot.get_yaxis()
            yax.set_label_coords(x=-0.18, y=0.5)
            self.subplot.set_xlabel(var_xaxis)
            self.subplot.set_ylabel(var_yaxis)
            self.subplot.set_rasterization_zorder(-1)
            self.pp.savefig()
            
    def potato_plots(self):
        logging.info('start potato-plotting...')
        fig = plt.figure()
        self.subplot = fig.add_axes([0.2, 0.15, 0.7, 0.75]) # List is [left, bottom, width, height]
        
        potato = np.sort(np.unique(self.potatos_fz_mx + self.potatos_mx_my + self.potatos_fz_my + self.potatos_fy_mx + self.potatos_mx_mz + self.potatos_my_mz))
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
            if station in self.potatos_fy_mx:
                var_xaxis='Fy [N]'
                var_yaxis='Mx [Nm]'
                dof_xaxis=1
                dof_yaxis=3
                self.potato_plot_nicely(station, station, dof_xaxis, dof_yaxis, var_xaxis, var_yaxis)
            if station in self.potatos_mx_mz:
                var_xaxis='Mx [Nm]'
                var_yaxis='Mz [Nm]'
                dof_xaxis=3
                dof_yaxis=5
                self.potato_plot_nicely(station, station, dof_xaxis, dof_yaxis, var_xaxis, var_yaxis)
            if station in self.potatos_my_mz:
                var_xaxis='My [Nm]'
                var_yaxis='Mz [Nm]'
                dof_xaxis=4
                dof_yaxis=5
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
            self.subplot.plot(offsets[:,1],loads[:,:,i_cuttingforce], color='cornflowerblue', linestyle='-', marker='.', zorder=-2)
            for i_station in range(len(self.cuttingforces_wing)):
                # verticalalignment or va 	[ 'center' | 'top' | 'bottom' | 'baseline' ]
                # max
                plt.scatter(offsets[i_station,1],loads[i_station,i_max[i_station],i_cuttingforce], color='r')
                plt.text(   offsets[i_station,1],loads[i_station,i_max[i_station],i_cuttingforce], str(self.monstations[list(self.monstations)[0]][subcase_string][i_max[i_station]]), fontsize=4, verticalalignment='bottom' )
                # min
                plt.scatter(offsets[i_station,1],loads[i_station,i_min[i_station],i_cuttingforce], color='r')
                plt.text(   offsets[i_station,1],loads[i_station,i_min[i_station],i_cuttingforce], str(self.monstations[list(self.monstations)[0]][subcase_string][i_min[i_station]]), fontsize=4, verticalalignment='top' )

            self.subplot.set_title('Wing')        
            self.subplot.ticklabel_format(style='sci', axis='y', scilimits=(-2,2))
            self.subplot.grid(b=True, which='major', axis='both')
            self.subplot.minorticks_on()
            yax = self.subplot.get_yaxis()
            yax.set_label_coords(x=-0.18, y=0.5)
            self.subplot.set_xlabel('y [m]')
            self.subplot.set_ylabel(cuttingforces[i_cuttingforce])
            self.subplot.set_rasterization_zorder(-1)
            self.pp.savefig()
        plt.close(fig)
              
    def plot_monstations_time(self, filename_pdf):
        logging.info('start plotting cutting forces over time ...')
        pp = PdfPages(filename_pdf)
        potato = np.sort(np.unique(self.potatos_fz_mx + self.potatos_mx_my + self.potatos_fz_my + self.potatos_fy_mx + self.potatos_mx_mz + self.potatos_my_mz))
        for station in potato:
            monstation = self.monstations[station]
            fig, ax = plt.subplots(6, sharex=True, figsize=(8,10) )
            for i_simcase in range(len(self.jcl.simcase)):
                loads = np.array(monstation['loads'][i_simcase])
                t = monstation['t'][i_simcase]
                ax[0].plot(t, loads[:,0], 'k', zorder=-2)
                ax[1].plot(t, loads[:,1], 'k', zorder=-2)
                ax[2].plot(t, loads[:,2], 'k', zorder=-2)
                ax[3].plot(t, loads[:,3], 'k', zorder=-2)
                ax[4].plot(t, loads[:,4], 'k', zorder=-2)
                ax[5].plot(t, loads[:,5], 'k', zorder=-2)
            # make plots nice
            ax[0].set_position([0.2, 0.83, 0.7, 0.12])
            ax[0].title.set_text(station)
            ax[0].set_ylabel('Fx [N]')
            ax[0].get_yaxis().set_label_coords(x=-0.18, y=0.5)
            ax[0].grid(b=True, which='major', axis='both')
            ax[0].minorticks_on()
            ax[0].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            
            ax[1].set_position([0.2, 0.68, 0.7, 0.12])
            ax[1].set_ylabel('Fy [N]')
            ax[1].get_yaxis().set_label_coords(x=-0.18, y=0.5)
            ax[1].grid(b=True, which='major', axis='both')
            ax[1].minorticks_on()
            ax[1].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            
            ax[2].set_position([0.2, 0.53, 0.7, 0.12])
            ax[2].set_ylabel('Fz [N]')
            ax[2].get_yaxis().set_label_coords(x=-0.18, y=0.5)
            ax[2].grid(b=True, which='major', axis='both')
            ax[2].minorticks_on()
            ax[2].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            
            ax[3].set_position([0.2, 0.38, 0.7, 0.12])
            ax[3].set_ylabel('Mx [Nm]')
            ax[3].get_yaxis().set_label_coords(x=-0.18, y=0.5)
            ax[3].grid(b=True, which='major', axis='both')
            ax[3].minorticks_on()
            ax[3].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            
            ax[4].set_position([0.2, 0.23, 0.7, 0.12])
            ax[4].set_ylabel('My [Nm]')
            ax[4].get_yaxis().set_label_coords(x=-0.18, y=0.5)
            ax[4].grid(b=True, which='major', axis='both')
            ax[4].minorticks_on()
            ax[4].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            
            ax[5].set_position([0.2, 0.08, 0.7, 0.12])
            ax[5].set_ylabel('Mz [Nm]')
            ax[5].get_yaxis().set_label_coords(x=-0.18, y=0.5)
            ax[5].grid(b=True, which='major', axis='both')
            ax[5].minorticks_on()
            ax[5].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            
            ax[5].set_xlabel('t [sec]')
            
            ax[0].set_rasterization_zorder(-1)
            ax[1].set_rasterization_zorder(-1)
            ax[2].set_rasterization_zorder(-1)
            ax[3].set_rasterization_zorder(-1)
            ax[4].set_rasterization_zorder(-1)
            ax[5].set_rasterization_zorder(-1)
            
            pp.savefig()
            plt.close()
        pp.close()
        logging.info('plots saved as ' + filename_pdf)

    def plot_fluttercurves(self):
        logging.info('start plotting flutter curves...')
        for response in self.responses:
            trimcase   = self.jcl.trimcase[response['i']]
            i_mass     = self.model.mass['key'].index(trimcase['mass'])
            i_atmo     = self.model.atmo['key'].index(trimcase['altitude'])
            #Plot boundaries
            freqs = np.real(self.model.mass['Khh'][i_mass].diagonal())**0.5 /2/np.pi
            fmin = 0
            fmax = 5 * np.ceil(freqs.max() / 5)
            Vtrim = tas2eas(self.model.atmo['a'][i_atmo] * trimcase['Ma'], self.model.atmo['h'][i_atmo])
            Vmin = 0
            Vmax = 5 * np.ceil(Vtrim*2.0 / 5)
            gmin = -0.1
            gmax = 0.1

            colors = itertools.cycle(( plt.cm.tab20c(np.linspace(0, 1, 20)) ))
            markers = itertools.cycle(('+', 'o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'x', 'D',))
            
            fig, ax = plt.subplots(2, sharex=True )
            for j in range(response['freqs'].shape[1]): 
                marker = next(markers)
                color = next(colors)
                ax[0].plot(tas2eas(response['Vtas'][:, j], self.model.atmo['h'][i_atmo]), response['freqs'][:, j],   marker=marker, markersize=2.0, linewidth=1.0, color=color)
                ax[1].plot(tas2eas(response['Vtas'][:, j], self.model.atmo['h'][i_atmo]), response['damping'][:, j], marker=marker, markersize=2.0, linewidth=1.0, color=color)
            
            # make plots nice
            ax[0].set_position([0.15, 0.55, 0.75, 0.35])
            ax[0].title.set_text(trimcase['desc'])
            ax[0].title.set_fontsize(16)
            ax[0].set_ylabel('f [Hz]')
            ax[0].get_yaxis().set_label_coords(x=-0.13, y=0.5)
            ax[0].grid(b=True, which='major', axis='both')
            ax[0].minorticks_on()
            ax[0].axis([Vmin, Vmax, fmin, fmax])
            ax[1].set_position([0.15, 0.15, 0.75, 0.35])
            ax[1].set_ylabel('g [-]')
            ax[1].get_yaxis().set_label_coords(x=-0.13, y=0.5)
            ax[1].grid(b=True, which='major', axis='both')
            ax[1].minorticks_on()
            ax[1].axis([Vmin, Vmax, gmin, gmax])
            ax[1].set_xlabel('$V_{eas} [m/s]$')
            
            # additional axis for Vtas
            ax_vtas = ax[1].twiny()
            ax_vtas.set_position([0.15, 0.15, 0.75, 0.35])
            ax_vtas.xaxis.set_ticks_position('bottom') # set the position of the second x-axis to bottom
            ax_vtas.xaxis.set_label_position('bottom') # set the position of the second x-axis to bottom
            ax_vtas.spines['bottom'].set_position(('outward', 60))
            x1, x2 = ax[1].get_xlim()
            ax_vtas.set_xlim(( eas2tas(x1, self.model.atmo['h'][i_atmo]), eas2tas(x2, self.model.atmo['h'][i_atmo]) ))
            ax_vtas.minorticks_on()
            ax_vtas.set_xlabel('$V_{tas} [m/s]$')

        plt.show()
