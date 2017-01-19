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
import csv, time, os, copy, logging
import build_aero, write_functions


class plotting:
    def __init__(self, jcl, model, response=None):
        self.jcl = jcl
        self.model = model
        self.response = response
        
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
            self.cuttingforces_wing = ['MON1', 'MON2', 'MON3', 'MON33', 'MON8']
            self.f_scale = 0.002 # vectors
            self.p_scale = 0.1 # points
        # Discus2c
        elif self.jcl.general['aircraft'] == 'Discus2c':
            self.potatos_Fz_Mx = ['MON646', 'MON644', 'MON641', 'MON546', 'MON544', 'MON541', 'MON348', 'MON346', 'MON102']
            self.potatos_Mx_My = ['MON646', 'MON644', 'MON641', 'MON546', 'MON544', 'MON541', 'MON348', 'MON346', 'MON102']
            self.potatos_Fz_My = ['MON102']
            self.cuttingforces_wing = ['MON646', 'MON644', 'MON641', 'MON541', 'MON544', 'MON546']
            self.f_scale = 0.1 # vectors
            self.p_scale = 0.2 # points
        # FLEXOP
        elif self.jcl.general['aircraft'] == 'FLEXOP':
            self.potatos_Fz_Mx = ['MON1']
            self.potatos_Mx_My = ['MON1']
            self.potatos_Fz_My = ['MON1']
            self.cuttingforces_wing = ['MON1']
            self.f_scale = 0.1 # vectors
            self.p_scale = 0.05 # points
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
            ax = build_aero.plot_aerogrid(self.model.aerogrid, cp, 'jet', -0.5, 0.5)
            ax.set_title('Cp for {:s}'.format(trimcase['desc']))
#             ax.set_xlim(0, 16)
#             ax.set_ylim(-8, 8)
            F = response['Pk_idrag'][self.model.aerogrid['set_k'][:,0]]
            cp = F / (rho/2.0*Vtas**2) / self.model.aerogrid['A']
            ax = build_aero.plot_aerogrid(self.model.aerogrid, cp, 'jet', -0.01, 0.03)
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
            mlab.quiver3d(x, y, z, response['Pg'][self.model.strcgrid['set'][:,0]], response['Pg'][self.model.strcgrid['set'][:,1]], response['Pg'][self.model.strcgrid['set'][:,2]], color=(1,1,0), scale_factor=self.f_scale)
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
           
    def write_critical_trimcases(self, filename_csv, dyn2stat=False):
        # eigentlich gehoert diese Funtion eher zum post-processing als zum
        # plotten, kann aber erst nach dem plotten ausgefuehrt werden...
        if dyn2stat:
            crit_trimcases = list(set([int(crit_trimcase.split('_')[0]) for crit_trimcase in self.crit_trimcases])) # extract original subcase number
        else: 
            crit_trimcases = self.crit_trimcases
        crit_trimcases_info = []
        for i_case in range(len(self.jcl.trimcase)):
            if self.jcl.trimcase[i_case]['subcase'] in crit_trimcases:
                trimcase = copy.deepcopy(self.jcl.trimcase[i_case])
                if dyn2stat:
                    trimcase.update(self.jcl.simcase[i_case]) # merge infos from simcase with trimcase
                crit_trimcases_info.append(trimcase)
                
        logging.info('writing critical trimcases cases to: ' + filename_csv)
        with open(filename_csv, 'wb') as fid:
            w = csv.DictWriter(fid, crit_trimcases_info[0].keys())
            w.writeheader()
            w.writerows(crit_trimcases_info)
        return

    def save_dyn2stat(self, dyn2stat, filename):
        # eigentlich gehoert diese Funtion eher zum post-processing als zum
        # plotten, kann aber erst nach dem plotten ausgefuehrt werden...
        logging.info('saving dyn2stat nodal loads as Nastarn cards...')
        with open(filename+'_Pg_dyn2stat', 'w') as fid: 
            for i_case in range(len(dyn2stat['subcases'])):
                if dyn2stat['subcases'][i_case] in self.crit_trimcases:
                    write_functions.write_force_and_moment_cards(fid, self.model.strcgrid, dyn2stat['Pg'][i_case], dyn2stat['subcases_ID'][i_case])
        with open(filename+'_subcases_dyn2stat', 'w') as fid:         
            for i_case in range(len(dyn2stat['subcases'])):
                if dyn2stat['subcases'][i_case] in self.crit_trimcases:
                    write_functions.write_subcases(fid, dyn2stat['subcases_ID'][i_case], dyn2stat['subcases'][i_case])
    
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
        
    def plot_time_data(self, animation_dimensions = '2D'):
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
            plt.figure()
            plt.plot(self.response[i_simcase]['t'], Pb_gust[:,2], 'b-')
            plt.plot(self.response[i_simcase]['t'], Pb_unsteady[:,2], 'r-')
            plt.plot(self.response[i_simcase]['t'], Pb_gust[:,2] + Pb_unsteady[:,2], 'b--')
            plt.plot(self.response[i_simcase]['t'], Pb_aero[:,2], 'g-')
            plt.plot(self.response[i_simcase]['t'], Pb_aero[:,2] - Pb_unsteady[:,2], 'k--')
            plt.xlabel('t [sec]')
            plt.ylabel('Pb [N]')
            plt.grid('on')
            plt.legend(['Pb_gust', 'Pb_unsteady', 'Pb_gust+unsteady', 'Pb_aero'])
            
            plt.figure()
            plt.subplot(3,1,1)
            plt.plot(self.response[i_simcase]['t'], self.response[i_simcase]['p1'])
            plt.plot(self.response[i_simcase]['t'], self.response[i_simcase]['X'][:,95:98], '--')
            plt.legend(('p1 MLG1', 'p1 MLG2', 'p1 NLG', 'p2 MLG1', 'p2 MLG2', 'p2 NLG'), loc='best')
            plt.xlabel('t [s]')
            plt.ylabel('p1,2 [m]')
            plt.grid('on')
            plt.subplot(3,1,2)
            plt.plot(self.response[i_simcase]['t'], self.response[i_simcase]['F1'])
            plt.plot(self.response[i_simcase]['t'], self.response[i_simcase]['F2'], '--')
            plt.legend(('F1 MLG1', 'F1 MLG2', 'F1 NLG', 'F2 MLG1', 'F2 MLG2', 'F2 NLG'), loc='best')
            plt.xlabel('t [s]')
            plt.ylabel('F1,2 [N]')
            plt.grid('on')
            
            plt.subplot(3,1,3)
            plt.plot(self.response[i_simcase]['t'], self.response[i_simcase]['dp1'])
            plt.plot(self.response[i_simcase]['t'], self.response[i_simcase]['X'][:,98:101], '--')
            plt.legend(('dp1 MLG1', 'dp1 MLG2', 'dp1 NLG', 'dp2 MLG1', 'dp2 MLG2', 'dp2 NLG'), loc='best')
            plt.xlabel('t [s]')
            plt.ylabel('dp1,2 [m/s]')
            plt.grid('on')
        
            plt.figure()
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
            
            
            plt.figure()
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
            
            plt.figure()
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
            
            plt.figure()
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
            
            plt.figure()
            plt.subplot(2,1,1)
            plt.plot(self.response[i_simcase]['t'], self.response[i_simcase]['dUcg_dt'][:,0], 'b-')
            plt.plot(self.response[i_simcase]['t'], self.response[i_simcase]['dUcg_dt'][:,1], 'g-')
            plt.plot(self.response[i_simcase]['t'], self.response[i_simcase]['dUcg_dt'][:,2], 'r-')
            plt.xlabel('t [sec]')
            plt.ylabel('[m/s]')
            plt.grid('on')
            plt.legend(['u_body', 'v_body', 'w_body'])
            plt.subplot(2,1,2)
            plt.plot(self.response[i_simcase]['t'], self.response[i_simcase]['dUcg_dt'][:,3]/np.pi*180.0, 'b-')
            plt.plot(self.response[i_simcase]['t'], self.response[i_simcase]['dUcg_dt'][:,4]/np.pi*180.0, 'g-')
            plt.plot(self.response[i_simcase]['t'], self.response[i_simcase]['dUcg_dt'][:,5]/np.pi*180.0, 'r-')
            plt.xlabel('t [sec]')
            plt.ylabel('[deg/s]')
            plt.grid('on')
            plt.legend(['p_body', 'q_body', 'r_body'])
            
            plt.figure()
            plt.subplot(2,1,1)
            plt.plot(self.response[i_simcase]['t'], self.response[i_simcase]['dUcg_dt'][:,0], 'b-')
            plt.plot(self.response[i_simcase]['t'], self.response[i_simcase]['dUcg_dt'][:,1], 'g-')
            plt.plot(self.response[i_simcase]['t'], self.response[i_simcase]['dUcg_dt'][:,2], 'r-')
            plt.xlabel('t [sec]')
            plt.ylabel('[m/s^2]')
            plt.grid('on')
            plt.legend(['du_body', 'dv_body', 'dw_body'])
            plt.subplot(2,1,2)
            plt.plot(self.response[i_simcase]['t'], self.response[i_simcase]['dUcg_dt'][:,3]/np.pi*180.0, 'b-')
            plt.plot(self.response[i_simcase]['t'], self.response[i_simcase]['dUcg_dt'][:,4]/np.pi*180.0, 'g-')
            plt.plot(self.response[i_simcase]['t'], self.response[i_simcase]['dUcg_dt'][:,5]/np.pi*180.0, 'r-')
            plt.xlabel('t [sec]')
            plt.ylabel('[deg/s^2]')
            plt.grid('on')
            plt.legend(['dp_body', 'dq_body', 'dr_body'])
            
            # show time plots
            plt.ion()
            plt.show()
            plt.ioff()
            
            # plot animation
            if animation_dimensions == '2D':
                # option if mayavi is not installed
                self.plot_time_animation_2d(i_simcase)
            elif animation_dimensions == '3D':
                # preferred
                self.plot_time_animation_3d(i_simcase)
            plt.close('all')
            
    def plot_time_animation_2d(self, i_simcase):        
            if self.jcl.general['aircraft'] == 'ALLEGRA':
                lim=25.0
                length=45
            elif self.jcl.general['aircraft'] in ['DLR F-19-S', 'MULDICON']:
                lim=10.0
                length=15
            elif self.jcl.general['aircraft'] == 'Discus2c':
                lim=10.0
                length=5.0
            else:
                logging.error('Unknown aircraft: ' + str(self.jcl.general['aircraft']))
                return
            
            def update_line(num, data, line1, line2, line3, ax2, ax3, t, time_text, length):
                line1.set_data(data[1,num,:], data[2,num,:])
                line2.set_data(data[0,num,:], data[2,num,:])
                line3.set_data(data[0,num,:], data[1,num,:])
                ax2.set_xlim((-10+data[0,num,0], length+data[0,num,0]))
                ax3.set_xlim((-10+data[0,num,0], length+data[0,num,0]))
                time_text.set_text('Time = ' + str(t[num,0]))   
                
            # Set up data
            x = self.model.strcgrid['offset'][:,0] + self.response[i_simcase]['Ug'][:,self.model.strcgrid['set'][:,0]]
            y = self.model.strcgrid['offset'][:,1] + self.response[i_simcase]['Ug'][:,self.model.strcgrid['set'][:,1]]
            z = self.model.strcgrid['offset'][:,2] + self.response[i_simcase]['Ug'][:,self.model.strcgrid['set'][:,2]]
            data = np.array([x,y,z])
            t = self.response[i_simcase]['t']
            # Set up plot
            fig = plt.figure()
            ax1 = plt.subplot(1,3,1)
            line1, = ax1.plot([], [], 'r.')
            time_text = ax1.text(-lim+2,lim-2, '')
            ax1.set_xlim((-lim, lim))
            ax1.set_ylim((-lim, lim))
            ax1.grid('on')
            ax1.set_title('Back')
            
            ax2 = plt.subplot(1,3,2)
            line2, = ax2.plot([], [], 'r.')
            ax2.set_ylim((-lim, lim))
            ax2.set_title('Side')
            #ax2.grid('on')
            #update_line(0,data,line1, line2, t, time_text)
            
            ax3 = plt.subplot(1,3,3)
            line3, = ax3.plot([], [], 'r.')
            ax3.set_ylim((-lim, lim))
            ax3.set_title('Top')
    
            line_ani = animation.FuncAnimation(fig, update_line, fargs=(data, line1, line2, line3, ax2, ax3, t, time_text, length), frames=len(t), interval=50, repeat=True, repeat_delay=3000)
            # Set up formatting for the movie files
    #         Writer = animation.writers['ffmpeg']
    #         writer = Writer(fps=20, bitrate=2000)        
    #         line_ani.save('/scratch/Discus2c_LoadsKernel/Elev3211_B_4sec.mp4', writer) 
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
                  
    def plot_time_animation_3d(self, i_trimcase, path_output='./', speedup_factor=1.0, make_movie=False):
        # To Do: show simulation time in animation
        from mayavi import mlab
        
        response   = self.response[i_trimcase]
        trimcase   = self.jcl.trimcase[i_trimcase]
        simcase    = self.jcl.simcase[i_trimcase] 
        
        @mlab.animate(delay=int(speedup_factor*simcase['dt']*1000.0), ui=True)
        def anim(self):
            # internal function that actually updates the animation
            while True:
                for (x, y, z, u1, v1, w1, u2, v2, w2, u3, v3, w3) in zip(self.x, self.y, self.z, self.u1, self.v1, self.w1, self.u2, self.v2, self.w2, self.u3, self.v3, self.w3):
                    self.fig.scene.disable_render = True
                    #self.points.mlab_source.set(x=x, y=y, z=z)
                    self.src.outputs[0].points.from_array(np.array([x, y, z]).T)
                    self.vectors1.mlab_source.set(x=x, y=y, z=z, u=u1, v=v1, w=w1)
                    self.cones1.mlab_source.set(x=x+u1, y=y+v1, z=z+w1, u=u1, v=v1, w=w1)
                    self.vectors2.mlab_source.set(x=x, y=y, z=z, u=u2, v=v2, w=w2)
                    self.cones2.mlab_source.set(x=x+u2, y=y+v2, z=z+w2, u=u2, v=v2, w=w2)
                    self.vectors3.mlab_source.set(x=x, y=y, z=z, u=u3, v=v3, w=w3)
                    self.cones3.mlab_source.set(x=x+u3, y=y+v3, z=z+w3, u=u3, v=v3, w=w3)
                    self.fig.scene.disable_render = False
                    #time.sleep(0.01)
                    yield
                
        def movie(self):
            # internal function that actually updates the animation
            self.fig.scene.disable_render = True
            i_frame = 0
            for (x, y, z, u1, v1, w1, u2, v2, w2, u3, v3, w3) in zip(self.x, self.y, self.z, self.u1, self.v1, self.w1, self.u2, self.v2, self.w2, self.u3, self.v3, self.w3):
                #self.points.mlab_source.set(x=x, y=y, z=z)
                self.src.outputs[0].points.from_array(np.array([x, y, z]).T)
                self.vectors1.mlab_source.set(x=x, y=y, z=z, u=u1, v=v1, w=w1)
                self.cones1.mlab_source.set(x=x+u1, y=y+v1, z=z+w1, u=u1, v=v1, w=w1)
                self.vectors2.mlab_source.set(x=x, y=y, z=z, u=u2, v=v2, w=w2)
                self.cones2.mlab_source.set(x=x+u2, y=y+v2, z=z+w2, u=u2, v=v2, w=w2)
                self.vectors3.mlab_source.set(x=x, y=y, z=z, u=u3, v=v3, w=w3)
                self.cones3.mlab_source.set(x=x+u3, y=y+v3, z=z+w3, u=u3, v=v3, w=w3)
                self.fig.scene.render()
                self.fig.scene.save_png('{}anim/subcase_{}_frame_{:06d}.png'.format(path_output, trimcase['subcase'], i_frame))
                i_frame += 1        
            
        # get deformations and forces
        # x-component without rigid body motion so that the aircraft does not fly out of sight
        self.x = self.model.strcgrid['offset'][:,0] + response['Ug'][:,self.model.strcgrid['set'][:,0]] - response['X'][:,0].repeat(self.model.strcgrid['n']).reshape(-1,self.model.strcgrid['n'])
        self.y = self.model.strcgrid['offset'][:,1] + response['Ug'][:,self.model.strcgrid['set'][:,1]]
        self.z = self.model.strcgrid['offset'][:,2] + response['Ug'][:,self.model.strcgrid['set'][:,2]]# - response['X'][:,2].repeat(self.model.strcgrid['n']).reshape(-1,self.model.strcgrid['n'])
      
        # Skalieren der Kraefte mit n-ter Wurzel bzw. Exponent
        exponent = 0.5
        uvw1 = np.linalg.norm(response['Pg_aero_global'][:,self.model.strcgrid['set'][:,(0,1,2)]], axis=2)
        uvw2 = np.linalg.norm(response['Pg_iner_global'][:,self.model.strcgrid['set'][:,(0,1,2)]], axis=2)
        uvw3 = np.linalg.norm(response['Pg_ext_global'][:,self.model.strcgrid['set'][:,(0,1,2)]], axis=2)
        uvw1_e = uvw1**exponent
        uvw2_e = uvw2**exponent
        uvw3_e = uvw3**exponent
        
        u1 = response['Pg_aero_global'][:,self.model.strcgrid['set'][:,0]] / uvw1 * uvw1_e
        v1 = response['Pg_aero_global'][:,self.model.strcgrid['set'][:,1]] / uvw1 * uvw1_e
        w1 = response['Pg_aero_global'][:,self.model.strcgrid['set'][:,2]] / uvw1 * uvw1_e
        u2 = response['Pg_iner_global'][:,self.model.strcgrid['set'][:,0]] / uvw2 * uvw2_e
        v2 = response['Pg_iner_global'][:,self.model.strcgrid['set'][:,1]] / uvw2 * uvw2_e
        w2 = response['Pg_iner_global'][:,self.model.strcgrid['set'][:,2]] / uvw2 * uvw2_e
        u3 = response['Pg_ext_global'][:,self.model.strcgrid['set'][:,0]] / uvw3 * uvw3_e
        v3 = response['Pg_ext_global'][:,self.model.strcgrid['set'][:,1]] / uvw3 * uvw3_e
        w3 = response['Pg_ext_global'][:,self.model.strcgrid['set'][:,2]] / uvw3 * uvw3_e
        
        # guard for NaNs due to pervious division by uvw
        u1[np.isnan(u1)] = 0.0
        v1[np.isnan(v1)] = 0.0
        w1[np.isnan(w1)] = 0.0
        u2[np.isnan(u2)] = 0.0
        v2[np.isnan(v2)] = 0.0
        w2[np.isnan(w2)] = 0.0
        u3[np.isnan(u3)] = 0.0
        v3[np.isnan(v3)] = 0.0
        w3[np.isnan(w3)] = 0.0
        
        # maximale Ist-Laenge eines Vektors
        r_max = np.max([(u1**2.0 + v1**2.0 + w1**2.0)**0.5, (u2**2.0 + v2**2.0 + w2**2.0)**0.5, (u3**2.0 + v3**2.0 + w3**2.0)**0.5])
        # maximale Soll-Laenge eines Vektors, abgeleitet von der Ausdehnung des Modells
        r_scale = np.max([self.model.strcgrid['offset'][:,0].max() - self.model.strcgrid['offset'][:,0].min(), self.model.strcgrid['offset'][:,1].max() - self.model.strcgrid['offset'][:,1].min(), self.model.strcgrid['offset'][:,2].max() - self.model.strcgrid['offset'][:,2].min()])
 
        # skalieren der Vektoren
        self.u1 = u1 / r_max * r_scale
        self.v1 = v1 / r_max * r_scale
        self.w1 = w1 / r_max * r_scale
        self.u2 = u2 / r_max * r_scale
        self.v2 = v2 / r_max * r_scale
        self.w2 = w2 / r_max * r_scale
        self.u3 = u3 / r_max * r_scale
        self.v3 = v3 / r_max * r_scale
        self.w3 = w3 / r_max * r_scale

        # set up animation
        if make_movie:
            logging.info('rendering offscreen simulation {:s} ...'.format(trimcase['desc']))
            mlab.options.offscreen = True
            self.fig = mlab.figure(size=(1920, 1080))
        else: 
            logging.info('interactive plotting of forces and deformations for simulation {:s}'.format(trimcase['desc']))
            self.fig = mlab.figure()
        mlab.points3d(self.x[0,:], self.y[0,:], self.z[0,:], color=(1,1,1), opacity=0.4, scale_factor=self.p_scale) # intital position of aircraft, remains as a shadow in the animation for better comparision
        # BUG: mlab_source.set() funktioniert nicht bei points3d, daher der direkte Weg ueber eine pipeline
        #self.points = mlab.points3d(self.x[0,:], self.y[0,:], self.z[0,:], color=(0,0,1), scale_factor=self.p_scale)
        self.src = mlab.pipeline.scalar_scatter(self.x[0,:], self.y[0,:], self.z[0,:])
        pts = mlab.pipeline.glyph(self.src, color=(0,0,1), scale_factor=self.p_scale)
        self.vectors1 = mlab.quiver3d(self.x[0,:],self.y[0,:], self.z[0,:], self.u1[0,:], self.v1[0,:], self.w1[0,:], color=(0,1,0),  mode='2ddash', opacity=0.4,  scale_mode='vector', scale_factor=1.0)
        self.cones1   = mlab.quiver3d(self.x[0,:]+self.u1[0,:], self.y[0,:]+self.v1[0,:], self.z[0,:]+self.w1[0,:], self.u1[0,:], self.v1[0,:], self.w1[0,:], color=(0,1,0),  mode='cone', opacity=0.4, scale_mode='vector', scale_factor=0.1, resolution=16)
        self.vectors2 = mlab.quiver3d(self.x[0,:],self.y[0,:], self.z[0,:], self.u2[0,:], self.v2[0,:], self.w2[0,:], color=(0,1,1),  mode='2ddash', opacity=0.4,  scale_mode='vector', scale_factor=1.0)
        self.cones2   = mlab.quiver3d(self.x[0,:]+self.u2[0,:], self.y[0,:]+self.v2[0,:], self.z[0,:]+self.w2[0,:], self.u2[0,:], self.v2[0,:], self.w2[0,:], color=(0,1,1),  mode='cone', opacity=0.4, scale_mode='vector', scale_factor=0.1, resolution=16)       
        self.vectors3 = mlab.quiver3d(self.x[0,:],self.y[0,:], self.z[0,:], self.u3[0,:], self.v3[0,:], self.w3[0,:], color=(1,1,0),  mode='2ddash', opacity=0.4,  scale_mode='vector', scale_factor=1.0)
        self.cones3   = mlab.quiver3d(self.x[0,:]+self.u3[0,:], self.y[0,:]+self.v3[0,:], self.z[0,:]+self.w3[0,:], self.u3[0,:], self.v3[0,:], self.w3[0,:], color=(1,1,0),  mode='cone', opacity=0.4, scale_mode='vector', scale_factor=0.1, resolution=16)       

        mlab.orientation_axes()

        #mlab.view(azimuth=180.0, elevation=90.0, roll=-90.0, distance=70.0, focalpoint=np.array([self.x.mean(),self.y.mean(),self.z.mean()])) # back view
        distance = 1.5*((self.x.max()-self.x.min())**2 + (self.y.max()-self.y.min())**2 + (self.z.max()-self.z.min())**2)**0.5
        mlab.view(azimuth=135.0, elevation=120.0, roll=-120.0, distance=distance, focalpoint=np.array([self.x.mean(),self.y.mean(),self.z.mean()])) # view from right and above

        if make_movie:
            if not os.path.exists('{}anim/'.format(path_output)):
                os.makedirs('{}anim/'.format(path_output))
            movie(self) # launch animation
            mlab.close()
            # h.246
            os.system('ffmpeg -framerate {} -i {}anim/subcase_{}_frame_%06d.png  -r 30 -y {}anim/subcase_{}.mov'.format( speedup_factor/simcase['dt'], path_output, trimcase['subcase'], path_output, trimcase['subcase']) )
            # MPEG-4 - besser geeignet fuer PowerPoint & Co.
            os.system('ffmpeg -framerate {} -i {}anim/subcase_{}_frame_%06d.png -c:v mpeg4 -q:v 3 -r 30 -y {}anim/subcase_{}.avi'.format( speedup_factor/simcase['dt'], path_output, trimcase['subcase'], path_output, trimcase['subcase']) )
        else:
            anim(self) # launch animation
            mlab.show()
            
