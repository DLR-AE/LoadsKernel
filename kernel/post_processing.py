# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 13:34:50 2015

@author: voss_ar
"""
import numpy as np
import matplotlib.pyplot as plt
#from mayavi import mlab
from matplotlib.backends.backend_pdf import PdfPages
from scipy.spatial import ConvexHull
import os, csv

from trim_tools import *
import write_functions
from grid_trafo import *

class post_processing:
    def __init__(self, jcl, model, response):
        self.jcl = jcl
        self.model = model
        self.response = response
    
    def force_summation_method(self):
        print 'calculating forces & moments on structural set (force summation method)...'
        for i_trimcase in range(len(self.jcl.trimcase)):
            response   = self.response[i_trimcase]
            trimcase   = self.jcl.trimcase[i_trimcase]
            
            i_atmo     = self.model.atmo['key'].index(trimcase['altitude'])
            i_mass     = self.model.mass['key'].index(trimcase['mass'])
            Mgg        = self.model.mass['MGG'][i_mass]
            PHIf_strc  = self.model.mass['PHIf_strc'][i_mass]
            PHIstrc_cg = self.model.mass['PHIstrc_cg'][i_mass]
            PHInorm_cg = self.model.mass['PHInorm_cg'][i_mass]
            PHIcg_norm = self.model.mass['PHIcg_norm'][i_mass]
            n_modes    = self.model.mass['n_modes'][i_mass]
            
            # Formel bezogen auf die linearen Bewegungsgleichungen Nastrans. 
            # Fuer Bewegungsgleichungen z.B. von Waszack muessen die zusaetzlichen Terme hier ebenfalls beruecksichtigt werden!
            d2Ug_dt2_r = PHIstrc_cg.dot( np.hstack((response['d2Ucg_dt2'][0:3] - response['g_cg'], response['d2Ucg_dt2'][3:6])) )
            response['Pg_iner_r'] = - Mgg.dot(d2Ug_dt2_r)
            
            d2Ug_dt2_f = PHIf_strc.T.dot(response['d2Uf_dt2'])
            response['Pg_iner_f'] = - Mgg.dot(d2Ug_dt2_f)
            #response['Ug_flex'] = PHIf_strc.T.dot(response['Uf'])
            #response['Pg_flex'] = self.model.Kgg.dot(response['Ug_flex']) * 0.0
            
            response['Pg'] = response['Pg_aero'] + response['Pg_iner_r'] + response['Pg_iner_f']

            # das muss raus kommen:
            #np.dot(self.model.mass['Mb'][i_mass],np.hstack((response['d2Ucg_dt2'][0:3] - response['g_cg'], response['d2Ucg_dt2'][3:6])))
            #PHIstrc_cg.T.dot(response['Pg_aero'])
            # das kommt raus:
            #PHIstrc_cg.T.dot(response['Pg_iner_r'])
            
            
            Uf = response['X'][12:12+n_modes]
            response['Ug_f'] = np.dot(self.model.mass['PHIf_strc'][i_mass].T, Uf.T).T
            
            Tgeo2body = np.zeros((6,6))
            Tgeo2body[0:3,0:3] = calc_drehmatrix(response['X'][3], response['X'][4], response['X'][5])
            Tgeo2body[3:6,3:6] = calc_drehmatrix(response['X'][3], response['X'][4], response['X'][5])
            height = self.model.atmo['h'][i_atmo] # correction of height to zero to allow plotting in one diagram
            response['Ug_r'] = PHIstrc_cg.dot( np.dot(PHIcg_norm,np.dot(Tgeo2body, response['X'][0:6]+[0,0,height,0,0,0])) )
            
            response['Ug'] = response['Ug_r'] + response['Ug_f']

        
    def cuttingforces(self):
        print 'calculating cutting forces & moments...'
        for i_trimcase in range(len(self.jcl.trimcase)):
            self.response[i_trimcase]['Pmon_global'] = self.model.PHIstrc_mon.T.dot(self.response[i_trimcase]['Pg'])
            self.response[i_trimcase]['Pmon_local'] = force_trafo(self.model.mongrid, self.model.coord, self.response[i_trimcase]['Pmon_global'])
                            
        
    def gather_monstations(self):
        print 'gathering information on monitoring stations from respone(s)...'
        self.monstations = {}
        for i_station in range(self.model.mongrid['n']):
            monstation = {'CD': self.model.mongrid['CD'][i_station] ,
                          'CP': self.model.mongrid['CP'][i_station], 
                          'offset': self.model.mongrid['offset'][i_station], 
                          'subcase': [],  
                          'loads':[],
                         }
            for i_trimcase in range(len(self.jcl.trimcase)):
                monstation['loads'].append(self.response[i_trimcase]['Pmon_local'][self.model.mongrid['set'][i_station,:]])
                monstation['subcase'].append(self.jcl.trimcase[i_trimcase]['subcase'])
            
            if not self.model.mongrid.has_key('name'):
                name = 'MON{:s}'.format(str(int(self.model.mongrid['ID'][i_station]))) # make up a name
            else:
                name = self.model.mongrid['name'][i_station] # take name from mongrid
            self.monstations[name] = monstation
            

    def save_monstations(self, filename):
        print 'saving monitoring stations as Nastarn cards...'
        with open(filename, 'w') as fid: 
            for i_trimcase in range(len(self.jcl.trimcase)):
                write_functions.write_force_and_moment_cards(fid, self.model.mongrid, self.response[i_trimcase]['Pmon_local'], i_trimcase+1)
    
    def save_nodaldefo(self, filename):
        print 'saving nodal deformations as dat file...'
        with open(filename+'_undeformed.dat', 'w') as fid:             
            np.savetxt(fid, np.hstack((self.model.strcgrid['ID'].reshape(-1,1), self.model.strcgrid['offset'])))
            #np.savetxt(fid, self.model.strcgrid['offset'])
        
        for i_trimcase in range(len(self.jcl.trimcase)):
            with open(filename+'_subcase_'+str(i_trimcase+1)+'_Uf_x10.dat', 'w') as fid: 
                defo = np.hstack((self.model.strcgrid['ID'].reshape(-1,1), self.model.strcgrid['offset'] + self.response[i_trimcase]['Ug_f'][self.model.strcgrid['set'][:,0:3]] * 10.0 ))
                np.savetxt(fid, defo)
                #np.savetxt(fid, defo[:,1:4])
                
    def save_nodalloads(self, filename):
        print 'saving nodal loads as Nastarn cards...'
        with open(filename+'_Pg', 'w') as fid: 
            for i_trimcase in range(len(self.jcl.trimcase)):
                write_functions.write_force_and_moment_cards(fid, self.model.strcgrid, self.response[i_trimcase]['Pg'], i_trimcase+1)
#        with open(filename+'_Pg_aero', 'w') as fid: 
#            for i_trimcase in range(len(self.jcl.trimcase)):
#                write_functions.write_force_and_moment_cards(fid, self.model.strcgrid, self.response[i_trimcase]['Pg_aero'], i_trimcase+1)
#        with open(filename+'_Pg_iner_r', 'w') as fid: 
#            for i_trimcase in range(len(self.jcl.trimcase)):
#                write_functions.write_force_and_moment_cards(fid, self.model.strcgrid, self.response[i_trimcase]['Pg_iner_r'], i_trimcase+1) 
#        with open(filename+'_Pg_iner_f', 'w') as fid: 
#            for i_trimcase in range(len(self.jcl.trimcase)):
#                write_functions.write_force_and_moment_cards(fid, self.model.strcgrid, self.response[i_trimcase]['Pg_iner_f'], i_trimcase+1)
    
    
    def plot_forces_deformation_interactive(self):
        from mayavi import mlab

        for i_trimcase in range(len(self.jcl.trimcase)):
            response   = self.response[i_trimcase]
            trimcase   = self.jcl.trimcase[i_trimcase]
            print 'interactive plotting of forces and deformations for trim {:s}'.format(trimcase['desc'])

            x = self.model.aerogrid['offset_k'][:,0]
            y = self.model.aerogrid['offset_k'][:,1]
            z = self.model.aerogrid['offset_k'][:,2]
            fx, fy, fz = response['Pk_rbm'][self.model.aerogrid['set_k'][:,0]],response['Pk_rbm'][self.model.aerogrid['set_k'][:,1]], response['Pk_rbm'][self.model.aerogrid['set_k'][:,2]]
            f_scale = 0.002 # vectors
            p_scale = 0.3 # points
            mlab.figure()
            mlab.points3d(x, y, z, scale_factor=p_scale)
            mlab.quiver3d(x, y, z, response['Pk_rbm'][self.model.aerogrid['set_k'][:,0]], response['Pk_rbm'][self.model.aerogrid['set_k'][:,1]], response['Pk_rbm'][self.model.aerogrid['set_k'][:,2]], color=(0,1,0), scale_factor=f_scale)            
            #mlab.quiver3d(x, y, z, fx*f_scale, fy*f_scale, fz*f_scale , color=(0,1,0),  mode='2ddash', opacity=0.4,  scale_mode='vector', scale_factor=1.0)
            #mlab.quiver3d(x+fx*f_scale, y+fy*f_scale, z+fz*f_scale,fx*f_scale, fy*f_scale, fz*f_scale , color=(0,1,0),  mode='cone', scale_mode='scalar', scale_factor=0.5, resolution=16)
            mlab.title('Pk_rbm', size=0.2, height=0.95)
            
            mlab.figure() 
            mlab.points3d(x, y, z, scale_factor=p_scale)
            mlab.quiver3d(x, y, z, response['Pk_cam'][self.model.aerogrid['set_k'][:,0]], response['Pk_cam'][self.model.aerogrid['set_k'][:,1]], response['Pk_cam'][self.model.aerogrid['set_k'][:,2]], color=(0,1,1), scale_factor=f_scale)            
            mlab.title('Pk_camber_twist', size=0.2, height=0.95)
            
            mlab.figure()        
            mlab.points3d(x, y, z, scale_factor=p_scale)
            mlab.quiver3d(x, y, z, response['Pk_cs'][self.model.aerogrid['set_k'][:,0]], response['Pk_cs'][self.model.aerogrid['set_k'][:,1]], response['Pk_cs'][self.model.aerogrid['set_k'][:,2]], color=(1,0,0), scale_factor=f_scale)
            mlab.title('Pk_cs', size=0.2, height=0.95)
            
            mlab.figure()   
            mlab.points3d(x, y, z, scale_factor=p_scale)
            mlab.quiver3d(x, y, z, response['Pk_f'][self.model.aerogrid['set_k'][:,0]], response['Pk_f'][self.model.aerogrid['set_k'][:,1]], response['Pk_f'][self.model.aerogrid['set_k'][:,2]], color=(1,0,1), scale_factor=f_scale)
            mlab.title('Pk_flex', size=0.2, height=0.95)
            
            mlab.figure()   
            mlab.points3d(x, y, z, scale_factor=p_scale)
            mlab.quiver3d(x, y, z, response['Pk_cfd'][self.model.aerogrid['set_k'][:,0]], response['Pk_cfd'][self.model.aerogrid['set_k'][:,1]], response['Pk_cfd'][self.model.aerogrid['set_k'][:,2]], color=(1,1,1), scale_factor=f_scale)
            mlab.title('Pk_cfd', size=0.2, height=0.95)
            
            x = self.model.strcgrid['offset'][:,0]
            y = self.model.strcgrid['offset'][:,1]
            z = self.model.strcgrid['offset'][:,2]
            x_r = self.model.strcgrid['offset'][:,0] + response['Ug_r'][self.model.strcgrid['set'][:,0]]
            y_r = self.model.strcgrid['offset'][:,1] + response['Ug_r'][self.model.strcgrid['set'][:,1]]
            z_r = self.model.strcgrid['offset'][:,2] + response['Ug_r'][self.model.strcgrid['set'][:,2]]
            x_f = self.model.strcgrid['offset'][:,0] + response['Ug_f'][self.model.strcgrid['set'][:,0]] * 10.0
            y_f = self.model.strcgrid['offset'][:,1] + response['Ug_f'][self.model.strcgrid['set'][:,1]] * 10.0
            z_f = self.model.strcgrid['offset'][:,2] + response['Ug_f'][self.model.strcgrid['set'][:,2]] * 10.0
            
            mlab.figure()
            mlab.points3d(x, y, z, color=(0,0,0), scale_factor=p_scale)
            mlab.points3d(x_r, y_r, z_r, color=(0,1,0), scale_factor=p_scale)
            mlab.points3d(x_f, y_f, z_f, color=(0,0,1), scale_factor=p_scale)
            mlab.title('rbm (green) and flexible deformation x10 (blue)', size=0.2, height=0.95)
            
            mlab.figure()   
            mlab.points3d(x, y, z, color=(0,0,0), scale_factor=p_scale)
            mlab.quiver3d(x, y, z, response['Pg_aero'][self.model.strcgrid['set'][:,0]], response['Pg_aero'][self.model.strcgrid['set'][:,1]], response['Pg_aero'][self.model.strcgrid['set'][:,2]], color=(0,1,0), scale_factor=f_scale)
            mlab.title('Pg_aero', size=0.2, height=0.95)
            
            mlab.show()
                

    def plot_monstations(self, monstations, filename_pdf):
        # Allegra
        potatos_Fz_Mx = ['ZWCUT01', 'ZWCUT04', 'ZWCUT08', 'ZWCUT12', 'ZWCUT16', 'ZWCUT20', 'ZWCUT24', 'ZHCUT01' ]
        potatos_Mx_My = ['ZWCUT01', 'ZWCUT04', 'ZWCUT08', 'ZWCUT12', 'ZWCUT16', 'ZWCUT20', 'ZWCUT24', 'ZHCUT01' ]
        potatos_Fz_My = ['ZWCUT01', 'ZWCUT04', 'ZWCUT08', 'ZWCUT12', 'ZWCUT16', 'ZWCUT20', 'ZWCUT24', 'ZHCUT01', 'ZFCUT27', 'ZFCUT28']
        cuttingforces_wing = ['ZWCUT01', 'ZWCUT04', 'ZWCUT08', 'ZWCUT12', 'ZWCUT16', 'ZWCUT20', 'ZWCUT24', ]
        # DLR-F19
#        potatos_Fz_Mx = ['MON1', 'MON2', 'MON3', 'MON33', 'MON8', 'MON6', 'MON7', 'MON9']
#        potatos_Mx_My = ['MON1', 'MON2', 'MON3', 'MON33', 'MON8', 'MON6', 'MON7', 'MON9']
#        potatos_Fz_My = ['MON1', 'MON2', 'MON3', 'MON33', 'MON4', 'MON5']
#        cuttingforces_wing = ['MON1', 'MON2', 'MON3', 'MON33', 'MON8']

        print 'start potato-plotting...'
        # get data needed for plotting from monstations
        loads = []
        offsets = []
        potato = np.unique(potatos_Fz_Mx + potatos_Mx_My + potatos_Fz_My)
        for station in potato:
            loads.append(monstations[station]['loads'])
            offsets.append(monstations[station]['offset'])
        loads = np.array(loads)
        offsets = np.array(offsets)
        
        pp = PdfPages(filename_pdf)
        self.crit_trimcases = []
        for i_station in range(len(potato)):
            
            if potato[i_station] in potatos_Fz_Mx:
                points = np.vstack((loads[i_station][:,2], loads[i_station][:,3])).T
                labels = ['Fz [N]', 'Mx [Nm]' ]
                # --- plot ---
                hull = ConvexHull(points) # calculated convex hull from scattered points
                plt.figure()
                plt.scatter(points[:,0], points[:,1], color='cornflowerblue') # plot points
                for simplex in hull.simplices:                   # plot convex hull
                    plt.plot(points[simplex,0], points[simplex,1], color='cornflowerblue', linewidth=2.0, linestyle='--')
                for i_case in range(hull.nsimplex):              # plot text   
                    plt.text(points[hull.vertices[i_case],0], points[hull.vertices[i_case],1], str(monstations[potato[i_station]]['subcase'][hull.vertices[i_case]]), fontsize=8)
                    self.crit_trimcases.append(monstations[potato[i_station]]['subcase'][hull.vertices[i_case]])
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
                hull = ConvexHull(points) # calculated convex hull from scattered points
                plt.figure()
                plt.scatter(points[:,0], points[:,1], color='cornflowerblue') # plot points
                for simplex in hull.simplices:                   # plot convex hull
                    plt.plot(points[simplex,0], points[simplex,1], color='cornflowerblue', linewidth=2.0, linestyle='--')
                for i_case in range(hull.nsimplex):              # plot text   
                    plt.text(points[hull.vertices[i_case],0], points[hull.vertices[i_case],1], str(monstations[potato[i_station]]['subcase'][hull.vertices[i_case]]), fontsize=8)
                    self.crit_trimcases.append(monstations[potato[i_station]]['subcase'][hull.vertices[i_case]])
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
                hull = ConvexHull(points) # calculated convex hull from scattered points
                plt.figure()
                plt.scatter(points[:,0], points[:,1], color='cornflowerblue') # plot points
                for simplex in hull.simplices:                   # plot convex hull
                    plt.plot(points[simplex,0], points[simplex,1], color='cornflowerblue', linewidth=2.0, linestyle='--')
                for i_case in range(hull.nsimplex):              # plot text   
                    plt.text(points[hull.vertices[i_case],0], points[hull.vertices[i_case],1], str(monstations[potato[i_station]]['subcase'][hull.vertices[i_case]]), fontsize=8)
                    self.crit_trimcases.append(monstations[potato[i_station]]['subcase'][hull.vertices[i_case]])
                plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
                plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
                plt.title(potato[i_station])
                plt.grid('on')
                plt.xlabel(labels[0])
                plt.ylabel(labels[1])
                pp.savefig()
                plt.close()   
        
        
        print 'start plotting cutting forces...'
        cuttingforces = ['Fx [N]', 'Fy [N]', 'Fz [N]', 'Mx [Nm]', 'My [Nm]', 'Mz [Nm]']
        
        loads = []
        offsets = []
        for station in cuttingforces_wing:
            loads.append(monstations[station]['loads'])
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
                plt.text(   offsets[i_station,1],loads[i_station,i_max[i_station],i_cuttingforce], str(monstations[monstations.keys()[0]]['subcase'][i_max[i_station]]), fontsize=8, verticalalignment='bottom' )
                # min
                plt.scatter(offsets[i_station,1],loads[i_station,i_min[i_station],i_cuttingforce], color='r')
                plt.text(   offsets[i_station,1],loads[i_station,i_min[i_station],i_cuttingforce], str(monstations[monstations.keys()[0]]['subcase'][i_min[i_station]]), fontsize=8, verticalalignment='top' )
            plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            plt.title('Wing')        
            plt.grid('on')
            plt.xlabel('y [m]')
            plt.ylabel(cuttingforces[i_cuttingforce]) 
            #plt.show()
            pp.savefig()
            plt.close()
            
        pp.close()
        print 'plots saved as ' + filename_pdf
        #print 'opening '+ filename_pdf
        #os.system('evince ' + filename_pdf + ' &')
        return 
        
        
    def write_critical_trimcases(self, crit_trimcases, trimcases, filename_csv):
        
        crit_trimcases_info = []
        for trimcase in trimcases:
            if trimcase['subcase'] in crit_trimcases:
                crit_trimcases_info.append(trimcase)
        print 'writing critical trimcases cases to: ' + filename_csv
        with open(filename_csv, 'wb') as fid:
            w = csv.DictWriter(fid, crit_trimcases_info[0].keys())
            w.writeheader()
            w.writerows(crit_trimcases_info)
        return
        
        
        
        
        