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
import os

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

            self.monstations['MON{:s}'.format(str(int(self.model.mongrid['ID'][i_station])))] = monstation
            

    def save_monstations(self, filename):
        print 'saving monitoring stations as Nastarn cards...'
        with open(filename, 'w') as fid: 
            for i_trimcase in range(len(self.jcl.trimcase)):
                write_functions.write_force_and_moment_cards(fid, self.model.mongrid, self.response[i_trimcase]['Pmon_local'], i_trimcase+1)
    
    def save_nodaldefo(self, filename):
        print 'saving nodal deformations as dat file...'
        with open(filename+'_undeformed.dat', 'w') as fid:             
            np.savetxt(fid, np.hstack((self.model.strcgrid['ID'].reshape(-1,1), self.model.strcgrid['offset'])))
        
        for i_trimcase in range(len(self.jcl.trimcase)):
            with open(filename+'_subcase_'+str(i_trimcase+1)+'_Uf_x10.dat', 'w') as fid: 
                defo = np.hstack((self.model.strcgrid['ID'].reshape(-1,1), self.model.strcgrid['offset'] + self.response[i_trimcase]['Ug_f'][self.model.strcgrid['set'][:,0:3]] * 10.0 ))
                np.savetxt(fid, defo)
                
                
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
    
            mlab.figure()
            mlab.points3d(x, y, z, scale_factor=0.1)
            mlab.quiver3d(x, y, z, fx*0.01, fy*0.01, fz*0.01 , color=(0,1,0),  mode='2ddash', opacity=0.4,  scale_mode='vector', scale_factor=1.0)
            mlab.quiver3d(x+fx*0.01, y+fy*0.01, z+fz*0.01,fx*0.01, fy*0.01, fz*0.01 , color=(0,1,0),  mode='cone', scale_mode='scalar', scale_factor=0.5, resolution=16)
            mlab.title('Pk_rbm', size=0.2, height=0.95)
            
            mlab.figure() 
            mlab.points3d(x, y, z, scale_factor=0.1)
            mlab.quiver3d(x, y, z, response['Pk_cam'][self.model.aerogrid['set_k'][:,0]], response['Pk_cam'][self.model.aerogrid['set_k'][:,1]], response['Pk_cam'][self.model.aerogrid['set_k'][:,2]], color=(0,1,1), scale_factor=0.01)            
            mlab.title('Pk_camber_twist', size=0.2, height=0.95)
            
            mlab.figure()        
            mlab.points3d(x, y, z, scale_factor=0.1)
            mlab.quiver3d(x, y, z, response['Pk_cs'][self.model.aerogrid['set_k'][:,0]], response['Pk_cs'][self.model.aerogrid['set_k'][:,1]], response['Pk_cs'][self.model.aerogrid['set_k'][:,2]], color=(1,0,0), scale_factor=0.01)
            mlab.title('Pk_cs', size=0.2, height=0.95)
            
            mlab.figure()   
            mlab.points3d(x, y, z, scale_factor=0.1)
            mlab.quiver3d(x, y, z, response['Pk_f'][self.model.aerogrid['set_k'][:,0]], response['Pk_f'][self.model.aerogrid['set_k'][:,1]], response['Pk_f'][self.model.aerogrid['set_k'][:,2]], color=(1,0,1), scale_factor=0.01)
            mlab.title('Pk_flex', size=0.2, height=0.95)
            
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
            mlab.points3d(x, y, z,  scale_factor=0.1)
            mlab.points3d(x_r, y_r, z_r, color=(0,1,0), scale_factor=0.1)
            mlab.points3d(x_f, y_f, z_f, color=(0,0,1), scale_factor=0.1)
            mlab.title('rbm (green) and flexible deformation x10 (blue)', size=0.2, height=0.95)
            mlab.show()
                

    def plot_monstations(self, monstations, filename_pdf):
        
        stations_wing = ['Mon1', 'Mon2', 'Mon3', 'Mon8']
        stations_fuselage = ['Mon6', 'Mon7']
        stations_to_plot = stations_wing + stations_fuselage
        
        print 'start potato-plotting...'
        # get data needed for plotting from monstations
        loads = []
        offsets = []
        for station in stations_to_plot:
            loads.append(monstations[station]['loads'])
            offsets.append(monstations[station]['offset'])
        loads = np.array(loads)
        offsets = np.array(offsets)
        
        pp = PdfPages(filename_pdf)
        for i_station in range(len(stations_to_plot)):
            # calculated convex hull from scattered points
            if stations_to_plot[i_station] in stations_wing:
                points = np.vstack((loads[i_station][:,2], loads[i_station][:,3])).T
                labels = ['Fz [N]', 'Mx [Nm]' ]
            elif stations_to_plot[i_station] in stations_fuselage:
                points = np.vstack((loads[i_station][:,2], loads[i_station][:,4])).T
                labels = ['Fz [N]', 'My [Nm]' ]
            hull = ConvexHull(points)
            
            plt.figure()
            # plot points
            plt.scatter(points[:,0], points[:,1], color='b')
            # plot hull and lable points
            plt.scatter(points[hull.vertices,0], points[hull.vertices,1], color='r')  
            for simplex in hull.simplices: 
                plt.plot(points[simplex,0], points[simplex,1], 'r--')
            for i_case in range(hull.nsimplex):                    
                plt.text(points[hull.vertices[i_case],0], points[hull.vertices[i_case],1], str(monstations[stations_to_plot[i_station]]['subcase'][hull.vertices[i_case]]), fontsize=8)
        
            plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
            plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            plt.title(stations_to_plot[i_station])
            plt.grid('on')
            plt.xlabel(labels[0])
            plt.ylabel(labels[1])
            
            #plt.show()
            pp.savefig()
            plt.close()
        
        print 'start plotting cutting forces...'
        for i_case in range(len(monstations[stations_to_plot[0]]['subcase'])):
            #print i_case
            loads_subcase = loads[:,i_case,:]
            plt.figure()
            # for every subcase, plot Fx, Fy, Fz, Mx, My, Mz in 3x2 subplots
            plt.subplot(2,3,1)
            plt.plot(offsets[:,1], loads_subcase[:,0], '.-')
            plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            plt.grid('on')
            plt.xlabel('y [m]')
            plt.ylabel('Fx [N]')   
            plt.subplot(2,3,2)
            plt.plot(offsets[:,1], loads_subcase[:,1], '.-')  
            plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            plt.grid('on')
            plt.xlabel('y [m]')
            plt.ylabel('Fy [N]')   
            plt.subplot(2,3,3)
            plt.plot(offsets[:,1], loads_subcase[:,2], '.-')  
            plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            plt.grid('on')
            plt.xlabel('y [m]')
            plt.ylabel('Fz [N]')   
            plt.subplot(2,3,4)
            plt.plot(offsets[:,1], loads_subcase[:,3], '.-') 
            plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            plt.grid('on')
            plt.xlabel('y [m]')
            plt.ylabel('Mx [Nm]')   
            plt.subplot(2,3,5)
            plt.plot(offsets[:,1], loads_subcase[:,4], '.-') 
            plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            plt.grid('on')
            plt.xlabel('y [m]')
            plt.ylabel('My [Nm]')  
            plt.subplot(2,3,6)
            plt.plot(offsets[:,1], loads_subcase[:,5], '.-') 
            plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            plt.grid('on')
            plt.xlabel('y [m]')
            plt.ylabel('Mz [Nm]')
            #plt.tight_layout()
            plt.suptitle('Subcase ' + str(monstations['Mon1']['subcase'][i_case]))
            plt.subplots_adjust(left=0.1, right=0.95, top=0.90, bottom=0.1, wspace = 0.4, hspace = 0.3)
            
            #plt.show()
            pp.savefig()
            plt.close()
            
        pp.close()
        print 'saved as ' + filename_pdf
        print 'opening '+ filename_pdf
        os.system('evince ' + filename_pdf + ' &')
        return
        
        
        
        
        
        