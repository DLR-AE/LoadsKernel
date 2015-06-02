# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 13:34:50 2015

@author: voss_ar
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.spatial import ConvexHull
import os

import write_functions

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
            
            i_mass     = self.model.mass['key'].index(trimcase['mass'])
            Mgg        = self.model.mass['MGG'][i_mass]
            PHIf_strc  = self.model.mass['PHIf_strc'][i_mass]
            PHIstrc_cg  = self.model.mass['PHIstrc_cg'][i_mass]
            
            # Formel bezogen auf die linearen Bewegungsgleichungen Nastrans. 
            # Fuer Bewegungsgleichungen z.B. von Waszack muessen die zusaetzlichen Terme hier ebenfalls beruecksichtigt werden!
            d2Ug_dt2_r = PHIstrc_cg.dot( np.hstack((response['d2Ucg_dt2'][0:3] - response['g_cg'], response['d2Ucg_dt2'][3:6])) )
            response['Pg_iner_r'] = - Mgg.dot(d2Ug_dt2_r)
            
            d2Ug_dt2_f = PHIf_strc.T.dot(response['d2Uf_dt2'])
            response['Pg_iner_f'] = - Mgg.dot(d2Ug_dt2_f)  * 0.0
            #response['Ug_flex'] = PHIf_strc.T.dot(response['Uf'])
            #response['Pg_flex'] = self.model.Kgg.dot(response['Ug_flex']) * 0.0
            
            response['Pg'] = response['Pg_aero'] + response['Pg_iner_r'] + response['Pg_iner_f']

            # das muss raus kommen:
            #np.dot(self.model.mass['Mb'][i_mass],np.hstack((response['d2Ucg_dt2'][0:3] - response['g_cg'], response['d2Ucg_dt2'][3:6])))
            #PHIstrc_cg.T.dot(response['Pg_aero'])
            # das kommt raus:
            #PHIstrc_cg.T.dot(response['Pg_iner_r'])

            plotting = False
            if plotting:
                from mayavi import mlab
                x, y, z = self.model.strcgrid['offset'][:,0], self.model.strcgrid['offset'][:,1], self.model.strcgrid['offset'][:,2]
                
                mlab.figure() 
                mlab.points3d(x, y, z, scale_factor=0.05)
                #mlab.quiver3d(x, y, z, response['Pg_iner_r'][self.model.strcgrid['set'][:,0]], response['Pg_iner_r'][self.model.strcgrid['set'][:,1]], response['Pg_iner_r'][self.model.strcgrid['set'][:,2]], color=(1,0,0), scale_factor=0.001)            
                #mlab.quiver3d(x, y, z, Pg_iner_f[self.model.strcgrid['set'][:,0]], Pg_iner_f[self.model.strcgrid['set'][:,1]], Pg_iner_f[self.model.strcgrid['set'][:,2]], color=(0,1,0), scale_factor=0.01)            
                
                #mlab.quiver3d(x, y, z, Pg_flex[self.model.strcgrid['set'][:,0]]*1, Pg_flex[self.model.strcgrid['set'][:,1]]*0, Pg_flex[self.model.strcgrid['set'][:,2]]*0, color=(1,0,0), scale_factor=0.01)         
                #mlab.quiver3d(x, y, z, Pg_flex[self.model.strcgrid['set'][:,0]]*0, Pg_flex[self.model.strcgrid['set'][:,1]]*1, Pg_flex[self.model.strcgrid['set'][:,2]]*0, color=(0,1,0), scale_factor=0.01)         
                #mlab.quiver3d(x, y, z, Pg_flex[self.model.strcgrid['set'][:,0]]*0, Pg_flex[self.model.strcgrid['set'][:,1]]*0, Pg_flex[self.model.strcgrid['set'][:,2]]*1, color=(0,0,1), scale_factor=0.01)                 
        
                mlab.quiver3d(x, y, z, response['Pg_aero'][self.model.strcgrid['set'][:,0]], response['Pg_aero'][self.model.strcgrid['set'][:,1]], response['Pg_aero'][self.model.strcgrid['set'][:,2]], color=(0,0,1), scale_factor=0.001)            
         
                #x_f = self.model.strcgrid['offset'][:,0] + Ug_flex[self.model.strcgrid['set'][:,0]] * 100
                #y_f = self.model.strcgrid['offset'][:,1] + Ug_flex[self.model.strcgrid['set'][:,1]] * 100
                #z_f = self.model.strcgrid['offset'][:,2] + Ug_flex[self.model.strcgrid['set'][:,2]] * 100
                
                #mlab.figure()
                #mlab.points3d(x, y, z,  scale_factor=0.1)
                #mlab.points3d(x_f, y_f, z_f, color=(0,0,1), scale_factor=0.1)
                #mlab.title('flexible deformation', size=0.2, height=0.95)
    
                mlab.show()

        
    def cuttingforces(self):
        print 'calculating cutting forces & moments...'
        for i_trimcase in range(len(self.jcl.trimcase)):
            self.response[i_trimcase]['Pmon'] = self.model.PHIstrc_mon.T.dot(self.response[i_trimcase]['Pg'])
        
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
                monstation['loads'].append(self.response[i_trimcase]['Pmon'][self.model.mongrid['set'][i_station,:]])
                monstation['subcase'].append(self.jcl.trimcase[i_trimcase]['desc'])

            self.monstations['MON{:0>2s}'.format(str(int(self.model.mongrid['ID'][i_station])))] = monstation
            

    def save_monstations(self, filename):
        print 'saving monitoring stations as Nastarn cards...'
        with open(filename, 'w') as fid: 
            for i_trimcase in range(len(self.jcl.trimcase)):
                write_functions.write_force_and_moment_cards(fid, self.model.mongrid, self.response[i_trimcase]['Pmon'], i_trimcase+1)

    def save_nodalloads(self, filename):
        print 'saving nodal loads as Nastarn cards...'
        with open(filename, 'w') as fid: 
            for i_trimcase in range(len(self.jcl.trimcase)):
                write_functions.write_force_and_moment_cards(fid, self.model.strcgrid, self.response[i_trimcase]['Pg'], i_trimcase+1)
    
    def plot_monstations(self, monstations, filename_pdf):
        
        stations_to_plot = ['MON01', 'MON02', 'MON03', 'MON00']
        
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
            points = np.vstack((loads[i_station][:,2], loads[i_station][:,3])).T
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
            plt.xlabel('Fz [N]')
            plt.ylabel('Mx [Nm]')
            
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
            plt.suptitle('Subcase ' + str(monstations['MON00']['subcase'][i_case]))
            plt.subplots_adjust(left=0.1, right=0.95, top=0.90, bottom=0.1, wspace = 0.4, hspace = 0.3)
            
            #plt.show()
            pp.savefig()
            plt.close()
            
        pp.close()
        print 'saved as ' + filename_pdf
        print 'opening '+ filename_pdf
        os.system('evince ' + filename_pdf + ' &')
        return
        
        
        
        
        
        