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
import os, csv, getpass, time, platform, copy

from trim_tools import *
import write_functions
from grid_trafo import *

class post_processing:
    def __init__(self, jcl, model, response):
        self.jcl = jcl
        self.model = model
        self.response = response
    
    def force_summation_method(self):
        print 'calculating forces & moments on structural set (force summation method) and apply euler angles...'
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
            Mb         = self.model.mass['Mb'][i_mass]
            
            # Unterscheidung zwischen Trim und Zeit-Simulation, da die Dimensionen der response anders sind (n_step x n_value)
            if len(response['t']) > 1:
                response['Pg_iner_r']   = np.zeros((len(response['t']), 6*self.model.strcgrid['n']))
                response['Pg_iner_f']   = np.zeros((len(response['t']), 6*self.model.strcgrid['n']))
                response['Pg_aero']     = np.zeros((len(response['t']), 6*self.model.strcgrid['n']))
                response['Pg']          = np.zeros((len(response['t']), 6*self.model.strcgrid['n']))
                response['Pg_iner_global']   = np.zeros((len(response['t']), 6*self.model.strcgrid['n']))
                response['Pg_aero_global']   = np.zeros((len(response['t']), 6*self.model.strcgrid['n']))
                response['Pg_global']   = np.zeros((len(response['t']), 6*self.model.strcgrid['n']))
                response['Ug_f']        = np.zeros((len(response['t']), 6*self.model.strcgrid['n']))
                response['Ug_r']        = np.zeros((len(response['t']), 6*self.model.strcgrid['n']))
                response['Ug']          = np.zeros((len(response['t']), 6*self.model.strcgrid['n']))
                response['d2Ug_dt2_f']  = np.zeros((len(response['t']), 6*self.model.strcgrid['n']))
                response['d2Ug_dt2_r']  = np.zeros((len(response['t']), 6*self.model.strcgrid['n']))
                response['d2Ug_dt2']    = np.zeros((len(response['t']), 6*self.model.strcgrid['n']))
    
                for i_step in range(len(response['t'])):
                    if hasattr(self.jcl,'eom') and self.jcl.eom['version'] == 'waszak':
                        # Fuer Bewegungsgleichungen z.B. von Waszack muessen die zusaetzlichen Terme hier ebenfalls beruecksichtigt werden!
                        d2Ug_dt2_r = PHIstrc_cg.dot( np.hstack((response['d2Ucg_dt2'][i_step,0:3] - response['g_cg'][i_step,:] - np.cross(response['dUcg_dt'][i_step,0:3], response['dUcg_dt'][i_step,3:6]), 
                                                                response['d2Ucg_dt2'][i_step,3:6] ))  ) 
                    else:
                        d2Ug_dt2_r = PHIstrc_cg.dot( np.hstack((response['d2Ucg_dt2'][i_step,0:3] - response['g_cg'][i_step,:], response['d2Ucg_dt2'][i_step,3:6])) ) # Nastran
                    response['d2Ug_dt2_r'][i_step,:] = d2Ug_dt2_r
                    response['Pg_iner_r'][i_step,:] = - Mgg.dot(d2Ug_dt2_r)
                    response['Pg_aero'][i_step,:] = np.dot(self.model.PHIk_strc.T, response['Pk_aero'][i_step,:])
                    d2Ug_dt2_f = PHIf_strc.T.dot(response['d2Uf_dt2'][i_step,:])
                    response['d2Ug_dt2_f'][i_step,:] = d2Ug_dt2_f
                    response['Pg_iner_f'][i_step,:] = - Mgg.dot(d2Ug_dt2_f)
                    response['Pg'][i_step,:] = response['Pg_aero'][i_step,:] + response['Pg_iner_r'][i_step,:] + response['Pg_iner_f'][i_step,:]

                    # Including rotation about euler angles in calculation of Ug_r and Ug_f
                    # This is mainly done for plotting and time animation.
                    
                    # setting up coordinate system
                    coord_tmp = copy.deepcopy(self.model.coord)
                    coord_tmp['ID'].append(1000000)
                    coord_tmp['RID'].append(0)
                    coord_tmp['dircos'].append(PHIcg_norm[0:3,0:3].dot(calc_drehmatrix(response['X'][i_step,:][3], response['X'][i_step,:][4], response['X'][i_step,:][5])))
                    coord_tmp['offset'].append(response['X'][i_step,:][0:3] + np.array([0., 0., self.model.atmo['h'][i_atmo]])) # correction of height to zero to allow plotting in one diagram]))

                    # apply transformation to strcgrid
                    strcgrid_tmp = copy.deepcopy(self.model.strcgrid)
                    grid_trafo(strcgrid_tmp, coord_tmp, 1000000)
                    response['Ug_r'][i_step,self.model.strcgrid['set'][:,0]] = strcgrid_tmp['offset'][:,0] - self.model.strcgrid['offset'][:,0]
                    response['Ug_r'][i_step,self.model.strcgrid['set'][:,1]] = strcgrid_tmp['offset'][:,1] - self.model.strcgrid['offset'][:,1]
                    response['Ug_r'][i_step,self.model.strcgrid['set'][:,2]] = strcgrid_tmp['offset'][:,2] - self.model.strcgrid['offset'][:,2]
                    # apply transformation to flexible deformations vector
                    Uf = response['X'][i_step,:][12:12+n_modes]
                    Ug_f_body = np.dot(self.model.mass['PHIf_strc'][i_mass].T, Uf.T).T
                    strcgrid_tmp = copy.deepcopy(self.model.strcgrid)
                    strcgrid_tmp['CD'] = np.repeat(1000000, self.model.strcgrid['n'])
                    response['Ug_f'][i_step,:] = force_trafo(strcgrid_tmp, coord_tmp, Ug_f_body)
                    response['Pg_aero_global'][i_step,:] = force_trafo(strcgrid_tmp, coord_tmp, response['Pg_aero'][i_step,:])
                    response['Pg_iner_global'][i_step,:] = force_trafo(strcgrid_tmp, coord_tmp, response['Pg_iner_r'][i_step,:] + response['Pg_iner_f'][i_step,:])
                    response['Pg_global'][i_step,:] = force_trafo(strcgrid_tmp, coord_tmp, response['Pg'][i_step,:])
                    response['Ug'][i_step,:] = response['Ug_r'][i_step,:] + response['Ug_f'][i_step,:]
                    response['d2Ug_dt2'][i_step,:] = response['d2Ug_dt2_r'][i_step,:] + response['d2Ug_dt2_f'][i_step,:]
            else:
                if hasattr(self.jcl,'eom') and self.jcl.eom['version'] == 'waszak':
                    # Fuer Bewegungsgleichungen z.B. von Waszack muessen die zusaetzlichen Terme hier ebenfalls beruecksichtigt werden!
                    d2Ug_dt2_r = PHIstrc_cg.dot( np.hstack((response['d2Ucg_dt2'][0:3] - response['g_cg'] - np.cross(response['dUcg_dt'][0:3], response['dUcg_dt'][3:6]), 
                                                            response['d2Ucg_dt2'][3:6] ))  ) 
                else:
                    d2Ug_dt2_r = PHIstrc_cg.dot( np.hstack((response['d2Ucg_dt2'][0:3] - response['g_cg'], response['d2Ucg_dt2'][3:6])) ) # Nastran                
                response['Pg_iner_r'] = - Mgg.dot(d2Ug_dt2_r)
                response['d2Ug_dt2_r'] = d2Ug_dt2_r
                d2Ug_dt2_f = PHIf_strc.T.dot(response['d2Uf_dt2'])
                response['d2Ug_dt2_f'] = d2Ug_dt2_f
                response['Pg_iner_f'] = - Mgg.dot(d2Ug_dt2_f)
                #response['Ug_flex'] = PHIf_strc.T.dot(response['Uf'])
                #response['Pg_flex'] = self.model.Kgg.dot(response['Ug_flex']) * 0.0
                
                response['Pg_aero'] = np.dot(self.model.PHIk_strc.T, response['Pk_aero'])

                response['Pg'] = response['Pg_aero'] + response['Pg_iner_r'] + response['Pg_iner_f']
    
                # das muss raus kommen:
                #np.dot(self.model.mass['Mb'][i_mass],np.hstack((response['d2Ucg_dt2'][0:3] - response['g_cg'], response['d2Ucg_dt2'][3:6])))
                #PHIstrc_cg.T.dot(response['Pg_aero'])
                # das kommt raus:
                #PHIstrc_cg.T.dot(response['Pg_iner_r'])

                # Including rotation about euler angles in calculation of Ug_r and Ug_f
                # This is mainly done for plotting and time animation.
                
                # setting up coordinate system
                coord_tmp = copy.deepcopy(self.model.coord)
                coord_tmp['ID'].append(1000000)
                coord_tmp['RID'].append(0)
                coord_tmp['dircos'].append(PHIcg_norm[0:3,0:3].dot(calc_drehmatrix(response['X'][3], response['X'][4], response['X'][5])))
                coord_tmp['offset'].append(response['X'][0:3] + np.array([0., 0., self.model.atmo['h'][i_atmo]])) # correction of height to zero to allow plotting in one diagram]))
                # apply transformation to strcgrid
                strcgrid_tmp = copy.deepcopy(self.model.strcgrid)
                grid_trafo(strcgrid_tmp, coord_tmp, 1000000)
                response['Ug_r'] = np.zeros(response['Pg'].shape)
                response['Ug_r'][self.model.strcgrid['set'][:,0]] = strcgrid_tmp['offset'][:,0] - self.model.strcgrid['offset'][:,0]
                response['Ug_r'][self.model.strcgrid['set'][:,1]] = strcgrid_tmp['offset'][:,1] - self.model.strcgrid['offset'][:,1]
                response['Ug_r'][self.model.strcgrid['set'][:,2]] = strcgrid_tmp['offset'][:,2] - self.model.strcgrid['offset'][:,2]
                # apply transformation to flexible deformations and forces & moments vectors
                Uf = response['X'][12:12+n_modes]
                Ug_f_body = np.dot(self.model.mass['PHIf_strc'][i_mass].T, Uf.T).T
                strcgrid_tmp = copy.deepcopy(self.model.strcgrid)
                strcgrid_tmp['CD'] = np.repeat(1000000, self.model.strcgrid['n'])
                response['Ug_f'] = force_trafo(strcgrid_tmp, coord_tmp, Ug_f_body)
                response['Pg_aero_global'] = force_trafo(strcgrid_tmp, coord_tmp, response['Pg_aero'])
                response['Pg_iner_global'] = force_trafo(strcgrid_tmp, coord_tmp, response['Pg_iner_r'] + response['Pg_iner_f'])
                response['Pg_global'] = force_trafo(strcgrid_tmp, coord_tmp, response['Pg'])
#                 x = self.model.strcgrid['offset'][:,0] + response['Ug_r'][self.model.strcgrid['set'][:,0]]
#                 y = self.model.strcgrid['offset'][:,1] + response['Ug_r'][self.model.strcgrid['set'][:,1]]
#                 z = self.model.strcgrid['offset'][:,2] + response['Ug_r'][self.model.strcgrid['set'][:,2]]
#                 Ufx = response['Ug_f'][self.model.strcgrid['set'][:,0]] * 10.0
#                 Ufy = response['Ug_f'][self.model.strcgrid['set'][:,1]] * 10.0
#                 Ufz = response['Ug_f'][self.model.strcgrid['set'][:,2]] * 10.0
#                 from mayavi import mlab
#                 mlab.figure()
#                 mlab.points3d(x,y,z, scale_factor=0.2)
#                 mlab.points3d(x+Ufx,y+Ufy,z+Ufz, color=(0,0,1), scale_factor=0.2)
#                 mlab.show()
                
                response['Ug'] = response['Ug_r'] + response['Ug_f']
                response['d2Ug_dt2'] = response['d2Ug_dt2_r'] + response['d2Ug_dt2_f']
           
        
    def cuttingforces(self):
        print 'calculating cutting forces & moments...'
        for i_trimcase in range(len(self.jcl.trimcase)):        
            response = self.response[i_trimcase]
            # Unterscheidung zwischen Trim und Zeit-Simulation, da die Dimensionen der response anders sind (n_step x n_value)
            if len(response['t']) > 1:
                response['Pmon_global'] = np.zeros((len(response['t']), 6*self.model.mongrid['n']))
                response['Pmon_local'] = np.zeros((len(response['t']), 6*self.model.mongrid['n']))  
                for i_step in range(len(response['t'])):
                    response['Pmon_global'][i_step,:] = self.model.PHIstrc_mon.T.dot(response['Pg'][i_step,:])
                    response['Pmon_local'][i_step,:] = force_trafo(self.model.mongrid, self.model.coord, response['Pmon_global'][i_step,:])
            else:
                response['Pmon_global'] = self.model.PHIstrc_mon.T.dot(response['Pg'])
                response['Pmon_local'] = force_trafo(self.model.mongrid, self.model.coord, response['Pmon_global'])
        
    def gather_monstations(self):
        print 'gathering information on monitoring stations from respone(s)...'
        self.monstations = {}
        for i_station in range(self.model.mongrid['n']):
            monstation = {'CD': self.model.mongrid['CD'][i_station] ,
                          'CP': self.model.mongrid['CP'][i_station], 
                          'offset': self.model.mongrid['offset'][i_station], 
                          'subcase': [],  
                          'loads':[],
                          't':[]
                         }
            for i_trimcase in range(len(self.jcl.trimcase)):
                monstation['subcase'].append(self.jcl.trimcase[i_trimcase]['subcase'])
                monstation['t'].append(self.response[i_trimcase]['t'])
                response = self.response[i_trimcase]
                # Unterscheidung zwischen Trim und Zeit-Simulation, da die Dimensionen der response anders sind (n_step x n_value)
                if len(self.response[i_trimcase]['t']) > 1:
                    monstation['loads'].append(response['Pmon_local'][:,self.model.mongrid['set'][i_station,:]])
                else:
                    monstation['loads'].append(response['Pmon_local'][self.model.mongrid['set'][i_station,:]])
                
            if not self.model.mongrid.has_key('name'):
                name = 'MON{:s}'.format(str(int(self.model.mongrid['ID'][i_station]))) # make up a name
            else:
                name = self.model.mongrid['name'][i_station] # take name from mongrid
            self.monstations[name] = monstation
    
    def dyn2stat(self):
        print 'searching min/max of Fz/Mx/My in time data at {} monitoring stations and gathering loads (dyn2stat)...'.format(len(self.monstations.keys()))
        Pg_dyn2stat = []
        Pg_dyn2stat_desc = []  
        for key in self.monstations.keys():
            # Schnittlasten an den Monitoring Stationen raus schreiben zum Plotten
            # Knotenlasten raus schreiben
            loads_dyn2stat = []
            subcases_dyn2stat = []
            for i_case in range(len(self.monstations[key]['subcase'])):
                pos_max_loads_over_time = np.argmax(self.monstations[key]['loads'][i_case], 0)
                pos_min_loads_over_time = np.argmin(self.monstations[key]['loads'][i_case], 0)
                # Fz max und min
                loads_dyn2stat.append(self.monstations[key]['loads'][i_case][pos_max_loads_over_time[2],:])
                Pg_dyn2stat.append(self.response[i_case]['Pg'][pos_max_loads_over_time[2],:])
                subcases_dyn2stat.append(str(self.monstations[key]['subcase'][i_case]) + '_' + key + '_Fz_max')
                loads_dyn2stat.append(self.monstations[key]['loads'][i_case][pos_min_loads_over_time[2],:])
                Pg_dyn2stat.append(self.response[i_case]['Pg'][pos_min_loads_over_time[2],:])
                subcases_dyn2stat.append(str(self.monstations[key]['subcase'][i_case]) + '_' + key + '_Fz_min')
                # Mx max und min
                loads_dyn2stat.append(self.monstations[key]['loads'][i_case][pos_max_loads_over_time[3],:])
                Pg_dyn2stat.append(self.response[i_case]['Pg'][pos_max_loads_over_time[3],:])
                subcases_dyn2stat.append(str(self.monstations[key]['subcase'][i_case]) + '_' + key + '_Mx_max')
                loads_dyn2stat.append(self.monstations[key]['loads'][i_case][pos_min_loads_over_time[3],:])
                Pg_dyn2stat.append(self.response[i_case]['Pg'][pos_min_loads_over_time[3],:])
                subcases_dyn2stat.append(str(self.monstations[key]['subcase'][i_case]) + '_' + key + '_Mx_min')
                # My max und min
                loads_dyn2stat.append(self.monstations[key]['loads'][i_case][pos_max_loads_over_time[4],:])
                Pg_dyn2stat.append(self.response[i_case]['Pg'][pos_max_loads_over_time[4],:])
                subcases_dyn2stat.append(str(self.monstations[key]['subcase'][i_case]) + '_' + key + '_My_max')
                loads_dyn2stat.append(self.monstations[key]['loads'][i_case][pos_min_loads_over_time[4],:])
                Pg_dyn2stat.append(self.response[i_case]['Pg'][pos_min_loads_over_time[4],:])
                subcases_dyn2stat.append(str(self.monstations[key]['subcase'][i_case]) + '_' + key + '_My_min')
            self.monstations[key]['loads_dyn2stat'] = np.array(loads_dyn2stat)
            self.monstations[key]['subcases_dyn2stat'] = np.array(subcases_dyn2stat)
            Pg_dyn2stat_desc += subcases_dyn2stat
        # save dyn2stat
        self.dyn2stat = {'Pg': np.array(Pg_dyn2stat), 
                         'desc': Pg_dyn2stat_desc,
                        }
            
    def save_dyn2stat(self, filename):
        print 'saving dyn2stat nodal loads as Nastarn cards...'
        with open(filename+'_Pg_dyn2stat', 'w') as fid: 
            for i_case in range(len(self.dyn2stat['desc'])):
                write_functions.write_force_and_moment_cards(fid, self.model.strcgrid, self.dyn2stat['Pg'][i_case,:], int(self.dyn2stat['desc'][i_case].split('_')[0])*1000000+i_case)
        with open(filename+'_subcases_dyn2stat', 'w') as fid:         
            for i_case in range(len(self.dyn2stat['desc'])):
                write_functions.write_subcases(fid, int(self.dyn2stat['desc'][i_case].split('_')[0])*1000000+i_case, self.dyn2stat['desc'][i_case])


    def save_monstations(self, filename):
        print 'saving monitoring stations as Nastarn cards...'
        with open(filename, 'w') as fid: 
            for i_trimcase in range(len(self.jcl.trimcase)):
                write_functions.write_force_and_moment_cards(fid, self.model.mongrid, self.response[i_trimcase]['Pmon_local'], i_trimcase+1)
    
    def save_nodaldefo(self, filename):
        print 'saving nodal flexible deformations as dat file...'
        with open(filename+'_undeformed.dat', 'w') as fid:             
            np.savetxt(fid, np.hstack((self.model.strcgrid['ID'].reshape(-1,1), self.model.strcgrid['offset'])))
            #np.savetxt(fid, self.model.strcgrid['offset'])
        
        for i_trimcase in range(len(self.jcl.trimcase)):
            with open(filename+'_subcase_'+str(self.jcl.trimcase[i_trimcase]['subcase'])+'_Uf_x10.dat', 'w') as fid: 
                defo = np.hstack((self.model.strcgrid['ID'].reshape(-1,1), self.model.strcgrid['offset'] + self.response[i_trimcase]['Ug_f'][self.model.strcgrid['set'][:,0:3]] * 10.0 ))
                np.savetxt(fid, defo)
                #np.savetxt(fid, defo[:,1:4])
                
    def save_nodalloads(self, filename):
        print 'saving nodal loads as Nastarn cards...'
        with open(filename+'_Pg', 'w') as fid: 
            for i_trimcase in range(len(self.jcl.trimcase)):
                write_functions.write_force_and_moment_cards(fid, self.model.strcgrid, self.response[i_trimcase]['Pg'], self.jcl.trimcase[i_trimcase]['subcase'])
        with open(filename+'_subcases', 'w') as fid:         
            for i_trimcase in range(len(self.jcl.trimcase)):
                write_functions.write_subcases(fid, self.jcl.trimcase[i_trimcase]['subcase'], self.jcl.trimcase[i_trimcase]['desc'])
#        with open(filename+'_Pg_aero', 'w') as fid: 
#            for i_trimcase in range(len(self.jcl.trimcase)):
#                write_functions.write_force_and_moment_cards(fid, self.model.strcgrid, self.response[i_trimcase]['Pg_aero'], self.jcl.trimcase[i_trimcase]['subcase'])
#        with open(filename+'_Pg_iner_r', 'w') as fid: 
#            for i_trimcase in range(len(self.jcl.trimcase)):
#                write_functions.write_force_and_moment_cards(fid, self.model.strcgrid, self.response[i_trimcase]['Pg_iner_r'], self.jcl.trimcase[i_trimcase]['subcase']) 
#        with open(filename+'_Pg_iner_f', 'w') as fid: 
#            for i_trimcase in range(len(self.jcl.trimcase)):
#                write_functions.write_force_and_moment_cards(fid, self.model.strcgrid, self.response[i_trimcase]['Pg_iner_f'], self.jcl.trimcase[i_trimcase]['subcase'])
    
    def save_cpacs_header(self):
        
        self.cf.addElem('/cpacs/header', 'name', self.jcl.general['aircraft'], 'text')
        self.cf.addElem('/cpacs/header', 'creator', getpass.getuser() + ' on ' + platform.node() + ' (' + platform.platform() +')', 'text')
        self.cf.addElem('/cpacs/header', 'description', 'This is a file generated by Loads Kernel.', 'text')
        self.cf.addElem('/cpacs/header', 'timestamp', time.strftime("%Y-%m-%d %H:%M", time.localtime()), 'text' )
        
    def save_cpacs_flightLoadCases(self):
        # create flighLoadCases
        self.cf.createPath('/cpacs/vehicles/aircraft/model/analysis', 'loadAnalysis/loadCases/flightLoadCases')
        path_flightLoadCases = '/cpacs/vehicles/aircraft/model/analysis/loadAnalysis/loadCases/flightLoadCases'
        
        # create nodal + cut loads for each trim case
        for i_trimcase in range(len(self.jcl.trimcase)):
            # info on current load case
            self.tixi.createElement(path_flightLoadCases, 'flightLoadCase')
            self.cf.addElem(path_flightLoadCases+'/flightLoadCase['+str(i_trimcase+1)+']', 'name',  'subcase ' + str(self.jcl.trimcase[i_trimcase]['subcase']), 'double')
            self.cf.addElem(path_flightLoadCases+'/flightLoadCase['+str(i_trimcase+1)+']', 'uID', self.jcl.trimcase[i_trimcase]['desc'], 'text')  
            desc_string = 'calculated with Loads Kernel on ' + time.strftime("%Y-%m-%d %H:%M", time.localtime()) + ' by ' + getpass.getuser() + ' on ' + platform.node() + ' (' + platform.platform() +')'
            self.cf.addElem(path_flightLoadCases+'/flightLoadCase['+str(i_trimcase+1)+']', 'description', desc_string  , 'text')
            # nodal loads
            self.cf.createPath(path_flightLoadCases+'/flightLoadCase['+str(i_trimcase+1)+']', 'nodalLoads/wingNodalLoad')
            path_nodalLoads =       path_flightLoadCases+'/flightLoadCase['+str(i_trimcase+1)+']'+'/nodalLoads/wingNodalLoad'
            self.cf.addElem(path_nodalLoads, 'parentUID', 'complete aircraft', 'text')
            self.cf.write_cpacs_loadsvector(path_nodalLoads, self.model.strcgrid, self.response[i_trimcase]['Pg'] )
            # cut loads
            self.cf.createPath(path_flightLoadCases+'/flightLoadCase['+str(i_trimcase+1)+']', 'cutLoads/wingCutLoad')
            path_cutLoads =         path_flightLoadCases+'/flightLoadCase['+str(i_trimcase+1)+']'+'/cutLoads/wingCutLoad'
            self.cf.addElem(path_cutLoads, 'parentUID', 'complete aircraft', 'text')
            self.cf.write_cpacs_loadsvector(path_cutLoads, self.model.mongrid, self.response[i_trimcase]['Pmon_local'])
    
    def save_cpacs_dynamicAircraftModelPoints(self):
        # save structural grid points to CPACS
        self.cf.createPath('/cpacs/vehicles/aircraft/model/wings/wing/dynamicAircraftModel', 'dynamicAircraftModelPoints')
        path_dynamicAircraftModelPoints = '/cpacs/vehicles/aircraft/model/wings/wing/dynamicAircraftModel/dynamicAircraftModelPoints'
        self.cf.write_cpacs_grid(path_dynamicAircraftModelPoints, self.model.strcgrid)
        
    def save_cpacs_CutLoadIntegrationPoints(self):
        # save monitoring stations to CPACS
        self.cf.createPath('/cpacs/vehicles/aircraft/model/wings/wing/dynamicAircraftModel', 'cutLoadIntegrationPoints')
        path_cutLoadIntegrationPoints = '/cpacs/vehicles/aircraft/model/wings/wing/dynamicAircraftModel/cutLoadIntegrationPoints'
        self.cf.write_cpacs_grid(path_cutLoadIntegrationPoints, self.model.mongrid)
        #self.cf.write_cpacs_grid_orientation(path_CutLoadIntegrationPoints, self.model.mongrid, self.model.coord)
            
        
    def save_cpacs(self, filename):
        print 'saving nodal loads and monitoring stations as CPACS...'
        from tixiwrapper import Tixi
        self.tixi = Tixi()
        self.tixi.create('cpacs')
        self.cf = write_functions.cpacs_functions(self.tixi)
        
        # These paths might already exist when writing into a given CPACS-file...        
        self.cf.createPath('/cpacs', 'header')
        self.save_cpacs_header()
        
        self.cf.createPath('/cpacs', 'vehicles/aircraft/model/analysis') 
        self.cf.createPath('/cpacs/vehicles/aircraft/model', 'wings/wing/dynamicAircraftModel')
        self.tixi.addTextAttribute('/cpacs/vehicles/aircraft/model/wings/wing', 'uID', 'complete aircraft')
        self.tixi.addTextElement('/cpacs/vehicles/aircraft/model/wings/wing', 'description', 'complete aircraft as used in Loads Kernel - without distinction of components' )
        
        self.save_cpacs_dynamicAircraftModelPoints()
        self.save_cpacs_CutLoadIntegrationPoints()
        self.save_cpacs_flightLoadCases()

        self.tixi.save(filename)
        
        
        