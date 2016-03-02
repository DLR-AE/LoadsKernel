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
            
            # Unterscheidung zwischen Trim und Zeit-Simulation, da die Dimensionen der response anders sind (n_step x n_value)
            if len(response['t']) > 1:
                response['Pg_iner_r']   = np.zeros((len(response['t']), 6*self.model.strcgrid['n']))
                response['Pg_iner_f']   = np.zeros((len(response['t']), 6*self.model.strcgrid['n']))
                response['Pg_aero']     = np.zeros((len(response['t']), 6*self.model.strcgrid['n']))
                response['Pg']          = np.zeros((len(response['t']), 6*self.model.strcgrid['n']))
                response['Ug_f']        = np.zeros((len(response['t']), 6*self.model.strcgrid['n']))
                response['Ug_r']        = np.zeros((len(response['t']), 6*self.model.strcgrid['n']))
                response['Ug']          = np.zeros((len(response['t']), 6*self.model.strcgrid['n']))
    
                for i_step in range(len(response['t'])):
                    # Formel bezogen auf die linearen Bewegungsgleichungen Nastrans. 
                    # Fuer Bewegungsgleichungen z.B. von Waszack muessen die zusaetzlichen Terme hier ebenfalls beruecksichtigt werden!
                    d2Ug_dt2_r = PHIstrc_cg.dot( np.hstack((response['d2Ucg_dt2'][i_step,0:3] - response['g_cg'][i_step,:], response['d2Ucg_dt2'][i_step,3:6])) )
                    response['Pg_iner_r'][i_step,:] = - Mgg.dot(d2Ug_dt2_r)
                    response['Pg_aero'][i_step,:] = np.dot(self.model.PHIk_strc.T, response['Pk_aero'][i_step,:])
                    d2Ug_dt2_f = PHIf_strc.T.dot(response['d2Uf_dt2'][i_step,:])
                    response['Pg_iner_f'][i_step,:] = - Mgg.dot(d2Ug_dt2_f)
                    response['Pg'][i_step,:] = response['Pg_aero'][i_step,:] + response['Pg_iner_r'][i_step,:] + response['Pg_iner_f'][i_step,:]
                    
                    Uf = response['X'][i_step,12:12+n_modes]
                    response['Ug_f'][i_step,:] = np.dot(self.model.mass['PHIf_strc'][i_mass].T, Uf.T).T
                    
                    Tgeo2body = np.zeros((6,6))
                    Tgeo2body[0:3,0:3] = calc_drehmatrix(response['X'][i_step,3], response['X'][i_step,4], response['X'][i_step,5])
                    Tgeo2body[3:6,3:6] = calc_drehmatrix(response['X'][i_step,3], response['X'][i_step,4], response['X'][i_step,5])
                    height = self.model.atmo['h'][i_atmo] # correction of height to zero to allow plotting in one diagram
                    response['Ug_r'][i_step,:] = PHIstrc_cg.dot( np.dot(PHIcg_norm, response['X'][i_step,0:6]+[0,0,height,0,0,0]) )
                    
                    response['Ug'][i_step,:] = response['Ug_r'][i_step,:] + response['Ug_f'][i_step,:]
            else:
                # Formel bezogen auf die linearen Bewegungsgleichungen Nastrans. 
                # Fuer Bewegungsgleichungen z.B. von Waszack muessen die zusaetzlichen Terme hier ebenfalls beruecksichtigt werden!
                d2Ug_dt2_r = PHIstrc_cg.dot( np.hstack((response['d2Ucg_dt2'][0:3] - response['g_cg'], response['d2Ucg_dt2'][3:6])) )
                response['Pg_iner_r'] = - Mgg.dot(d2Ug_dt2_r)
                
                d2Ug_dt2_f = PHIf_strc.T.dot(response['d2Uf_dt2'])
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
            response = self.response[i_trimcase]
            # Unterscheidung zwischen Trim und Zeit-Simulation, da die Dimensionen der response anders sind (n_step x n_value)
            if len(response['t']) > 1:
                response['Pmon_global'] = np.zeros((len(response['t']), 6*self.model.mongrid['n']))
                response['Pmon_local'] = np.zeros((len(response['t']), 6*self.model.mongrid['n']))  
                for i_step in range(len(response['t'])):
                    response['Pmon_global'][i_step,:] = self.model.PHIstrc_mon.T.dot(response['Pg'][i_step,:])
                    response['Pmon_local'][i_step,:] = force_trafo(self.model.mongrid, self.model.coord, response['Pmon_global'][i_step,:])
            else:
                response = self.response[i_trimcase]
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
            with open(filename+'_subcase_'+str(self.jcl.trimcase[i_trimcase]['subcase'])+'_Uf_x10.dat', 'w') as fid: 
                defo = np.hstack((self.model.strcgrid['ID'].reshape(-1,1), self.model.strcgrid['offset'] + self.response[i_trimcase]['Ug_f'][self.model.strcgrid['set'][:,0:3]] * 10.0 ))
                np.savetxt(fid, defo)
                #np.savetxt(fid, defo[:,1:4])
                
    def save_nodalloads(self, filename):
        print 'saving nodal loads as Nastarn cards...'
        with open(filename+'_Pg', 'w') as fid: 
            for i_trimcase in range(len(self.jcl.trimcase)):
                write_functions.write_force_and_moment_cards(fid, self.model.strcgrid, self.response[i_trimcase]['Pg'], self.jcl.trimcase[i_trimcase]['subcase'])
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
        self.cpacs_functions.addElem('/cpacs/header', 'name', self.jcl.general['aircraft'], 'text')
        self.cpacs_functions.addElem('/cpacs/header', 'creator', 'Loads Kernel', 'text')
        self.cpacs_functions.addElem('/cpacs/header', 'description', 'This is a file generated by Loads Kernel.', 'text')
        
    def save_cpacs_flightLoadCases(self):
        # create flighLoadCases
        self.cpacs_functions.createPath('/cpacs/vehicles/aircraft/model/analysis', 'loadAnalysis/loadCases/flightLoadCases')
        path_flightLoadCases = '/cpacs/vehicles/aircraft/model/analysis/loadAnalysis/loadCases/flightLoadCases'
        
        # create nodal + cut loads for each trim case
        for i_trimcase in range(len(self.jcl.trimcase)):
            # nodal loads
            self.tixi.createElement(path_flightLoadCases, 'flightLoadCase')
            self.tixi.createElement(path_flightLoadCases+'/flightLoadCase['+str(i_trimcase+1)+']', 'nodalLoads')
            path_nodalLoads =       path_flightLoadCases+'/flightLoadCase['+str(i_trimcase+1)+']'+'/nodalLoads'
            self.cpacs_functions.write_cpacs_loadsvector(path_nodalLoads, self.model.strcgrid, self.response[i_trimcase]['Pg'], self.jcl.trimcase[i_trimcase]['subcase'], self.jcl.trimcase[i_trimcase]['desc'])
            # cut loads
            self.tixi.createElement(path_flightLoadCases+'/flightLoadCase['+str(i_trimcase+1)+']', 'cutLoads')
            path_cutLoads =         path_flightLoadCases+'/flightLoadCase['+str(i_trimcase+1)+']'+'/cutLoads'
            self.cpacs_functions.write_cpacs_loadsvector(path_cutLoads, self.model.mongrid, self.response[i_trimcase]['Pmon_local'], self.jcl.trimcase[i_trimcase]['subcase'], self.jcl.trimcase[i_trimcase]['desc'])
    
    def save_cpacs_dynamicAircraftModelPoints(self):
        # save structural grid points to CPACS
        self.cpacs_functions.createPath('/cpacs/vehicles/aircraft/model/wings/wing/dynamicAircraftModel', 'dynamicAircraftModelPoints')
        path_dynamicAircraftModelPoints = '/cpacs/vehicles/aircraft/model/wings/wing/dynamicAircraftModel/dynamicAircraftModelPoints'
        self.cpacs_functions.write_cpacs_grid(path_dynamicAircraftModelPoints, self.model.strcgrid)
        
    def save_cpacs_CutLoadIntegrationPoints(self):
        # save monitoring stations to CPACS
        self.cpacs_functions.createPath('/cpacs/vehicles/aircraft/model/wings/wing/dynamicAircraftModel', 'CutLoadIntegrationPoints')
        path_CutLoadIntegrationPoints = '/cpacs/vehicles/aircraft/model/wings/wing/dynamicAircraftModel/CutLoadIntegrationPoints'
        self.cpacs_functions.write_cpacs_grid(path_CutLoadIntegrationPoints, self.model.mongrid)
        #self.cpacs_functions.write_cpacs_grid_orientation(path_CutLoadIntegrationPoints, self.model.mongrid, self.model.coord)
            
        
    def save_cpacs(self, filename):
        print 'saving nodal loads and monitoring stations as CPACS...'
        from tixiwrapper import Tixi
        self.tixi = Tixi()
        self.tixi.create('cpacs')
        self.cpacs_functions = write_functions.cpacs_functions(self.tixi)
        
        # These paths might already exist when writing into a given CPACS-file...        
        self.cpacs_functions.createPath('/cpacs', 'header')
        self.cpacs_functions.createPath('/cpacs', 'vehicles/aircraft/model/analysis') 
        self.cpacs_functions.createPath('/cpacs/vehicles/aircraft/model', 'wings/wing/dynamicAircraftModel')
        
        self.save_cpacs_header()
        self.save_cpacs_dynamicAircraftModelPoints()
        self.save_cpacs_CutLoadIntegrationPoints()
        self.save_cpacs_flightLoadCases()

        self.tixi.save(filename)
        
        
        