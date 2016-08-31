
import write_functions
import numpy as np
import copy, getpass, platform, time
from grid_trafo import *

class auxiliary_output:
    #===========================================================================
    # This class provides functions to save data of trim calculations. 
    #===========================================================================
    def __init__(self, jcl, model, trimcase, response):
        self.jcl = jcl
        self.model = model
        self.trimcase = trimcase
        self.response = response
 
    def save_nodaldefo(self, filename):
        # deformations are given in 9300 coord
        strcgrid_tmp = copy.deepcopy(self.model.strcgrid)
        grid_trafo(strcgrid_tmp, self.model.coord, 9300)
        print 'saving nodal flexible deformations as dat file...'
        with open(filename+'_undeformed.dat', 'w') as fid:             
            np.savetxt(fid, np.hstack((self.model.strcgrid['ID'].reshape(-1,1), strcgrid_tmp['offset'])))
        
        for i_trimcase in range(len(self.jcl.trimcase)):
            with open(filename+'_subcase_'+str(self.jcl.trimcase[i_trimcase]['subcase'])+'_Ug.dat', 'w') as fid: 
                defo = np.hstack((self.model.strcgrid['ID'].reshape(-1,1), self.model.strcgrid['offset'] + self.response[i_trimcase]['Ug_r'][self.model.strcgrid['set'][:,0:3]] + + self.response[i_trimcase]['Ug_f'][self.model.strcgrid['set'][:,0:3]] * 500.0))
                np.savetxt(fid, defo)
                
    def save_nodalloads(self, filename):
        print 'saving nodal loads as Nastarn cards...'
        with open(filename+'_Pg', 'w') as fid: 
            for i_trimcase in range(len(self.jcl.trimcase)):
                write_functions.write_force_and_moment_cards(fid, self.model.strcgrid, self.response[i_trimcase]['Pg'], self.jcl.trimcase[i_trimcase]['subcase'])
        with open(filename+'_subcases', 'w') as fid:         
            for i_trimcase in range(len(self.jcl.trimcase)):
                write_functions.write_subcases(fid, self.jcl.trimcase[i_trimcase]['subcase'], self.jcl.trimcase[i_trimcase]['desc'])
    
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
        