
import write_functions
import numpy as np
import copy, getpass, platform, time, logging, csv
from grid_trafo import *

class auxiliary_output:
    #===========================================================================
    # This class provides functions to save data of trim calculations. 
    #===========================================================================
    def __init__(self, jcl, model, trimcase):
        self.jcl = jcl
        self.model = model
        self.trimcase = trimcase
 
    def save_nodaldefo(self, filename):
        # deformations are given in 9300 coord
        strcgrid_tmp = copy.deepcopy(self.model.strcgrid)
        grid_trafo(strcgrid_tmp, self.model.coord, 9300)
        logging.info( 'saving nodal flexible deformations as dat file...')
        with open(filename+'_undeformed.dat', 'w') as fid:             
            np.savetxt(fid, np.hstack((self.model.strcgrid['ID'].reshape(-1,1), strcgrid_tmp['offset'])))
        
        for i_trimcase in range(len(self.jcl.trimcase)):
            with open(filename+'_subcase_'+str(self.jcl.trimcase[i_trimcase]['subcase'])+'_Ug.dat', 'w') as fid: 
                defo = np.hstack((self.model.strcgrid['ID'].reshape(-1,1), self.model.strcgrid['offset'] + self.response[i_trimcase]['Ug_r'][self.model.strcgrid['set'][:,0:3]] + + self.response[i_trimcase]['Ug_f'][self.model.strcgrid['set'][:,0:3]] * 500.0))
                np.savetxt(fid, defo)
                
    def write_all_nodalloads(self, filename):
        logging.info( 'saving all nodal loads as Nastarn cards...')
        with open(filename+'_Pg', 'w') as fid: 
            for i_trimcase in range(len(self.jcl.trimcase)):
                write_functions.write_force_and_moment_cards(fid, self.model.strcgrid, self.response[i_trimcase]['Pg'], self.jcl.trimcase[i_trimcase]['subcase'])
        with open(filename+'_subcases', 'w') as fid:         
            for i_trimcase in range(len(self.jcl.trimcase)):
                write_functions.write_subcases(fid, self.jcl.trimcase[i_trimcase]['subcase'], self.jcl.trimcase[i_trimcase]['desc'])
    
    
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
    
    def write_critical_nodalloads(self, filename, dyn2stat=False): 
        logging.info( 'saving critical nodal loads as Nastarn cards...')
        if dyn2stat:
            # This is quite a complicated sorting because the subcases from dyn2stat may contain non-numeric characters. 
            # A "normal" sorting returns an undesired sequence, leading IDs in a non-ascending sequence. This a not allowed by Nastran. 
            subcases_IDs = [self.dyn2stat_data['subcases_ID'][self.dyn2stat_data['subcases'].index(crit_trimcase)] for crit_trimcase in np.unique(self.crit_trimcases) ]
            subcases_IDs = np.sort(subcases_IDs)
            with open(filename+'_Pg', 'w') as fid: 
                for subcase_ID in subcases_IDs:
                    idx = self.dyn2stat_data['subcases_ID'].index(subcase_ID)
                    write_functions.write_force_and_moment_cards(fid, self.model.strcgrid, self.dyn2stat_data['Pg'][idx], self.dyn2stat_data['subcases_ID'][idx])
            with open(filename+'_subcases', 'w') as fid:  
                for subcase_ID in subcases_IDs:
                    idx = self.dyn2stat_data['subcases_ID'].index(subcase_ID)
                    write_functions.write_subcases(fid, self.dyn2stat_data['subcases_ID'][idx], self.dyn2stat_data['subcases'][idx])
        else:
            crit_trimcases = self.crit_trimcases
            with open(filename+'_Pg', 'w') as fid: 
                for i_case in range(len(self.jcl.trimcase)):
                    if self.jcl.trimcase[i_case]['subcase'] in crit_trimcases:
                        write_functions.write_force_and_moment_cards(fid, self.model.strcgrid, self.response[i_case]['Pg'], self.jcl.trimcase[i_case]['subcase'])
            with open(filename+'_subcases', 'w') as fid:         
                for i_case in range(len(self.jcl.trimcase)):
                    if self.jcl.trimcase[i_case]['subcase'] in crit_trimcases:
                        write_functions.write_subcases(fid, self.jcl.trimcase[i_case]['subcase'], self.jcl.trimcase[i_case]['desc'])
    
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
        logging.info( 'saving nodal loads and monitoring stations as CPACS...')
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
        