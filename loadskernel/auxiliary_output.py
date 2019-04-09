
import io_functions.specific_functions
import io_functions.nastran_functions
import numpy as np
import copy, getpass, platform, time, logging, csv
from grid_trafo import *

class AuxiliaryOutput:
    #===========================================================================
    # This class provides functions to save data of trim calculations. 
    #===========================================================================
    def __init__(self, jcl, model, trimcase):
        self.jcl = jcl
        self.model = model
        self.trimcase = trimcase
        self.responses = []
 
    def save_nodaldefo(self, filename):
        # deformations are given in 9300 coord
        strcgrid_tmp = copy.deepcopy(self.model.strcgrid)
        grid_trafo(strcgrid_tmp, self.model.coord, 9300)
        logging.info( 'saving nodal flexible deformations as dat file...')
        with open(filename+'_undeformed.dat', 'w') as fid:             
            np.savetxt(fid, np.hstack((self.model.strcgrid['ID'].reshape(-1,1), strcgrid_tmp['offset'])))
        
        for i_trimcase in range(len(self.jcl.trimcase)):
            with open(filename+'_subcase_'+str(self.jcl.trimcase[i_trimcase]['subcase'])+'_Ug.dat', 'w') as fid: 
                defo = np.hstack((self.model.strcgrid['ID'].reshape(-1,1), self.model.strcgrid['offset'] + self.responses[i_trimcase]['Ug_r'][self.model.strcgrid['set'][:,0:3]] + + self.responses[i_trimcase]['Ug_f'][self.model.strcgrid['set'][:,0:3]] * 500.0))
                np.savetxt(fid, defo)
                
    def write_all_nodalloads(self, filename):
        logging.info( 'saving all nodal loads as Nastarn cards...')
        with open(filename+'_Pg', 'w') as fid: 
            for i_trimcase in range(len(self.jcl.trimcase)):
                io_functions.nastran_functions.write_force_and_moment_cards(fid, self.model.strcgrid, self.responses[i_trimcase]['Pg'], self.jcl.trimcase[i_trimcase]['subcase'])
        with open(filename+'_subcases', 'w') as fid:         
            for i_trimcase in range(len(self.jcl.trimcase)):
                io_functions.nastran_functions.write_subcases(fid, self.jcl.trimcase[i_trimcase]['subcase'], self.jcl.trimcase[i_trimcase]['desc'])
    
    def write_trimresults(self, filename):
        trimresults = []
        for i_case in range(len(self.jcl.trimcase)):
            trimresult = self.assemble_trimresult(i_case)
            if trimresult != False:
                trimresults.append(trimresult)
            
    def assemble_trimresult(self, i_case):
        response = self.responses[i_case]
        if response['successful']:
            trimresult = {'subcase':  self.jcl.trimcase[i_case]['subcase'],
                          'desc':     self.jcl.trimcase[i_case]['desc'],}
            i_mass  = self.model.mass['key'].index(self.jcl.trimcase[i_case]['mass'])
            n_modes = self.model.mass['n_modes'][i_mass]

            # get trimmed states
            trimresult['u'] = response['X'][6]
            trimresult['v'] = response['X'][7]
            trimresult['w'] = response['X'][8]
            trimresult['p'] = response['X'][9]
            trimresult['q'] = response['X'][10]
            trimresult['r'] = response['X'][11]
            trimresult['du'] = response['Y'][6]
            trimresult['dv'] = response['Y'][7]
            trimresult['dw'] = response['Y'][8]
            trimresult['dp'] = response['Y'][9]
            trimresult['dq'] = response['Y'][10]
            trimresult['dr'] = response['Y'][11]
            trimresult['command_xi [deg]']   = response['X'][12+2*n_modes]/np.pi*180.0
            trimresult['command_eta [deg]']  = response['X'][13+2*n_modes]/np.pi*180.0
            trimresult['command_zeta [deg]'] = response['X'][14+2*n_modes]/np.pi*180.0
            trimresult['Nz'] = response['Y'][12+2*n_modes+3]
            trimresult['Vtas'] = response['Y'][12+2*n_modes+4]
            trimresult['q_dyn'] = response['q_dyn']
            trimresult['alpha [deg]'] = response['alpha']/np.pi*180.0
            trimresult['beta [deg]'] = response['beta']/np.pi*180.0
            
            # calculate additional aero coefficients
            Pmac_rbm  = np.dot(self.model.Dkx1.T, response['Pk_rbm'])
            Pmac_cam  = np.dot(self.model.Dkx1.T, response['Pk_cam'])
            Pmac_cs   = np.dot(self.model.Dkx1.T, response['Pk_cs'])
            Pmac_f    = np.dot(self.model.Dkx1.T, response['Pk_f'])
            Pmac_idrag = np.dot(self.model.Dkx1.T, response['Pk_idrag'])
            A = self.jcl.general['A_ref'] #sum(self.model.aerogrid['A'][:])
            AR = self.jcl.general['b_ref']**2.0 / self.jcl.general['A_ref']
            Pmac_c = response['Pmac']/response['q_dyn']/A
            # um alpha drehen, um Cl und Cd zu erhalten
            Cl = Pmac_c[2]*np.cos(response['alpha'])+Pmac_c[0]*np.sin(response['alpha'])
            Cd = Pmac_c[2]*np.sin(response['alpha'])+Pmac_c[0]*np.cos(response['alpha'])
            Cd_ind_theo = Cl**2.0/np.pi/AR
            
            trimresult['Cz_rbm'] = Pmac_rbm[2]/response['q_dyn']/A
            trimresult['Cz_cam'] = Pmac_cam[2]/response['q_dyn']/A
            trimresult['Cz_cs'] = Pmac_cs[2]/response['q_dyn']/A
            trimresult['Cz_f'] = Pmac_f[2]/response['q_dyn']/A
            trimresult['Cx'] = Pmac_c[0]
            trimresult['Cy'] = Pmac_c[1]
            trimresult['Cz'] = Pmac_c[2]
            trimresult['Cmx'] = Pmac_c[3]/self.model.macgrid['b_ref']
            trimresult['Cmy'] = Pmac_c[4]/self.model.macgrid['c_ref']
            trimresult['Cmz'] = Pmac_c[5]/self.model.macgrid['b_ref']
            trimresult['Cl'] = Cl
            trimresult['Cd'] = Cd
            trimresult['E'] = Cl/Cd
            trimresult['Cd_ind'] = Pmac_idrag[0]/response['q_dyn']/A
            trimresult['Cmz_ind'] = Pmac_idrag[5]/response['q_dyn']/A/self.model.macgrid['b_ref']
            trimresult['e'] = Cd_ind_theo/(Pmac_idrag[0]/response['q_dyn']/A)
        else:
            trimresult = False
        return trimresult   

    def write_successful_trimcases(self, filename_csv):
        sucessfull_trimcases_info = []
        for i_case in range(len(self.jcl.trimcase)):
            trimcase = {'subcase':  self.jcl.trimcase[i_case]['subcase'],
                        'desc':     self.jcl.trimcase[i_case]['desc'],}
            if self.responses[i_case]['successful']:
                sucessfull_trimcases_info.append(trimcase)
        logging.info('writing successful trimcases cases to: ' + filename_csv)
        io_functions.specific_functions.write_dictionary(sucessfull_trimcases_info, filename_csv)
        
    def write_failed_trimcases(self, filename_csv):
        failed_trimcases_info = []
        for i_case in range(len(self.jcl.trimcase)):
            trimcase = {'subcase':  self.jcl.trimcase[i_case]['subcase'],
                        'desc':     self.jcl.trimcase[i_case]['desc'],}
            if not self.responses[i_case]['successful']:
                failed_trimcases_info.append(trimcase)
        logging.info('writing failed trimcases cases to: ' + filename_csv)
        io_functions.specific_functions.write_dictionary(failed_trimcases_info, filename_csv)
    
    def write_critical_trimcases(self, filename_csv, dyn2stat=False):
        # eigentlich gehoert diese Funtion eher zum post-processing als zum
        # plotten, kann aber erst nach dem plotten ausgefuehrt werden...
        if dyn2stat:
            crit_trimcases = list(set([int(str(crit_trimcase).split('_')[0]) for crit_trimcase in self.crit_trimcases])) # extract original subcase number
        else: 
            crit_trimcases = self.crit_trimcases
        crit_trimcases_info = []
        for i_case in range(len(self.jcl.trimcase)):
            if self.jcl.trimcase[i_case]['subcase'] in crit_trimcases:
                trimcase = {'subcase':  self.jcl.trimcase[i_case]['subcase'],
                            'desc':     self.jcl.trimcase[i_case]['desc'],}
#                 does not work if maneuver and time simulations are handled simultaneously
#                 trimcase = copy.deepcopy(self.jcl.trimcase[i_case])
#                 if dyn2stat:
#                     trimcase.update(self.jcl.simcase[i_case]) # merge infos from simcase with trimcase
                crit_trimcases_info.append(trimcase)
                
        logging.info('writing critical trimcases cases to: ' + filename_csv)
        io_functions.specific_functions.write_dictionary(crit_trimcases_info, filename_csv)
    
    def write_critical_nodalloads(self, filename, dyn2stat=False): 
        logging.info( 'saving critical nodal loads as Nastarn cards...')
        if dyn2stat:
            # This is quite a complicated sorting because the subcases from dyn2stat may contain non-numeric characters. 
            # A "normal" sorting returns an undesired sequence, leading IDs in a non-ascending sequence. This a not allowed by Nastran. 
            subcases_IDs = [self.dyn2stat_data['subcases_ID'][self.dyn2stat_data['subcases'].index(str(crit_trimcase))] for crit_trimcase in np.unique(self.crit_trimcases) ]
            subcases_IDs = np.sort(subcases_IDs)
            with open(filename+'_Pg', 'w') as fid: 
                for subcase_ID in subcases_IDs:
                    idx = self.dyn2stat_data['subcases_ID'].index(subcase_ID)
                    io_functions.nastran_functions.write_force_and_moment_cards(fid, self.model.strcgrid, self.dyn2stat_data['Pg'][idx], self.dyn2stat_data['subcases_ID'][idx])
            with open(filename+'_subcases', 'w') as fid:  
                for subcase_ID in subcases_IDs:
                    idx = self.dyn2stat_data['subcases_ID'].index(subcase_ID)
                    io_functions.nastran_functions.write_subcases(fid, self.dyn2stat_data['subcases_ID'][idx], self.dyn2stat_data['subcases'][idx])
        else:
            crit_trimcases = self.crit_trimcases
            with open(filename+'_Pg', 'w') as fid: 
                for i_case in range(len(self.jcl.trimcase)):
                    if self.jcl.trimcase[i_case]['subcase'] in crit_trimcases:
                        io_functions.nastran_functions.write_force_and_moment_cards(fid, self.model.strcgrid, self.responses[i_case]['Pg'], self.jcl.trimcase[i_case]['subcase'])
            with open(filename+'_subcases', 'w') as fid:         
                for i_case in range(len(self.jcl.trimcase)):
                    if self.jcl.trimcase[i_case]['subcase'] in crit_trimcases:
                        io_functions.nastran_functions.write_subcases(fid, self.jcl.trimcase[i_case]['subcase'], self.jcl.trimcase[i_case]['desc'])
    
    def save_cpacs_header(self):
        
        self.cf.add_elem('/cpacs/header', 'name', self.jcl.general['aircraft'], 'text')
        self.cf.add_elem('/cpacs/header', 'creator', getpass.getuser() + ' on ' + platform.node() + ' (' + platform.platform() +')', 'text')
        self.cf.add_elem('/cpacs/header', 'description', 'This is a file generated by Loads Kernel.', 'text')
        self.cf.add_elem('/cpacs/header', 'timestamp', time.strftime("%Y-%m-%d %H:%M", time.localtime()), 'text' )
        
    def save_cpacs_flight_load_cases(self):
        # create flighLoadCases
        self.cf.create_path('/cpacs/vehicles/aircraft/model/analysis', 'loadAnalysis/loadCases/flightLoadCases')
        path_flight_load_cases = '/cpacs/vehicles/aircraft/model/analysis/loadAnalysis/loadCases/flightLoadCases'
        
        # create nodal + cut loads for each trim case
        for i_trimcase in range(len(self.jcl.trimcase)):
            # info on current load case
            self.tixi.createElement(path_flight_load_cases, 'flightLoadCase')
            self.cf.add_elem(path_flight_load_cases+'/flightLoadCase['+str(i_trimcase+1)+']', 'name',  'subcase ' + str(self.jcl.trimcase[i_trimcase]['subcase']), 'double')
            self.cf.add_elem(path_flight_load_cases+'/flightLoadCase['+str(i_trimcase+1)+']', 'uID', self.jcl.trimcase[i_trimcase]['desc'], 'text')  
            desc_string = 'calculated with Loads Kernel on ' + time.strftime("%Y-%m-%d %H:%M", time.localtime()) + ' by ' + getpass.getuser() + ' on ' + platform.node() + ' (' + platform.platform() +')'
            self.cf.add_elem(path_flight_load_cases+'/flightLoadCase['+str(i_trimcase+1)+']', 'description', desc_string  , 'text')
            # nodal loads
            self.cf.create_path(path_flight_load_cases+'/flightLoadCase['+str(i_trimcase+1)+']', 'nodalLoads/wingNodalLoad')
            path_nodalLoads =       path_flight_load_cases+'/flightLoadCase['+str(i_trimcase+1)+']'+'/nodalLoads/wingNodalLoad'
            self.cf.add_elem(path_nodalLoads, 'parentUID', 'complete aircraft', 'text')
            self.cf.write_cpacs_loadsvector(path_nodalLoads, self.model.strcgrid, self.responses[i_trimcase]['Pg'] )
            # cut loads
            self.cf.create_path(path_flight_load_cases+'/flightLoadCase['+str(i_trimcase+1)+']', 'cutLoads/wingCutLoad')
            path_cutLoads =         path_flight_load_cases+'/flightLoadCase['+str(i_trimcase+1)+']'+'/cutLoads/wingCutLoad'
            self.cf.add_elem(path_cutLoads, 'parentUID', 'complete aircraft', 'text')
            self.cf.write_cpacs_loadsvector(path_cutLoads, self.model.mongrid, self.responses[i_trimcase]['Pmon_local'])
    
    def save_cpacs_dynamic_aircraft_model_points(self):
        # save structural grid points to CPACS
        self.cf.create_path('/cpacs/vehicles/aircraft/model/wings/wing/dynamicAircraftModel', 'dynamicAircraftModelPoints')
        path_dynamic_aircraft_model_points = '/cpacs/vehicles/aircraft/model/wings/wing/dynamicAircraftModel/dynamicAircraftModelPoints'
        self.cf.write_cpacs_grid(path_dynamic_aircraft_model_points, self.model.strcgrid)
        
    def save_cpacs_cut_load_integration_points(self):
        # save monitoring stations to CPACS
        self.cf.create_path('/cpacs/vehicles/aircraft/model/wings/wing/dynamicAircraftModel', 'cutLoadIntegrationPoints')
        path_cut_load_integration_points = '/cpacs/vehicles/aircraft/model/wings/wing/dynamicAircraftModel/cutLoadIntegrationPoints'
        self.cf.write_cpacs_grid(path_cut_load_integration_points, self.model.mongrid)
        #self.cf.write_cpacs_grid_orientation(path_CutLoadIntegrationPoints, self.model.mongrid, self.model.coord)
            
        
    def save_cpacs(self, filename):
        logging.info( 'saving nodal loads and monitoring stations as CPACS...')
        from tixiwrapper import Tixi
        self.tixi = Tixi()
        self.tixi.create('cpacs')
        self.cf = io_functions.cpacs_functions.CpacsFunctions(self.tixi)
        
        # These paths might already exist when writing into a given CPACS-file...        
        self.cf.create_path('/cpacs', 'header')
        self.save_cpacs_header()
        
        self.cf.create_path('/cpacs', 'vehicles/aircraft/model/analysis') 
        self.cf.create_path('/cpacs/vehicles/aircraft/model', 'wings/wing/dynamicAircraftModel')
        self.tixi.addTextAttribute('/cpacs/vehicles/aircraft/model/wings/wing', 'uID', 'complete aircraft')
        self.tixi.addTextElement('/cpacs/vehicles/aircraft/model/wings/wing', 'description', 'complete aircraft as used in Loads Kernel - without distinction of components' )
        
        self.save_cpacs_dynamic_aircraft_model_points()
        self.save_cpacs_cut_load_integration_points()
        self.save_cpacs_flight_load_cases()

        self.tixi.save(filename)
        