# -*- coding: utf-8 -*-
"""
Created on Thu Nov 27 14:00:31 2014

@author: voss_ar
"""
import getpass, platform, logging, sys, copy
import numpy as np

import loadskernel.io_functions.specific_functions as specific_io
from loadskernel import auxiliary_output
from loadskernel import plotting_standard
from loadskernel import kernel

class Merge:
    def __init__(self, path_input, path_output):
        self.datasets = {   'ID':[], 
                            'jcl':[],
                            'monstations':[],
                            'response':[],
                            'dyn2stat':[],
                            'desc': [],
                            'color':[],
                            'n': 0,
                        }
        self.common_monstations = np.array([])
        
        self.path_input = specific_io.check_path(path_input) 
        self.path_output = specific_io.check_path(path_output) 
    
    def load_job(self, job_name):
        # load jcl
        jcl = specific_io.load_jcl(job_name, self.path_input, jcl=None)
        
        logging.info( '--> Loading monstations(s).' )
#         with open(self.path_output + 'monstations_' + job_name + '.pickle', 'rb') as f:
#             monstations = specific_io.load_pickle(f)
        monstations = specific_io.load_hdf5(self.path_output + 'monstations_' + job_name + '.hdf5')

        logging.info( '--> Loading dyn2stat.'  )
#         with open(self.path_output + 'dyn2stat_' + job_name + '.pickle', 'rb') as f:
#             dyn2stat_data = specific_io.load_pickle(f)
        dyn2stat_data = specific_io.load_hdf5(self.path_output + 'dyn2stat_' + job_name + '.hdf5')
        
        # save into data structure
        self.datasets['ID'].append(self.datasets['n'])  
        self.datasets['jcl'].append(jcl)
        self.datasets['monstations'].append(monstations)
        self.datasets['dyn2stat'].append(dyn2stat_data)
        self.datasets['desc'].append('dataset '+ str(self.datasets['n']))
        self.datasets['n'] += 1
    
    def load_jobs(self, jobs_to_merge):
        for job_name in jobs_to_merge:
            logging.info('job:' + job_name)
            self.load_job(job_name)
        self.update_fields()
        
    def update_fields(self):
        keys = [list(monstations) for monstations in self.datasets['monstations']]
        self.common_monstations = np.unique(keys)
            
    def run_merge(self, job_name, jobs_to_merge):
           
        k = kernel.Kernel(job_name, path_input=self.path_input, path_output=self.path_output)
        k.setup()
        k.setup_logger()
        logging.info( 'Starting Loads Merge')
        logging.info( 'user ' + getpass.getuser() + ' on ' + platform.node() + ' (' + platform.platform() +')')
        self.model = specific_io.load_model(jobs_to_merge[0], self.path_output)
        self.load_jobs(jobs_to_merge)
        self.build_new_dataset()
        self.plot_monstations(job_name)
        self.build_auxiliary_output(job_name)

        logging.info('Loads Merge finished.')
        k.print_logo()
    
    def build_new_dataset(self):
        # Init new datastructure
        new_monstations = {}
        new_dyn2stat = {}
        # Take first jcl as baseline, clear out trim- and simcases
        new_jcl = copy.deepcopy(self.datasets['jcl'][0])
        new_jcl.trimcase=[]
        new_jcl.simcase=[]
        
        # Merge datasets
        for x in range(self.datasets['n']):
            logging.info('Working on {} ...'.format(self.datasets['desc'][x]))
            # Append trimcases
            new_jcl.trimcase += self.datasets['jcl'][x].trimcase
            new_jcl.simcase += self.datasets['jcl'][x].simcase
            # Append dyn2stat
            for key in self.datasets['dyn2stat'][x].keys():
                if key not in new_dyn2stat.keys():
                    new_dyn2stat[key] = []
                new_dyn2stat[key] += list(self.datasets['dyn2stat'][x][key][()])
                    
            # Handle monstations
            for station in self.common_monstations:
                if station not in new_monstations:
                    # create (empty) entries for new monstation
                    new_monstations[station] = {'CD': self.datasets['monstations'][x][station]['CD'][()],
                                                'CP': self.datasets['monstations'][x][station]['CP'][()],
                                                'offset': self.datasets['monstations'][x][station]['offset'][()],
                                                'subcases': [],
                                                'loads':[],
                                                't':[],
                                                }
                # Check for dynamic loads.
                if np.size(self.datasets['monstations'][x][station]['t'][0]) == 1:
                    # Scenario 1: There are only static loads.
                    loads_string   = 'loads'
                    subcase_string = 'subcases'
                    t_string = 't'
                    logging.info('- {}: found {} static loads'.format(station, self.datasets['monstations'][x][station][subcase_string].__len__() ))
                elif (np.size(self.datasets['monstations'][x][station]['t'][0]) > 1) and ('loads_dyn2stat' in self.datasets['monstations'][x][station].keys()) and (self.datasets['monstations'][x][station]['loads_dyn2stat'] != []):
                    # Scenario 2: Dynamic loads have been converted to quasi-static time slices / snapshots.
                    loads_string   = 'loads_dyn2stat'
                    subcase_string = 'subcases_dyn2stat'
                    t_string = 't_dyn2stat'
                    logging.info('- {}: found {} dyn2stat loads -> discarding dynamic loads'.format(station, self.datasets['monstations'][x][station][subcase_string].__len__() ))
                else:
                    # Scenario 3: There are only dynamic loads. 
                    return
                # Merge.   
                new_monstations[station]['loads']           += list(self.datasets['monstations'][x][station][loads_string][()])
                new_monstations[station]['subcases']        += list(self.datasets['monstations'][x][station][subcase_string][()])
                new_monstations[station]['t']               += list(self.datasets['monstations'][x][station][t_string][()])

        # Save into existing data structure.
        self.new_dataset_id = self.datasets['n']
        self.datasets['ID'].append(self.new_dataset_id)  
        self.datasets['monstations'].append(new_monstations)
        self.datasets['dyn2stat'].append(new_dyn2stat)
        self.datasets['jcl'].append(new_jcl)
        self.datasets['desc'].append('dataset '+ str(self.datasets['n']))
        self.datasets['n'] += 1

    def plot_monstations(self, job_name):
        logging.info( '--> Drawing some plots.' ) 
        jcl           = self.datasets['jcl'][self.new_dataset_id]
        monstations   = self.datasets['monstations'][self.new_dataset_id]
        plt = plotting_standard.StandardPlots(jcl, model=None)
        # determine crit trimcases graphically
        plt.add_monstations(monstations)
        plt.plot_monstations(self.path_output + 'monstations_' + job_name + '.pdf')
        # store crit trimcases
        self.crit_trimcases = plt.crit_trimcases
        
    def build_auxiliary_output(self, job_name):
        logging.info( '--> Saving auxiliary output data.')
        jcl           = self.datasets['jcl'][self.new_dataset_id]
        dyn2stat_data = self.datasets['dyn2stat'][self.new_dataset_id]
        monstations   = self.datasets['monstations'][self.new_dataset_id]
        
        aux_out = auxiliary_output.AuxiliaryOutput(jcl=jcl, model=self.model, trimcase=jcl.trimcase)
        aux_out.crit_trimcases = self.crit_trimcases
        aux_out.dyn2stat_data = dyn2stat_data
        aux_out.monstations = monstations
        
        aux_out.write_critical_trimcases(self.path_output + 'crit_trimcases_' + job_name + '.csv', dyn2stat=True) 
        #aux_out.write_critical_nodalloads(self.path_output + 'nodalloads_' + job_name + '.bdf', dyn2stat=True) 
        aux_out.write_critical_sectionloads(self.path_output + 'monstations_' + job_name + '.pickle', dyn2stat=True) 
    
if __name__ == "__main__":
    jobs_to_merge = ['jcl_HAP-O6-loop1_maneuver', 'jcl_HAP-O6-loop1_gust_unsteady_FMU', 'jcl_HAP-O6-loop1_landing', 'jcl_HAP-O6-loop1_prop']
    m = Merge(path_input='/scratch/HAP_workingcopy/JCLs', path_output='/scratch/HAP_LoadsKernel')
    m.run_merge('jcl_HAP-O6_merged_loop1', jobs_to_merge)
    