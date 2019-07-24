
import numpy as np
import logging

class monstations:
    #===========================================================================
    # In this class actually no calculation is done, it merely gathers data.
    # From the response, the monstations are assembled in a more convenient order and format.
    # From the response of a dynamic simulation, the peaks are identified and saved as snapshots (dyn2stat).
    #===========================================================================
    def __init__(self, jcl, model):
        self.jcl = jcl
        self.model = model
        # init monstation structure to be filled later
        self.monstations = {}
        for i_station in range(self.model.mongrid['n']):
            name = self.get_monstation_name(i_station)
            self.monstations[name] = {'CD': self.model.mongrid['CD'][i_station],
                                      'CP': self.model.mongrid['CP'][i_station],
                                      'offset': self.model.mongrid['offset'][i_station],
                                      'subcase': [],
                                      'loads':[],
                                      't':[],
                                      'loads_dyn2stat':[],
                                      'subcases_dyn2stat':[],
                                      't_dyn2stat':[],
                                     }
        self.dyn2stat = {'Pg': [], 
                         'subcases': [],
                         'subcases_ID': [],
                        }     
        
    def get_monstation_name(self, i_station):
        if not 'name' in self.model.mongrid:
                name = 'MON{:s}'.format(str(int(self.model.mongrid['ID'][i_station]))) # make up a name
        else:
            name = self.model.mongrid['name'][i_station] # take name from mongrid
        return name
    
    def gather_monstations(self, trimcase, response):
        logging.info('gathering information on monitoring stations from respone(s)...')
        for i_station in range(self.model.mongrid['n']):
            name = self.get_monstation_name(i_station)
            self.monstations[name]['subcase'].append(str(trimcase['subcase']))
            self.monstations[name]['t'].append(response['t'])
            # Unterscheidung zwischen Trim und Zeit-Simulation, da die Dimensionen der response anders sind (n_step x n_value)
            if len(response['t']) > 1:
                self.monstations[name]['loads'].append(response['Pmon_local'][:,self.model.mongrid['set'][i_station,:]])
            else:
                self.monstations[name]['loads'].append(response['Pmon_local'][self.model.mongrid['set'][i_station,:]])

    
    def gather_dyn2stat(self, i_case, response, mode='time-based'):
        """
        Schnittlasten an den Monitoring Stationen raus schreiben (z.B. zum Plotten)
        Knotenlasten raus schreiben (weiterverarbeitung z.B. als FORCE und MOMENT Karten fuer Nastran)
        """
        logging.info('searching min/max in time data at {} monitoring stations and gathering loads (dyn2stat)...'.format(len(self.monstations.keys())))
        if mode == 'time-based':
            timeslices_dyn2stat = np.array([],dtype=int)
            for key in self.monstations.keys():
                pos_max_loads_over_time = np.argmax(self.monstations[key]['loads'][i_case], 0)
                pos_min_loads_over_time = np.argmin(self.monstations[key]['loads'][i_case], 0)
                """
                Although the time-based approach considers all DoFs, it might lead to fewer time slices / snapshots
                compared to Fz,min/max, Mx,min/max, ...,  because identical time slices are avoided.
                """
                timeslices_dyn2stat = np.concatenate((timeslices_dyn2stat, pos_max_loads_over_time, pos_min_loads_over_time)) # remember identified time slices 
            logging.info('reducing dyn2stat data...')       
            timeslices_dyn2stat = np.unique(timeslices_dyn2stat)
            for pos in timeslices_dyn2stat: 
                # save nodal loads Pg for this time slice
                self.dyn2stat['Pg'].append(response['Pg'][pos,:])
                subcases_dyn2stat_string = str(self.monstations[key]['subcase'][i_case]) + '_t={:05.3f}'.format(self.monstations[key]['t'][i_case][pos,0])
                self.dyn2stat['subcases'].append(subcases_dyn2stat_string)
                """
                Generate unique IDs for subcases:
                Take first digits from original subcase, then add a running number.
                This is only important for Nastran output as Nastran does not like subcases in the style 'xy_t=0.123'
                """
                self.dyn2stat['subcases_ID'].append( int(subcases_dyn2stat_string.replace('_t=', '').replace('.', '')) )
                # save section loads to monstations
                for key in self.monstations.keys():
                    self.monstations[key]['loads_dyn2stat'].append(self.monstations[key]['loads'][i_case][pos,:])
                    self.monstations[key]['subcases_dyn2stat'].append(subcases_dyn2stat_string)
                    self.monstations[key]['t_dyn2stat'].append(response['t'][pos,:])

        elif mode == 'stat2stat':
            self.dyn2stat['Pg'].append(response['Pg'])
            self.dyn2stat['subcases'].append(str(self.jcl.trimcase[i_case]['subcase']))
            self.dyn2stat['subcases_ID'].append(int(self.jcl.trimcase[i_case]['subcase']))
                
                
            