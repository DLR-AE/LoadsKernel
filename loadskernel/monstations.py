
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
            self.monstations[name]['subcase'].append(trimcase['subcase'])
            self.monstations[name]['t'].append(response['t'])
            # Unterscheidung zwischen Trim und Zeit-Simulation, da die Dimensionen der response anders sind (n_step x n_value)
            if len(response['t']) > 1:
                self.monstations[name]['loads'].append(response['Pmon_local'][:,self.model.mongrid['set'][i_station,:]])
            else:
                self.monstations[name]['loads'].append(response['Pmon_local'][self.model.mongrid['set'][i_station,:]])

    
    def gather_dyn2stat(self, i_case, response, mode='time-based'):
        # Schnittlasten an den Monitoring Stationen raus schreiben (zum Plotten)
        # Knotenlasten raus schreiben (weiterverarbeitung z.B. als FORCE und MOMENT Karten fuer Nastran)
        logging.info('searching min/max in time data at {} monitoring stations and gathering loads (dyn2stat)...'.format(len(self.monstations.keys())))
        all_subcases_dyn2stat = []
        Pg_dyn2stat = []
        for key in self.monstations.keys():
            loads_dyn2stat = []
            subcases_dyn2stat = []
            t_dyn2stat = []
            pos_max_loads_over_time = np.argmax(self.monstations[key]['loads'][i_case], 0)
            pos_min_loads_over_time = np.argmin(self.monstations[key]['loads'][i_case], 0)
            # Although the time-based approach considers all DoFs, it might lead to fewer time slices / snapshots, 
            # because identical time slices are avoided.
            if mode == 'time-based':
                #unique_pos = np.unique( np.concatenate((pos_max_loads_over_time ,pos_min_loads_over_time)) )
                for pos in np.concatenate((pos_max_loads_over_time ,pos_min_loads_over_time)): 
                    loads_dyn2stat.append(self.monstations[key]['loads'][i_case][pos,:])
                    Pg_dyn2stat.append(response['Pg'][pos,:])
                    subcases_dyn2stat.append(str(self.monstations[key]['subcase'][i_case]) + '_t={:05.3f}'.format(self.monstations[key]['t'][i_case][pos,0]))
                    t_dyn2stat.append(response['t'][pos,:])
            elif mode == 'origin-based':
                # Fz max und min
                loads_dyn2stat.append(self.monstations[key]['loads'][i_case][pos_max_loads_over_time[2],:])
                Pg_dyn2stat.append(response['Pg'][pos_max_loads_over_time[2],:])
                subcases_dyn2stat.append(str(self.monstations[key]['subcase'][i_case]) + '_' + key + '_Fz_max')
                loads_dyn2stat.append(self.monstations[key]['loads'][i_case][pos_min_loads_over_time[2],:])
                Pg_dyn2stat.append(response['Pg'][pos_min_loads_over_time[2],:])
                subcases_dyn2stat.append(str(self.monstations[key]['subcase'][i_case]) + '_' + key + '_Fz_min')
                # Mx max und min
                loads_dyn2stat.append(self.monstations[key]['loads'][i_case][pos_max_loads_over_time[3],:])
                Pg_dyn2stat.append(response['Pg'][pos_max_loads_over_time[3],:])
                subcases_dyn2stat.append(str(self.monstations[key]['subcase'][i_case]) + '_' + key + '_Mx_max')
                loads_dyn2stat.append(self.monstations[key]['loads'][i_case][pos_min_loads_over_time[3],:])
                Pg_dyn2stat.append(response['Pg'][pos_min_loads_over_time[3],:])
                subcases_dyn2stat.append(str(self.monstations[key]['subcase'][i_case]) + '_' + key + '_Mx_min')
                # My max und min
                loads_dyn2stat.append(self.monstations[key]['loads'][i_case][pos_max_loads_over_time[4],:])
                Pg_dyn2stat.append(response['Pg'][pos_max_loads_over_time[4],:])
                subcases_dyn2stat.append(str(self.monstations[key]['subcase'][i_case]) + '_' + key + '_My_max')
                loads_dyn2stat.append(self.monstations[key]['loads'][i_case][pos_min_loads_over_time[4],:])
                Pg_dyn2stat.append(response['Pg'][pos_min_loads_over_time[4],:])
                subcases_dyn2stat.append(str(self.monstations[key]['subcase'][i_case]) + '_' + key + '_My_min')
            elif mode == 'stat2stat':
                loads_dyn2stat.append(self.monstations[key]['loads'][i_case])
                Pg_dyn2stat.append(response['Pg'])
                subcases_dyn2stat.append(str(self.monstations[key]['subcase'][i_case]))# + '_t={:05.3f}'.format(self.monstations[key]['t'][i_case][0]))
                t_dyn2stat.append(response['t'][0])
                
            # save to monstations
            self.monstations[key]['loads_dyn2stat'] += loads_dyn2stat
            self.monstations[key]['subcases_dyn2stat'] += subcases_dyn2stat
            self.monstations[key]['t_dyn2stat'] += t_dyn2stat
            all_subcases_dyn2stat += subcases_dyn2stat
        
        # save to dyn2stat
        
        logging.info('reducing dyn2stat data...')
        pos = [ all_subcases_dyn2stat.index(subcase) for subcase in set(all_subcases_dyn2stat) ]
        self.dyn2stat['Pg'] += [Pg_dyn2stat[p] for p in pos]
        self.dyn2stat['subcases'] += [all_subcases_dyn2stat[p] for p in pos]
        # generate unique IDs for subcases
        # take first digits from original subcase, then add a running number
        if mode == 'time-based':
            self.dyn2stat['subcases_ID'] += [ int(all_subcases_dyn2stat[p].replace('_t=', '').replace('.', '')) for p in pos]
        elif mode == 'origin-based':
            self.dyn2stat['subcases_ID'] += [ int(all_subcases_dyn2stat[i].split('_')[0])*1000+i  for i in range(len(all_subcases_dyn2stat))]
        elif mode == 'stat2stat':
            self.dyn2stat['subcases_ID'] += [ int(all_subcases_dyn2stat[p]) for p in pos]
            
            