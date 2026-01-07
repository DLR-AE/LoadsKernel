import logging
import numpy as np

from loadskernel.io_functions.data_handling import load_hdf5_dict


class GatherLoads:
    """
    In this class actually no calculation is done, it merely gathers data.
    From the response, the monstations are assembled in a more convenient order and format.
    From the response of a dynamic simulation, the peaks are identified and saved as snapshots (dyn2stat).
    """

    def __init__(self, jcl, model):
        self.jcl = jcl
        self.model = model
        self.mongrid = load_hdf5_dict(self.model['mongrid'])
        # init monstation structure to be filled later
        self.monstations = {}
        for i_station in range(self.mongrid['n']):
            name = self.mongrid['name'][i_station]
            self.monstations[name] = {'ID': self.mongrid['ID'][i_station],
                                      'CD': self.mongrid['CD'][i_station],
                                      'CP': self.mongrid['CP'][i_station],
                                      'offset': self.mongrid['offset'][i_station],
                                      'subcases': [],
                                      'loads': [],
                                      't': [],
                                      'turbulence_loads': [],
                                      'correlations': [],
                                      }
        self.dyn2stat = {'Pg': [],
                         'subcases': [],
                         'subcases_ID': [],
                         }

    def gather_monstations(self, trimcase, response):
        if 'Pmon_local' in response:
            logging.info('gathering information on monitoring stations from response(s)...')
            for i_station in range(self.mongrid['n']):
                name = self.mongrid['name'][i_station]
                subcase = str(trimcase['subcase'])
                # Unterscheidung zwischen Trim und Zeit-Simulation, da die Dimensionen der response anders sind
                # (n_step x n_value)
                if len(response['t']) > 1:
                    loads = response['Pmon_local'][:, self.mongrid['set'][i_station, :]]
                    # save time data per subcase
                    self.monstations[name][str(response['i'])] = {'subcase': subcase,
                                                                  'loads': loads,
                                                                  't': response['t']}
                else:
                    loads = response['Pmon_local'][0, self.mongrid['set'][i_station, :]]
                    self.monstations[name]['subcases'].append(subcase)
                    self.monstations[name]['loads'].append(loads)
                    self.monstations[name]['t'].append(response['t'][0])

                if 'Pmon_turb' in response:
                    # Check if there are any limit turbulence loads available in the response.
                    # If yes, map them into the monstations.
                    loads = response['Pmon_turb'][0, self.mongrid['set'][i_station, :]]
                    correlations = response['correlations'][self.mongrid['set']
                                                            [i_station, :], :][:, self.mongrid['set'][i_station, :]]

                    self.monstations[name]['turbulence_loads'].append(loads)
                    self.monstations[name]['correlations'].append(correlations)

    def gather_dyn2stat(self, response):
        """
        Write out section loads as well and nodal loads, for example for plotting and for FORCE and MOMENT
        cards in Nastran. The procedure involves searching for min and max loads over time at each monitoring
        station and gathering the corresponding time slices / snapshots.
        The dyn2stat loads are also useful if a dynamic simulations are performed and the response file becomes
        too large to handle.
        """
        if 'Pg' in response:
            logging.info('searching min/max in time data at %s monitoring stations '
                         'and gathering loads (dyn2stat)...', len(self.monstations.keys()))
            if len(response['t']) > 1:
                i_case = str(response['i'])
                timeslices_dyn2stat = np.array([], dtype=int)
                for key, monstation in self.monstations.items():
                    pos_max_loads_over_time = np.argmax(monstation[i_case]['loads'], 0)
                    pos_min_loads_over_time = np.argmin(monstation[i_case]['loads'], 0)
                    """
                    Although the time-based approach considers all DoFs, it might lead to fewer time slices / snapshots
                    compared to Fz,min/max, Mx,min/max, ...,  because identical time slices are avoided.
                    """
                    # Remember identified time slices
                    timeslices_dyn2stat = np.concatenate((timeslices_dyn2stat, pos_max_loads_over_time,
                                                          pos_min_loads_over_time))
                logging.info('reducing dyn2stat data...')
                timeslices_dyn2stat = np.unique(timeslices_dyn2stat)
                nastran_subcase_running_number = 1
                for pos in timeslices_dyn2stat:
                    # Save nodal loads Pg for this time slice
                    self.dyn2stat['Pg'].append(response['Pg'][pos, :])
                    # For the next operation, just take the first monstation to get the subcase name and time
                    key = list(self.monstations.keys())[0]
                    subcases_dyn2stat_string = f'{self.monstations[key][i_case]['subcase']}_t={self.monstations[key]
                                                                                               [i_case]['t'][pos, 0]:.3f}'
                    self.dyn2stat['subcases'].append(subcases_dyn2stat_string)
                    """
                    Generate unique IDs for subcases:
                    Take first digits from original subcase, then add a running number. This is a really stupid approach,
                    but we are limited to 8 digits and need to make the best out of that... Using 5 digits for the subcase
                    and 3 digits for the running number appears to work for most cases. This is only important for Nastran
                    Force and Moment cards as Nastran does not like subcases in the style 'xy_t=0.123' and requires numbers
                    in ascending order.
                    """
                    self.dyn2stat['subcases_ID'].append(int(self.monstations[key][i_case]['subcase'])
                                                        * 1000 + nastran_subcase_running_number)
                    # Save section loads to monstations
                    for key, monstation in self.monstations.items():
                        monstation['subcases'].append(subcases_dyn2stat_string)
                        monstation['loads'].append(monstation[i_case]['loads'][pos, :])
                        monstation['t'].append(monstation[i_case]['t'][pos, :])
                    # Increase counter of running number by 1
                    nastran_subcase_running_number += 1

            else:
                i_case = response['i']
                self.dyn2stat['Pg'].append(response['Pg'][0, :])
                self.dyn2stat['subcases'].append(str(self.jcl.trimcase[i_case]['subcase']))
                self.dyn2stat['subcases_ID'].append(int(self.jcl.trimcase[i_case]['subcase']))
