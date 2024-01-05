import logging
from collections import OrderedDict
import numpy as np

from loadskernel.io_functions import write_mona, data_handling
from loadskernel.io_functions.data_handling import load_hdf5_dict


class AuxiliaryOutput():
    """
    This class provides functions to save data of trim calculations.
    """

    def __init__(self, jcl, model, trimcase):
        self.jcl = jcl
        self.model = model
        self.trimcase = trimcase
        self.responses = []
        self.crit_trimcases = []
        self.dyn2stat_data = {}
        self.monstations = {}

        self.strcgrid = load_hdf5_dict(self.model['strcgrid'])
        self.mongrid = load_hdf5_dict(self.model['mongrid'])
        self.macgrid = load_hdf5_dict(self.model['macgrid'])
        self.coord = load_hdf5_dict(self.model['coord'])

        self.Dkx1 = self.model['Dkx1'][()]

    def write_all_nodalloads(self, filename):
        logging.info('saving all nodal loads as Nastarn cards...')
        with open(filename + '_Pg', 'w') as fid:
            for i_trimcase in range(len(self.jcl.trimcase)):
                write_mona.write_force_and_moment_cards(fid, self.strcgrid,
                                                        self.responses[i_trimcase]['Pg'][0, :],
                                                        self.jcl.trimcase[i_trimcase]['subcase'])
        with open(filename + '_subcases', 'w') as fid:
            for i_trimcase in range(len(self.jcl.trimcase)):
                write_mona.write_subcases(fid, self.jcl.trimcase[i_trimcase]['subcase'],
                                          self.jcl.trimcase[i_trimcase]['desc'])

    def write_trimresults(self, filename_csv):
        trimresults = []
        for response in self.responses:
            trimresult = self.assemble_trimresult(response)
            trimresults.append(trimresult)
        logging.info('writing trim results to: %s', filename_csv)
        data_handling.write_list_of_dictionaries(trimresults, filename_csv)

    def assemble_trimresult(self, response):
        trimresult = OrderedDict({'subcase': self.jcl.trimcase[response['i'][()]]['subcase'],
                                  'desc': self.jcl.trimcase[response['i'][()]]['desc'],
                                  })
        n_modes = self.model['mass'][self.jcl.trimcase[response['i'][()]]['mass']]['n_modes'][()]
        # get trimmed states
        trimresult['x'] = response['X'][0, 0]
        trimresult['y'] = response['X'][0, 1]
        trimresult['z'] = response['X'][0, 2]
        trimresult['phi [deg]'] = response['X'][0, 3] / np.pi * 180.0
        trimresult['theta [deg]'] = response['X'][0, 4] / np.pi * 180.0
        trimresult['psi [deg]'] = response['X'][0, 5] / np.pi * 180.0
        trimresult['dx'] = response['Y'][0, 0]
        trimresult['dy'] = response['Y'][0, 1]
        trimresult['dz'] = response['Y'][0, 2]
        trimresult['u'] = response['X'][0, 6]
        trimresult['v'] = response['X'][0, 7]
        trimresult['w'] = response['X'][0, 8]
        trimresult['p'] = response['X'][0, 9]
        trimresult['q'] = response['X'][0, 10]
        trimresult['r'] = response['X'][0, 11]
        trimresult['du'] = response['Y'][0, 6]
        trimresult['dv'] = response['Y'][0, 7]
        trimresult['dw'] = response['Y'][0, 8]
        trimresult['dp'] = response['Y'][0, 9]
        trimresult['dq'] = response['Y'][0, 10]
        trimresult['dr'] = response['Y'][0, 11]
        trimresult['command_xi [deg]'] = response['X'][0, 12 + 2 * n_modes] / np.pi * 180.0
        trimresult['command_eta [deg]'] = response['X'][0, 13 + 2 * n_modes] / np.pi * 180.0
        trimresult['command_zeta [deg]'] = response['X'][0, 14 + 2 * n_modes] / np.pi * 180.0
        trimresult['thrust per engine [N]'] = response['X'][0, 15 + 2 * n_modes]
        trimresult['stabilizer [deg]'] = response['X'][0, 16 + 2 * n_modes] / np.pi * 180.0
        trimresult['flap setting [deg]'] = response['X'][0, 17 + 2 * n_modes] / np.pi * 180.0
        trimresult['Nz'] = response['Nxyz'][0, 2]
        trimresult['Vtas'] = response['Y'][0, -2]
        trimresult['q_dyn'] = response['q_dyn'][0, 0]
        trimresult['alpha [deg]'] = response['alpha'][0, 0] / np.pi * 180.0
        trimresult['beta [deg]'] = response['beta'][0, 0] / np.pi * 180.0

        # calculate additional aero coefficients
        Pmac_rbm = np.dot(self.Dkx1.T, response['Pk_rbm'][0, :])
        Pmac_cam = np.dot(self.Dkx1.T, response['Pk_cam'][0, :])
        Pmac_cs = np.dot(self.Dkx1.T, response['Pk_cs'][0, :])
        Pmac_f = np.dot(self.Dkx1.T, response['Pk_f'][0, :])
        Pmac_idrag = np.dot(self.Dkx1.T, response['Pk_idrag'][0, :])
        A = self.jcl.general['A_ref']  # sum(self.model.aerogrid['A'][:])
        AR = self.jcl.general['b_ref'] ** 2.0 / self.jcl.general['A_ref']
        Pmac_c = np.divide(response['Pmac'][0, :], response['q_dyn'][0]) / A
        # um alpha drehen, um Cl und Cd zu erhalten
        Cl = Pmac_c[2] * np.cos(response['alpha'][0, 0]) + Pmac_c[0] * np.sin(response['alpha'][0, 0])
        Cd = Pmac_c[2] * np.sin(response['alpha'][0, 0]) + Pmac_c[0] * np.cos(response['alpha'][0, 0])
        Cd_ind_theo = Cl ** 2.0 / np.pi / AR

        trimresult['Cz_rbm'] = Pmac_rbm[2] / response['q_dyn'][0, 0] / A
        trimresult['Cz_cam'] = Pmac_cam[2] / response['q_dyn'][0, 0] / A
        trimresult['Cz_cs'] = Pmac_cs[2] / response['q_dyn'][0, 0] / A
        trimresult['Cz_f'] = Pmac_f[2] / response['q_dyn'][0, 0] / A
        trimresult['Cx'] = Pmac_c[0]
        trimresult['Cy'] = Pmac_c[1]
        trimresult['Cz'] = Pmac_c[2]
        trimresult['Cmx'] = Pmac_c[3] / self.macgrid['b_ref']
        trimresult['Cmy'] = Pmac_c[4] / self.macgrid['c_ref']
        trimresult['Cmz'] = Pmac_c[5] / self.macgrid['b_ref']
        trimresult['Cl'] = Cl
        trimresult['Cd'] = Cd
        trimresult['E'] = Cl / Cd
        trimresult['Cd_ind'] = Pmac_idrag[0] / response['q_dyn'][0, 0] / A
        trimresult['Cmz_ind'] = Pmac_idrag[5] / response['q_dyn'][0, 0] / A / self.macgrid['b_ref']
        trimresult['e'] = Cd_ind_theo / (Pmac_idrag[0] / response['q_dyn'][0, 0] / A)

        return trimresult

    def write_successful_trimcases(self, filename_sucessfull, filename_failed):
        # Get the index of all sucessfull responses. This relies on the mechanism, that when loading the responses from HDF5,
        # only the successfull ones are returned.
        i_cases_sucessfull = [response['i'][()] for response in self.responses]
        # Init two empty lists
        trimcases_sucessfull = []
        trimcases_failed = []
        # Loop over all trim cases and sort them into the two lists
        for i_case, _ in enumerate(self.jcl.trimcase):
            trimcase = {'subcase': self.jcl.trimcase[i_case]['subcase'],
                        'desc': self.jcl.trimcase[i_case]['desc'], }
            if i_case in i_cases_sucessfull:
                trimcases_sucessfull.append(trimcase)
            else:
                trimcases_failed.append(trimcase)
        logging.info('writing successful trimcases cases to: %s', filename_sucessfull)
        data_handling.write_list_of_dictionaries(trimcases_sucessfull, filename_sucessfull)
        logging.info('writing failed trimcases cases to: %s', filename_failed)
        data_handling.write_list_of_dictionaries(trimcases_failed, filename_failed)

    def write_critical_trimcases(self, filename_csv):
        # eigentlich gehoert diese Funtion eher zum post-processing als zum
        # plotten, kann aber erst nach dem plotten ausgefuehrt werden...

        # extract original subcase number
        crit_trimcases = list(set([crit_trimcase.split('_')[0] for crit_trimcase in self.crit_trimcases]))

        crit_trimcases_info = []
        for i_case, _ in enumerate(self.jcl.trimcase):
            if str(self.jcl.trimcase[i_case]['subcase']) in crit_trimcases:
                trimcase = {'subcase': self.jcl.trimcase[i_case]['subcase'],
                            'desc': self.jcl.trimcase[i_case]['desc'], }
                crit_trimcases_info.append(trimcase)

        logging.info('writing critical trimcases cases to: %s', filename_csv)
        data_handling.write_list_of_dictionaries(crit_trimcases_info, filename_csv)

    def write_critical_nodalloads(self, filename):
        logging.info('saving critical nodal loads as Nastarn cards...')
        # This is quite a complicated sorting because the subcases from dyn2stat may contain non-numeric characters.
        # A "normal" sorting returns an undesired sequence, leading IDs in a non-ascending sequence. This a not
        # allowed by Nastran.
        subcases_IDs = list(self.dyn2stat_data['subcases_ID'][:])
        if isinstance(self.dyn2stat_data['subcases'], list):
            # This is an exception if source is not a hdf5 file.
            # For example, the monstations have been pre-processed by a merge script and are lists already.
            subcases = self.dyn2stat_data['subcases']
        else:
            # make sure this is a list of strings
            subcases = list(self.dyn2stat_data['subcases'].asstr()[:])
        crit_ids = [subcases_IDs[subcases.index(str(crit_trimcase))] for crit_trimcase in np.unique(self.crit_trimcases)]
        crit_ids = np.sort(crit_ids)
        with open(filename + '_Pg', 'w') as fid:
            for subcase_ID in crit_ids:
                idx = subcases_IDs.index(subcase_ID)
                write_mona.write_force_and_moment_cards(fid, self.strcgrid, self.dyn2stat_data['Pg'][idx][:],
                                                        subcases_IDs[idx])
        with open(filename + '_subcases', 'w') as fid:
            for subcase_ID in crit_ids:
                idx = subcases_IDs.index(subcase_ID)
                write_mona.write_subcases(fid, subcases_IDs[idx], subcases[idx])

    def write_critical_sectionloads(self, base_filename):
        crit_trimcases = np.unique(self.crit_trimcases)
        crit_monstations = {}
        for key, monstation in self.monstations.items():
            # create an empty monstation
            crit_monstations[key] = {}
            crit_monstations[key]['CD'] = monstation['CD']
            crit_monstations[key]['CP'] = monstation['CP']
            crit_monstations[key]['offset'] = monstation['offset']
            crit_monstations[key]['subcases'] = []
            crit_monstations[key]['loads'] = []
            crit_monstations[key]['t'] = []
            # copy only critical subcases into new monstation
            for subcase_id in monstation['subcases']:
                if subcase_id in crit_trimcases:
                    pos_to_copy = list(monstation['subcases']).index(subcase_id)
                    crit_monstations[key]['subcases'] += [monstation['subcases'][pos_to_copy]]
                    crit_monstations[key]['loads'] += [monstation['loads'][pos_to_copy]]
                    crit_monstations[key]['t'] += [monstation['t'][pos_to_copy]]
        logging.info('saving critical monstation(s).')
        with open(base_filename + '.pickle', 'wb') as f:
            data_handling.dump_pickle(crit_monstations, f)
        data_handling.dump_hdf5(base_filename + '.hdf5', crit_monstations)
