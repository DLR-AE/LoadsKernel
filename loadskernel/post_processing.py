# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 13:34:50 2015

@author: voss_ar
"""
import numpy as np
import copy, logging

from loadskernel.solution_tools import *
from loadskernel.grid_trafo import *

class PostProcessing:
    #===========================================================================
    # In this class calculations are made that follow up on every simulation.
    # The functions should be able to handle both trim calculations and time simulations.  
    #===========================================================================
    def __init__(self, jcl, model, trimcase, response):
        self.jcl = jcl
        self.model = model
        self.trimcase = trimcase
        self.response = response
        
    
    def force_summation_method(self):
        logging.info('calculating forces & moments on structural set (force summation method)...')
        response   = self.response
        trimcase   = self.trimcase
        
        i_mass     = self.model.mass['key'].index(trimcase['mass'])
        Mgg        = self.model.mass['MGG'][i_mass]
        PHIf_strc  = self.model.mass['PHIf_strc'][i_mass]
        PHIstrc_cg = self.model.mass['PHIstrc_cg'][i_mass]

        response['Pg_iner']        = np.zeros((len(response['t']), 6*self.model.strcgrid['n']))
        response['Pg_aero']        = np.zeros((len(response['t']), 6*self.model.strcgrid['n']))
        #response['Pg_unsteady']    = np.zeros((len(response['t']), 6*self.model.strcgrid['n']))
        #response['Pg_gust']        = np.zeros((len(response['t']), 6*self.model.strcgrid['n']))
        #response['Pg_cs']          = np.zeros((len(response['t']), 6*self.model.strcgrid['n']))
        #response['Pg_idrag']       = np.zeros((len(response['t']), 6*self.model.strcgrid['n']))
        response['Pg_ext']         = np.zeros((len(response['t']), 6*self.model.strcgrid['n']))
        response['Pg_cfd']         = np.zeros((len(response['t']), 6*self.model.strcgrid['n']))
        response['Pg']             = np.zeros((len(response['t']), 6*self.model.strcgrid['n']))
        response['d2Ug_dt2']       = np.zeros((len(response['t']), 6*self.model.strcgrid['n']))
        for i_step in range(len(response['t'])):
            if hasattr(self.jcl,'eom') and self.jcl.eom['version'] == 'waszak':
                # Fuer Bewegungsgleichungen z.B. von Waszack muessen die zusaetzlichen Terme hier ebenfalls beruecksichtigt werden!
                d2Ug_dt2_r = PHIstrc_cg.dot( np.hstack((response['d2Ucg_dt2'][i_step,0:3] - response['g_cg'][i_step,:] - np.cross(response['dUcg_dt'][i_step,0:3], response['dUcg_dt'][i_step,3:6]), 
                                                        response['d2Ucg_dt2'][i_step,3:6] ))  ) 
            else:
                d2Ug_dt2_r = PHIstrc_cg.dot( np.hstack((response['d2Ucg_dt2'][i_step,0:3] - response['g_cg'][i_step,:], response['d2Ucg_dt2'][i_step,3:6])) ) # Nastran

            d2Ug_dt2_f = PHIf_strc.T.dot(response['d2Uf_dt2'][i_step,:])
            Pg_iner_r = - Mgg.dot(d2Ug_dt2_r)
            Pg_iner_f = - Mgg.dot(d2Ug_dt2_f)
            response['Pg_iner'][i_step,:] = Pg_iner_r + Pg_iner_f
            response['Pg_aero'][i_step,:] = self.model.PHIk_strc.T.dot(response['Pk_aero'][i_step,:])
            #response['Pg_gust'][i_step,:] = self.model.PHIk_strc.T.dot(response['Pk_gust'][i_step,:])
            #response['Pg_unsteady'][i_step,:]   = self.model.PHIk_strc.T.dot(response['Pk_unsteady'][i_step,:])
            #response['Pg_cs'][i_step,:]   = self.model.PHIk_strc.T.dot(response['Pk_cs'][i_step,:])
            #response['Pg_idrag'][i_step,:]= self.model.PHIk_strc.T.dot(response['Pk_idrag'][i_step,:])
            if self.jcl.aero['method'] == 'cfd_steady':
                response['Pg_cfd'][i_step,:] = self.model.PHIcfd_strc.T.dot(response['Pcfd'][i_step,:])
            if hasattr(self.jcl, 'landinggear') or hasattr(self.jcl, 'engine'):
                response['Pg_ext'][i_step,self.model.extragrid['set_strcgrid']] = response['Pextra'][i_step,self.model.extragrid['set']]
            response['Pg'][i_step,:] = response['Pg_aero'][i_step,:] + response['Pg_iner'][i_step,:] + response['Pg_ext'][i_step,:] + response['Pg_cfd'][i_step,:]
            response['d2Ug_dt2'][i_step,:] = d2Ug_dt2_r + d2Ug_dt2_f
    
    def modal_displacement_method(self):
        logging.info('calculating forces & moments on structural set (modal displacement method)...')
        logging.warning('using the modal displacement method is not recommended, use force summation method instead.')
        response   = self.response
        trimcase   = self.trimcase
        
        i_mass     = self.model.mass['key'].index(trimcase['mass'])
        Kgg        = self.model.KGG
        PHIf_strc  = self.model.mass['PHIf_strc'][i_mass]
        n_modes    = self.model.mass['n_modes'][i_mass]

        response['Pg'] = np.zeros((len(response['t']), 6*self.model.strcgrid['n']))
        for i_step in range(len(response['t'])):
            Uf = response['X'][i_step,:][12:12+n_modes]
            Ug_f_body = PHIf_strc.T.dot(Uf)
            # MDM: p = K*u
            response['Pg'][i_step,:] = Kgg.dot(Ug_f_body)

    def euler_transformation(self):
        logging.info('apply euler angles...')
        response   = self.response
        trimcase   = self.trimcase
        
        i_mass     = self.model.mass['key'].index(trimcase['mass'])
        PHIcg_norm = self.model.mass['PHIcg_norm'][i_mass]
        n_modes    = self.model.mass['n_modes'][i_mass]

        response['Pg_iner_global'] = np.zeros((len(response['t']), 6*self.model.strcgrid['n']))
        response['Pg_aero_global'] = np.zeros((len(response['t']), 6*self.model.strcgrid['n']))
        #response['Pg_gust_global'] = np.zeros((len(response['t']), 6*self.model.strcgrid['n']))
        #response['Pg_unsteady_global'] = np.zeros((len(response['t']), 6*self.model.strcgrid['n']))
        #response['Pg_cs_global']   = np.zeros((len(response['t']), 6*self.model.strcgrid['n']))
        #response['Pg_idrag_global']= np.zeros((len(response['t']), 6*self.model.strcgrid['n']))
        response['Pg_ext_global']  = np.zeros((len(response['t']), 6*self.model.strcgrid['n']))
        response['Pg_cfd_global']  = np.zeros((len(response['t']), 6*self.model.strcgrid['n']))
        response['Ug_r']           = np.zeros((len(response['t']), 6*self.model.strcgrid['n']))
        response['Ug_f']           = np.zeros((len(response['t']), 6*self.model.strcgrid['n']))
        response['Ug']             = np.zeros((len(response['t']), 6*self.model.strcgrid['n']))

        for i_step in range(len(response['t'])):
            # Including rotation about euler angles in calculation of Ug_r and Ug_f
            # This is mainly done for plotting and time animation.

            # setting up coordinate system
            coord_tmp = copy.deepcopy(self.model.coord)
            coord_tmp['ID'].append(1000000)
            coord_tmp['RID'].append(0)
            coord_tmp['dircos'].append(PHIcg_norm[0:3,0:3].dot(calc_drehmatrix(response['X'][i_step,:][3], response['X'][i_step,:][4], response['X'][i_step,:][5])))
            coord_tmp['offset'].append(response['X'][i_step,:][0:3])#self.model.atmo['h'][i_atmo]])) # correction of height to zero to allow plotting in one diagram]))
            
            coord_tmp['ID'].append(1000001)
            coord_tmp['RID'].append(0)
            coord_tmp['dircos'].append(np.eye(3))
            coord_tmp['offset'].append(-self.model.mass['cggrid'][i_mass]['offset'][0])
            
            # apply transformation to strcgrid
            strcgrid_tmp = copy.deepcopy(self.model.strcgrid)
            strcgrid_tmp['CP'] = np.repeat(1000001, self.model.strcgrid['n'])
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
            #response['Pg_gust_global'][i_step,:] = force_trafo(strcgrid_tmp, coord_tmp, response['Pg_gust'][i_step,:])
            #response['Pg_unsteady_global'][i_step,:] = force_trafo(strcgrid_tmp, coord_tmp, response['Pg_unsteady'][i_step,:])
            #response['Pg_cs_global'][i_step,:]   = force_trafo(strcgrid_tmp, coord_tmp, response['Pg_cs'][i_step,:])
            #response['Pg_idrag_global'][i_step,:]   = force_trafo(strcgrid_tmp, coord_tmp, response['Pg_idrag'][i_step,:])
            response['Pg_iner_global'][i_step,:] = force_trafo(strcgrid_tmp, coord_tmp, response['Pg_iner'][i_step,:])
            response['Pg_ext_global'][i_step,:]  = force_trafo(strcgrid_tmp, coord_tmp, response['Pg_ext'][i_step,:])
            response['Pg_cfd_global'][i_step,:]  = force_trafo(strcgrid_tmp, coord_tmp, response['Pg_cfd'][i_step,:])
            #response['Pg_global'][i_step,:] = force_trafo(strcgrid_tmp, coord_tmp, response['Pg'][i_step,:])
            response['Ug'][i_step,:] = response['Ug_r'][i_step,:] + response['Ug_f'][i_step,:]

    def cuttingforces(self):
        logging.info('calculating cutting forces & moments...')
        response   = self.response
        response['Pmon_local'] = np.zeros((len(response['t']), 6*self.model.mongrid['n']))  
        for i_step in range(len(response['t'])):
            response['Pmon_local'][i_step,:] = self.model.PHIstrc_mon.T.dot(response['Pg'][i_step,:])
