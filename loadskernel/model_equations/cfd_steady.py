'''
Created on Aug 2, 2019

@author: voss_ar
'''

import numpy as np

from loadskernel.trim_tools import * 
from loadskernel.model_equations.steady import Steady

class CfdSteady(Steady):

    def equations(self, X, t, modus):
        self.counter += 1
        # recover states
        Tgeo2body, Tbody2geo    = self.geo2body(X)
        dUcg_dt, Uf, dUf_dt     = self.recover_states(X)
        Vtas, q_dyn             = self.recover_Vtas(X)
        onflow                  = self.recover_onflow(X)
        alpha, beta, gamma      = self.windsensor(X, Vtas)
        Ux2 = self.get_Ux2(X)   
             
        # --------------------   
        # --- aerodynamics ---   
        # --------------------
        self.tau_update_para(X[6:12])
        self.tau_prepare_meshdefo(Uf, Ux2)
        self.tau_run()
        Pcfd = self.tau_last_solution()
        
        Pk_rbm      = np.zeros(6*self.model.aerogrid['n'])
        Pk_cam      = Pk_rbm*0.0
        Pk_cs       = Pk_rbm*0.0
        Pk_f        = Pk_rbm*0.0
        Pk_gust     = Pk_rbm*0.0
        Pk_idrag    = Pk_rbm*0.0
        Pk_unsteady = Pk_rbm*0.0 
        
        # -------------------------------  
        # --- correction coefficients ---   
        # -------------------------------
        Pb_corr = self.correctioon_coefficients(alpha, beta, q_dyn)
        
        # ---------------------------   
        # --- summation of forces ---   
        # ---------------------------
        Pk_aero = Pk_rbm + Pk_cam + Pk_cs + Pk_f + Pk_gust + Pk_idrag + Pk_unsteady
        Pmac = np.dot(self.Dkx1.T, Pk_aero)
        Pb = np.dot(self.PHImac_cg.T, Pmac) + Pb_corr + np.dot(self.PHIcfd_cg.T, Pcfd)
        
        g_cg = gravitation_on_earth(self.PHInorm_cg, Tgeo2body)
               
        # -----------   
        # --- EoM ---   
        # -----------
        d2Ucg_dt2, Nxyz = self.rigid_EoM(dUcg_dt, Pb, g_cg, modus)
        Pf = np.dot(self.PHIkf.T, Pk_aero) + self.Mfcg.dot( np.hstack((d2Ucg_dt2[0:3] - g_cg, d2Ucg_dt2[3:6])) ) + np.dot(self.PHIcfd_f.T, Pcfd)
        d2Uf_dt2 = self.flexible_EoM(dUf_dt, Uf, Pf)
        
        # ----------------------
        # --- CS derivatives ---
        # ----------------------
        dcommand = self.get_command_derivatives(t, dUcg_dt, X, Vtas, gamma, alpha, beta)

        # --------------   
        # --- output ---   
        # --------------
        Y = np.hstack((np.dot(Tbody2geo,X[6:12]), 
                       np.dot(self.PHIcg_norm,  d2Ucg_dt2), 
                       dUf_dt, 
                       d2Uf_dt2, 
                       dcommand, 
                       Nxyz[2],
                       Vtas,
                       beta,                       
                     )) 
        
        if modus in ['trim', 'sim']:
            return Y
        elif modus in ['trim_full_output', 'sim_full_output']:
            # calculate translations, velocities and accelerations of some additional points
            # (might also be used for sensors in a closed-loop system
            if hasattr(self.jcl, 'landinggear') and self.jcl.landinggear['method'] == 'generic':
                PHIextra_cg = self.model.mass['PHIextra_cg'][self.model.mass['key'].index(self.trimcase['mass'])]
                PHIf_extra = self.model.mass['PHIf_extra'][self.model.mass['key'].index(self.trimcase['mass'])]
                p1   = (PHIextra_cg.dot(np.dot(self.PHInorm_cg, X[0:6 ])) + PHIf_extra.T.dot(X[12:12+self.n_modes])                 )[self.model.extragrid['set'][:,2]] # position LG attachment point over ground
                dp1  = (PHIextra_cg.dot(np.dot(self.PHInorm_cg, X[6:12])) + PHIf_extra.T.dot(X[12+self.n_modes:12+self.n_modes*2]))[self.model.extragrid['set'][:,2]] # velocity LG attachment point 
                ddp1 = (PHIextra_cg.dot(np.dot(self.PHInorm_cg, Y[6:12])) + PHIf_extra.T.dot(Y[12+self.n_modes:12+self.n_modes*2]))[self.model.extragrid['set'][:,2]] # acceleration LG attachment point 
                Pextra  = np.zeros(self.model.extragrid['n']*6)
                F1   = np.zeros(self.model.extragrid['n']) 
                F2   = np.zeros(self.model.extragrid['n']) 
            else:
                p1 = ''
                dp1 = ''
                ddp1 = ''
                Pextra = ''
                F1 = ''
                F2 = ''
            response = {'X': X, 
                        'Y': Y,
                        't': np.array([t]),
                        'Pk_rbm': Pk_rbm,
                        'Pk_cam': Pk_cam,
                        'Pk_aero': Pk_aero,
                        'Pk_cs': Pk_cs,
                        'Pk_f': Pk_f,
                        'Pk_gust': Pk_gust,
                        'Pk_unsteady': Pk_unsteady,
                        'Pk_idrag': Pk_idrag,
                        'q_dyn': np.array([q_dyn]),
                        'Pb': Pb,
                        'Pmac': Pmac,
                        'Pf': Pf,
                        'alpha': np.array([alpha]),
                        'beta': np.array([beta]),
                        #'Pg_aero': np.dot(PHIk_strc.T, Pk_aero),
                        'Ux2': Ux2,
                        'dUcg_dt': dUcg_dt,
                        'd2Ucg_dt2': d2Ucg_dt2,
                        'Uf': Uf,
                        'dUf_dt': dUf_dt,
                        'd2Uf_dt2': d2Uf_dt2,
                        'Nxyz': Nxyz,
                        'g_cg': g_cg,
                        'Pextra': Pextra,
                        'p1': p1,
                        'dp1': dp1,
                        'ddp1': ddp1,
                        'F1': F1,
                        'F2': F2,
                        'Pcfd': Pcfd,
                       }
            return response        
        