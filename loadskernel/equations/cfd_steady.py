'''
Created on Aug 2, 2019

@author: voss_ar
'''
import numpy as np

from loadskernel.solution_tools import * 
from loadskernel.equations.steady import Steady

class CfdSteady(Steady):

    def equations(self, X, t, modus):
        self.counter += 1
        # recover states
        Tgeo2body, Tbody2geo    = self.geo2body(X)
        dUcg_dt, Uf, dUf_dt     = self.recover_states(X)
        Vtas, q_dyn             = self.recover_Vtas(X)
        onflow                  = self.recover_onflow(X)
        alpha, beta, gamma      = self.windsensor(X, Vtas, Uf, dUf_dt)
        Ux2 = self.get_Ux2(X)        
        # --------------------   
        # --- aerodynamics ---   
        # --------------------
        self.cfd_interface.update_para(X[6:12])
        self.cfd_interface.prepare_meshdefo(Uf, Ux2)
        self.cfd_interface.run_solver()
        Pcfd = self.cfd_interface.get_last_solution()
        
        Pk_rbm      = np.zeros(6*self.model.aerogrid['n'])
        Pk_cam      = Pk_rbm*0.0
        Pk_cs       = Pk_rbm*0.0
        Pk_f        = Pk_rbm*0.0
        Pk_gust     = Pk_rbm*0.0
        Pk_idrag    = Pk_rbm*0.0
        Pk_unsteady = Pk_rbm*0.0 
        
        Pextra, Pb_ext, Pf_ext = self.engine(X)
        
        # -------------------------------  
        # --- correction coefficients ---   
        # -------------------------------
        Pb_corr = self.correctioon_coefficients(alpha, beta, q_dyn)
        
        # ---------------------------   
        # --- summation of forces ---   
        # ---------------------------
        Pk_aero = Pk_rbm + Pk_cam + Pk_cs + Pk_f + Pk_gust + Pk_idrag + Pk_unsteady
        Pmac = np.dot(self.Dkx1.T, Pk_aero)
        Pb = np.dot(self.PHImac_cg.T, Pmac) + Pb_corr + Pb_ext + np.dot(self.PHIcfd_cg.T, Pcfd)
        
        g_cg = gravitation_on_earth(self.PHInorm_cg, Tgeo2body)
               
        # -----------   
        # --- EoM ---   
        # -----------
        d2Ucg_dt2, Nxyz = self.rigid_EoM(dUcg_dt, Pb, g_cg, modus)
        Pf = np.dot(self.PHIkf.T, Pk_aero) + self.Mfcg.dot( np.hstack((d2Ucg_dt2[0:3] - g_cg, d2Ucg_dt2[3:6])) ) + Pf_ext + np.dot(self.PHIcfd_f.T, Pcfd)
        d2Uf_dt2 = self.flexible_EoM(dUf_dt, Uf, Pf)
        
        # ----------------------
        # --- CS derivatives ---
        # ----------------------
        dcommand = self.get_command_derivatives(t, X, Vtas, gamma, alpha, beta, Nxyz, np.dot(Tbody2geo,X[6:12])[0:3])

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
        elif modus in ['trim_full_output']:
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
                        'Pcfd': Pcfd,
                       }
            return response        
        