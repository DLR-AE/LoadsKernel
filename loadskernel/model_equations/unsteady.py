'''
Created on Aug 2, 2019

@author: voss_ar
'''
import numpy as np

from loadskernel.trim_tools import * 
from loadskernel.model_equations.common import Common

class Unsteady(Common):

    def equations(self, X, t, modus):
        self.counter += 1
        # recover states
        Tgeo2body, Tbody2geo    = self.geo2body(X)
        dUcg_dt, Uf, dUf_dt     = self.recover_states(X)
        Vtas, q_dyn             = self.recover_Vtas(self.X0)
        onflow, alpha, beta, my, gamma = self.recover_onflow(X)
        Ux2 = self.get_Ux2(X)        
        # --------------------   
        # --- aerodynamics ---   
        # --------------------
        Pk_rbm,  wj_rbm  = self.rbm(onflow, alpha, q_dyn, Vtas)
        Pk_cam,  wj_cam  = self.camber_twist(q_dyn)
        Pk_cs,   wj_cs   = self.cs(X, Ux2, q_dyn)
        Pk_f,    wj_f    = self.flexible(Uf, dUf_dt, onflow, q_dyn, Vtas)
        Pk_gust, wj_gust = self.gust(X, q_dyn)
        
        wj = wj_rbm + wj_cam + wj_cs + wj_f + wj_gust
        Pk_idrag         = self.idrag(wj, q_dyn)
        Pk_unsteady, dlag_states_dt = self.unsteady(X, t, wj, Uf, dUf_dt, onflow, q_dyn, Vtas)
        
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
        Pb = np.dot(self.PHImac_cg.T, Pmac) + Pb_corr + Pb_ext
        
        g_cg = gravitation_on_earth(self.PHInorm_cg, Tgeo2body)
               
        # -----------   
        # --- EoM ---   
        # -----------
        d2Ucg_dt2, Nxyz = self.rigid_EoM(dUcg_dt, Pb, g_cg, modus)
        Pf = np.dot(self.PHIkf.T, Pk_aero) + self.Mfcg.dot( np.hstack((d2Ucg_dt2[0:3] - g_cg, d2Ucg_dt2[3:6])) ) + Pf_ext # viel schneller!
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
                       dlag_states_dt,
                       Nxyz[2],
                       Vtas,
                       beta,
                     )) 
        
        if modus in ['trim', 'sim']:
            return Y
        elif modus in ['trim_full_output', 'sim_full_output']:
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
                       }
            return response        
    
    def eval_equations(self, X_free, time, modus='trim_full_output'):
        if modus in[ 'sim', 'sim_full_output']:
            X = X_free
        
        # evaluate model equations
        if modus=='sim':
            Y = self.equations(X, time, 'sim')
            return Y[self.trim.idx_state_derivatives+self.trim.idx_input_derivatives+self.trim.idx_lag_derivatives] # Nz ist eine Rechengroesse und keine Simulationsgroesse!
            
        elif modus=='sim_full_output':
            response = self.equations(X, time, 'sim_full_output')
            return response
      