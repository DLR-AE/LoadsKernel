'''
Created on Aug 2, 2019

@author: voss_ar
'''
import numpy as np
from scipy import linalg
import logging

from loadskernel.trim_tools import * 
from loadskernel.model_equations.common import Common

class Steady(Common):

    def equations(self, X, t, modus):
        self.counter += 1
        # recover states
        Tgeo2body, Tbody2geo    = self.geo2body(X)
        dUcg_dt, Uf, dUf_dt     = self.recover_states(X)
        Vtas, q_dyn             = self.recover_Vtas(X)
        onflow, alpha, beta, my = self.recover_onflow(X)
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
        dcommand = self.get_command_derivatives(t, dUcg_dt, X)

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
        # this is a wrapper for the model equations 'eqn_basic'
        if modus in ['trim', 'trim_full_output']:
            # get inputs from trimcond and apply inputs from fsolve 
            X = np.array(self.trimcond_X[:,2], dtype='float')
            X[np.where((self.trimcond_X[:,1] == 'free'))[0]] = X_free
        elif modus in[ 'sim', 'sim_full_output']:
            X = X_free
        
        # evaluate model equations
        if modus=='trim':
            Y = self.equations(X, time, 'trim')
            # get the current values from Y and substract tamlab.figure()
            # fsolve only finds the roots; Y = 0
            Y_target_ist = Y[np.where((self.trimcond_Y[:,1] == 'target'))[0]]
            Y_target_soll = np.array(self.trimcond_Y[:,2], dtype='float')[np.where((self.trimcond_Y[:,1] == 'target'))[0]]
            out = Y_target_ist - Y_target_soll
            return out
        
        elif modus=='sim':
            Y = self.equations(X, time, 'sim')
            return Y[self.trim.idx_state_derivatives+self.trim.idx_input_derivatives] # Nz ist eine Rechengroesse und keine Simulationsgroesse!
            
        elif modus=='sim_full_output':
            response = self.equations(X, time, 'sim_full_output')
            return response
            
        elif modus=='trim_full_output':
            response = self.equations(X, time, 'trim_full_output')
            # do something with this output, e.g. plotting, animations, saving, etc.            
            logging.debug('')        
            logging.debug('X: ')
            logging.debug('--------------------')
            for i_X in range(len(response['X'])):
                logging.debug(self.trimcond_X[:,0][i_X] + ': %.4f' % float(response['X'][i_X]))
            logging.debug('Y: ')
            logging.debug('--------------------')
            for i_Y in range(len(response['Y'])):
                logging.debug(self.trimcond_Y[:,0][i_Y] + ': %.4f' % float(response['Y'][i_Y]))

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
            logging.debug('')
            logging.debug('--------------------')
            logging.debug('q_dyn: %.4f [Pa]' % float(response['q_dyn']))
            logging.debug('--------------------')
            logging.debug('aero derivatives:')
            logging.debug('--------------------')
            logging.debug('Cz_rbm: %.4f' % float(Pmac_rbm[2]/response['q_dyn']/A))
            logging.debug('Cz_cam: %.4f' % float(Pmac_cam[2]/response['q_dyn']/A))
            logging.debug('Cz_cs: %.4f' % float(Pmac_cs[2]/response['q_dyn']/A))
            logging.debug('Cz_f: %.4f' % float(Pmac_f[2]/response['q_dyn']/A))
            logging.debug('--------------')
            logging.debug('Cx: %.4f' % float(Pmac_c[0]))
            logging.debug('Cy: %.4f' % float(Pmac_c[1]))
            logging.debug('Cz: %.4f' % float(Pmac_c[2]))
            logging.debug('Cmx: %.6f' % float(Pmac_c[3]/self.model.macgrid['b_ref']))
            logging.debug('Cmy: %.6f' % float(Pmac_c[4]/self.model.macgrid['c_ref']))
            logging.debug('Cmz: %.6f' % float(Pmac_c[5]/self.model.macgrid['b_ref']))
            logging.debug('alpha: %.4f [deg]' % float(response['alpha']/np.pi*180))
            logging.debug('beta: %.4f [deg]' % float(response['beta']/np.pi*180))
            logging.debug('Cd: %.4f' % float(Cd))
            logging.debug('Cl: %.4f' % float(Cl))
            logging.debug('E: %.4f' % float(Cl/Cd))
            logging.debug('Cd_ind: %.6f' % float(Pmac_idrag[0]/response['q_dyn']/A))
            logging.debug('Cmz_ind: %.6f' % float(Pmac_idrag[5]/response['q_dyn']/A/self.model.macgrid['b_ref']))
            logging.debug('e: %.4f' % float(Cd_ind_theo/(Pmac_idrag[0]/response['q_dyn']/A)))
            logging.debug('command_xi: %.4f [rad] / %.4f [deg]' % (float( response['X'][np.where(self.trimcond_X[:,0]=='command_xi')[0][0]]), float( response['X'][np.where(self.trimcond_X[:,0]=='command_xi')[0][0]])/np.pi*180.0 ))
            logging.debug('command_eta: %.4f [rad] / %.4f [deg]' % (float( response['X'][np.where(self.trimcond_X[:,0]=='command_eta')[0][0]]), float( response['X'][np.where(self.trimcond_X[:,0]=='command_eta')[0][0]])/np.pi*180.0 ))
            logging.debug('command_zeta: %.4f [rad] / %.4f [deg]' % (float( response['X'][np.where(self.trimcond_X[:,0]=='command_zeta')[0][0]]), float( response['X'][np.where(self.trimcond_X[:,0]=='command_zeta')[0][0]])/np.pi*180.0 ))
            logging.debug('thrust: %.4f [percent]' % (float( response['X'][np.where(self.trimcond_X[:,0]=='thrust')[0][0]]*100.0) ))
            logging.debug('CS deflections [deg]: ' + str(response['Ux2']/np.pi*180))
            logging.debug('--------------------')
            
            return response
        
    def eval_equations_iteratively(self, X_free, time, modus='trim_full_output'):
        # this is a wrapper for the model equations
        i_mass = self.model.mass['key'].index(self.trimcase['mass'])
        n_modes = self.model.mass['n_modes'][i_mass]
        
        # get inputs from trimcond and apply inputs from fsolve 
        X = np.array(self.trimcond_X[:,2], dtype='float')
        X[np.where((self.trimcond_X[:,1] == 'free'))[0]] = X_free
        logging.debug('X_free: {}'.format(X_free))
        converged = False
        inner_loops = 0
        while not converged:
            inner_loops += 1
            response = self.equations(X, time, 'trim_full_output')
            logging.info('Inner iteration {:>3d}, calculate structural deformations...'.format(self.counter))
            Uf_new = linalg.solve(self.Kff, response['Pf'])

            # recover Uf_old from last step and blend with Uf_now
            f_relax = 0.8
            Uf_old = [self.trimcond_X[np.where((self.trimcond_X[:,0] == 'Uf'+str(i_mode)))[0][0],2] for i_mode in range(n_modes)]
            Uf_old = np.array(Uf_old, dtype='float')
            Uf_new = Uf_new*f_relax + Uf_old*(1.0-f_relax)

            # set new values for Uf in trimcond for next loop and store in response
            for i_mode in range(n_modes):
                self.trimcond_X[np.where((self.trimcond_X[:,0] == 'Uf'+str(i_mode)))[0][0],2] = '{:g}'.format(Uf_new[i_mode])
                response['X'][12+i_mode] = Uf_new[i_mode]
            
            # convergence parameter for iterative evaluation  
            Ug_f_body = np.dot(self.PHIf_strc.T, Uf_new.T).T
            defo_new = Ug_f_body[self.model.strcgrid['set'][:,(0,1,2)]].max() # Groesste Verformung, meistens Fluegelspitze
            ddefo = defo_new - self.defo_old
            self.defo_old = np.copy(defo_new)
            if np.abs(ddefo) < self.jcl.general['b_ref']*1.0e-5:
                converged = True
                logging.info('Inner iteration {:>3d}, defo_new: {:< 10.6g}, ddefo: {:< 10.6g}, converged.'.format(self.counter, defo_new, ddefo))
            else:
                logging.info('Inner iteration {:>3d}, defo_new: {:< 10.6g}, ddefo: {:< 10.6g}'.format(self.counter, defo_new, ddefo))
            if inner_loops > 20:
                raise ConvergenceError('No convergence of structural deformation achieved after {} inner loops. Check convergence of Tau solution and/or convergence criterion "ddefo".'.format(inner_loops))
        # get the current values from Y and substract tamlab.figure()
        # fsolve only finds the roots; Y = 0
        Y_target_ist = response['Y'][np.where((self.trimcond_Y[:,1] == 'target'))[0]]
        Y_target_soll = np.array(self.trimcond_Y[:,2], dtype='float')[np.where((self.trimcond_Y[:,1] == 'target'))[0]]
        out = Y_target_ist - Y_target_soll
        
        if modus in ['trim']:
            return out
        elif modus=='trim_full_output':
            # do something with this output, e.g. plotting, animations, saving, etc.            
            logging.debug('')        
            logging.debug('X: ')
            logging.debug('--------------------')
            for i_X in range(len(response['X'])):
                logging.debug(self.trimcond_X[:,0][i_X] + ': %.4f' % float(response['X'][i_X]))
            logging.debug('Y: ')
            logging.debug('--------------------')
            for i_Y in range(len(response['Y'])):
                logging.debug(self.trimcond_Y[:,0][i_Y] + ': %.4f' % float(response['Y'][i_Y]))

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
            logging.debug('')
            logging.debug('--------------------')
            logging.debug('q_dyn: %.4f [Pa]' % float(response['q_dyn']))
            logging.debug('--------------------')
            logging.debug('aero derivatives:')
            logging.debug('--------------------')
            logging.debug('Cz_rbm: %.4f' % float(Pmac_rbm[2]/response['q_dyn']/A))
            logging.debug('Cz_cam: %.4f' % float(Pmac_cam[2]/response['q_dyn']/A))
            logging.debug('Cz_cs: %.4f' % float(Pmac_cs[2]/response['q_dyn']/A))
            logging.debug('Cz_f: %.4f' % float(Pmac_f[2]/response['q_dyn']/A))
            logging.debug('--------------')
            logging.debug('Cx: %.4f' % float(Pmac_c[0]))
            logging.debug('Cy: %.4f' % float(Pmac_c[1]))
            logging.debug('Cz: %.4f' % float(Pmac_c[2]))
            logging.debug('Cmx: %.6f' % float(Pmac_c[3]/self.model.macgrid['b_ref']))
            logging.debug('Cmy: %.6f' % float(Pmac_c[4]/self.model.macgrid['c_ref']))
            logging.debug('Cmz: %.6f' % float(Pmac_c[5]/self.model.macgrid['b_ref']))
            logging.debug('alpha: %.4f [deg]' % float(response['alpha']/np.pi*180))
            logging.debug('beta: %.4f [deg]' % float(response['beta']/np.pi*180))
            logging.debug('Cd: %.4f' % float(Cd))
            logging.debug('Cl: %.4f' % float(Cl))
            logging.debug('Cd_ind: %.6f' % float(Pmac_idrag[0]/response['q_dyn']/A))
            logging.debug('Cmz_ind: %.6f' % float(Pmac_idrag[5]/response['q_dyn']/A/self.model.macgrid['b_ref']))
            logging.debug('command_xi: %.4f [rad] / %.4f [deg]' % (float( response['X'][np.where(self.trimcond_X[:,0]=='command_xi')[0][0]]), float( response['X'][np.where(self.trimcond_X[:,0]=='command_xi')[0][0]])/np.pi*180.0 ))
            logging.debug('command_eta: %.4f [rad] / %.4f [deg]' % (float( response['X'][np.where(self.trimcond_X[:,0]=='command_eta')[0][0]]), float( response['X'][np.where(self.trimcond_X[:,0]=='command_eta')[0][0]])/np.pi*180.0 ))
            logging.debug('command_zeta: %.4f [rad] / %.4f [deg]' % (float( response['X'][np.where(self.trimcond_X[:,0]=='command_zeta')[0][0]]), float( response['X'][np.where(self.trimcond_X[:,0]=='command_zeta')[0][0]])/np.pi*180.0 ))
            logging.debug('CS deflections [deg]: ' + str(response['Ux2']/np.pi*180))
            logging.debug('--------------------')
            
            return response
