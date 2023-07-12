'''
Created on Aug 2, 2019

@author: voss_ar
'''
import numpy as np
import importlib, logging, os, sys
from scipy import interpolate, linalg

from loadskernel.solution_tools import * 
from loadskernel.cfd_interfaces import tau_interface, su2_interface
from loadskernel.engine_interfaces import engine, propeller

class Common():
    def __init__(self, solution, X0=''):
        logging.info('Init model equations of type "{}"'.format(self.__class__.__name__))
        self.model      = solution.model
        self.jcl        = solution.jcl
        self.trimcase   = solution.trimcase
        self.X0         = X0
        self.simcase    = solution.simcase
        self.trimcond_X = solution.trimcond_X
        self.trimcond_Y = solution.trimcond_Y
        self.solution   = solution
        self.counter    = 0
        
        self.i_atmo     = self.model.atmo['key'].index(self.trimcase['altitude'])
        self.i_aero     = self.model.aero['key'].index(self.trimcase['aero'])
        self.i_mass     = self.model.mass['key'].index(self.trimcase['mass'])
        
        self.Qjj         = self.model.aero['Qjj'][self.i_aero]  
       
        self.PHImac_cg   = self.model.mass[self.i_mass]['PHImac_cg']
        self.PHIcg_mac   = self.model.mass[self.i_mass]['PHIcg_mac']
        self.PHInorm_cg  = self.model.mass[self.i_mass]['PHInorm_cg']
        self.PHIcg_norm  = self.model.mass[self.i_mass]['PHIcg_norm']
        self.Mb          = self.model.mass[self.i_mass]['Mb']
        self.Mff         = self.model.mass[self.i_mass]['Mff']
        self.Kff         = self.model.mass[self.i_mass]['Kff']
        self.Dff         = self.model.mass[self.i_mass]['Dff']
        self.Mhh         = self.model.mass[self.i_mass]['Mhh']
        self.Khh         = self.model.mass[self.i_mass]['Khh']
        self.Dhh         = self.model.mass[self.i_mass]['Dhh']
        self.PHIf_strc   = self.model.mass[self.i_mass]['PHIf_strc']
        self.PHIstrc_cg  = self.model.mass[self.i_mass]['PHIstrc_cg']
        self.Mgg         = self.model.mass[self.i_mass]['MGG']
        self.Mfcg        = self.model.mass[self.i_mass]['Mfcg']
        self.PHIjf       = self.model.mass[self.i_mass]['PHIjf']
        self.PHIkf       = self.model.mass[self.i_mass]['PHIkf']
        self.PHIlf       = self.model.mass[self.i_mass]['PHIlf']
        self.PHIjh       = self.model.mass[self.i_mass]['PHIjh']
        self.PHIkh       = self.model.mass[self.i_mass]['PHIkh']
        self.PHIlh       = self.model.mass[self.i_mass]['PHIlh']
        self.n_modes     = self.model.mass[self.i_mass]['n_modes'] 
        
        self.PHIk_strc   = self.model.PHIk_strc
        self.Djx1        = self.model.Djx1
        self.Dkx1        = self.model.Dkx1
        
        # set hingeline for cs deflections       
        if 'hingeline' in self.jcl.aero and self.jcl.aero['hingeline'] == 'z':
            self.hingeline = 'z'
        else: # default
            self.hingeline = 'y'
 
        # import aircraft-specific class from efcs.py dynamically 
        if 'path' in self.jcl.efcs and sys.version_info[0] >= 3:
            # If a path is specified, import module from that path.
            spec = importlib.util.spec_from_file_location(self.jcl.efcs['version'], os.path.join(self.jcl.efcs['path'], self.jcl.efcs['version']+'.py' ))
            efcs_module = spec.loader.load_module()
        elif 'path' in self.jcl.efcs and sys.version_info[0] < 3:
            import imp
            efcs_module = imp.load_source(self.jcl.efcs['version'], os.path.join(self.jcl.efcs['path'], self.jcl.efcs['version']+'.py' ))
        else: 
            logging.error('Please provide an EFCS, see ./doc/jcl_template.py and efcs_template.py for reference.')
        # init efcs
        self.efcs =  efcs_module.Efcs()
        logging.info('Init EFCS "{}"'.format(efcs_module.__name__))
        
        # in case we work with cpacs-mona, init the standard mapping
        if self.jcl.efcs['version'] == 'cpacsmona_standard_efcs':
            self.efcs.apply_cpasmona_standard_mapping(self.model.x2grid, self.solution.n_inputs)

        # get cfd splining matrices and cfd solver interface
        if self.jcl.aero['method'] in ['cfd_steady', 'cfd_unsteady']:
            # get cfd splining matrices
            self.PHIcfd_strc = self.model.PHIcfd_strc
            self.PHIcfd_cg   = self.model.mass[self.i_mass]['PHIcfd_cg'] 
            self.PHIcfd_f    = self.model.mass[self.i_mass]['PHIcfd_f']
            # initialize the interface to a cfd solver
            if self.jcl.aero['cfd_solver'].lower() == 'tau' and self.jcl.aero['method'] == 'cfd_steady':
                self.cfd_interface = tau_interface.TauInterface(self.solution)
            elif self.jcl.aero['cfd_solver'].lower() == 'su2' and self.jcl.aero['method'] == 'cfd_steady':
                self.cfd_interface = su2_interface.SU2InterfaceGridVelocity(self.solution)
            elif self.jcl.aero['cfd_solver'].lower() == 'su2' and self.jcl.aero['method'] == 'cfd_unsteady':
                self.cfd_interface = su2_interface.SU2InterfaceFarfieldOnflow(self.solution)
            else:
                logging.error('Interface for CFD solver "{}" and "{}" not implemented!'.format(self.jcl.aero['cfd_solver'], self.jcl.aero['method']))

        # set-up 1-cos gust   
        # Vtas aus solution condition berechnen
        uvw = np.array(self.trimcond_X[6:9,2], dtype='float')
        Vtas = sum(uvw**2)**0.5
        if 'gust' in self.simcase and self.simcase['gust']:
            # calculate and set the gust velocities
            V_D = self.model.atmo['a'][self.i_atmo] * self.simcase['gust_para']['MD'] 
            self.s0 = self.simcase['gust_para']['T1'] * Vtas 
            if 'WG_TAS' not in self.simcase.keys():
                self.WG_TAS, U_ds, V_gust = design_gust_cs_25_341(self.simcase, self.model.atmo['h'][self.i_atmo], self.model.atmo['rho'][self.i_atmo], Vtas, V_D)
            else:
                self.WG_TAS = self.simcase['WG_TAS']
            # write some user information / confirmation
            logging.info('Gust set up with initial Vtas = {:.4f}, t1 = {}, WG_tas = {:.4f}'.format(Vtas, self.simcase['gust_para']['T1'], self.WG_TAS))
            
        elif ('turbulence' in self.simcase or 'limit_turbulence' in self.simcase) and (self.simcase['turbulence'] or self.simcase['limit_turbulence']):
            V_C = self.model.atmo['a'][self.i_atmo] * self.simcase['gust_para']['MC']
            V_D = self.model.atmo['a'][self.i_atmo] * self.simcase['gust_para']['MD']
            if 'u_sigma' not in self.simcase.keys():
                self.u_sigma = turbulence_cs_25_341(self.model.atmo['h'][self.i_atmo], self.simcase['gust_para']['Z_mo'], Vtas, V_C, V_D, self.simcase['gust_para']['MLW'], self.simcase['gust_para']['MTOW'], self.simcase['gust_para']['MZFW'])
            else:
                self.u_sigma = self.simcase['u_sigma']
            logging.info('Turbulence set up with initial Vtas = {:.4f} and u_sigma = {:.4f}'.format(Vtas, self.u_sigma))

        # init cs_signal
        if 'cs_signal' in self.simcase and self.simcase['cs_signal']:
            self.efcs.cs_signal_init(self.trimcase['desc'])
        
        # init controller
        if 'controller' in self.simcase and self.simcase['controller']:
            """
            The controller might be set-up in different ways, e.g. to maintain a certain angular acceleration of velocity.
            Example: self.efcs.controller_init(np.array((0.0,0.0,0.0)), 'angular accelerations')
            Example: self.efcs.controller_init(np.dot(self.PHIcg_norm[3:6,3:6],np.dot(calc_drehmatrix_angular(float(self.trimcond_X[3,2]), float(self.trimcond_X[4,2]), float(self.trimcond_X[5,2])), np.array(self.trimcond_X[9:12,2], dtype='float'))), 'angular velocities')
            """
            if self.jcl.efcs['version'] in ['HAP']:
                self.efcs.controller_init(command_0=X0[self.solution.idx_inputs], 
                                          setpoint_v=float(self.trimcond_Y[np.where(self.trimcond_Y[:,0]=='Vtas')[0][0], 2]),
                                          setpoint_h=-X0[np.where(self.trimcond_X[:,0]=='z')[0][0]],
                                         )
            elif self.jcl.efcs['version'] in ['HAP_FMU']:
                setpoint = {'pqr':          X0[self.solution.idx_states[9:12]],
                            'PhiThetaPsi':  X0[self.solution.idx_states[3:6]],
                            'z':            X0[self.solution.idx_states[2]],
                            'Vtas':         float(self.trimcond_Y[np.where(self.trimcond_Y[:,0]=='Vtas')[0][0], 2]), 
                            'commands':     X0[self.solution.idx_inputs],
                            'Nxyz':         np.array([0.0, 0.0, float(self.trimcond_Y[np.where(self.trimcond_Y[:,0]=='Nz')[0][0], 2])]),
                            'dz':           float(self.trimcond_Y[np.where(self.trimcond_Y[:,0]=='dz')[0][0], 2]),
                           } 
                
                self.efcs.fmu_init(filename_fmu=self.jcl.efcs['filename_fmu'],
                                   filename_actuator=self.jcl.efcs['filename_actuator'],
                                   setpoint=setpoint)
            else:
                logging.error('Unknown EFCS: {}'.format(self.jcl.efcs['version']))
                        
        # convergence parameter for iterative evaluation
        self.defo_old = 0.0    
        
        if self.jcl.aero['method'] in ['mona_steady', 'mona_unsteady', 'freq_dom']:
            self.Djf_1 = self.model.aerogrid['Nmat'].dot(self.model.aerogrid['Rmat'].dot(self.model.mass[self.i_mass]['PHIjf']))
            self.Djf_2 = self.model.aerogrid['Nmat'].dot(self.model.mass[self.i_mass]['PHIjf'])* -1.0
            self.Djh_1 = self.model.aerogrid['Nmat'].dot(self.model.aerogrid['Rmat'].dot(self.model.mass[self.i_mass]['PHIjh']))
            self.Djh_2 = self.model.aerogrid['Nmat'].dot(self.model.mass[self.i_mass]['PHIjh']) * -1.0
            
        if hasattr(self.jcl, 'engine'):
            self.engine_loads = engine.EngineLoads()
            if self.jcl.engine['method'] == 'propellerdisk':
                self.propeller_precession_loads = propeller.PropellerPrecessionLoads()
            elif self.jcl.engine['method'] == 'pyPropMat':
                self.propeller_aero_loads = propeller.PyPropMat4Loads(self.jcl.engine['propeller_input_file'])
                self.propeller_precession_loads = propeller.PropellerPrecessionLoads()
            elif self.jcl.engine['method'] == 'VLM4Prop':
                self.propeller_aero_loads = propeller.VLM4PropLoads(self.model.prop, self.i_atmo)
                self.propeller_precession_loads = propeller.PropellerPrecessionLoads()
    
    def ode_arg_sorter(self, t, X):
        return self.eval_equations(X, t, 'sim')
    
    def calc_Pk_nonlin(self, dUmac_dt, wj):
        Ujx1 = np.dot(self.Djx1,dUmac_dt)
        q = Ujx1[self.model.aerogrid['set_j'][:,(0,1,2)]]
        r = self.model.aerogrid['r']
        rho = self.model.atmo['rho'][self.i_atmo]
        Gamma = self.model.aero['Gamma_jj'][self.i_aero]
        Pl = np.zeros(self.model.aerogrid['n']*6)
        Pl[self.model.aerogrid['set_l'][:,0]] = rho * Gamma.dot(wj) * np.cross(q, r)[:,0]
        Pl[self.model.aerogrid['set_l'][:,1]] = rho * Gamma.dot(wj) * np.cross(q, r)[:,1]
        Pl[self.model.aerogrid['set_l'][:,2]] = rho * Gamma.dot(wj) * np.cross(q, r)[:,2]
        Pk = self.model.PHIlk.T.dot(Pl)
        return Pk

    def calc_Pk(self, q_dyn, wj):
        fl = q_dyn * self.model.aerogrid['N'].T*self.model.aerogrid['A']*np.dot(self.Qjj, wj)
        Pl = np.zeros(self.model.aerogrid['n']*6)
        Pl[self.model.aerogrid['set_l'][:,0]] = fl[0,:]
        Pl[self.model.aerogrid['set_l'][:,1]] = fl[1,:]
        Pl[self.model.aerogrid['set_l'][:,2]] = fl[2,:]
        if self.trimcase['Ma'] >= 1.0:
            # supersonic aero with NP at 50% (ZONA51)
            Pk = Pl
        else:
            # subsonic aero with NP at 25% (VLM/DLM)
            Pk = self.model.PHIlk.T.dot(Pl)
        return Pk

    def rbm_nonlin(self, dUcg_dt, alpha, Vtas):
        dUmac_dt = np.dot(self.PHImac_cg, dUcg_dt) # auch bodyfixed
        Ujx1 = np.dot(self.Djx1,dUmac_dt)
        # der downwash wj ist nur die Komponente von Uj, welche senkrecht zum Panel steht! 
        # --> mit N multiplizieren und danach die Norm bilden    
        wj = np.sum(self.model.aerogrid['N'][:] * Ujx1[self.model.aerogrid['set_j'][:,(0,1,2)]],axis=1)
        Pk = self.calc_Pk_nonlin(dUmac_dt, wj)
        return Pk, wj
    
    def rbm(self, dUcg_dt, alpha, q_dyn, Vtas):
        dUmac_dt = np.dot(self.PHImac_cg, dUcg_dt) # auch bodyfixed
        Ujx1 = np.dot(self.Djx1,dUmac_dt)
        # der downwash wj ist nur die Komponente von Uj, welche senkrecht zum Panel steht! 
        # --> mit N multiplizieren und danach die Norm bilden    
        wj = np.sum(self.model.aerogrid['N'][:] * Ujx1[self.model.aerogrid['set_j'][:,(0,1,2)]],axis=1) / Vtas * -1 
        Pk = self.calc_Pk(q_dyn, wj)
        return Pk, wj
    
    def camber_twist_nonlin(self, dUcg_dt):
        wj = np.sin(self.model.camber_twist['cam_rad'] ) * -1.0 
        dUmac_dt = np.dot(self.PHImac_cg, dUcg_dt) # auch bodyfixed
        Pk = self.calc_Pk_nonlin(dUmac_dt, wj)
        return Pk, wj
    
    def camber_twist(self, q_dyn):
        """
        Multiplication with the sign of the z-direction of the panel normal vector ensures compatibility 
        with aerodynamic models that are NOT constructed from the left to the right. However, this is not 
        the ultimate solution, because it will fail for vertical surfaces with camber + twist. Fortunately, I can't 
        think of any twisted vertical tail or a vertical tail has a cambered airfoil. 
        """
        wj = np.sin(self.model.camber_twist['cam_rad']) * np.sign(self.model.aerogrid['N'][:,2])
        Pk = self.calc_Pk(q_dyn, wj)
        return Pk, wj
    
    def cs_nonlin(self, dUcg_dt, X, Ux2, Vtas):
        wj = np.zeros(self.model.aerogrid['n'])
        # Hier gibt es zwei Wege und es wird je Steuerflaeche unterschieden:
        # a) es liegen Daten in der AeroDB vor -> Kraefte werden interpoliert, dann zu Pk addiert, downwash vector bleibt unveraendert
        # b) der downwash der Steuerflaeche wird berechnet, zum downwash vector addiert 
        for i_x2 in range(len(self.efcs.keys)):
            # b) use DLM solution
            if self.hingeline == 'y':
                Ujx2 = np.dot(self.model.Djx2[i_x2],[0,0,0,0,Ux2[i_x2],0])
            elif self.hingeline == 'z':
                Ujx2 = np.dot(self.model.Djx2[i_x2],[0,0,0,0,0,Ux2[i_x2]])
            # Rotationen ry und rz verursachen Luftkraefte. Rotation rx hat keinen Einfluss, wenn die Stoemung von vorne kommt...
            # Mit der Norm von wj geht das Vorzeichen verloren - dies ist aber fuer den Steuerflaechenausschlag wichtig.
            wj += self.model.x2grid['eff'][i_x2] * np.sign(Ux2[i_x2]) * np.sqrt(np.sin(Ujx2[self.model.aerogrid['set_j'][:,4]])**2.0 + \
                                                                                np.sin(Ujx2[self.model.aerogrid['set_j'][:,5]])**2.0) * -Vtas
        dUmac_dt = np.dot(self.PHImac_cg, dUcg_dt) # auch bodyfixed
        Pk = self.calc_Pk_nonlin(dUmac_dt, wj)
        return Pk, wj
    
    def cs(self, X, Ux2, q_dyn):
        wj = np.zeros(self.model.aerogrid['n'])
        # Hier gibt es zwei Wege und es wird je Steuerflaeche unterschieden:
        # a) es liegen Daten in der AeroDB vor -> Kraefte werden interpoliert, dann zu Pk addiert, downwash vector bleibt unveraendert
        # b) der downwash der Steuerflaeche wird berechnet, zum downwash vector addiert 
        for i_x2 in range(len(self.efcs.keys)):
            # b) use DLM solution
            if self.hingeline == 'y':
                Ujx2 = np.dot(self.model.Djx2[i_x2],[0,0,0,0,Ux2[i_x2],0])
            elif self.hingeline == 'z':
                Ujx2 = np.dot(self.model.Djx2[i_x2],[0,0,0,0,0,Ux2[i_x2]])
            wj += self.model.x2grid['eff'][i_x2] * np.sum(self.model.aerogrid['N'][:] * np.cross(Ujx2[self.model.aerogrid['set_j'][:,(3,4,5)]], np.array([-1.,0.,0.])),axis=1)
        Pk = self.calc_Pk(q_dyn, wj)
        return Pk, wj
    
    def flexible_nonlin(self, dUcg_dt, Uf, dUf_dt, Vtas):
        if 'flex' in self.jcl.aero and self.jcl.aero['flex']:
            dUmac_dt = np.dot(self.PHImac_cg, dUcg_dt)
             # modale Verformung
            Ujf = np.dot(self.PHIjf, Uf )
            wjf_1 = np.sum(self.model.aerogrid['N'][:] * np.cross(Ujf[self.model.aerogrid['set_j'][:,(3,4,5)]], dUmac_dt[0:3]),axis=1) * -1 
            # modale Bewegung
            # zu Ueberpruefen, da Trim bisher in statischer Ruhelage und somit  dUf_dt = 0
            dUjf_dt = np.dot(self.PHIjf, dUf_dt ) # viel schneller!
            wjf_2 = np.sum(self.model.aerogrid['N'][:] * dUjf_dt[self.model.aerogrid['set_j'][:,(0,1,2)]],axis=1)
            
            wj = wjf_1 + wjf_2
            Pk = self.calc_Pk_nonlin(dUmac_dt, wj)
        else:
            Pk = np.zeros(self.model.aerogrid['n']*6)
            wj = np.zeros(self.model.aerogrid['n'])
        return Pk, wj
        
    def flexible(self, Uf, dUf_dt, dUcg_dt, q_dyn, Vtas):
        wj = np.zeros(self.model.aerogrid['n'])
        if 'flex' in self.jcl.aero and self.jcl.aero['flex']:
            dUmac_dt = np.dot(self.PHImac_cg, dUcg_dt)
             # modale Verformung
            Ujf = np.dot(self.PHIjf, Uf )
            wjf_1 = np.sum(self.model.aerogrid['N'][:] * np.cross(Ujf[self.model.aerogrid['set_j'][:,(3,4,5)]], dUmac_dt[0:3]),axis=1) / Vtas
            # modale Bewegung
            # zu Ueberpruefen, da Trim bisher in statischer Ruhelage und somit  dUf_dt = 0
            dUjf_dt = np.dot(self.PHIjf, dUf_dt ) # viel schneller!
            wjf_2 = np.sum(self.model.aerogrid['N'][:] * dUjf_dt[self.model.aerogrid['set_j'][:,(0,1,2)]],axis=1) / Vtas * -1 
            wj = wjf_1 + wjf_2
        Pk = self.calc_Pk(q_dyn, wj)
        return Pk, wj
    
    def gust(self, X, q_dyn):
        wj = np.zeros(self.model.aerogrid['n'])
        if 'gust' in self.simcase and self.simcase['gust']:
            # Eintauchtiefe in die Boe berechnen
            s_gust = (X[0] - self.model.aerogrid['offset_j'][:,0] - self.s0)
            # downwash der 1-cos Boe auf ein jedes Panel berechnen
            wj_gust = self.WG_TAS * 0.5 * (1-np.cos(np.pi * s_gust / self.simcase['gust_gradient']))
            wj_gust[np.where(s_gust <= 0.0)] = 0.0
            wj_gust[np.where(s_gust > 2*self.simcase['gust_gradient'])] = 0.0
            # Ausrichtung der Boe fehlt noch
            gust_direction_vector = np.sum(self.model.aerogrid['N'] * np.dot(np.array([0,0,1]), calc_drehmatrix( self.simcase['gust_orientation']/180.0*np.pi, 0.0, 0.0 )), axis=1)
            wj = wj_gust *  gust_direction_vector
        Pk = self.calc_Pk(q_dyn, wj)
        return Pk, wj
    
    def windsensor(self, X, Vtas, Uf, dUf_dt):
        """
        Definitions
        alpha positiv = w positiv, wind from below
        beta positiv = v positive, wind from the right side
        """
        if hasattr(self.jcl, 'sensor') and 'wind' in self.jcl.sensor['key']:
            # calculate onflow at sensor location
            i_sensor = self.jcl.sensor['key'].index('wind')
            u, v, w = self.get_sensor_onflow(i_sensor, X, Vtas, Uf, dUf_dt)
        else:
            # if no sensors are present, then take only rigid body motion as input
            u, v, w  = X[6:9] # u v w bodyfixed

        alpha = np.arctan(w/u)
        beta  = np.arctan(v/u)
        gamma = X[4] - alpha # alpha = theta - gamma
        return alpha, beta, gamma
    
    def get_sensor_onflow(self, i_sensor, X, Vtas, Uf, dUf_dt):
        PHIsensor_cg = self.model.mass[self.i_mass]['PHIsensor_cg']
        PHIf_sensor = self.model.mass[self.i_mass]['PHIf_sensor']
        # rigid
        u, v, w = PHIsensor_cg.dot(X[6:12])[self.model.sensorgrid['set'][i_sensor,0:3]] # velocity sensor attachment point
        # additional wind from flexible deformation
        uf_1, vf_1, wf_1 = np.cross(PHIf_sensor.T.dot(Uf)[self.model.sensorgrid['set'][i_sensor,(3,4,5)]], X[6:9]).dot(self.PHIcg_norm[:3,:3])
        # additional wind from flexible velocity
        uf_2, vf_2, wf_2 = PHIf_sensor.T.dot(dUf_dt)[self.model.sensorgrid['set'][i_sensor,0:3]].dot(self.PHIcg_norm[:3,:3])
        
        v += vf_1 + vf_2
        w += wf_1 + wf_2

        if 'gust' in self.simcase and self.simcase['gust']:
            # Eintauchtiefe in die Boe berechnen, analog zu gust()
            s_gust = (X[0] - self.model.sensorgrid['offset'][i_sensor,0] - self.s0)
            # downwash der 1-cos Boe an der Sensorposition, analog zu gust()
            wj_gust = self.WG_TAS * 0.5 * (1-np.cos(np.pi * s_gust / self.simcase['gust_gradient']))
            if s_gust <= 0.0: 
                wj_gust = 0.0
            if s_gust > 2*self.simcase['gust_gradient']:
                wj_gust = 0.0
            # Ausrichtung und Skalierung der Boe
            u_gust, v_gust, w_gust = Vtas * wj_gust * np.dot(np.array([0,0,1]), calc_drehmatrix( self.simcase['gust_orientation']/180.0*np.pi, 0.0, 0.0 ))
            v -= v_gust
            w += w_gust
        return u, v, w
    
    def idrag(self, wj, q_dyn):
        if self.jcl.aero['method_AIC'] in ['vlm', 'dlm', 'ae'] and 'induced_drag' in self.jcl.aero and self.jcl.aero['induced_drag']:
            Bjj = self.model.aero['Bjj'][self.i_aero]   
            cp = np.dot(self.Qjj, wj) # gesammtes cp durch gesammten downwash wj
            wj_ind = np.dot(Bjj, cp)
            cf = -wj_ind*cp
            fld = q_dyn*self.model.aerogrid['A']*cf
            Pld = np.zeros(self.model.aerogrid['n']*6)
            # Der induzierte Widerstand greift in x-Richtung an. Gibt es hierfuer vielleicht eine bessere/generische Loesung?
            Pld[self.model.aerogrid['set_l'][:,0]] = fld
            
            Pk_idrag = self.model.PHIlk.T.dot(Pld)
        else:
            Pk_idrag = np.zeros(self.model.aerogrid['n']*6) 
            
        return Pk_idrag
    
    def unsteady(self, X, t, wj, Uf, dUf_dt, onflow, q_dyn, Vtas):
        if 'method_rfa' in self.jcl.aero and self.jcl.aero['method_rfa'] == 'generalized':
            logging.error('Generalized RFA not yet implemented.')
        if 'method_rfa' in self.jcl.aero and self.jcl.aero['method_rfa'] == 'halfgeneralized':
            Pk_unsteady, Pk_unsteady_B, Pk_unsteady_D, dlag_states_dt =  self.unsteady_halfgeneralized(X, t, Uf, dUf_dt, onflow, q_dyn, Vtas)
        else: # 'physical'
#             dUmac_dt = np.dot(self.PHImac_cg, onflow)
#             # modale Verformung
#             Ujf = np.dot(self.PHIjf, Uf )
#             wjf_1 = np.sum(self.model.aerogrid['N'][:] * np.cross(Ujf[self.model.aerogrid['set_j'][:,(3,4,5)]], dUmac_dt[0:3]),axis=1) / Vtas
#             # modale Bewegung
#             dUjf_dt = np.dot(self.PHIjf, dUf_dt ) # viel schneller!
#             wjf_2 = np.sum(self.model.aerogrid['N'][:] * dUjf_dt[self.model.aerogrid['set_j'][:,(0,1,2)]],axis=1) / Vtas * -1.0
#             wjf = wjf_1 + wjf_2
            Pk_unsteady, Pk_unsteady_B, Pk_unsteady_D, dlag_states_dt = self.unsteady_pyhsical(X, t, wj, q_dyn, Vtas)
            
        return Pk_unsteady, dlag_states_dt

    def unsteady_halfgeneralized(self, X, t, Uf, dUf_dt, dUcg_dt, q_dyn, Vtas):
        n_modes     = self.n_modes
        n_poles     = self.model.aero['n_poles']
        betas       = self.model.aero['betas']
        ABCD        = self.model.aero['ABCD'][self.i_aero]
        c_ref       = self.jcl.general['c_ref']
        # There are lag states for the rotational motion (_1) and for the translational motion (_2).
        # This is to separate the lag states as the AIC matrices need to be generalized differently for the two cases.
        # In addition, the lag states depend on the generalized velocity and the generalized acceleration.
        lag_states_1 = X[self.solution.idx_lag_states[:int(self.solution.n_lag_states/2)]].reshape((self.n_modes,n_poles))
        lag_states_2 = X[self.solution.idx_lag_states[int(self.solution.n_lag_states/2):]].reshape((self.n_modes,n_poles))
        c_over_Vtas = (0.5*c_ref)/Vtas
        if t <= 0.0: # initial step
            self.t_old  = np.copy(t) 
            self.dUf_dt_old = np.copy(dUf_dt)
            self.d2Uf_d2t_old = np.zeros(n_modes)
            
        dt = t - self.t_old

        # d2Uf_d2t mittels "backward differences" berechnen
        if dt > 0.0: # solver laeuft vorwaerts
            d2Uf_d2t = (dUf_dt-self.dUf_dt_old) / dt
            self.d2Uf_d2t_old = np.copy(d2Uf_d2t)
            
        else: # solver bleibt stehen oder laeuft zurueck
            d2Uf_d2t = self.d2Uf_d2t_old

        # save for next step
        self.t_old  = np.copy(t)
        self.dUf_dt_old = np.copy(dUf_dt)

        # B - Daemfungsterm
        cp_unsteady = ABCD[1,:,:].dot(self.Djf_1).dot(dUf_dt) * c_over_Vtas \
                    + ABCD[1,:,:].dot(self.Djf_2).dot(d2Uf_d2t) / Vtas * -1.0 * c_over_Vtas 
        flunsteady = q_dyn * self.model.aerogrid['N'].T*self.model.aerogrid['A']*cp_unsteady
        Plunsteady = np.zeros((6*self.model.aerogrid['n']))
        Plunsteady[self.model.aerogrid['set_l'][:,0]] = flunsteady[0,:]
        Plunsteady[self.model.aerogrid['set_l'][:,1]] = flunsteady[1,:]
        Plunsteady[self.model.aerogrid['set_l'][:,2]] = flunsteady[2,:]
        Pk_unsteady_B = self.model.PHIlk.T.dot(Plunsteady)
        
        # C - Beschleunigungsterm -entfaellt -
        
        # D1-Dn - lag states
        dwff_dt_1 = dUf_dt
        dlag_states_dt_1 = dwff_dt_1.repeat(n_poles).reshape((n_modes, n_poles)) - betas*lag_states_1/c_over_Vtas
        dlag_states_dt_1 = dlag_states_dt_1.reshape((-1))
        
        dwff_dt_2 = d2Uf_d2t / Vtas * -1.0
        dlag_states_dt_2 = dwff_dt_2.repeat(n_poles).reshape((n_modes, n_poles)) - betas*lag_states_2/c_over_Vtas
        dlag_states_dt_2 = dlag_states_dt_2.reshape((-1))
        
        dlag_states_dt = np.concatenate((dlag_states_dt_1, dlag_states_dt_2))
  
        D_dot_lag = np.zeros(self.model.aerogrid['n'])
        for i_pole in np.arange(0,self.model.aero['n_poles']):
            D_dot_lag += ABCD[3+i_pole,:,:].dot(self.Djf_1).dot(lag_states_1[:,i_pole]) \
                       + ABCD[3+i_pole,:,:].dot(self.Djf_2).dot(lag_states_2[:,i_pole])
        cp_unsteady = D_dot_lag 
        flunsteady = q_dyn * self.model.aerogrid['N'].T*self.model.aerogrid['A']*cp_unsteady
        Plunsteady = np.zeros(6*self.model.aerogrid['n'])
        Plunsteady[self.model.aerogrid['set_l'][:,0]] = flunsteady[0,:]
        Plunsteady[self.model.aerogrid['set_l'][:,1]] = flunsteady[1,:]
        Plunsteady[self.model.aerogrid['set_l'][:,2]] = flunsteady[2,:]
        Pk_unsteady_D = self.model.PHIlk.T.dot(Plunsteady)
       
        Pk_unsteady =  Pk_unsteady_D + Pk_unsteady_B
        return Pk_unsteady, Pk_unsteady_B, Pk_unsteady_D, dlag_states_dt
    
    def unsteady_pyhsical(self, X, t, wj, q_dyn, Vtas):
        n_j         = self.model.aerogrid['n']
        n_poles     = self.model.aero['n_poles']
        betas       = self.model.aero['betas']
        ABCD        = self.model.aero['ABCD'][self.i_aero]
        c_ref       = self.jcl.general['c_ref']
    
        lag_states = X[self.solution.idx_lag_states].reshape((n_j,n_poles))
        c_over_Vtas = (0.5*c_ref)/Vtas
        if t <= 0.0: # initial step
            self.t_old  = np.copy(t) 
            self.wj_old = np.copy(wj) 
            self.dwj_dt_old = np.zeros(n_j)
            self.dlag_states_dt_old = np.zeros(n_j*n_poles)
            
        dt = t - self.t_old

        # dwj_dt mittels "backward differences" berechnen
        if dt > 0.0: # solver laeuft vorwaerts
            dwj_dt = (wj - self.wj_old) / dt
            self.dwj_dt_old = np.copy(dwj_dt)
        else: # solver bleibt stehen oder laeuft zurueck
            dwj_dt = self.dwj_dt_old
            
        #         dwj_dt = (wj - self.wj_old) / dt
        #         # guard for NaNs and Infs as we divide by dt, which might be zero...
        #         dwj_dt[np.isnan(dwj_dt)] = 0.0 
        #         dwj_dt[np.isinf(dwj_dt)] = 0.0
         
        # save for next step
        self.t_old  = np.copy(t)
        self.wj_old = np.copy(wj)
        
        # B - Daemfungsterm
        cp_unsteady = ABCD[1,:,:].dot(dwj_dt) * c_over_Vtas 
        flunsteady = q_dyn * self.model.aerogrid['N'].T*self.model.aerogrid['A']*cp_unsteady
        Plunsteady = np.zeros((6*self.model.aerogrid['n']))
        Plunsteady[self.model.aerogrid['set_l'][:,0]] = flunsteady[0,:]
        Plunsteady[self.model.aerogrid['set_l'][:,1]] = flunsteady[1,:]
        Plunsteady[self.model.aerogrid['set_l'][:,2]] = flunsteady[2,:]
        Pk_unsteady_B = self.model.PHIlk.T.dot(Plunsteady)
        
        # C - Beschleunigungsterm -entfaellt -
        
        # D1-Dn - lag states
        dlag_states_dt = dwj_dt.repeat(n_poles).reshape((n_j, n_poles)) - betas*lag_states/c_over_Vtas
        dlag_states_dt = dlag_states_dt.reshape((-1))
 
        D_dot_lag = np.zeros(n_j)
        for i_pole in np.arange(0,self.model.aero['n_poles']):
            D_dot_lag += ABCD[3+i_pole,:,:].dot(lag_states[:,i_pole])
        cp_unsteady = D_dot_lag 
        flunsteady = q_dyn * self.model.aerogrid['N'].T*self.model.aerogrid['A']*cp_unsteady
        Plunsteady = np.zeros(6*self.model.aerogrid['n'])
        Plunsteady[self.model.aerogrid['set_l'][:,0]] = flunsteady[0,:]
        Plunsteady[self.model.aerogrid['set_l'][:,1]] = flunsteady[1,:]
        Plunsteady[self.model.aerogrid['set_l'][:,2]] = flunsteady[2,:]
        Pk_unsteady_D = self.model.PHIlk.T.dot(Plunsteady)

        Pk_unsteady = Pk_unsteady_D + Pk_unsteady_B 
        return Pk_unsteady, Pk_unsteady_B, Pk_unsteady_D, dlag_states_dt
    
    def correctioon_coefficients(self, alpha, beta, q_dyn):
        Pb_corr = np.zeros(6)
        if 'Cm_alpha_corr' in self.jcl.aero:
            Pb_corr[4] += self.jcl.aero['Cm_alpha_corr'][self.i_aero]*q_dyn*self.jcl.general['A_ref']*self.jcl.general['c_ref']*alpha
        if 'Cn_beta_corr' in self.jcl.aero:
            Pb_corr[5] += self.jcl.aero['Cn_beta_corr'][self.i_aero]*q_dyn*self.jcl.general['A_ref']*self.jcl.general['b_ref']*beta
        return Pb_corr    
    
    def vdrag(self, alpha, q_dyn):
        Pmac = np.zeros(6)
        if 'viscous_drag' in self.jcl.aero and self.jcl.aero['viscous_drag'] == 'coefficients':
            if 'Cd_0' in self.jcl.aero: Cd0 = self.jcl.aero['Cd_0'][self.i_aero]
            else:                             Cd0 = 0.012
            if 'Cd_0' in self.jcl.aero: Cd_alpha_sq = self.jcl.aero['Cd_alpha^2'][self.i_aero]
            else:                             Cd_alpha_sq = 0.018
            Pmac[0] = q_dyn*self.jcl.general['A_ref'] * (Cd0 + Cd_alpha_sq*alpha**2.0)
        return Pmac    
    
    def landinggear(self, X, Tbody2geo):
        Pextra = np.zeros(self.model.extragrid['n']*6)
        F1 = np.zeros(self.model.extragrid['n'])
        F2 = np.zeros(self.model.extragrid['n'])
        Fx = np.zeros(self.model.extragrid['n'])
        My = np.zeros(self.model.extragrid['n'])
        p2 = np.zeros(self.model.extragrid['n'])
        dp2 = np.zeros(self.model.extragrid['n'])
        ddp2 = np.zeros(self.model.extragrid['n'])
        if 'landinggear' in self.simcase and self.simcase['landinggear']:
            # init
            PHIextra_cg = self.model.mass[self.i_mass]['PHIextra_cg']
            PHIf_extra = self.model.mass[self.i_mass]['PHIf_extra']
            p1   = -self.model.mass[self.i_mass]['cggrid']['offset'][:,2] + self.model.extragrid['offset'][:,2] + PHIextra_cg.dot(np.dot(self.PHInorm_cg, X[0:6 ]))[self.model.extragrid['set'][:,2]] + PHIf_extra.T.dot(X[12:12+self.n_modes])[self.model.extragrid['set'][:,2]] # position LG attachment point over ground
            dp1  = PHIextra_cg.dot(np.dot(self.PHInorm_cg, np.dot(Tbody2geo, X[6:12])))[self.model.extragrid['set'][:,2]]  + PHIf_extra.T.dot(X[12+self.n_modes:12+self.n_modes*2])[self.model.extragrid['set'][:,2]] # velocity LG attachment point 
            
            if self.jcl.landinggear['method'] in ['generic']:
                p2  = X[self.solution.idx_lg_states[:self.model.extragrid['n']]]  # position Tire center over ground
                dp2 = X[self.solution.idx_lg_states[self.model.extragrid['n']:]] # velocity Tire center
                # loop over every landing gear
                for i in range(self.model.extragrid['n']):
                    # calculate pre-stress F0 in gas spring
                    # assumption: gas spring is compressed by 2/3 when aircraft on ground 
                    F0 = self.jcl.landinggear['para'][i]['F_static'] / ((1.0-2.0/3.0)**(-self.jcl.landinggear['para'][i]['n']*self.jcl.landinggear['para'][i]['ck'])) # N
                    # gas spring and damper
                    stroke = p2[i] - p1[i] + self.jcl.landinggear['para'][i]['stroke_length'] + self.jcl.landinggear['para'][i]['fitting_length']
                    if stroke > 0.001:
                        Ff = F0*(1.0-stroke/self.jcl.landinggear['para'][i]['stroke_length'])**(-self.jcl.landinggear['para'][i]['n']*self.jcl.landinggear['para'][i]['ck'])
                        Fd = np.sign(dp2[i]-dp1[i])*self.jcl.landinggear['para'][i]['d2']*(dp2[i]-dp1[i])**2.0 
                    elif stroke < -0.001:
                        Ff = -F0
                        Fd = 0.0 #np.sign(dp2[i]-dp1[i])*self.jcl.landinggear['para'][i]['d2']*(dp2[i]-dp1[i])**2.0
                    else:
                        Ff = 0.0
                        Fd = 0.0
                    # tire
                    if p2[i] < self.jcl.landinggear['para'][i]['r_tire']:
                        Fz = self.jcl.landinggear['para'][i]['c1_tire']*(self.jcl.landinggear['para'][i]['r_tire'] - p2[i]) + self.jcl.landinggear['para'][i]['d1_tire']*(-dp2[i]) 
                    else:
                        Fz = 0.0
                    Fg_tire = 0.0 #self.jcl.landinggear['para'][i]['m_tire'] * 9.81
                    
                    # in case of retracted landing gear no forces apply
                    if self.simcase['landinggear_state'][i] == 'extended':
                        F1[i]=(Ff+Fd)
                        F2[i]=(-Fg_tire-(Ff+Fd)+Fz)
                        Fx[i]=(0.25*Fz) # CS 25.479(d)(1)
                        ddp2[i]=(1.0/self.jcl.landinggear['para'][i]['m_tire']*(-Fg_tire-(Ff+Fd)+Fz))
                    else: 
                        F1[i]=(0.0)
                        F2[i]=(0.0)
                        Fx[i]=(0.0)
                        ddp2[i]=(0.0)
            
            elif self.jcl.landinggear['method'] in ['skid']:
                # loop over every landing gear
                for i in range(self.model.extragrid['n']):
                    stroke = self.jcl.landinggear['para'][i]['r_tire'] - p1[i]
                    if (stroke > 0.0) and (self.simcase['landinggear_state'][i] == 'extended'):
                        # Forces only apply with extended landing gear and tire on ground
                        coeff_friction = 0.4 # landing skid on grass
                        Fz_i = stroke * self.jcl.landinggear['para'][i]['c1_tire']
                        Fx_i = coeff_friction*Fz_i
                        My_i = -p1[i]*Fx_i
                    else:
                        # In case of retracted landing gear or tire is still in the air, no forces apply
                        Fz_i = 0.0
                        Fx_i = 0.0
                        My_i = 0.0

                    F1[i]=Fz_i
                    Fx[i]=Fx_i
                    My[i]=My_i
                p2   = []
                dp2  = []
                ddp2 = []

            # insert forces in 6dof vector Pextra
            Pextra[self.model.extragrid['set'][:,0]] = Fx 
            Pextra[self.model.extragrid['set'][:,2]] = F1
            Pextra[self.model.extragrid['set'][:,4]] = My
            
        return Pextra, p2, dp2, np.array(ddp2), np.array(F1), np.array(F2)
    
    def apply_support_condition(self, modus, d2Ucg_dt2):
        # With the support option, the acceleration of the selected DoFs (0,1,2,3,4,5) is set to zero.
        # Trimcase and simcase can have different support conditions.
        
        # get support conditions from trimcase or simcase
        if modus in ['solution', 'trim_full_output'] and 'support' in self.trimcase:
            support = self.trimcase['support']
        elif modus in ['sim', 'sim_full_output'] and self.simcase!= '' and 'support' in self.simcase:
            support = self.simcase['support']
        else:
            support = []
        # apply support conditions in-place
        if 0 in support:
            d2Ucg_dt2[0] = 0.0
        if 1 in support:
            d2Ucg_dt2[1] = 0.0
        if 2 in support:
            d2Ucg_dt2[2] = 0.0
        if 3 in support:
            d2Ucg_dt2[3] = 0.0
        if 4 in support:
            d2Ucg_dt2[4] = 0.0
        if 5 in support:
            d2Ucg_dt2[5] = 0.0      
    
    def geo2body(self, X):
        Tgeo2body = np.zeros((6,6))
        Tgeo2body[0:3,0:3] = calc_drehmatrix(X[3], X[4], X[5])
        Tgeo2body[3:6,3:6] = calc_drehmatrix_angular(X[3], X[4], X[5])
        Tbody2geo = np.zeros((6,6))
        Tbody2geo[0:3,0:3] = calc_drehmatrix(X[3], X[4], X[5]).T
        Tbody2geo[3:6,3:6] = calc_drehmatrix_angular_inv(X[3], X[4], X[5])
        return Tgeo2body, Tbody2geo
    
    def recover_states(self, X):
        dUcg_dt = np.dot(self.PHInorm_cg, X[6:12]) # u v w p q r bodyfixed
        Uf = np.array(X[12:12+self.n_modes])
        dUf_dt = np.array(X[12+self.n_modes:12+self.n_modes*2])
        return dUcg_dt, Uf, dUf_dt
    
    def recover_Vtas(self, X):
        # aktuelle Vtas und q_dyn berechnen
        dxyz = X[6:9]
        Vtas = sum(dxyz**2)**0.5
        rho = self.model.atmo['rho'][self.i_atmo]
        q_dyn = rho/2.0*Vtas**2
        return Vtas, q_dyn
    
    def recover_onflow(self, X):
        onflow  = np.dot(self.PHInorm_cg, X[6:12]) # u v w p q r bodyfixed
        return onflow
    
    def get_Ux2(self, X):
        # Steuerflaechenausschlaege vom efcs holen
        Ux2 = self.efcs.cs_mapping(X[self.solution.idx_inputs])
        return Ux2
    
    def rigid_EoM(self, dUcg_dt, Pb, g_cg, modus):
        d2Ucg_dt2 = np.zeros(dUcg_dt.shape)
        if hasattr(self.jcl,'eom') and self.jcl.eom['version'] == 'waszak':
            # # non-linear EoM, bodyfixed / Waszak
            d2Ucg_dt2[0:3] = np.cross(dUcg_dt[0:3], dUcg_dt[3:6]) + np.dot(np.linalg.inv(self.Mb)[0:3,0:3], Pb[0:3]) + g_cg 
            d2Ucg_dt2[3:6] = np.dot(np.linalg.inv(self.Mb[3:6,3:6]) , Pb[3:6] - np.cross(dUcg_dt[3:6], np.dot(self.Mb[3:6,3:6], dUcg_dt[3:6])) )
            self.apply_support_condition(modus, d2Ucg_dt2)
            Nxyz = (d2Ucg_dt2[0:3] - g_cg - np.cross(dUcg_dt[0:3], dUcg_dt[3:6]) )/9.8066  
        else:
            # linear EoM, bodyfixed / Nastran
            d2Ucg_dt2[0:3] = np.dot(np.linalg.inv(self.Mb)[0:3,0:3], Pb[0:3]) + g_cg 
            d2Ucg_dt2[3:6] = np.dot(np.linalg.inv(self.Mb)[3:6,3:6], Pb[3:6] )
            self.apply_support_condition(modus, d2Ucg_dt2)
            Nxyz = (d2Ucg_dt2[0:3] - g_cg) /9.8066 
        return d2Ucg_dt2, Nxyz
    
    def flexible_EoM(self, dUf_dt, Uf, Pf):
        d2Uf_dt2 = np.dot( -np.linalg.inv(self.Mff),  ( np.dot(self.Dff, dUf_dt) + np.dot(self.Kff, Uf) - Pf  ) )
        return d2Uf_dt2
    
    def get_command_derivatives(self, t, X, Vtas, gamma, alpha, beta, Nxyz, dxyz):
        if 'cs_signal' in self.simcase and self.simcase['cs_signal']:
            dcommand = self.efcs.cs_signal(t)
        elif 'controller' in self.simcase and self.simcase['controller']:
            feedback = {'pqr':          X[self.solution.idx_states[9:12]],
                        'PhiThetaPsi':  X[self.solution.idx_states[3:6]],
                        'z':            X[self.solution.idx_states[2]],
                        'Vtas':         Vtas, 
                        'gamma':        gamma,
                        'alpha':        alpha,
                        'beta':         beta,
                        'commands':     X[self.solution.idx_inputs],
                        'Nxyz':         Nxyz,
                        'dz':           dxyz[2],
                       } 

            dcommand = self.efcs.controller(t, feedback)
        else:
            dcommand = np.zeros(self.solution.n_input_derivatives)
        return dcommand
        
    def engine(self, X, Vtas, q_dyn, Uf, dUf_dt, t):
        if hasattr(self.jcl, 'engine'):
            # get thrust setting
            thrust = X[np.where(self.trimcond_X[:,0]=='thrust')[0][0]]
            # get all matrices for the extra -set
            PHIextra_cg = self.model.mass[self.i_mass]['PHIextra_cg']
            PHIf_extra = self.model.mass[self.i_mass]['PHIf_extra']
            dUcg_dt, Uf, dUf_dt = self.recover_states(X)
            # calculate velocity at extra / engine attachment point 
            dUextra_dt = PHIextra_cg.dot(dUcg_dt) + PHIf_extra.T.dot(dUf_dt) 
            # init an empty force vector for all extra points
            Pextra = np.zeros(self.model.extragrid['n']*6)
            # loop over all engines
            for i_engine in range(self.jcl.engine['key'].__len__()):
                # assemble a dictionary that contains all engine-relevant parameters
                parameter_dict = {'thrust_vector': np.array(self.jcl.engine['thrust_vector'][i_engine]),
                                  'Vtas': Vtas,
                                  'q_dyn': q_dyn,
                                  'Ma': self.jcl.aero['Ma'][self.i_aero],
                                  'rho': self.model.atmo['rho'][self.i_atmo],
                                 }
                # get engine thrust vector   
                P_thrust = self.engine_loads.thrust_forces(parameter_dict, thrust)
                # add thrust to extra point force vector
                Pextra[self.model.extragrid['set'][i_engine,:]] += P_thrust
                
                if self.jcl.engine['method'] in ['propellerdisk']:
                    # expand the parameter dictionary with the propellerdisk-relevant parameters
                    parameter_dict['RPM'] = self.trimcase['RPM']
                    parameter_dict['power'] = self.trimcase['power']
                    parameter_dict['pqr'] =  dUextra_dt[self.model.extragrid['set'][i_engine,(3,4,5)]]
                    parameter_dict['rotation_inertia'] = self.jcl.engine['rotation_inertia'][i_engine]
                    parameter_dict['rotation_vector'] = np.array(self.jcl.engine['rotation_vector'][i_engine])
                    
                    # get engine load vector(s)
                    P_engine_torque     = self.engine_loads.torque_moments(parameter_dict) 
                    P_precessions       = self.propeller_precession_loads.precession_moments(parameter_dict)
                    
                    # add engine and propeller loads to extra point force vector
                    Pextra[self.model.extragrid['set'][i_engine,:]] += P_precessions + P_engine_torque
                
                elif self.jcl.engine['method'] in ['pyPropMat', 'VLM4Prop']:
                    # find the sensor that corresponds to the engine
                    i_sensor = self.jcl.sensor['key'].index(self.jcl.engine['key'][i_engine])
                    # calculate the sensor onflow angles with alpha = np.arctan(w/u) and beta = np.arctan(v/u)
                    # Note that the onflow sensor considers also a gust velocities.
                    u, v, w = self.get_sensor_onflow(i_sensor, X, Vtas, Uf, dUf_dt)
                    # expand the parameter dictionary with the propellerdisk-relevant parameters
                    parameter_dict['RPM'] = self.trimcase['RPM']
                    parameter_dict['power'] = self.trimcase['power']
                    parameter_dict['alpha'] = np.arctan(w/u)
                    parameter_dict['beta']  = np.arctan(v/u)
                    parameter_dict['pqr'] =  dUextra_dt[self.model.extragrid['set'][i_engine,(3,4,5)]]
                    parameter_dict['diameter'] = self.jcl.engine['diameter'][i_engine]
                    parameter_dict['n_blades'] = self.jcl.engine['n_blades'][i_engine]
                    parameter_dict['rotation_inertia'] = self.jcl.engine['rotation_inertia'][i_engine]
                    parameter_dict['rotation_vector'] = np.array(self.jcl.engine['rotation_vector'][i_engine])
                    parameter_dict['t'] = t
                    
                    # get engine load vector(s)
                    P_engine_torque     = self.engine_loads.torque_moments(parameter_dict) 
                    P_precessions       = self.propeller_precession_loads.precession_moments(parameter_dict)
                    P_prop              = self.propeller_aero_loads.calc_loads(parameter_dict)
                    
                    # add engine and propeller loads to extra point force vector
                    Pextra[self.model.extragrid['set'][i_engine,:]] += P_precessions + P_engine_torque + P_prop
                    
            Pb_ext = PHIextra_cg.T.dot(Pextra)
            Pf_ext = PHIf_extra.dot(Pextra)
        else:
            Pextra = np.array([])
            Pb_ext = np.zeros(6)
            Pf_ext = np.zeros(self.n_modes)
            
        return Pextra, Pb_ext, Pf_ext

    def finalize(self):
        """
        This function is called each time a trim is finished and the model equations are no longer used.
        The background is that in case the model equations are pure Python code, we can rely on the automatic 
        memory management. In cases where other, external code is involved, it might become necessary to 
        release the memory manually, for example with the CFD solver SU2.Of course other, final operations 
        may be performed as needed. 
        """
        pass
    
class ConvergenceError(Exception):
    '''Raise when structural deformation does not converge after xx loops'''
