import numpy as np
import importlib, logging
#import time
from trim_tools import * 
from scipy import interpolate
       

class common():
    def __init__(self, model, jcl, trimcase, trimcond_X, trimcond_Y, simcase = False, X0=''):
        logging.info('Init model equations.')
        self.model = model
        self.jcl = jcl
        self.trimcase = trimcase
        self.simcase = simcase
        self.trimcond_X = trimcond_X
        self.trimcond_Y = trimcond_Y
        self.counter = 0
        
        self.i_atmo     = self.model.atmo['key'].index(self.trimcase['altitude'])
        self.i_aero     = self.model.aero['key'].index(self.trimcase['aero'])
        self.i_mass     = self.model.mass['key'].index(self.trimcase['mass'])
        
        self.Qjj         = self.model.aero['Qjj'][self.i_aero]  
       
        self.PHImac_cg   = self.model.mass['PHImac_cg'][self.i_mass]
        self.PHIcg_mac   = self.model.mass['PHIcg_mac'][self.i_mass]
        self.PHInorm_cg  = self.model.mass['PHInorm_cg'][self.i_mass]
        self.PHIcg_norm  = self.model.mass['PHIcg_norm'][self.i_mass]
        self.Mb          = self.model.mass['Mb'][self.i_mass]
        self.Mff         = self.model.mass['Mff'][self.i_mass]
        self.Kff         = self.model.mass['Kff'][self.i_mass]
        self.Dff         = self.model.mass['Dff'][self.i_mass]
        self.PHIf_strc   = self.model.mass['PHIf_strc'][self.i_mass]
        self.PHIstrc_cg  = self.model.mass['PHIstrc_cg'][self.i_mass]
        self.Mgg         = self.model.mass['MGG'][self.i_mass]
        self.Mfcg        = self.model.mass['Mfcg'][self.i_mass]
        self.PHIjf       = self.model.mass['PHIjf'][self.i_mass]
        self.PHIkf       = self.model.mass['PHIkf'][self.i_mass]
        self.n_modes     = self.model.mass['n_modes'][self.i_mass] 
        
        self.PHIk_strc   = self.model.PHIk_strc
        self.Djx1        = self.model.Djx1
        self.Dkx1        = self.model.Dkx1
        
        # set hingeline for cs deflections       
        if self.jcl.aero.has_key('hingeline') and self.jcl.aero['hingeline'] == 'y':
            self.hingeline = 'y'
        elif self.jcl.aero.has_key('hingeline') and self.jcl.aero['hingeline'] == 'z':
            self.hingeline = 'z'
        else: # default
            self.hingeline = 'y'
 
        # import aircraft-specific class from efcs.py dynamically 
        module = importlib.import_module('efcs')
        efcs_class = getattr(module, jcl.efcs['version'])
        # init efcs
        self.efcs =  efcs_class() 
        
        # set-up 1-cos gust           
        if self.simcase and self.simcase['gust']:
            # Vtas aus trim condition berechnen
            uvw = np.array(self.trimcond_X[6:9,2], dtype='float')
            Vtas = sum(uvw**2)**0.5
        
            V_D = self.model.atmo['a'][self.i_atmo] * self.simcase['gust_para']['MD']
            self.x0 = self.simcase['gust_para']['T1'] * Vtas 
            self.WG_TAS, U_ds, V_gust = DesignGust_CS_25_341(self.simcase['gust_gradient'], self.model.atmo['h'][self.i_atmo], self.model.atmo['rho'][self.i_atmo], Vtas, self.simcase['gust_para']['Z_mo'], V_D, self.simcase['gust_para']['MLW'], self.simcase['gust_para']['MTOW'], self.simcase['gust_para']['MZFW'])
            logging.info('Gust set up with initial Vtas = {}, t1 = {}, WG_tas = {}'.format(Vtas, self.simcase['gust_para']['T1'], self.WG_TAS))
        
        # init cs_signal
        if self.simcase and self.simcase['cs_signal']:
            self.efcs.cs_signal_init(self.trimcase['desc'])
        
        # init controller
        if self.simcase and self.simcase['controller']:
            #self.efcs.controller_init(np.array((0.0,0.0,0.0)), 'angular accelerations')
            #self.efcs.controller_init(np.dot(self.PHIcg_norm[3:6,3:6],np.dot(calc_drehmatrix_angular(float(self.trimcond_X[3,2]), float(self.trimcond_X[4,2]), float(self.trimcond_X[5,2])), np.array(self.trimcond_X[9:12,2], dtype='float'))), 'angular velocities')
            self.efcs.controller_init(command_0=X0[12+self.n_modes*2:12+self.n_modes*2+3], setpoint_q=float(self.trimcond_X[np.where(self.trimcond_X[:,0]=='q')[0][0], 2]) )
                
        # init aero db for hybrid aero: alpha
        if self.jcl.aero['method'] == 'hybrid' and self.model.aerodb.has_key('alpha') and self.trimcase['aero'] in self.model.aerodb['alpha']:
            self.correct_alpha = True
            self.aerodb_alpha = self.model.aerodb['alpha'][self.trimcase['aero']]
            # interp1d:  x has to be an array of monotonically increasing values
            self.aerodb_alpha_interpolation_function = interpolate.interp1d(np.array(self.aerodb_alpha['values'])/180.0*np.pi, np.array(self.aerodb_alpha['Pk']), axis=0, bounds_error = True )
            logging.info('Hybrid aero is used for alpha.')
            #print 'Forces from aero db ({}) will be scaled from q_dyn = {:.2f} to current q_dyn = {:.2f}.'.format(self.trimcase['aero'], self.aerodb_alpha['q_dyn'], self.q_dyn)        
        else:
            self.correct_alpha = False      
        
        # init aero db for hybrid aero: control surfaces x2
        # because there are several control surfaces, lists are used
        self.correct_x2 = []
        self.aerodb_x2 = []
        self.aerodb_x2_interpolation_function = []
        for x2_key in self.efcs.keys:        
            if self.jcl.aero['method'] == 'hybrid' and self.model.aerodb.has_key(x2_key) and self.trimcase['aero'] in self.model.aerodb[x2_key]:
                self.correct_x2.append(x2_key)
                self.aerodb_x2.append(self.model.aerodb[x2_key][self.trimcase['aero']])
                self.aerodb_x2_interpolation_function.append(interpolate.interp1d(np.array(self.aerodb_x2[-1]['values'])/180.0*np.pi, np.array(self.aerodb_x2[-1]['Pk']), axis=0, bounds_error = True ))
                logging.info('Hybrid aero is used for {}.'.format(x2_key))
                #print 'Forces from aero db ({}) will be scaled from q_dyn = {:.2f} to current q_dyn = {:.2f}.'.format(self.trimcase['aero'], self.aerodb_x2[-1]['q_dyn'], self.q_dyn)       
    
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
        Pk = self.model.Dlk.T.dot(Pl)
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
        if self.correct_alpha:
            # Anstellwinkel alpha von der DLM-Loesung abziehen
            alpha = self.efcs.alpha_protetcion(alpha)
            drehmatrix = np.zeros((6,6))
            drehmatrix[0:3,0:3] = calc_drehmatrix(0.0, -alpha, 0.0) 
            drehmatrix[3:6,3:6] = calc_drehmatrix(0.0, -alpha, 0.0) 
            dUcg_dt = np.dot(drehmatrix, dUcg_dt)
            
        dUmac_dt = np.dot(self.PHImac_cg, dUcg_dt) # auch bodyfixed
        Ujx1 = np.dot(self.Djx1,dUmac_dt)
        # der downwash wj ist nur die Komponente von Uj, welche senkrecht zum Panel steht! 
        # --> mit N multiplizieren und danach die Norm bilden    
        wjx1 = np.sum(self.model.aerogrid['N'][:] * Ujx1[self.model.aerogrid['set_j'][:,(0,1,2)]],axis=1) / Vtas * -1 
        flx1 = q_dyn * self.model.aerogrid['N'].T*self.model.aerogrid['A']*np.dot(self.Qjj, wjx1)
        # Bemerkung: greifen die Luftkraefte bei j,l oder k an?
        # Dies wuerde das Cmy beeinflussen!
        # Gewaehlt: l
        Plx1 = np.zeros(self.model.aerogrid['n']*6)
        Plx1[self.model.aerogrid['set_l'][:,0]] = flx1[0,:]
        Plx1[self.model.aerogrid['set_l'][:,1]] = flx1[1,:]
        Plx1[self.model.aerogrid['set_l'][:,2]] = flx1[2,:]
        
        Pk_rbm = self.model.Dlk.T.dot(Plx1)
        return Pk_rbm, wjx1
        
    def camber_twist(self, q_dyn):
        if self.correct_alpha:
            Pk_cam = np.zeros(self.model.aerogrid['n']*6)
            wj_cam = np.zeros(self.model.aerogrid['n'])
        else:
            wj_cam = np.sin(self.model.camber_twist['cam_rad'] )
            flcam = q_dyn * self.model.aerogrid['N'].T*self.model.aerogrid['A']*np.dot(self.Qjj, wj_cam)
            Plcam = np.zeros(self.model.aerogrid['n']*6)
            Plcam[self.model.aerogrid['set_l'][:,0]] = flcam[0,:]
            Plcam[self.model.aerogrid['set_l'][:,1]] = flcam[1,:]
            Plcam[self.model.aerogrid['set_l'][:,2]] = flcam[2,:]
            
            Pk_cam = self.model.Dlk.T.dot(Plcam) 
        return Pk_cam, wj_cam
    
    def cs_nonlin(self, dUcg_dt, X, Ux2, Vtas):
        wj = np.zeros(self.model.aerogrid['n'])
        # Hier gibt es zwei Wege und es wird je Steuerflaeche unterschieden:
        # a) es liegen Daten in der AeroDB vor -> Kraefte werden interpoliert, dann zu Pk addiert, downwash vector bleibt unveraendert
        # b) der downwash der Steuerflaeche wird berechnet, zum downwash vector addiert 
        for i_x2 in range(len(self.efcs.keys)):
            if self.efcs.keys[i_x2] not in self.correct_x2:
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
        wjx2 = np.zeros(self.model.aerogrid['n'])
        # Hier gibt es zwei Wege und es wird je Steuerflaeche unterschieden:
        # a) es liegen Daten in der AeroDB vor -> Kraefte werden interpoliert, dann zu Pk addiert, downwash vector bleibt unveraendert
        # b) der downwash der Steuerflaeche wird berechnet, zum downwash vector addiert 
        for i_x2 in range(len(self.efcs.keys)):
            if self.efcs.keys[i_x2] not in self.correct_x2:
                # b) use DLM solution
                if self.hingeline == 'y':
                    Ujx2 = np.dot(self.model.Djx2[i_x2],[0,0,0,0,Ux2[i_x2],0])
                elif self.hingeline == 'z':
                    Ujx2 = np.dot(self.model.Djx2[i_x2],[0,0,0,0,0,Ux2[i_x2]])
                # Rotationen ry und rz verursachen Luftkraefte. Rotation rx hat keinen Einfluss, wenn die Stoemung von vorne kommt...
                # Mit der Norm von wj geht das Vorzeichen verloren - dies ist aber fuer den Steuerflaechenausschlag wichtig.
                wjx2 += self.model.x2grid['eff'][i_x2] * np.sign(Ux2[i_x2]) * np.sqrt(np.sin(Ujx2[self.model.aerogrid['set_j'][:,4]])**2.0 + \
                                                                                      np.sin(Ujx2[self.model.aerogrid['set_j'][:,5]])**2.0)  #* Vtas/Vtas
        flx2 = q_dyn * self.model.aerogrid['N'].T*self.model.aerogrid['A']*np.dot(self.Qjj, wjx2)
        Plx2 = np.zeros(self.model.aerogrid['n']*6)
        Plx2[self.model.aerogrid['set_l'][:,0]] = flx2[0,:]
        Plx2[self.model.aerogrid['set_l'][:,1]] = flx2[1,:]
        Plx2[self.model.aerogrid['set_l'][:,2]] = flx2[2,:]
    
        Pk_cs = self.model.Dlk.T.dot(Plx2)
        return Pk_cs, wjx2
    
    def flexible_nonlin(self, dUcg_dt, Uf, dUf_dt, Vtas):
        if self.jcl.aero.has_key('flex') and self.jcl.aero['flex']:
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
            q = np.zeros(self.model.aerogrid['n'])
        return Pk, wj
        
    def flexible(self, Uf, dUf_dt, dUcg_dt, q_dyn, Vtas):
        if self.jcl.aero.has_key('flex') and self.jcl.aero['flex']:
            dUmac_dt = np.dot(self.PHImac_cg, dUcg_dt)
             # modale Verformung
            Ujf = np.dot(self.PHIjf, Uf )
            wjf_1 = np.sum(self.model.aerogrid['N'][:] * np.cross(Ujf[self.model.aerogrid['set_j'][:,(3,4,5)]], dUmac_dt[0:3]),axis=1) / Vtas
            # modale Bewegung
            # zu Ueberpruefen, da Trim bisher in statischer Ruhelage und somit  dUf_dt = 0
            dUjf_dt = np.dot(self.PHIjf, dUf_dt ) # viel schneller!
            wjf_2 = np.sum(self.model.aerogrid['N'][:] * dUjf_dt[self.model.aerogrid['set_j'][:,(0,1,2)]],axis=1) / Vtas * -1 
            wjf = wjf_1 + wjf_2
            flf = q_dyn * self.model.aerogrid['N'].T*self.model.aerogrid['A']*np.dot(self.Qjj, wjf)        
            Plf = np.zeros(self.model.aerogrid['n']*6)
            Plf[self.model.aerogrid['set_l'][:,0]] = flf[0,:]
            Plf[self.model.aerogrid['set_l'][:,1]] = flf[1,:]
            Plf[self.model.aerogrid['set_l'][:,2]] = flf[2,:]
            
            Pk_f = self.model.Dlk.T.dot(Plf)
        else:
            Pk_f = np.zeros(self.model.aerogrid['n']*6)
            wjf = np.zeros(self.model.aerogrid['n'])
        return Pk_f, wjf
    
    def gust(self, X, q_dyn):
        if self.simcase and self.simcase['gust']:
            # Eintauchtiefe in die Boe berechnen
            s_gust = (X[0] - self.model.aerogrid['offset_j'][:,0] - self.x0)
            # downwash der 1-cos Boe auf ein jedes Panel berechnen
            wj_gust = self.WG_TAS * 0.5 * (1-np.cos(np.pi * s_gust / self.simcase['gust_gradient']))
            wj_gust[np.where(s_gust <= 0.0)] = 0.0
            wj_gust[np.where(s_gust > 2*self.simcase['gust_gradient'])] = 0.0
            # Ausrichtung der Boe fehlt noch
            gust_direction_vector = np.sum(self.model.aerogrid['N'] * np.dot(np.array([0,0,1]), calc_drehmatrix( self.simcase['gust_orientation']/180.0*np.pi, 0.0, 0.0 )), axis=1)
            wj_gust = wj_gust *  gust_direction_vector
            flgust = q_dyn * self.model.aerogrid['N'].T*self.model.aerogrid['A']*np.dot(self.Qjj, wj_gust)
            Plgust = np.zeros((6*self.model.aerogrid['n']))
            Plgust[self.model.aerogrid['set_l'][:,0]] = flgust[0,:]
            Plgust[self.model.aerogrid['set_l'][:,1]] = flgust[1,:]
            Plgust[self.model.aerogrid['set_l'][:,2]] = flgust[2,:]
            
            Pk_gust = self.model.Dlk.T.dot(Plgust)
        else:
            Pk_gust = np.zeros(self.model.aerogrid['n']*6)
            wj_gust = np.zeros(self.model.aerogrid['n'])
        return Pk_gust, wj_gust
    
    def idrag(self, wj, q_dyn):
        if self.jcl.aero['method_AIC'] in ['vlm', 'dlm', 'ae'] and self.jcl.aero.has_key('induced_drag') and self.jcl.aero['induced_drag']:
            Bjj = self.model.aero['Bjj'][self.i_aero]   
            cp = np.dot(self.Qjj, wj) # gesammtes cp durch gesammten downwash wj
            wj_ind = np.dot(Bjj, cp)
            cf = -wj_ind*cp
            fld = q_dyn*self.model.aerogrid['A']*cf
            Pld = np.zeros(self.model.aerogrid['n']*6)
            # Der induzierte Widerstand greift in x-Richtung an. Gibt es hierfuer vielleicht eine bessere/generische Loesung?
            Pld[self.model.aerogrid['set_l'][:,0]] = fld
            
            Pk_idrag = self.model.Dlk.T.dot(Pld)
        else:
            Pk_idrag = np.zeros(self.model.aerogrid['n']*6) 
            
        return Pk_idrag
    
    def unsteady(self, X, t, wj, q_dyn, Vtas):
        n_j         = self.model.aerogrid['n']
        n_poles     = self.model.aero['n_poles']
        betas       = self.model.aero['betas']
        ABCD        = self.model.aero['ABCD'][self.i_aero]
        c_ref       = self.jcl.general['c_ref']
    
        lag_states = X[12+self.n_modes*2+3:12+self.n_modes*2+3+n_j*n_poles].reshape((n_j,n_poles))
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
        Pk_unsteady_B = self.model.Dlk.T.dot(Plunsteady)
        
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
        Pk_unsteady_D = self.model.Dlk.T.dot(Plunsteady)

        Pk_unsteady = Pk_unsteady_D + Pk_unsteady_B 
        return Pk_unsteady, Pk_unsteady_B, Pk_unsteady_D, dlag_states_dt
        
    def cfd(self, alpha, X, Ux2, q_dyn):
        Pk_cfd = np.zeros(6*self.model.aerogrid['n'])
        if self.correct_alpha:
            # jetzt den Anstellwinkel durch die CFD-loesung addieren
            Pk_cfd += self.aerodb_alpha_interpolation_function(alpha) / self.aerodb_alpha['q_dyn'] * q_dyn
        
        # Hier gibt es zwei Wege und es wird je Steuerflaeche unterschieden:
        # a) es liegen Daten in der AeroDB vor -> Kraefte werden interpoliert, dann zu Pk addiert, downwash vector bleibt unveraendert
        # b) der downwash der Steuerflaeche wird berechnet, zum downwash vector addiert 
        for i_x2 in range(len(self.efcs.keys)):
            if self.efcs.keys[i_x2] in self.correct_x2:
                # a) use CFD solution
                pos = self.correct_x2.index(self.efcs.keys[i_x2])
                x2_interpolation_function = self.aerodb_x2_interpolation_function[pos]
                # Null-Loesung abziehen!
                # Extrapolation sollte durch passend eingestellte Grenzen im EFCS verhindert werden. 
                Pk_cfd += (x2_interpolation_function(Ux2[i_x2]) - x2_interpolation_function(0.0) ) / self.aerodb_x2[pos]['q_dyn'] * q_dyn
        return Pk_cfd
    
    def correctioon_coefficients(self, alpha, beta, q_dyn):
        Pb_corr = np.zeros(6)
        if self.jcl.aero.has_key('Cm_alpha_corr'):
            Pb_corr[4] += self.jcl.aero['Cm_alpha_corr'][self.i_aero]*q_dyn*self.jcl.general['A_ref']*self.jcl.general['c_ref']*alpha
        if self.jcl.aero.has_key('Cn_beta_corr'):
            Pb_corr[5] += self.jcl.aero['Cn_beta_corr'][self.i_aero]*q_dyn*self.jcl.general['A_ref']*self.jcl.general['b_ref']*beta
        return Pb_corr    
    
    def vdrag(self, alpha, q_dyn):
        Pmac = np.zeros(6)
        if self.jcl.aero.has_key('viscous_drag') and self.jcl.aero['viscous_drag'] == 'coefficients':
            if self.jcl.aero.has_key('Cd_0'): Cd0 = self.jcl.aero['Cd_0'][self.i_aero]
            else:                             Cd0 = 0.012
            if self.jcl.aero.has_key('Cd_0'): Cd_alpha_sq = self.jcl.aero['Cd_alpha^2'][self.i_aero]
            else:                             Cd_alpha_sq = 0.018
            Pmac[0] = q_dyn*self.jcl.general['A_ref'] * (Cd0 + Cd_alpha_sq*alpha**2.0)
        return Pmac    
    
    def landinggear(self, X, Tgeo2body):
        Plg = np.zeros(self.model.lggrid['n']*6)
        F1 = []
        F2 = []
        Fx = []
        p2 = []
        dp2 = []
        ddp2 = []
        if self.simcase and self.simcase['landinggear']:
            # init
            PHIlg_cg = self.model.mass['PHIlg_cg'][self.i_mass]
            PHIf_lg = self.model.mass['PHIf_lg'][self.i_mass]
            PHIlg_cg = self.model.mass['PHIlg_cg'][self.model.mass['key'].index(self.trimcase['mass'])]
            p1   = (PHIlg_cg.dot(np.dot(self.PHInorm_cg, X[0:6 ])) + PHIf_lg.T.dot(X[12:12+self.n_modes])                 )[self.model.lggrid['set'][:,2]] # position LG attachment point over ground
            dp1  = (PHIlg_cg.dot(np.dot(self.PHInorm_cg, X[6:12])) + PHIf_lg.T.dot(X[12+self.n_modes:12+self.n_modes*2]))[self.model.lggrid['set'][:,2]] # velocity LG attachment point 
            p2  = X[12+self.n_modes*2+3:12+self.n_modes*2+3+self.model.lggrid['n']]  # position Tire center over ground
            dp2 = X[12+self.n_modes*2+3+self.model.lggrid['n']:12+self.n_modes*2+3+self.model.lggrid['n']*2] # velocity Tire center
            # loop over every landing gear
            for i in range(self.model.lggrid['n']):
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
                    F1.append(Ff+Fd)
                    F2.append(-Fg_tire-(Ff+Fd)+Fz)
                    Fx.append(0.25*Fz) # CS 25.479(d)(1)
                    ddp2.append(1.0/self.jcl.landinggear['para'][i]['m_tire']*(-Fg_tire-(Ff+Fd)+Fz))
                else: 
                    F1.append(0.0)
                    F2.append(0.0)
                    Fx.append(0.0)
                    ddp2.append(0.0)
                    
            # insert forces in 6dof vector Plg
            Plg[self.model.lggrid['set'][:,0]] = Fx 
            Plg[self.model.lggrid['set'][:,2]] = F1
            
        return Plg, p2, dp2, np.array(ddp2), np.array(F1), np.array(F2)

class nonlin_steady(common):

    def equations(self, X, t, type):
        self.counter += 1
        # recover states
        Tgeo2body = np.zeros((6,6))
        Tgeo2body[0:3,0:3] = calc_drehmatrix(X[3], X[4], X[5])
        Tgeo2body[3:6,3:6] = calc_drehmatrix_angular(X[3], X[4], X[5])
        Tbody2geo = np.zeros((6,6))
        Tbody2geo[0:3,0:3] = calc_drehmatrix(X[3], X[4], X[5]).T
        Tbody2geo[3:6,3:6] = calc_drehmatrix_angular_inv(X[3], X[4], X[5])
        dUcg_dt  = np.dot(self.PHInorm_cg, X[6:12]) # u v w p q r bodyfixed
        Uf = np.array(X[12:12+self.n_modes])
        dUf_dt = np.array(X[12+self.n_modes:12+self.n_modes*2])
               
        # aktuelle Vtas und q_dyn berechnen
        dxyz = X[6:9]
        Vtas = sum(dxyz**2)**0.5
        rho = self.model.atmo['rho'][self.i_atmo]
        q_dyn = rho/2.0*Vtas**2
        onflow  = np.dot(self.PHInorm_cg, X[6:12]) # u v w p q r bodyfixed
        alpha = np.arctan(onflow[2]/onflow[0]) #X[4] + np.arctan(X[8]/X[6]) # alpha = theta - gamma, Wind fehlt!
        beta  = np.arctan(onflow[1]/onflow[0]) #X[5] - np.arctan(X[7]/X[6])
        my    = 0.0
        
        # Steuerflaechenausschlaege vom efcs holen
        Ux2 = self.efcs.efcs(X[np.where(self.trimcond_X[:,0]=='command_xi')[0][0]], X[np.where(self.trimcond_X[:,0]=='command_eta')[0][0]], X[np.where(self.trimcond_X[:,0]=='command_zeta')[0][0]])
        
        # --------------------   
        # --- aerodynamics ---   
        # --------------------
        Pk_rbm,  wj_rbm  = self.rbm_nonlin(onflow, alpha, Vtas)
        Pk_cs,   wj_cs   = self.cs_nonlin(onflow, X, Ux2, Vtas)
        Pk_f,    wj_f    = self.flexible_nonlin(onflow, Uf, dUf_dt, Vtas)
#         Pk_rbm,  wj_rbm  = self.rbm(onflow, alpha, q_dyn, Vtas)
#         Pk_cs,   wj_cs   = self.cs(X, Ux2, q_dyn)
#         Pk_f,    wj_f    = self.flexible(Uf, dUf_dt, onflow, q_dyn, Vtas)
        
        wj = (wj_rbm + wj_cs + wj_f)/Vtas
        Pk_idrag         = self.idrag(wj, q_dyn)
        
        Pk_cam      = Pk_rbm*0.0
        Pk_gust     = Pk_rbm*0.0
        Pk_cfd      = Pk_rbm*0.0
        Pk_unsteady = Pk_rbm*0.0
        
        # -------------------------------  
        # --- correction coefficients ---   
        # -------------------------------
        Pb_corr = self.correctioon_coefficients(alpha, beta, q_dyn)
        Pmac_vdrag = self.vdrag(alpha, q_dyn)
        
        # ---------------------------   
        # --- summation of forces ---   
        # ---------------------------
        Pk_aero = Pk_rbm + Pk_cam + Pk_cs + Pk_f + Pk_gust + Pk_idrag + Pk_cfd + Pk_unsteady
        Pmac = np.dot(self.Dkx1.T, Pk_aero) + Pmac_vdrag
        Pb = np.dot(self.PHImac_cg.T, Pmac) + Pb_corr
        
        g = np.array([0.0, 0.0, 9.8066]) # erdfest, geodetic
        g_cg = np.dot(self.PHInorm_cg[0:3,0:3], np.dot(Tgeo2body[0:3,0:3],g)) # bodyfixed
               
        # -----------   
        # --- EoM ---   
        # -----------
        d2Ucg_dt2 = np.zeros(dUcg_dt.shape)
        if hasattr(self.jcl,'eom') and self.jcl.eom['version'] == 'waszak':
            # # non-linear EoM, bodyfixed / Waszak
            d2Ucg_dt2[0:3] = np.cross(dUcg_dt[0:3], dUcg_dt[3:6]) + np.dot(np.linalg.inv(self.Mb)[0:3,0:3], Pb[0:3]) + g_cg 
            d2Ucg_dt2[3:6] = np.dot(np.linalg.inv(self.Mb[3:6,3:6]) , Pb[3:6] - np.cross(dUcg_dt[3:6], np.dot(self.Mb[3:6,3:6], dUcg_dt[3:6])) )
            Nxyz = (d2Ucg_dt2[0:3] - g_cg - np.cross(dUcg_dt[0:3], dUcg_dt[3:6]) )/9.8066  
        else:
            # linear EoM, bodyfixed / Nastran
            d2Ucg_dt2[0:3] = np.dot(np.linalg.inv(self.Mb)[0:3,0:3], Pb[0:3]) + g_cg 
            d2Ucg_dt2[3:6] = np.dot(np.linalg.inv(self.Mb)[3:6,3:6], Pb[3:6] )
            Nxyz = (d2Ucg_dt2[0:3] - g_cg) /9.8066 
        
        Pf = np.dot(self.PHIkf.T, Pk_aero) + self.Mfcg.dot( np.hstack((d2Ucg_dt2[0:3] - g_cg, d2Ucg_dt2[3:6])) ) # viel schneller!
        # flexible EoM
        d2Uf_dt2 = np.dot( -np.linalg.inv(self.Mff),  ( np.dot(self.Dff, dUf_dt) + np.dot(self.Kff, Uf) - Pf  ) )
        
        # ----------------------
        # --- CS derivatives ---
        # ----------------------
        if self.simcase and self.simcase['cs_signal']:
            dcommand = self.efcs.cs_signal(t)
        elif self.simcase and self.simcase['controller']:
            #dcommand = self.efcs.controller(d2Ucg_dt2[3:6])
            #dcommand = self.efcs.controller(dUcg_dt[3:6])
            dcommand = self.efcs.controller(t=t, feedback_q=dUcg_dt[4], feedback_eta=X[np.where(self.trimcond_X[:,0]=='command_eta')[0][0]])
        else:
            dcommand = np.zeros(3)

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
                     ))
            
        if type in ['trim', 'sim']:
            return Y
        elif type in ['trim_full_output', 'sim_full_output']:
            # calculate translations, velocities and accelerations of some additional points
            # (might also be used for sensors in a closed-loop system
            if hasattr(self.jcl, 'landinggear') and self.jcl.landinggear['method'] == 'generic':
                PHIlg_cg = self.model.mass['PHIlg_cg'][self.model.mass['key'].index(self.trimcase['mass'])]
                PHIf_lg = self.model.mass['PHIf_lg'][self.model.mass['key'].index(self.trimcase['mass'])]
                p1   = (PHIlg_cg.dot(np.dot(self.PHInorm_cg, X[0:6 ])) + PHIf_lg.T.dot(X[12:12+self.n_modes])                 )[self.model.lggrid['set'][:,2]] # position LG attachment point over ground
                dp1  = (PHIlg_cg.dot(np.dot(self.PHInorm_cg, X[6:12])) + PHIf_lg.T.dot(X[12+self.n_modes:12+self.n_modes*2]))[self.model.lggrid['set'][:,2]] # velocity LG attachment point 
                ddp1 = (PHIlg_cg.dot(np.dot(self.PHInorm_cg, Y[6:12])) + PHIf_lg.T.dot(Y[12+self.n_modes:12+self.n_modes*2]))[self.model.lggrid['set'][:,2]] # acceleration LG attachment point 
                Plg  = np.zeros(self.model.lggrid['n']*6)
                F1   = np.zeros(self.model.lggrid['n']) 
                F2   = np.zeros(self.model.lggrid['n']) 
            else:
                p1 = ''
                dp1 = ''
                ddp1 = ''
                Plg = ''
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
                        'Pk_cfd': Pk_cfd,
                        'Pk_gust': Pk_gust,
                        'Pk_unsteady': Pk_unsteady,
                        'Pk_idrag': Pk_idrag,
                        'Pmac_vdrag': Pmac_vdrag,
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
                        'Plg': Plg,
                        'p1': p1,
                        'dp1': dp1,
                        'ddp1': ddp1,
                        'F1': F1,
                        'F2': F2,
                       }
            return response        
    def ode_arg_sorter(self, t, X):
        return self.eval_equations(X, t, 'sim')
            
    def eval_equations(self, X_free, time, type='trim_full_output'):
        # this is a wrapper for the model equations 'eqn_basic'
        if type in ['trim', 'trim_full_output']:
            # get inputs from trimcond and apply inputs from fsolve 
            X = np.array(self.trimcond_X[:,2], dtype='float')
            X[np.where((self.trimcond_X[:,1] == 'free'))[0]] = X_free
        elif type in[ 'sim', 'sim_full_output']:
            X = X_free
        
        # evaluate model equations
        if type=='trim':
            Y = self.equations(X, time, 'trim')
            # get the current values from Y and substract tamlab.figure()
            # fsolve only finds the roots; Y = 0
            Y_target_ist = Y[np.where((self.trimcond_Y[:,1] == 'target'))[0]]
            Y_target_soll = np.array(self.trimcond_Y[:,2], dtype='float')[np.where((self.trimcond_Y[:,1] == 'target'))[0]]
            out = Y_target_ist - Y_target_soll
            return out
        
        elif type=='sim':
            Y = self.equations(X, time, 'sim')
            return Y[:-2] # Nz ist eine Rechengroesse und keine Simulationsgroesse!
            
        elif type=='sim_full_output':
            response = self.equations(X, time, 'sim_full_output')
            return response
            
        elif type=='trim_full_output':
            response = self.equations(X, time, 'trim_full_output')
            # do something with this output, e.g. plotting, animations, saving, etc.            
            logging.info('')        
            logging.info('X: ')
            logging.info('--------------------')
            for i_X in range(len(response['X'])):
                logging.info(self.trimcond_X[:,0][i_X] + ': %.4f' % float(response['X'][i_X]))
            logging.info('Y: ')
            logging.info('--------------------')
            for i_Y in range(len(response['Y'])):
                logging.info(self.trimcond_Y[:,0][i_Y] + ': %.4f' % float(response['Y'][i_Y]))

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
            logging.info('')
            logging.info('--------------------')
            logging.info('q_dyn: %.4f [Pa]' % float(response['q_dyn']))
            logging.info('--------------------')
            logging.info('aero derivatives:')
            logging.info('--------------------')
            logging.info('Cz_rbm: %.4f' % float(Pmac_rbm[2]/response['q_dyn']/A))
            logging.info('Cz_cam: %.4f' % float(Pmac_cam[2]/response['q_dyn']/A))
            logging.info('Cz_cs: %.4f' % float(Pmac_cs[2]/response['q_dyn']/A))
            logging.info('Cz_f: %.4f' % float(Pmac_f[2]/response['q_dyn']/A))
            logging.info('--------------')
            logging.info('Cx: %.4f' % float(Pmac_c[0]))
            logging.info('Cy: %.4f' % float(Pmac_c[1]))
            logging.info('Cz: %.4f' % float(Pmac_c[2]))
            logging.info('Cmx: %.6f' % float(Pmac_c[3]/self.model.macgrid['b_ref']))
            logging.info('Cmy: %.6f' % float(Pmac_c[4]/self.model.macgrid['c_ref']))
            logging.info('Cmz: %.6f' % float(Pmac_c[5]/self.model.macgrid['b_ref']))
            #logging.info('dCmz_dbeta: %.6f' % float(Pmac_c[5]/self.model.macgrid['b_ref']/response['beta'])
            logging.info('alpha: %.4f [deg]' % float(response['alpha']/np.pi*180))
            logging.info('beta: %.4f [deg]' % float(response['beta']/np.pi*180))
            logging.info('Cd: %.4f' % float(Cd))
            logging.info('Cl: %.4f' % float(Cl))
            logging.info('E: %.4f' % float(Cl/Cd))
            logging.info('Cd_vis: %.6f' % float(response['Pmac_vdrag'][0]/response['q_dyn']/A))
            logging.info('Cd_ind: %.6f' % float(Pmac_idrag[0]/response['q_dyn']/A))
            logging.info('Cmz_ind: %.6f' % float(Pmac_idrag[5]/response['q_dyn']/A/self.model.macgrid['b_ref']))
            logging.info('e: %.4f' % float(Cd_ind_theo/(Pmac_idrag[0]/response['q_dyn']/A)))
            logging.info('command_xi: %.4f [rad] / %.4f [deg]' % (float( response['X'][np.where(self.trimcond_X[:,0]=='command_xi')[0][0]]), float( response['X'][np.where(self.trimcond_X[:,0]=='command_xi')[0][0]])/np.pi*180.0 ))
            logging.info('command_eta: %.4f [rad] / %.4f [deg]' % (float( response['X'][np.where(self.trimcond_X[:,0]=='command_eta')[0][0]]), float( response['X'][np.where(self.trimcond_X[:,0]=='command_eta')[0][0]])/np.pi*180.0 ))
            logging.info('command_zeta: %.4f [rad] / %.4f [deg]' % (float( response['X'][np.where(self.trimcond_X[:,0]=='command_zeta')[0][0]]), float( response['X'][np.where(self.trimcond_X[:,0]=='command_zeta')[0][0]])/np.pi*180.0 ))
            logging.info('CS deflections [deg]: ' + str(response['Ux2']/np.pi*180))
            #logging.info('dCz_da: %.4f' % float(Pmac_c[2]/response['alpha']))
            #logging.info('dCmy_da: %.4f' % float(Pmac_c[4]/self.model.macgrid['c_ref']/response['alpha']))
            #logging.info('dCmz_db: %.4f' % float(Pmac_c[4]/self.model.macgrid['b_ref']/response['beta']))
            logging.info('--------------------')
            
            return response
    
class steady(common):

    def equations(self, X, t, type):
        self.counter += 1
        # recover states
        Tgeo2body = np.zeros((6,6))
        Tgeo2body[0:3,0:3] = calc_drehmatrix(X[3], X[4], X[5])
        Tgeo2body[3:6,3:6] = calc_drehmatrix_angular(X[3], X[4], X[5])
        Tbody2geo = np.zeros((6,6))
        Tbody2geo[0:3,0:3] = calc_drehmatrix(X[3], X[4], X[5]).T
        Tbody2geo[3:6,3:6] = calc_drehmatrix_angular_inv(X[3], X[4], X[5])
        dUcg_dt  = np.dot(self.PHInorm_cg, X[6:12]) # u v w p q r bodyfixed
        Uf = np.array(X[12:12+self.n_modes])
        dUf_dt = np.array(X[12+self.n_modes:12+self.n_modes*2])
               
        # aktuelle Vtas und q_dyn berechnen
        dxyz = X[6:9]
        Vtas = sum(dxyz**2)**0.5
        rho = self.model.atmo['rho'][self.i_atmo]
        q_dyn = rho/2.0*Vtas**2
        onflow  = np.dot(self.PHInorm_cg, X[6:12]) # u v w p q r bodyfixed
        alpha = np.arctan(onflow[2]/onflow[0]) #X[4] + np.arctan(X[8]/X[6]) # alpha = theta - gamma, Wind fehlt!
        beta  = np.arctan(onflow[1]/onflow[0]) #X[5] - np.arctan(X[7]/X[6])
        my    = 0.0
        
        # Steuerflaechenausschlaege vom efcs holen
        Ux2 = self.efcs.efcs(X[np.where(self.trimcond_X[:,0]=='command_xi')[0][0]], X[np.where(self.trimcond_X[:,0]=='command_eta')[0][0]], X[np.where(self.trimcond_X[:,0]=='command_zeta')[0][0]])
        
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
        Pk_cfd           = self.cfd( alpha, X, Ux2, q_dyn)
        
        Pk_unsteady = Pk_rbm*0.0
        
        # -------------------------------  
        # --- correction coefficients ---   
        # -------------------------------
        Pb_corr = self.correctioon_coefficients(alpha, beta, q_dyn)
        
        # ---------------------------   
        # --- summation of forces ---   
        # ---------------------------
        Pk_aero = Pk_rbm + Pk_cam + Pk_cs + Pk_f + Pk_gust + Pk_idrag + Pk_cfd + Pk_unsteady
        Pmac = np.dot(self.Dkx1.T, Pk_aero)
        Pb = np.dot(self.PHImac_cg.T, Pmac) + Pb_corr
        
        g = np.array([0.0, 0.0, 9.8066]) # erdfest, geodetic
        g_cg = np.dot(self.PHInorm_cg[0:3,0:3], np.dot(Tgeo2body[0:3,0:3],g)) # bodyfixed
               
        # -----------   
        # --- EoM ---   
        # -----------
        d2Ucg_dt2 = np.zeros(dUcg_dt.shape)
        if hasattr(self.jcl,'eom') and self.jcl.eom['version'] == 'waszak':
            # # non-linear EoM, bodyfixed / Waszak
            d2Ucg_dt2[0:3] = np.cross(dUcg_dt[0:3], dUcg_dt[3:6]) + np.dot(np.linalg.inv(self.Mb)[0:3,0:3], Pb[0:3]) + g_cg 
            d2Ucg_dt2[3:6] = np.dot(np.linalg.inv(self.Mb[3:6,3:6]) , Pb[3:6] - np.cross(dUcg_dt[3:6], np.dot(self.Mb[3:6,3:6], dUcg_dt[3:6])) )
            Nxyz = (d2Ucg_dt2[0:3] - g_cg - np.cross(dUcg_dt[0:3], dUcg_dt[3:6]) )/9.8066  
        else:
            # linear EoM, bodyfixed / Nastran
            d2Ucg_dt2[0:3] = np.dot(np.linalg.inv(self.Mb)[0:3,0:3], Pb[0:3]) + g_cg 
            d2Ucg_dt2[3:6] = np.dot(np.linalg.inv(self.Mb)[3:6,3:6], Pb[3:6] )
            Nxyz = (d2Ucg_dt2[0:3] - g_cg) /9.8066 
        
        Pf = np.dot(self.PHIkf.T, Pk_aero) + self.Mfcg.dot( np.hstack((d2Ucg_dt2[0:3] - g_cg, d2Ucg_dt2[3:6])) ) # viel schneller!
        # flexible EoM
        d2Uf_dt2 = np.dot( -np.linalg.inv(self.Mff),  ( np.dot(self.Dff, dUf_dt) + np.dot(self.Kff, Uf) - Pf  ) )
        
        # ----------------------
        # --- CS derivatives ---
        # ----------------------
        if self.simcase and self.simcase['cs_signal']:
            dcommand = self.efcs.cs_signal(t)
        elif self.simcase and self.simcase['controller']:
            #dcommand = self.efcs.controller(d2Ucg_dt2[3:6])
            #dcommand = self.efcs.controller(dUcg_dt[3:6])
            dcommand = self.efcs.controller(t=t, feedback_q=dUcg_dt[4], feedback_eta=X[np.where(self.trimcond_X[:,0]=='command_eta')[0][0]])
        else:
            dcommand = np.zeros(3)

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
                     )) 
        
        if type in ['trim', 'sim']:
            return Y
        elif type in ['trim_full_output', 'sim_full_output']:
            # calculate translations, velocities and accelerations of some additional points
            # (might also be used for sensors in a closed-loop system
            if hasattr(self.jcl, 'landinggear') and self.jcl.landinggear['method'] == 'generic':
                PHIlg_cg = self.model.mass['PHIlg_cg'][self.model.mass['key'].index(self.trimcase['mass'])]
                PHIf_lg = self.model.mass['PHIf_lg'][self.model.mass['key'].index(self.trimcase['mass'])]
                p1   = (PHIlg_cg.dot(np.dot(self.PHInorm_cg, X[0:6 ])) + PHIf_lg.T.dot(X[12:12+self.n_modes])                 )[self.model.lggrid['set'][:,2]] # position LG attachment point over ground
                dp1  = (PHIlg_cg.dot(np.dot(self.PHInorm_cg, X[6:12])) + PHIf_lg.T.dot(X[12+self.n_modes:12+self.n_modes*2]))[self.model.lggrid['set'][:,2]] # velocity LG attachment point 
                ddp1 = (PHIlg_cg.dot(np.dot(self.PHInorm_cg, Y[6:12])) + PHIf_lg.T.dot(Y[12+self.n_modes:12+self.n_modes*2]))[self.model.lggrid['set'][:,2]] # acceleration LG attachment point 
                Plg  = np.zeros(self.model.lggrid['n']*6)
                F1   = np.zeros(self.model.lggrid['n']) 
                F2   = np.zeros(self.model.lggrid['n']) 
            else:
                p1 = ''
                dp1 = ''
                ddp1 = ''
                Plg = ''
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
                        'Pk_cfd': Pk_cfd,
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
                        'Plg': Plg,
                        'p1': p1,
                        'dp1': dp1,
                        'ddp1': ddp1,
                        'F1': F1,
                        'F2': F2,
                       }
            return response        
    def ode_arg_sorter(self, t, X):
        return self.eval_equations(X, t, 'sim')
            
    def eval_equations(self, X_free, time, type='trim_full_output'):
        # this is a wrapper for the model equations 'eqn_basic'
        if type in ['trim', 'trim_full_output']:
            # get inputs from trimcond and apply inputs from fsolve 
            X = np.array(self.trimcond_X[:,2], dtype='float')
            X[np.where((self.trimcond_X[:,1] == 'free'))[0]] = X_free
        elif type in[ 'sim', 'sim_full_output']:
            X = X_free
        
        # evaluate model equations
        if type=='trim':
            Y = self.equations(X, time, 'trim')
            # get the current values from Y and substract tamlab.figure()
            # fsolve only finds the roots; Y = 0
            Y_target_ist = Y[np.where((self.trimcond_Y[:,1] == 'target'))[0]]
            Y_target_soll = np.array(self.trimcond_Y[:,2], dtype='float')[np.where((self.trimcond_Y[:,1] == 'target'))[0]]
            out = Y_target_ist - Y_target_soll
            return out
        
        elif type=='sim':
            Y = self.equations(X, time, 'sim')
            return Y[:-2] # Nz ist eine Rechengroesse und keine Simulationsgroesse!
            
        elif type=='sim_full_output':
            response = self.equations(X, time, 'sim_full_output')
            return response
            
        elif type=='trim_full_output':
            response = self.equations(X, time, 'trim_full_output')
            # do something with this output, e.g. plotting, animations, saving, etc.            
            logging.info('')        
            logging.info('Y: ')
            logging.info('--------------------')
            for i_Y in range(len(response['Y'])):
                logging.info(self.trimcond_Y[:,0][i_Y] + ': %.4f' % float(response['Y'][i_Y]))

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
            logging.info('')
            logging.info('--------------------')
            logging.info('q_dyn: %.4f [Pa]' % float(response['q_dyn']))
            logging.info('--------------------')
            logging.info('aero derivatives:')
            logging.info('--------------------')
            logging.info('Cz_rbm: %.4f' % float(Pmac_rbm[2]/response['q_dyn']/A))
            logging.info('Cz_cam: %.4f' % float(Pmac_cam[2]/response['q_dyn']/A))
            logging.info('Cz_cs: %.4f' % float(Pmac_cs[2]/response['q_dyn']/A))
            logging.info('Cz_f: %.4f' % float(Pmac_f[2]/response['q_dyn']/A))
            logging.info('--------------')
            logging.info('Cx: %.4f' % float(Pmac_c[0]))
            logging.info('Cy: %.4f' % float(Pmac_c[1]))
            logging.info('Cz: %.4f' % float(Pmac_c[2]))
            logging.info('Cmx: %.6f' % float(Pmac_c[3]/self.model.macgrid['b_ref']))
            logging.info('Cmy: %.6f' % float(Pmac_c[4]/self.model.macgrid['c_ref']))
            logging.info('Cmz: %.6f' % float(Pmac_c[5]/self.model.macgrid['b_ref']))
            #logging.info('dCmz_dbeta: %.6f' % float(Pmac_c[5]/self.model.macgrid['b_ref']/response['beta'])
            logging.info('alpha: %.4f [deg]' % float(response['alpha']/np.pi*180))
            logging.info('beta: %.4f [deg]' % float(response['beta']/np.pi*180))
            logging.info('Cd: %.4f' % float(Cd))
            logging.info('Cl: %.4f' % float(Cl))
            logging.info('Cd_ind: %.6f' % float(Pmac_idrag[0]/response['q_dyn']/A))
            logging.info('Cmz_ind: %.6f' % float(Pmac_idrag[5]/response['q_dyn']/A/self.model.macgrid['b_ref']))
            #logging.info('e: %.4f' % float(Cd_ind_theo/(Pmac_idrag[0]/response['q_dyn']/A)))
            logging.info('command_xi: %.4f [rad] / %.4f [deg]' % (float( response['X'][np.where(self.trimcond_X[:,0]=='command_xi')[0][0]]), float( response['X'][np.where(self.trimcond_X[:,0]=='command_xi')[0][0]])/np.pi*180.0 ))
            logging.info('command_eta: %.4f [rad] / %.4f [deg]' % (float( response['X'][np.where(self.trimcond_X[:,0]=='command_eta')[0][0]]), float( response['X'][np.where(self.trimcond_X[:,0]=='command_eta')[0][0]])/np.pi*180.0 ))
            logging.info('command_zeta: %.4f [rad] / %.4f [deg]' % (float( response['X'][np.where(self.trimcond_X[:,0]=='command_zeta')[0][0]]), float( response['X'][np.where(self.trimcond_X[:,0]=='command_zeta')[0][0]])/np.pi*180.0 ))
            logging.info('CS deflections [deg]: ' + str(response['Ux2']/np.pi*180))
            #logging.info('dCz_da: %.4f' % float(Pmac_c[2]/response['alpha']))
            #logging.info('dCmy_da: %.4f' % float(Pmac_c[4]/self.model.macgrid['c_ref']/response['alpha']))
            #logging.info('dCmz_db: %.4f' % float(Pmac_c[4]/self.model.macgrid['b_ref']/response['beta']))
            logging.info('--------------------')
            
            return response
        
class unsteady(common):

    def equations(self, X, t, type):
        self.counter += 1
        # recover states
        Tgeo2body = np.zeros((6,6))
        Tgeo2body[0:3,0:3] = calc_drehmatrix(X[3], X[4], X[5])
        Tgeo2body[3:6,3:6] = calc_drehmatrix_angular(X[3], X[4], X[5])
        Tbody2geo = np.zeros((6,6))
        Tbody2geo[0:3,0:3] = calc_drehmatrix(X[3], X[4], X[5]).T
        Tbody2geo[3:6,3:6] = calc_drehmatrix_angular_inv(X[3], X[4], X[5])
        dUcg_dt  = np.dot(self.PHInorm_cg, X[6:12]) # u v w p q r bodyfixed
        Uf = np.array(X[12:12+self.n_modes])
        dUf_dt = np.array(X[12+self.n_modes:12+self.n_modes*2])
               
        # aktuelle Vtas und q_dyn berechnen
        dxyz = X[6:9]
        Vtas = sum(dxyz**2)**0.5
        rho = self.model.atmo['rho'][self.i_atmo]
        q_dyn = rho/2.0*Vtas**2
        onflow  = np.dot(self.PHInorm_cg, X[6:12]) # u v w p q r bodyfixed
        alpha = np.arctan(onflow[2]/onflow[0]) #X[4] + np.arctan(X[8]/X[6]) # alpha = theta - gamma, Wind fehlt!
        beta  = np.arctan(onflow[1]/onflow[0]) #X[5] - np.arctan(X[7]/X[6])
        my    = 0.0
        
        # Steuerflaechenausschlaege vom efcs holen
        Ux2 = self.efcs.efcs(X[np.where(self.trimcond_X[:,0]=='command_xi')[0][0]], X[np.where(self.trimcond_X[:,0]=='command_eta')[0][0]], X[np.where(self.trimcond_X[:,0]=='command_zeta')[0][0]])
        
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
        Pk_unsteady, Pk_unsteady_B, Pk_unsteady_D, dlag_states_dt = self.unsteady(X, t, wj, q_dyn, Vtas)
        
        Pk_cfd = Pk_rbm*0.0
        
        # -------------------------------  
        # --- correction coefficients ---   
        # -------------------------------
        Pb_corr = self.correctioon_coefficients(alpha, beta, q_dyn)
        
        # ---------------------------   
        # --- summation of forces ---   
        # ---------------------------
        Pk_aero = Pk_rbm + Pk_cam + Pk_cs + Pk_f + Pk_gust + Pk_idrag + Pk_cfd + Pk_unsteady
        Pmac = np.dot(self.Dkx1.T, Pk_aero)
        Pb = np.dot(self.PHImac_cg.T, Pmac) + Pb_corr
        
        g = np.array([0.0, 0.0, 9.8066]) # erdfest, geodetic
        g_cg = np.dot(self.PHInorm_cg[0:3,0:3], np.dot(Tgeo2body[0:3,0:3],g)) # bodyfixed
               
        # -----------   
        # --- EoM ---   
        # -----------
        d2Ucg_dt2 = np.zeros(dUcg_dt.shape)
        if hasattr(self.jcl,'eom') and self.jcl.eom['version'] == 'waszak':
            # # non-linear EoM, bodyfixed / Waszak
            d2Ucg_dt2[0:3] = np.cross(dUcg_dt[0:3], dUcg_dt[3:6]) + np.dot(np.linalg.inv(self.Mb)[0:3,0:3], Pb[0:3]) + g_cg 
            d2Ucg_dt2[3:6] = np.dot(np.linalg.inv(self.Mb[3:6,3:6]) , Pb[3:6] - np.cross(dUcg_dt[3:6], np.dot(self.Mb[3:6,3:6], dUcg_dt[3:6])) )
            Nxyz = (d2Ucg_dt2[0:3] - g_cg - np.cross(dUcg_dt[0:3], dUcg_dt[3:6]) )/9.8066  
        else:
            # linear EoM, bodyfixed / Nastran
            d2Ucg_dt2[0:3] = np.dot(np.linalg.inv(self.Mb)[0:3,0:3], Pb[0:3]) + g_cg 
            d2Ucg_dt2[3:6] = np.dot(np.linalg.inv(self.Mb)[3:6,3:6], Pb[3:6] )
            Nxyz = (d2Ucg_dt2[0:3] - g_cg) /9.8066 
        
        Pf = np.dot(self.PHIkf.T, Pk_aero) + self.Mfcg.dot( np.hstack((d2Ucg_dt2[0:3] - g_cg, d2Ucg_dt2[3:6])) ) # viel schneller!
        # flexible EoM
        d2Uf_dt2 = np.dot( -np.linalg.inv(self.Mff),  ( np.dot(self.Dff, dUf_dt) + np.dot(self.Kff, Uf) - Pf  ) )
        
        # ----------------------
        # --- CS derivatives ---
        # ----------------------
        if self.simcase and self.simcase['cs_signal']:
            dcommand = self.efcs.cs_signal(t)
        elif self.simcase and self.simcase['controller']:
            #dcommand = self.efcs.controller(angular_acc=d2Ucg_dt2[3:6])
            dcommand = self.efcs.controller(t=t, feedback_q=dUcg_dt[4], feedback_eta=X[np.where(self.trimcond_X[:,0]=='command_eta')[0][0]])
        else:
            dcommand= np.zeros(3)

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
                     ))
             
        if type in ['trim', 'sim']:
            return Y
        elif type in ['trim_full_output', 'sim_full_output']:
            # calculate translations, velocities and accelerations of some additional points
            # (might also be used for sensors in a closed-loop system
            if hasattr(self.jcl, 'landinggear') and self.jcl.landinggear['method'] == 'generic':
                PHIlg_cg = self.model.mass['PHIlg_cg'][self.model.mass['key'].index(self.trimcase['mass'])]
                PHIf_lg = self.model.mass['PHIf_lg'][self.model.mass['key'].index(self.trimcase['mass'])]
                p1   = (PHIlg_cg.dot(np.dot(self.PHInorm_cg, X[0:6 ])) + PHIf_lg.T.dot(X[12:12+self.n_modes])                 )[self.model.lggrid['set'][:,2]] # position LG attachment point over ground
                dp1  = (PHIlg_cg.dot(np.dot(self.PHInorm_cg, X[6:12])) + PHIf_lg.T.dot(X[12+self.n_modes:12+self.n_modes*2]))[self.model.lggrid['set'][:,2]] # velocity LG attachment point 
                ddp1 = (PHIlg_cg.dot(np.dot(self.PHInorm_cg, Y[6:12])) + PHIf_lg.T.dot(Y[12+self.n_modes:12+self.n_modes*2]))[self.model.lggrid['set'][:,2]] # acceleration LG attachment point 
                Plg  = np.zeros(self.model.lggrid['n']*6)
                F1   = np.zeros(self.model.lggrid['n']) 
                F2   = np.zeros(self.model.lggrid['n']) 
            else:
                p1 = ''
                dp1 = ''
                ddp1 = ''
                Plg = ''
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
                        'Pk_cfd': Pk_cfd,
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
                        'Plg': Plg,
                        'p1': p1,
                        'dp1': dp1,
                        'ddp1': ddp1,
                        'F1': F1,
                        'F2': F2,
                       }
            return response        
    
    def ode_arg_sorter(self, t, X):
        return self.eval_equations(X, t, 'sim')
            
    def eval_equations(self, X_free, time, type='trim_full_output'):
        if type in[ 'sim', 'sim_full_output']:
            X = X_free
        
        # evaluate model equations
        if type=='sim':
            Y = self.equations(X, time, 'sim')
            return Y[:-2] # Nz ist eine Rechengroesse und keine Simulationsgroesse!
            
        elif type=='sim_full_output':
            response = self.equations(X, time, 'sim_full_output')
            return response
      
class landing(common):

    def equations(self, X, t, type):
        self.counter += 1
        # get additional tranlation matricies
        PHIlg_cg = self.model.mass['PHIlg_cg'][self.i_mass]
        PHIf_lg = self.model.mass['PHIf_lg'][self.i_mass]
        # recover states
        Tgeo2body = np.zeros((6,6))
        Tgeo2body[0:3,0:3] = calc_drehmatrix(X[3], X[4], X[5])
        Tgeo2body[3:6,3:6] = calc_drehmatrix_angular(X[3], X[4], X[5])
        Tbody2geo = np.zeros((6,6))
        Tbody2geo[0:3,0:3] = calc_drehmatrix(X[3], X[4], X[5]).T
        Tbody2geo[3:6,3:6] = calc_drehmatrix_angular_inv(X[3], X[4], X[5])
        dUcg_dt  = np.dot(self.PHInorm_cg, X[6:12]) # u v w p q r bodyfixed
        Uf = np.array(X[12:12+self.n_modes])
        dUf_dt = np.array(X[12+self.n_modes:12+self.n_modes*2])
               
        # aktuelle Vtas und q_dyn berechnen
        dxyz = X[6:9]
        Vtas = sum(dxyz**2)**0.5
        rho = self.model.atmo['rho'][self.i_atmo]
        q_dyn = rho/2.0*Vtas**2
        onflow  = np.dot(self.PHInorm_cg, X[6:12]) # u v w p q r bodyfixed
        alpha = np.arctan(onflow[2]/onflow[0]) #X[4] + np.arctan(X[8]/X[6]) # alpha = theta - gamma, Wind fehlt!
        beta  = np.arctan(onflow[1]/onflow[0]) #X[5] - np.arctan(X[7]/X[6])
        my    = 0.0
        
        # Steuerflaechenausschlaege vom efcs holen
        Ux2 = self.efcs.efcs(X[np.where(self.trimcond_X[:,0]=='command_xi')[0][0]], X[np.where(self.trimcond_X[:,0]=='command_eta')[0][0]], X[np.where(self.trimcond_X[:,0]=='command_zeta')[0][0]])
        
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
        Pk_cfd           = self.cfd( alpha, X, Ux2, q_dyn)
        
        Pk_unsteady = Pk_rbm*0.0
        
        # -------------------------------  
        # --- correction coefficients ---   
        # -------------------------------
        Pb_corr = self.correctioon_coefficients(alpha, beta, q_dyn)
        
        # -------------------- 
        # --- landing gear ---   
        # --------------------
        Plg, p2, dp2, ddp2, F1, F2 = self.landinggear(X, Tgeo2body)
        
        # ---------------------------   
        # --- summation of forces ---   
        # ---------------------------
        Pk_aero = Pk_rbm + Pk_cam + Pk_cs + Pk_f + Pk_gust + Pk_idrag + Pk_cfd + Pk_unsteady
        Pmac = np.dot(self.Dkx1.T, Pk_aero)
        Pb = np.dot(self.PHImac_cg.T, Pmac) + Pb_corr + np.dot(PHIlg_cg.T, Plg)
        
        g = np.array([0.0, 0.0, 9.8066]) # erdfest, geodetic
        g_cg = np.dot(self.PHInorm_cg[0:3,0:3], np.dot(Tgeo2body[0:3,0:3],g)) # bodyfixed
               
        # -----------   
        # --- EoM ---   
        # -----------
        d2Ucg_dt2 = np.zeros(dUcg_dt.shape)
        if hasattr(self.jcl,'eom') and self.jcl.eom['version'] == 'waszak':
            # # non-linear EoM, bodyfixed / Waszak
            d2Ucg_dt2[0:3] = np.cross(dUcg_dt[0:3], dUcg_dt[3:6]) + np.dot(np.linalg.inv(self.Mb)[0:3,0:3], Pb[0:3]) + g_cg 
            d2Ucg_dt2[3:6] = np.dot(np.linalg.inv(self.Mb[3:6,3:6]) , Pb[3:6] - np.cross(dUcg_dt[3:6], np.dot(self.Mb[3:6,3:6], dUcg_dt[3:6])) )
            Nxyz = (d2Ucg_dt2[0:3] - g_cg - np.cross(dUcg_dt[0:3], dUcg_dt[3:6]) )/9.8066  
        else:
            # linear EoM, bodyfixed / Nastran
            d2Ucg_dt2[0:3] = np.dot(np.linalg.inv(self.Mb)[0:3,0:3], Pb[0:3]) + g_cg 
            d2Ucg_dt2[3:6] = np.dot(np.linalg.inv(self.Mb)[3:6,3:6], Pb[3:6] )
            Nxyz = (d2Ucg_dt2[0:3] - g_cg) /9.8066 
        
        Pf = np.dot(self.PHIkf.T, Pk_aero) + self.Mfcg.dot( np.hstack((d2Ucg_dt2[0:3] - g_cg, d2Ucg_dt2[3:6])) ) + np.dot(PHIf_lg, Plg) # viel schneller!
        # flexible EoM
        d2Uf_dt2 = np.dot( -np.linalg.inv(self.Mff),  ( np.dot(self.Dff, dUf_dt) + np.dot(self.Kff, Uf) - Pf  ) )
        
        # ----------------------
        # --- CS derivatives ---
        # ----------------------
        if self.simcase and self.simcase['cs_signal']:
            dcommand = self.efcs.cs_signal(t)
        elif self.simcase and self.simcase['controller']:
            dcommand = self.efcs.controller(angular_acc=d2Ucg_dt2[3:6])
        else:
            dcommand= np.zeros(3)

        # --------------   
        # --- output ---   
        # --------------
        Y = np.hstack((np.dot(Tbody2geo,X[6:12]), 
                       np.dot(self.PHIcg_norm,  d2Ucg_dt2), 
                       dUf_dt, 
                       d2Uf_dt2, 
                       dcommand, 
                       np.hstack((dp2, ddp2)),
                       Nxyz[2],
                       Vtas, 
                     ))
             
        if type in ['trim', 'sim']:
            return Y
        elif type in ['trim_full_output', 'sim_full_output']:
            # calculate translations, velocities and accelerations of some additional points
            # (might also be used for sensors in a closed-loop system
            p1   = (PHIlg_cg.dot(np.dot(self.PHInorm_cg, X[0:6 ])) + PHIf_lg.T.dot(X[12:12+self.n_modes])               )[self.model.lggrid['set'][:,2]] # position LG attachment point over ground
            dp1  = (PHIlg_cg.dot(np.dot(self.PHInorm_cg, X[6:12])) + PHIf_lg.T.dot(X[12+self.n_modes:12+self.n_modes*2]))[self.model.lggrid['set'][:,2]] # velocity LG attachment point 
            ddp1 = (PHIlg_cg.dot(np.dot(self.PHInorm_cg, Y[6:12])) + PHIf_lg.T.dot(Y[12+self.n_modes:12+self.n_modes*2]))[self.model.lggrid['set'][:,2]] # acceleration LG attachment point 

            response = {'X': X, 
                        'Y': Y,
                        't': np.array([t]),
                        'Pk_rbm': Pk_rbm,
                        'Pk_cam': Pk_cam,
                        'Pk_aero': Pk_aero,
                        'Pk_cs': Pk_cs,
                        'Pk_f': Pk_f,
                        'Pk_cfd': Pk_cfd,
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
                        'Plg': Plg,
                        'p1': p1,
                        'dp1': dp1,
                        'ddp1': ddp1,
                        'F1': F1,
                        'F2': F2,
                        }
            return response        
    
    def ode_arg_sorter(self, t, X):
        return self.eval_equations(X, t, 'sim')  
        
    def eval_equations(self, X_free, time, type='trim_full_output'):
        if type in[ 'sim', 'sim_full_output']:
            X = X_free
        
        # evaluate model equations
        if type=='sim':
            Y = self.equations(X, time, 'sim')
            return Y[:-2] # Nz ist eine Rechengroesse und keine Simulationsgroesse!
            
        elif type=='sim_full_output':
            response = self.equations(X, time, 'sim_full_output')
            return response
