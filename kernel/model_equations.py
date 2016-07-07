# -*- coding: utf-8 -*-
"""
Created on Thu Nov 27 15:43:35 2014

@author: voss_ar
"""

import numpy as np
import importlib
#import time
from trim_tools import * 
from scipy import interpolate

class hybrid:
    def __init__(self, model, jcl, trimcase, trimcond_X, trimcond_Y, simcase = False):
        print 'Init hybrid model equations with linear EoM in Nastran-Style.'
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
        
        # q_dyn aus trim condition berechnen
        # es erfolg spaeter kein update mehr, sollte sich die Geschwindigkeit aendern !
        uvw = np.array(self.trimcond_X[6:9,2], dtype='float')
        self.Vtas = sum(uvw**2)**0.5
        rho = self.model.atmo['rho'][self.i_atmo]
        self.q_dyn = rho/2.0*self.Vtas**2
        
        if self.jcl.aero.has_key('flex') and self.jcl.aero['flex'] == True:
            self.k_flex = 1.0
        else:
            self.k_flex = 0.0
            
        if self.jcl.aero.has_key('hingeline') and self.jcl.aero['hingeline'] == 'y':
            self.hingeline = 'y'
        elif self.jcl.aero.has_key('hingeline') and self.jcl.aero['hingeline'] == 'z':
            self.hingeline = 'z'
        else: # default
            self.hingeline = 'y'
        
        # init aero db for hybrid aero: alpha
        if self.jcl.aero['method'] == 'hybrid' and self.model.aerodb.has_key('alpha') and self.trimcase['aero'] in self.model.aerodb['alpha']:
            self.correct_alpha = True
            self.aerodb_alpha = self.model.aerodb['alpha'][self.trimcase['aero']]
            # interp1d:  x has to be an array of monotonically increasing values
            self.aerodb_alpha_interpolation_function = interpolate.interp1d(np.array(self.aerodb_alpha['values'])/180.0*np.pi, np.array(self.aerodb_alpha['Pk']), axis=0, bounds_error = True )
            print 'Hybrid aero is used for alpha.'
            #print 'Forces from aero db ({}) will be scaled from q_dyn = {:.2f} to current q_dyn = {:.2f}.'.format(self.trimcase['aero'], self.aerodb_alpha['q_dyn'], self.q_dyn)        
        else:
            self.correct_alpha = False      
       
        # import aircraft-specific class from efcs.py dynamically 
        module = importlib.import_module('efcs')
        efcs_class = getattr(module, jcl.efcs['version'])
        
        # init efcs
        self.efcs =  efcs_class() 
        
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
                print 'Hybrid aero is used for {}.'.format(x2_key)
                #print 'Forces from aero db ({}) will be scaled from q_dyn = {:.2f} to current q_dyn = {:.2f}.'.format(self.trimcase['aero'], self.aerodb_x2[-1]['q_dyn'], self.q_dyn)       
           
        if self.simcase and self.simcase['gust']:
            V_D = self.model.atmo['a'][self.i_atmo] * self.simcase['gust_para']['MD']
            self.x0 = self.simcase['gust_para']['T1'] * self.Vtas 
            self.WG_TAS, U_ds, V_gust = DesignGust_CS_25_341(self.simcase['gust_gradient'], self.model.atmo['h'][self.i_atmo], rho, self.Vtas, self.simcase['gust_para']['Z_mo'], V_D, self.simcase['gust_para']['MLW'], self.simcase['gust_para']['MTOW'], self.simcase['gust_para']['MZFW'])

    def equations(self, X, t, type):
        self.counter += 1
        #t_start = time.time()
        #print 'flexible EoM in %.2f [sec].' % (time.time() - t_start)

        # Trim-spezifische Modelldaten holen, lange Namen abkuerzen
        Qjj        = self.model.aero['Qjj'][self.i_aero]    
       
        PHImac_cg  = self.model.mass['PHImac_cg'][self.i_mass]
        PHIcg_mac  = self.model.mass['PHIcg_mac'][self.i_mass]
        PHInorm_cg = self.model.mass['PHInorm_cg'][self.i_mass]
        PHIcg_norm = self.model.mass['PHIcg_norm'][self.i_mass]
        Mb         = self.model.mass['Mb'][self.i_mass]
        Mff        = self.model.mass['Mff'][self.i_mass]
        Kff        = self.model.mass['Kff'][self.i_mass]
        Dff        = self.model.mass['Dff'][self.i_mass]
        PHIf_strc  = self.model.mass['PHIf_strc'][self.i_mass]
        PHIstrc_cg = self.model.mass['PHIstrc_cg'][self.i_mass]
        Mgg        = self.model.mass['MGG'][self.i_mass]
        Mfcg        = self.model.mass['Mfcg'][self.i_mass]
        PHIjf      = self.model.mass['PHIjf'][self.i_mass]
        PHIkf      = self.model.mass['PHIkf'][self.i_mass]
        n_modes    = self.model.mass['n_modes'][self.i_mass] 
        
        PHIk_strc  = self.model.PHIk_strc
        Djx1       = self.model.Djx1
        Dkx1       = self.model.Dkx1
        
        # recover states
        Tgeo2body = np.zeros((6,6))
        Tgeo2body[0:3,0:3] = calc_drehmatrix(X[3], X[4], X[5])
        Tgeo2body[3:6,3:6] = calc_drehmatrix(X[3], X[4], X[5])
        Ucg      = np.dot(PHIcg_norm,X[0:6] ) # x, y, z, phi, theta, psi bodyfixed
        dUcg_dt  = np.dot(PHIcg_norm,np.dot(Tgeo2body, X[6:12])) # u v w p q r bodyfixed
        Uf = np.array(X[12:12+n_modes])
        dUf_dt = np.array(X[12+n_modes:12+n_modes*2])
        
        dUmac_dt = np.dot(PHImac_cg, dUcg_dt) # auch bodyfixed
        # ------------------------------    
        # --- aero rigid body motion ---   
        # ------------------------------ 
        alpha = np.arctan(dUcg_dt[2]/dUcg_dt[0]) #X[4] + np.arctan(X[8]/X[6]) # alpha = theta - gamma, Wind fehlt!
        beta  = np.arctan(dUcg_dt[1]/dUcg_dt[0]) #X[5] - np.arctan(X[7]/X[6])
        my    = 0.0
        
        # dUmac_dt und Ux1 unterscheiden sich durch 
        # - den induzierten Anstellwinkel aus der Flugbahn
        # - die Vorzeichen / Koordinatensystem
        
        if self.correct_alpha:
            # Anstellwinkel alpha von der DLM-Loesung abziehen
            alpha = self.efcs.alpha_protetcion(alpha)
            drehmatrix = np.zeros((6,6))
            drehmatrix[0:3,0:3] = calc_drehmatrix(0.0, -alpha, 0.0) 
            drehmatrix[3:6,3:6] = calc_drehmatrix(0.0, -alpha, 0.0) 
            dUcg_dt_tmp = np.dot(drehmatrix, dUcg_dt)
            dUmac_dt_tmp = np.dot(PHImac_cg, dUcg_dt_tmp)
            Ujx1 = np.dot(Djx1,dUmac_dt_tmp)
        else: 
            Ujx1 = np.dot(Djx1,dUmac_dt)
            
        # der downwash wj ist nur die Komponente von Uj, welche senkrecht zum Panel steht! 
        # --> mit N multiplizieren und danach die Norm bilden    
        wjx1 = np.sum(self.model.aerogrid['N'][:] * Ujx1[self.model.aerogrid['set_j'][:,(0,1,2)]],axis=1) / self.Vtas * -1 
        flx1 = self.q_dyn * self.model.aerogrid['N'].T*self.model.aerogrid['A']*np.dot(Qjj, wjx1)
        # Bemerkung: greifen die Luftkraefte bei j,l oder k an?
        # Dies w��rde das Cmy beeinflussen!
        # Gewaehlt: l
        Plx1 = np.zeros(np.shape(Ujx1))
        Plx1[self.model.aerogrid['set_l'][:,0]] = flx1[0,:]
        Plx1[self.model.aerogrid['set_l'][:,1]] = flx1[1,:]
        Plx1[self.model.aerogrid['set_l'][:,2]] = flx1[2,:]
        
        Pk_rbm = self.model.Dlk.T.dot(Plx1)
        
        # ----------------  
        # --- aero CFD ---   
        # ----------------
        
        if self.correct_alpha:
            # jetzt den Anstellwinkel durch die CFD-loesung addieren
            Pk_alpha = self.aerodb_alpha_interpolation_function(alpha) / self.aerodb_alpha['q_dyn'] * self.q_dyn
        else:
            Pk_alpha = np.zeros(Pk_rbm.shape)   
             
        Pk_cfd = Pk_alpha
       
        # -----------------------------   
        # --- aero camber and twist ---   
        # -----------------------------

        if self.correct_alpha:
            # Effekte von Camber und Twist sind in der CFD-Loesung durch die geometrische Form enthalten
            Pk_cam = np.zeros(Pk_rbm.shape)
        else:
            wj_cam = np.sin(self.model.camber_twist['cam_rad'] )
            flcam = self.q_dyn * self.model.aerogrid['N'].T*self.model.aerogrid['A']*np.dot(Qjj, wj_cam)
            Plcam = np.zeros(np.shape(Ujx1))
            Plcam[self.model.aerogrid['set_l'][:,0]] = flcam[0,:]
            Plcam[self.model.aerogrid['set_l'][:,1]] = flcam[1,:]
            Plcam[self.model.aerogrid['set_l'][:,2]] = flcam[2,:]
            
            Pk_cam = self.model.Dlk.T.dot(Plcam) 
        
        # -----------------------------   
        # --- aero control surfaces ---   
        # -----------------------------    
    
        Pk_cs = np.zeros(np.shape(Pk_rbm))
        wjx2 = np.zeros(np.shape(wjx1))
        Ux2 = self.efcs.efcs(X[np.where(self.trimcond_X[:,0]=='command_xi')[0][0]], X[np.where(self.trimcond_X[:,0]=='command_eta')[0][0]], X[np.where(self.trimcond_X[:,0]=='command_zeta')[0][0]])
        dcommand= np.zeros(3)        # Hier gibt es zwei Wege und es wird je Steuerflaeche unterschieden:
        # a) es liegen Daten in der AeroDB vor -> Kraefte werden interpoliert, dann zu Pk addiert, downwash vector bleibt unveraendert
        # b) der downwash der Steuerflaeche wird berechnet, zum downwash vector addiert 
        for i_x2 in range(len(self.efcs.keys)):
            if self.efcs.keys[i_x2] in self.correct_x2:
                pos = self.correct_x2.index(self.efcs.keys[i_x2])
                x2_interpolation_function = self.aerodb_x2_interpolation_function[pos]
                # Null-Loesung abziehen!
                # Extrapolation sollte durch passend eingestellte Grenzen im EFCS verhindert werden. 
                Pk_cfd += (x2_interpolation_function(Ux2[i_x2]) - x2_interpolation_function(0.0) ) / self.aerodb_x2[pos]['q_dyn'] * self.q_dyn
            else:
                # use DLM solution
                if self.hingeline == 'y':
                    Ujx2 = np.dot(self.model.Djx2[i_x2],[0,0,0,0,Ux2[i_x2],0])
                elif self.hingeline == 'z':
                    Ujx2 = np.dot(self.model.Djx2[i_x2],[0,0,0,0,0,Ux2[i_x2]])
                # Rotationen ry und rz verursachen Luftkraefte. Rotation rx hat keinen Einfluss, wenn die Stoemung von vorne kommt...
                wjx2 += np.sin(Ujx2[self.model.aerogrid['set_j'][:,4]])  #* Vtas/Vtas
                wjx2 += np.sin(Ujx2[self.model.aerogrid['set_j'][:,5]])  #* Vtas/Vtas
        flx2 = self.q_dyn * self.model.aerogrid['N'].T*self.model.aerogrid['A']*np.dot(Qjj, wjx2)
        Plx2 = np.zeros(np.shape(Plx1))
        Plx2[self.model.aerogrid['set_l'][:,0]] = flx2[0,:]
        Plx2[self.model.aerogrid['set_l'][:,1]] = flx2[1,:]
        Plx2[self.model.aerogrid['set_l'][:,2]] = flx2[2,:]
    
        Pk_cs += self.model.Dlk.T.dot(Plx2)
        
        # ---------------------   
        # --- aero flexible ---   
        # ---------------------  
               
        # modale Verformung
        Ujf = np.dot(PHIjf, Uf )
        wjf_1 = np.sum(self.model.aerogrid['N'][:] * np.cross(Ujf[self.model.aerogrid['set_j'][:,(3,4,5)]], dUmac_dt[0:3]),axis=1) / self.Vtas
        # modale Bewegung
        # zu Ueberpruefen, da Trim bisher in statischer Ruhelage und somit  dUf_dt = 0
        dUjf_dt = np.dot(PHIjf, dUf_dt ) # viel schneller!
        wjf_2 = np.sum(self.model.aerogrid['N'][:] * dUjf_dt[self.model.aerogrid['set_j'][:,(0,1,2)]],axis=1) / self.Vtas * -1 
        
        flf = self.q_dyn * self.model.aerogrid['N'].T*self.model.aerogrid['A']*np.dot(Qjj, wjf_1 + wjf_2)        
        Plf = np.zeros(np.shape(dUjf_dt))
        Plf[self.model.aerogrid['set_l'][:,0]] = flf[0,:]
        Plf[self.model.aerogrid['set_l'][:,1]] = flf[1,:]
        Plf[self.model.aerogrid['set_l'][:,2]] = flf[2,:]
        
        Pk_f = self.model.Dlk.T.dot(Plf) * self.k_flex

        # ------------   
        # --- Gust ---   
        # ------------ 
        if self.simcase and self.simcase['gust']:
            # Eintauchtiefe in die Boe berechnen
            s_gust = (X[0] - self.model.aerogrid['offset_j'][:,0] - self.x0)
            # downwash der 1-cos Boe auf ein jedes Panel berechnen
            wj_gust = self.WG_TAS * 0.5 * (1-np.cos(np.pi * s_gust / self.simcase['gust_gradient']))
            wj_gust[np.where(s_gust <= 0.0)] = 0.0
            wj_gust[np.where(s_gust > 2*self.simcase['gust_gradient'])] = 0.0
            # Ausrichtung der Boe fehlt noch
            gust_direction_vector = np.sum(self.model.aerogrid['N'] * np.dot(np.array([0,0,1]), calc_drehmatrix( my=self.simcase['gust_orientation']/180.0*np.pi, alpha=0.0, beta=0.0 )), axis=1)
            wj_gust = wj_gust *  gust_direction_vector
            flgust = self.q_dyn * self.model.aerogrid['N'].T*self.model.aerogrid['A']*np.dot(Qjj, wj_gust)
            Plgust = np.zeros((6*self.model.aerogrid['n']))
            Plgust[self.model.aerogrid['set_l'][:,0]] = flgust[0,:]
            Plgust[self.model.aerogrid['set_l'][:,1]] = flgust[1,:]
            Plgust[self.model.aerogrid['set_l'][:,2]] = flgust[2,:]
            
            Pk_gust = self.model.Dlk.T.dot(Plgust)
        else:
            Pk_gust = np.zeros(Pk_rbm.shape)
        
        
        #[ -Mff*omega**2 + 1j*Dff*omega + Kff ] * Uf = Pf_gust + self.qdyn * Qff * Uf
        #PHIkf =  np.dot(PHIk_strc, PHIf_strc.T)               
        #Pf_gust = np.dot(PHIkf.T, Pk_gust)
        #Qhh_times_Uf =  np.dot(PHIkf.T, Pk_f)
        #np.dot(-Mff*omega**2 + 1j*Dff*omega + Kff , Uf) = Pf_gust + Qhh_times_Uf

        # --------------------------------   
        # --- summation of forces, EoM ---   
        # --------------------------------
        Pk_aero = Pk_rbm + Pk_cam + Pk_cs + Pk_f + Pk_cfd + Pk_gust
        Pmac = np.dot(Dkx1.T, Pk_aero)
        Pb = np.dot(PHImac_cg.T, Pmac)

        # Bemerkung: 
        # Die AIC liefert Druecke auf einem Panel, daher stehen die Kraefte senkrecht 
        # zur Oberflaeche, sodass die Kraefte gleich im koerperfesten Koordinatensystem sind.
        # Anregung im aero coord., Reaktion im body-fixed coord.  

        g = np.array([0.0, 0.0, 9.8066]) # erdfest, geodetic
        g_cg = np.dot(PHInorm_cg[0:3,0:3], np.dot(Tgeo2body[0:3,0:3],g)) # bodyfixed
        
        # SPC 126
#        if np.any(Pb[[0,1,5]] != 0):
#            print str(Pb)
#            print 'enforcing SPC 126'
#            Pb[0] = 0.0
#            Pb[1] = 0.0
#            Pb[5] = 0.0
        
        # non-linear EoM, bodyfixed
        d2Ucg_dt2 = np.zeros(dUcg_dt.shape)
        #d2Ucg_dt2[0:3] = np.cross(dUcg_dt[0:3], dUcg_dt[3:6]) + np.dot(np.linalg.inv(Mb)[0:3,0:3], Pb[0:3]) + g_cg
        #d2Ucg_dt2[3:6] = np.dot(np.linalg.inv(Mb[3:6,3:6]) , np.dot( np.cross(-Mb[3:6,3:6], dUcg_dt[3:6]), dUcg_dt[3:6]) + Pb[3:6] )
        # Nastran
        d2Ucg_dt2[0:3] = np.dot(np.linalg.inv(Mb)[0:3,0:3], Pb[0:3]) + g_cg
        d2Ucg_dt2[3:6] = np.dot(np.linalg.inv(Mb)[3:6,3:6], Pb[3:6] )
        
        # Aero- und Inertialkraefte verursachen elastische Verformungen. 
        # Das System ist in einer statischen Ruhelagen, wenn die modalen Beschleunigungen gleich Null sind.

        #d2Ug_dt2_r = PHIstrc_cg.dot( np.hstack((d2Ucg_dt2[0:3] - g_cg, d2Ucg_dt2[3:6])) )
        #Pg_iner_r = - Mgg.dot(d2Ug_dt2_r)
        #Pg_aero = np.dot(PHIk_strc.T, Pk_aero)
        #Pf = np.dot(PHIf_strc, Pg_aero + Pg_iner_r )
        Pf = np.dot(PHIkf.T, Pk_aero) + Mfcg.dot( np.hstack((d2Ucg_dt2[0:3] - g_cg, d2Ucg_dt2[3:6])) ) # viel schneller!

        # linear, flexible EoM
        d2Uf_dt2 = np.dot( -np.linalg.inv(Mff),  ( np.dot(Dff, dUf_dt) + np.dot(Kff, Uf) - Pf  ) )

        # --------------   
        # --- output ---   
        # -------------- 
        # loadfactor im Sinne von Beschleunigung der Masse, Gravitation und Richtungsaenderung muessen abgezogen werden! 
        #Nxyz = (d2Ucg_dt2[0:3] - g_cg - np.cross(dUcg_dt[0:3], dUcg_dt[3:6]) )/9.8066  
        # Nastran
        Nxyz = (d2Ucg_dt2[0:3] - g_cg) /9.8066 
                
        Y = np.hstack((X[6:12], np.dot(PHIcg_norm, np.dot(Tgeo2body.T, d2Ucg_dt2)), dUf_dt, d2Uf_dt2, dcommand, Nxyz[2]))    
            
        if type == 'small_output':
            return Y
        elif type == 'full_output':
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
                        'q_dyn': np.array([self.q_dyn]),
                        'Pb': Pb,
                        'Pmac': Pmac,
                        'Pf': Pf,
                        'alpha': np.array([alpha]),
                        'beta': np.array([beta]),
                        #'Pg_aero': np.dot(PHIk_strc.T, Pk_aero),
                        'Ux2': Ux2,
                        'd2Ucg_dt2': d2Ucg_dt2,
                        'd2Uf_dt2': d2Uf_dt2,
                        'Uf': Uf,
                        'Nxyz': Nxyz,
                        'g_cg': g_cg,
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
            Y = self.equations(X, time, 'small_output')
            # get the current values from Y and substract tamlab.figure()
            # fsolve only finds the roots; Y = 0
            Y_target_ist = Y[np.where((self.trimcond_Y[:,1] == 'target'))[0]]
            Y_target_soll = np.array(self.trimcond_Y[:,2], dtype='float')[np.where((self.trimcond_Y[:,1] == 'target'))[0]]
            out = Y_target_ist - Y_target_soll
            return out
        
        elif type=='sim':
            Y = self.equations(X, time, 'small_output')
            return Y[:-1] # Nz ist eine Rechengroesse und keine Simulationsgroesse!
            
        elif type=='sim_full_output':
            response = self.equations(X, time, 'full_output')
            return response
            
        elif type=='trim_full_output':
            response = self.equations(X, time, 'full_output')
            # do something with this output, e.g. plotting, animations, saving, etc.            
            print ''            
            print 'Y: '         
            print '--------------------' 
            for i_Y in range(len(response['Y'])):
                print self.trimcond_Y[:,0][i_Y] + ': %.4f' % float(response['Y'][i_Y])

            Pmac_rbm  = np.dot(self.model.Dkx1.T, response['Pk_rbm'])
            Pmac_cam  = np.dot(self.model.Dkx1.T, response['Pk_cam'])
            Pmac_cs   = np.dot(self.model.Dkx1.T, response['Pk_cs'])
            Pmac_f    = np.dot(self.model.Dkx1.T, response['Pk_f'])
            Pmac_cfd  = np.dot(self.model.Dkx1.T, response['Pk_cfd'])
            
            A = self.jcl.general['A_ref'] #sum(self.model.aerogrid['A'][:])
            Pmac_c = response['Pmac']/response['q_dyn']/A
            # um alpha drehen, um Cl und Cd zu erhalten
            Cl = Pmac_c[2]*np.cos(response['alpha'])+Pmac_c[0]*np.sin(response['alpha'])
            Cd = Pmac_c[2]*np.sin(response['alpha'])+Pmac_c[0]*np.cos(response['alpha'])
            
            print ''
            print 'aero derivatives:'
            print '--------------------' 
            print 'Cz_rbm: %.4f' % float(Pmac_rbm[2]/response['q_dyn']/A)
            print 'Cz_cam: %.4f' % float(Pmac_cam[2]/response['q_dyn']/A)
            print 'Cz_cs: %.4f' % float(Pmac_cs[2]/response['q_dyn']/A)
            print 'Cz_f: %.4f' % float(Pmac_f[2]/response['q_dyn']/A)
            print 'Cz_cfd: %.4f' % float(Pmac_cfd[2]/response['q_dyn']/A)
            print '--------------'
            print 'Cx: %.4f' % float(Pmac_c[0])
            print 'Cy: %.4f' % float(Pmac_c[1])
            print 'Cz: %.4f' % float(Pmac_c[2])
            print 'Cmx: %.4f' % float(Pmac_c[3]/self.model.macgrid['b_ref'])
            print 'Cmy: %.4f' % float(Pmac_c[4]/self.model.macgrid['c_ref'])
            print 'Cmz: %.4f' % float(Pmac_c[5]/self.model.macgrid['b_ref'])
            print 'alpha: %.4f [deg]' % float(response['alpha']/np.pi*180)
            print 'Cd: %.4f' % float(Cd)
            print 'Cl: %.4f' % float(Cl)
            print 'command_xi: %.4f' % float( response['X'][np.where(self.trimcond_X[:,0]=='command_xi')[0][0]])
            print 'command_eta: %.4f' % float( response['X'][np.where(self.trimcond_X[:,0]=='command_eta')[0][0]])
            print 'command_zeta: %.4f' % float( response['X'][np.where(self.trimcond_X[:,0]=='command_zeta')[0][0]])
            print 'CS deflections [deg]: ' + str(response['Ux2']/np.pi*180)
            print 'dCz_da: %.4f' % float(Pmac_c[2]/response['alpha'])
            print '--------------------' 
            
            return response
        
class steady:
    def __init__(self, model, jcl, trimcase, trimcond_X, trimcond_Y, simcase = False):
        print 'Init steady model equations with linear EoM in Nastran-Style.'
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
        
        if self.jcl.aero.has_key('flex') and self.jcl.aero['flex'] == True:
            self.k_flex = 1.0
        else:
            self.k_flex = 0.0
        
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
                   
        if self.simcase and self.simcase['gust']:
            # Vtas aus trim condition berechnen
            uvw = np.array(self.trimcond_X[6:9,2], dtype='float')
            Vtas = sum(uvw**2)**0.5
        
            V_D = self.model.atmo['a'][self.i_atmo] * self.simcase['gust_para']['MD']
            self.x0 = self.simcase['gust_para']['T1'] * Vtas 
            self.WG_TAS, U_ds, V_gust = DesignGust_CS_25_341(self.simcase['gust_gradient'], self.model.atmo['h'][self.i_atmo], self.model.atmo['rho'][self.i_atmo], Vtas, self.simcase['gust_para']['Z_mo'], V_D, self.simcase['gust_para']['MLW'], self.simcase['gust_para']['MTOW'], self.simcase['gust_para']['MZFW'])
        
        if self.simcase and self.simcase['cs_signal']:
            self.efcs.cs_signal_init(self.trimcase['desc'])
        
    def equations(self, X, t, type):
        self.counter += 1

        Qjj         = self.model.aero['Qjj'][self.i_aero]    
       
        PHImac_cg   = self.model.mass['PHImac_cg'][self.i_mass]
        PHIcg_mac   = self.model.mass['PHIcg_mac'][self.i_mass]
        PHInorm_cg  = self.model.mass['PHInorm_cg'][self.i_mass]
        PHIcg_norm  = self.model.mass['PHIcg_norm'][self.i_mass]
        Mb          = self.model.mass['Mb'][self.i_mass]
        Mff         = self.model.mass['Mff'][self.i_mass]
        Kff         = self.model.mass['Kff'][self.i_mass]
        Dff         = self.model.mass['Dff'][self.i_mass]
        PHIf_strc   = self.model.mass['PHIf_strc'][self.i_mass]
        PHIstrc_cg  = self.model.mass['PHIstrc_cg'][self.i_mass]
        Mgg         = self.model.mass['MGG'][self.i_mass]
        Mfcg        = self.model.mass['Mfcg'][self.i_mass]
        PHIjf       = self.model.mass['PHIjf'][self.i_mass]
        PHIkf       = self.model.mass['PHIkf'][self.i_mass]
        n_modes     = self.model.mass['n_modes'][self.i_mass] 
        
        PHIk_strc   = self.model.PHIk_strc
        Djx1        = self.model.Djx1
        Dkx1        = self.model.Dkx1
        
        
        
        # recover states
        Tgeo2body = np.zeros((6,6))
        Tgeo2body[0:3,0:3] = calc_drehmatrix(X[3], X[4], X[5])
        Tgeo2body[3:6,3:6] = calc_drehmatrix(X[3], X[4], X[5])
        Ucg      = np.dot(PHIcg_norm,X[0:6] ) # x, y, z, phi, theta, psi bodyfixed
        dUcg_dt  = np.dot(PHIcg_norm,np.dot(Tgeo2body, X[6:12])) # u v w p q r bodyfixed
        Uf = np.array(X[12:12+n_modes])
        dUf_dt = np.array(X[12+n_modes:12+n_modes*2])
        
        dUmac_dt = np.dot(PHImac_cg, dUcg_dt) # auch bodyfixed
        
        # aktuelle Vtas und q_dyn berechnen
        uvw = X[6:9]
        Vtas = sum(uvw**2)**0.5
        rho = self.model.atmo['rho'][self.i_atmo]
        q_dyn = rho/2.0*Vtas**2
        #q_dyn = 482.0
        # ------------------------------    
        # --- aero rigid body motion ---   
        # ------------------------------ 
        alpha = np.arctan(dUcg_dt[2]/dUcg_dt[0]) #X[4] + np.arctan(X[8]/X[6]) # alpha = theta - gamma, Wind fehlt!
        beta  = np.arctan(dUcg_dt[1]/dUcg_dt[0]) #X[5] - np.arctan(X[7]/X[6])
        my    = 0.0
        
        # dUmac_dt und Ux1 unterscheiden sich durch 
        # - den induzierten Anstellwinkel aus der Flugbahn
        # - die Vorzeichen / Koordinatensystem
        
        Ujx1 = np.dot(Djx1,dUmac_dt)
            
        # der downwash wj ist nur die Komponente von Uj, welche senkrecht zum Panel steht! 
        # --> mit N multiplizieren und danach die Norm bilden    
        wjx1 = np.sum(self.model.aerogrid['N'][:] * Ujx1[self.model.aerogrid['set_j'][:,(0,1,2)]],axis=1) / Vtas * -1 
        flx1 = q_dyn * self.model.aerogrid['N'].T*self.model.aerogrid['A']*np.dot(Qjj, wjx1)
        # Bemerkung: greifen die Luftkraefte bei j,l oder k an?
        # Dies w��rde das Cmy beeinflussen!
        # Gewaehlt: l
        Plx1 = np.zeros(np.shape(Ujx1))
        Plx1[self.model.aerogrid['set_l'][:,0]] = flx1[0,:]
        Plx1[self.model.aerogrid['set_l'][:,1]] = flx1[1,:]
        Plx1[self.model.aerogrid['set_l'][:,2]] = flx1[2,:]
        
        Pk_rbm = self.model.Dlk.T.dot(Plx1)
             
        # -----------------------------   
        # --- aero camber and twist ---   
        # -----------------------------

        wj_cam = np.sin(self.model.camber_twist['cam_rad'] )
        flcam = q_dyn * self.model.aerogrid['N'].T*self.model.aerogrid['A']*np.dot(Qjj, wj_cam)
        Plcam = np.zeros(np.shape(Ujx1))
        Plcam[self.model.aerogrid['set_l'][:,0]] = flcam[0,:]
        Plcam[self.model.aerogrid['set_l'][:,1]] = flcam[1,:]
        Plcam[self.model.aerogrid['set_l'][:,2]] = flcam[2,:]
        
        Pk_cam = self.model.Dlk.T.dot(Plcam) 
        
        # -----------------------------   
        # --- aero control surfaces ---   
        # -----------------------------    
    
        Pk_cs = np.zeros(np.shape(Pk_rbm))
        wjx2 = np.zeros(np.shape(wjx1))
        Ux2 = self.efcs.efcs(X[np.where(self.trimcond_X[:,0]=='command_xi')[0][0]], X[np.where(self.trimcond_X[:,0]=='command_eta')[0][0]], X[np.where(self.trimcond_X[:,0]=='command_zeta')[0][0]])
        if self.simcase and self.simcase['cs_signal']:
            dcommand = self.efcs.cs_signal(t)
        else:
            dcommand= np.zeros(3)        
        # Hier gibt es zwei Wege und es wird je Steuerflaeche unterschieden:
        # a) es liegen Daten in der AeroDB vor -> Kraefte werden interpoliert, dann zu Pk addiert, downwash vector bleibt unveraendert
        # b) der downwash der Steuerflaeche wird berechnet, zum downwash vector addiert 
        for i_x2 in range(len(self.efcs.keys)):
            # use DLM solution
                if self.hingeline == 'y':
                    Ujx2 = np.dot(self.model.Djx2[i_x2],[0,0,0,0,Ux2[i_x2],0])
                elif self.hingeline == 'z':
                    Ujx2 = np.dot(self.model.Djx2[i_x2],[0,0,0,0,0,Ux2[i_x2]])
                # Rotationen ry und rz verursachen Luftkraefte. Rotation rx hat keinen Einfluss, wenn die Stoemung von vorne kommt...
                wjx2 += np.sin(Ujx2[self.model.aerogrid['set_j'][:,4]])  #* Vtas/Vtas
                wjx2 += np.sin(Ujx2[self.model.aerogrid['set_j'][:,5]])  #* Vtas/Vtas
        flx2 = q_dyn * self.model.aerogrid['N'].T*self.model.aerogrid['A']*np.dot(Qjj, wjx2)
        Plx2 = np.zeros(np.shape(Plx1))
        Plx2[self.model.aerogrid['set_l'][:,0]] = flx2[0,:]
        Plx2[self.model.aerogrid['set_l'][:,1]] = flx2[1,:]
        Plx2[self.model.aerogrid['set_l'][:,2]] = flx2[2,:]
    
        Pk_cs += self.model.Dlk.T.dot(Plx2)
        
        # ---------------------   
        # --- aero flexible ---   
        # ---------------------  
               
        # modale Verformung
        Ujf = np.dot(PHIjf, Uf )
        wjf_1 = np.sum(self.model.aerogrid['N'][:] * np.cross(Ujf[self.model.aerogrid['set_j'][:,(3,4,5)]], dUmac_dt[0:3]),axis=1) / Vtas
        # modale Bewegung
        # zu Ueberpruefen, da Trim bisher in statischer Ruhelage und somit  dUf_dt = 0
        dUjf_dt = np.dot(PHIjf, dUf_dt ) # viel schneller!
        wjf_2 = np.sum(self.model.aerogrid['N'][:] * dUjf_dt[self.model.aerogrid['set_j'][:,(0,1,2)]],axis=1) / Vtas * -1 
        
        flf = q_dyn * self.model.aerogrid['N'].T*self.model.aerogrid['A']*np.dot(Qjj, wjf_1 + wjf_2)        
        Plf = np.zeros(np.shape(dUjf_dt))
        Plf[self.model.aerogrid['set_l'][:,0]] = flf[0,:]
        Plf[self.model.aerogrid['set_l'][:,1]] = flf[1,:]
        Plf[self.model.aerogrid['set_l'][:,2]] = flf[2,:]
        
        Pk_f = self.model.Dlk.T.dot(Plf) * self.k_flex

        # ------------   
        # --- Gust ---   
        # ------------ 
        if self.simcase and self.simcase['gust']:
            # Eintauchtiefe in die Boe berechnen
            s_gust = (X[0] - self.model.aerogrid['offset_j'][:,0] - self.x0)
            # downwash der 1-cos Boe auf ein jedes Panel berechnen
            wj_gust = self.WG_TAS * 0.5 * (1-np.cos(np.pi * s_gust / self.simcase['gust_gradient']))
            wj_gust[np.where(s_gust <= 0.0)] = 0.0
            wj_gust[np.where(s_gust > 2*self.simcase['gust_gradient'])] = 0.0
            # Ausrichtung der Boe fehlt noch
            gust_direction_vector = np.sum(self.model.aerogrid['N'] * np.dot(np.array([0,0,1]), calc_drehmatrix( my=self.simcase['gust_orientation']/180.0*np.pi, alpha=0.0, beta=0.0 )), axis=1)
            wj_gust = wj_gust *  gust_direction_vector
            flgust = q_dyn * self.model.aerogrid['N'].T*self.model.aerogrid['A']*np.dot(Qjj, wj_gust)
            Plgust = np.zeros((6*self.model.aerogrid['n']))
            Plgust[self.model.aerogrid['set_l'][:,0]] = flgust[0,:]
            Plgust[self.model.aerogrid['set_l'][:,1]] = flgust[1,:]
            Plgust[self.model.aerogrid['set_l'][:,2]] = flgust[2,:]
            
            Pk_gust = self.model.Dlk.T.dot(Plgust)
        else:
            Pk_gust = np.zeros(Pk_rbm.shape)
        
        # --------------------------------   
        # --- summation of forces, EoM ---   
        # --------------------------------
        Pk_aero = Pk_rbm + Pk_cam + Pk_cs + Pk_f + Pk_gust
        Pmac = np.dot(Dkx1.T, Pk_aero)
        Pb = np.dot(PHImac_cg.T, Pmac)

        g = np.array([0.0, 0.0, 9.8066]) # erdfest, geodetic
        g_cg = np.dot(PHInorm_cg[0:3,0:3], np.dot(Tgeo2body[0:3,0:3],g)) # bodyfixed
               
        # non-linear EoM, bodyfixed
        d2Ucg_dt2 = np.zeros(dUcg_dt.shape)
        #d2Ucg_dt2[0:3] = np.cross(dUcg_dt[0:3], dUcg_dt[3:6]) + np.dot(np.linalg.inv(Mb)[0:3,0:3], Pb[0:3]) + g_cg
        #d2Ucg_dt2[3:6] = np.dot(np.linalg.inv(Mb[3:6,3:6]) , np.dot( np.cross(-Mb[3:6,3:6], dUcg_dt[3:6]), dUcg_dt[3:6]) + Pb[3:6] )
        # Nastran
        d2Ucg_dt2[0:3] = np.dot(np.linalg.inv(Mb)[0:3,0:3], Pb[0:3]) + g_cg
        d2Ucg_dt2[3:6] = np.dot(np.linalg.inv(Mb)[3:6,3:6], Pb[3:6] )

        Pf = np.dot(PHIkf.T, Pk_aero) + Mfcg.dot( np.hstack((d2Ucg_dt2[0:3] - g_cg, d2Ucg_dt2[3:6])) ) # viel schneller!

        # linear, flexible EoM
        d2Uf_dt2 = np.dot( -np.linalg.inv(Mff),  ( np.dot(Dff, dUf_dt) + np.dot(Kff, Uf) - Pf  ) )

        # --------------   
        # --- output ---   
        # -------------- 
        # loadfactor im Sinne von Beschleunigung der Masse, Gravitation und Richtungsaenderung muessen abgezogen werden! 
        #Nxyz = (d2Ucg_dt2[0:3] - g_cg - np.cross(dUcg_dt[0:3], dUcg_dt[3:6]) )/9.8066  
        # Nastran
        Nxyz = (d2Ucg_dt2[0:3] - g_cg) /9.8066 
                
        Y = np.hstack((X[6:12], np.dot(PHIcg_norm, np.dot(Tgeo2body.T, d2Ucg_dt2)), dUf_dt, d2Uf_dt2, dcommand, Nxyz[2] ))    
            
        if type == 'small_output':
            return Y
        elif type == 'full_output':
            response = {'X': X, 
                        'Y': Y,
                        't': np.array([t]),
                        'Pk_rbm': Pk_rbm,
                        'Pk_cam': Pk_cam,
                        'Pk_aero': Pk_aero,
                        'Pk_cs': Pk_cs,
                        'Pk_f': Pk_f,
                        'Pk_cfd': Pk_rbm * 0.0,
                        'Pk_gust': Pk_gust,
                        'Pk_unsteady': Pk_rbm * 0.0,
                        'Pk_unsteady_B': Pk_rbm * 0.0,
                        'Pk_unsteady_D': Pk_rbm * 0.0,
                        'q_dyn': np.array([q_dyn]),
                        'Pb': Pb,
                        'Pmac': Pmac,
                        'Pf': Pf,
                        'alpha': np.array([alpha]),
                        'beta': np.array([beta]),
                        #'Pg_aero': np.dot(PHIk_strc.T, Pk_aero),
                        'Ux2': Ux2,
                        'd2Ucg_dt2': d2Ucg_dt2,
                        'd2Uf_dt2': d2Uf_dt2,
                        'Uf': Uf,
                        'Nxyz': Nxyz,
                        'g_cg': g_cg,
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
            Y = self.equations(X, time, 'small_output')
            # get the current values from Y and substract tamlab.figure()
            # fsolve only finds the roots; Y = 0
            Y_target_ist = Y[np.where((self.trimcond_Y[:,1] == 'target'))[0]]
            Y_target_soll = np.array(self.trimcond_Y[:,2], dtype='float')[np.where((self.trimcond_Y[:,1] == 'target'))[0]]
            out = Y_target_ist - Y_target_soll
            return out
        
        elif type=='sim':
            Y = self.equations(X, time, 'small_output')
            return Y[:-1] # Nz ist eine Rechengroesse und keine Simulationsgroesse!
            
        elif type=='sim_full_output':
            response = self.equations(X, time, 'full_output')
            return response
            
        elif type=='trim_full_output':
            response = self.equations(X, time, 'full_output')
            # do something with this output, e.g. plotting, animations, saving, etc.            
            print ''            
            print 'Y: '         
            print '--------------------' 
            for i_Y in range(len(response['Y'])):
                print self.trimcond_Y[:,0][i_Y] + ': %.4f' % float(response['Y'][i_Y])

            Pmac_rbm  = np.dot(self.model.Dkx1.T, response['Pk_rbm'])
            Pmac_cam  = np.dot(self.model.Dkx1.T, response['Pk_cam'])
            Pmac_cs   = np.dot(self.model.Dkx1.T, response['Pk_cs'])
            Pmac_f    = np.dot(self.model.Dkx1.T, response['Pk_f'])
            
            A = self.jcl.general['A_ref'] #sum(self.model.aerogrid['A'][:])
            Pmac_c = response['Pmac']/response['q_dyn']/A
            # um alpha drehen, um Cl und Cd zu erhalten
            Cl = Pmac_c[2]*np.cos(response['alpha'])+Pmac_c[0]*np.sin(response['alpha'])
            Cd = Pmac_c[2]*np.sin(response['alpha'])+Pmac_c[0]*np.cos(response['alpha'])
            
            print ''
            print '--------------------' 
            print 'q_dyn: %.4f [Pa]' % float(response['q_dyn'])
            print '--------------------' 
            print 'aero derivatives:'
            print '--------------------' 
            print 'Cz_rbm: %.4f' % float(Pmac_rbm[2]/response['q_dyn']/A)
            print 'Cz_cam: %.4f' % float(Pmac_cam[2]/response['q_dyn']/A)
            print 'Cz_cs: %.4f' % float(Pmac_cs[2]/response['q_dyn']/A)
            print 'Cz_f: %.4f' % float(Pmac_f[2]/response['q_dyn']/A)
            print '--------------'
            print 'Cx: %.4f' % float(Pmac_c[0])
            print 'Cy: %.4f' % float(Pmac_c[1])
            print 'Cz: %.4f' % float(Pmac_c[2])
            print 'Cmx: %.4f' % float(Pmac_c[3]/self.model.macgrid['b_ref'])
            print 'Cmy: %.4f' % float(Pmac_c[4]/self.model.macgrid['c_ref'])
            print 'Cmz: %.4f' % float(Pmac_c[5]/self.model.macgrid['b_ref'])
            print 'alpha: %.4f [deg]' % float(response['alpha']/np.pi*180)
            print 'beta: %.4f [deg]' % float(response['beta']/np.pi*180)
            print 'Cd: %.4f' % float(Cd)
            print 'Cl: %.4f' % float(Cl)
            print 'command_xi: %.4f [rad] / %.4f [deg]' % (float( response['X'][np.where(self.trimcond_X[:,0]=='command_xi')[0][0]]), float( response['X'][np.where(self.trimcond_X[:,0]=='command_xi')[0][0]])/np.pi*180.0 )
            print 'command_eta: %.4f [rad] / %.4f [deg]' % (float( response['X'][np.where(self.trimcond_X[:,0]=='command_eta')[0][0]]), float( response['X'][np.where(self.trimcond_X[:,0]=='command_eta')[0][0]])/np.pi*180.0 )
            print 'command_zeta: %.4f [rad] / %.4f [deg]' % (float( response['X'][np.where(self.trimcond_X[:,0]=='command_zeta')[0][0]]), float( response['X'][np.where(self.trimcond_X[:,0]=='command_zeta')[0][0]])/np.pi*180.0 )
            print 'CS deflections [deg]: ' + str(response['Ux2']/np.pi*180)
            print 'dCz_da: %.4f' % float(Pmac_c[2]/response['alpha'])
            print 'dCmy_da: %.4f' % float(Pmac_c[4]/self.model.macgrid['c_ref']/response['alpha'])
            print '--------------------' 
            
            return response
      
class unsteady:
    def __init__(self, model, jcl, trimcase, trimcond_X, trimcond_Y, simcase = False):
        print 'Init unsteady model equations with linear EoM in Nastran-Style.'
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
        
        # q_dyn aus trim condition berechnen
        # es erfolg spaeter kein update mehr, sollte sich die Geschwindigkeit aendern !
        uvw = np.array(self.trimcond_X[6:9,2], dtype='float')
        self.Vtas = sum(uvw**2)**0.5
        rho = self.model.atmo['rho'][self.i_atmo]
        self.q_dyn = rho/2.0*self.Vtas**2
        
        if self.jcl.aero.has_key('flex') and self.jcl.aero['flex'] == True:
            self.k_flex = 1.0
        else:
            self.k_flex = 0.0
        
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
                   
        if self.simcase and self.simcase['gust']:
            V_D = self.model.atmo['a'][self.i_atmo] * self.simcase['gust_para']['MD']
            self.x0 = self.simcase['gust_para']['T1'] * self.Vtas 
            self.WG_TAS, U_ds, V_gust = DesignGust_CS_25_341(self.simcase['gust_gradient'], self.model.atmo['h'][self.i_atmo], rho, self.Vtas, self.simcase['gust_para']['Z_mo'], V_D, self.simcase['gust_para']['MLW'], self.simcase['gust_para']['MTOW'], self.simcase['gust_para']['MZFW'])
        
    def equations(self, X, t, type):
        self.counter += 1
        #t_start = time.time()
        #print 'flexible EoM in %.2f [sec].' % (time.time() - t_start)

        # Trim-spezifische Modelldaten holen, lange Namen abkuerzen
        c_ref       = self.jcl.general['c_ref']
        n_j = self.model.aerogrid['n']
        
        Qjj         = self.model.aero['Qjj'][self.i_aero]    
        n_poles     = self.model.aero['n_poles']
        betas       = self.model.aero['betas']
        ABCD        = self.model.aero['ABCD'][self.i_aero]
       
        PHImac_cg   = self.model.mass['PHImac_cg'][self.i_mass]
        PHIcg_mac   = self.model.mass['PHIcg_mac'][self.i_mass]
        PHInorm_cg  = self.model.mass['PHInorm_cg'][self.i_mass]
        PHIcg_norm  = self.model.mass['PHIcg_norm'][self.i_mass]
        Mb          = self.model.mass['Mb'][self.i_mass]
        Mff         = self.model.mass['Mff'][self.i_mass]
        Kff         = self.model.mass['Kff'][self.i_mass]
        Dff         = self.model.mass['Dff'][self.i_mass]
        PHIf_strc   = self.model.mass['PHIf_strc'][self.i_mass]
        PHIstrc_cg  = self.model.mass['PHIstrc_cg'][self.i_mass]
        Mgg         = self.model.mass['MGG'][self.i_mass]
        Mfcg        = self.model.mass['Mfcg'][self.i_mass]
        PHIjf       = self.model.mass['PHIjf'][self.i_mass]
        PHIkf       = self.model.mass['PHIkf'][self.i_mass]
        n_modes     = self.model.mass['n_modes'][self.i_mass] 
        
        PHIk_strc   = self.model.PHIk_strc
        Djx1        = self.model.Djx1
        Dkx1        = self.model.Dkx1
        
        
        
        # recover states
        Tgeo2body = np.zeros((6,6))
        Tgeo2body[0:3,0:3] = calc_drehmatrix(X[3], X[4], X[5])
        Tgeo2body[3:6,3:6] = calc_drehmatrix(X[3], X[4], X[5])
        Ucg      = np.dot(PHIcg_norm,X[0:6] ) # x, y, z, phi, theta, psi bodyfixed
        dUcg_dt  = np.dot(PHIcg_norm,np.dot(Tgeo2body, X[6:12])) # u v w p q r bodyfixed
        Uf = np.array(X[12:12+n_modes])
        dUf_dt = np.array(X[12+n_modes:12+n_modes*2])
        
        dUmac_dt = np.dot(PHImac_cg, dUcg_dt) # auch bodyfixed
        # ------------------------------    
        # --- aero rigid body motion ---   
        # ------------------------------ 
        alpha = np.arctan(dUcg_dt[2]/dUcg_dt[0]) #X[4] + np.arctan(X[8]/X[6]) # alpha = theta - gamma, Wind fehlt!
        beta  = np.arctan(dUcg_dt[1]/dUcg_dt[0]) #X[5] - np.arctan(X[7]/X[6])
        my    = 0.0
        
        # dUmac_dt und Ux1 unterscheiden sich durch 
        # - den induzierten Anstellwinkel aus der Flugbahn
        # - die Vorzeichen / Koordinatensystem
        
        Ujx1 = np.dot(Djx1,dUmac_dt)
            
        # der downwash wj ist nur die Komponente von Uj, welche senkrecht zum Panel steht! 
        # --> mit N multiplizieren und danach die Norm bilden    
        wjx1 = np.sum(self.model.aerogrid['N'][:] * Ujx1[self.model.aerogrid['set_j'][:,(0,1,2)]],axis=1) / self.Vtas * -1 
        flx1 = self.q_dyn * self.model.aerogrid['N'].T*self.model.aerogrid['A']*np.dot(Qjj, wjx1)
        # Bemerkung: greifen die Luftkraefte bei j,l oder k an?
        # Dies w��rde das Cmy beeinflussen!
        # Gewaehlt: l
        Plx1 = np.zeros(np.shape(Ujx1))
        Plx1[self.model.aerogrid['set_l'][:,0]] = flx1[0,:]
        Plx1[self.model.aerogrid['set_l'][:,1]] = flx1[1,:]
        Plx1[self.model.aerogrid['set_l'][:,2]] = flx1[2,:]
        
        Pk_rbm = self.model.Dlk.T.dot(Plx1)

        # ----------------  
        # --- aero CFD ---   
        # ----------------        
        
        Pk_cfd = np.zeros(Pk_rbm.shape)   
             
        # -----------------------------   
        # --- aero camber and twist ---   
        # -----------------------------

        wj_cam = np.sin(self.model.camber_twist['cam_rad'] )
        flcam = self.q_dyn * self.model.aerogrid['N'].T*self.model.aerogrid['A']*np.dot(Qjj, wj_cam)
        Plcam = np.zeros(np.shape(Ujx1))
        Plcam[self.model.aerogrid['set_l'][:,0]] = flcam[0,:]
        Plcam[self.model.aerogrid['set_l'][:,1]] = flcam[1,:]
        Plcam[self.model.aerogrid['set_l'][:,2]] = flcam[2,:]
        
        Pk_cam = self.model.Dlk.T.dot(Plcam) 
        
        # -----------------------------   
        # --- aero control surfaces ---   
        # -----------------------------    
    
        Pk_cs = np.zeros(np.shape(Pk_rbm))
        wjx2 = np.zeros(np.shape(wjx1))
        Ux2 = self.efcs.efcs(X[np.where(self.trimcond_X[:,0]=='command_xi')[0][0]], X[np.where(self.trimcond_X[:,0]=='command_eta')[0][0]], X[np.where(self.trimcond_X[:,0]=='command_zeta')[0][0]])
        dcommand= np.zeros(3)        
        # Hier gibt es zwei Wege und es wird je Steuerflaeche unterschieden:
        # a) es liegen Daten in der AeroDB vor -> Kraefte werden interpoliert, dann zu Pk addiert, downwash vector bleibt unveraendert
        # b) der downwash der Steuerflaeche wird berechnet, zum downwash vector addiert 
        for i_x2 in range(len(self.efcs.keys)):
           # use DLM solution
                if self.hingeline == 'y':
                    Ujx2 = np.dot(self.model.Djx2[i_x2],[0,0,0,0,Ux2[i_x2],0])
                elif self.hingeline == 'z':
                    Ujx2 = np.dot(self.model.Djx2[i_x2],[0,0,0,0,0,Ux2[i_x2]])
                # Rotationen ry und rz verursachen Luftkraefte. Rotation rx hat keinen Einfluss, wenn die Stoemung von vorne kommt...
                wjx2 += np.sin(Ujx2[self.model.aerogrid['set_j'][:,4]])  #* Vtas/Vtas
                wjx2 += np.sin(Ujx2[self.model.aerogrid['set_j'][:,5]])  #* Vtas/Vtas        flx2 = self.q_dyn * self.model.aerogrid['N'].T*self.model.aerogrid['A']*np.dot(Qjj, wjx2)
        flx2 = self.q_dyn * self.model.aerogrid['N'].T*self.model.aerogrid['A']*np.dot(Qjj, wjx2)
        Plx2 = np.zeros(np.shape(Plx1))
        Plx2[self.model.aerogrid['set_l'][:,0]] = flx2[0,:]
        Plx2[self.model.aerogrid['set_l'][:,1]] = flx2[1,:]
        Plx2[self.model.aerogrid['set_l'][:,2]] = flx2[2,:]
    
        Pk_cs += self.model.Dlk.T.dot(Plx2)
        
        # ---------------------   
        # --- aero flexible ---   
        # ---------------------  
               
        # modale Verformung
        Ujf = np.dot(PHIjf, Uf )
        wjf_1 = np.sum(self.model.aerogrid['N'][:] * np.cross(Ujf[self.model.aerogrid['set_j'][:,(3,4,5)]], dUmac_dt[0:3]),axis=1) / self.Vtas
        # modale Bewegung
        # zu Ueberpruefen, da Trim bisher in statischer Ruhelage und somit  dUf_dt = 0
        dUjf_dt = np.dot(PHIjf, dUf_dt ) # viel schneller!
        wjf_2 = np.sum(self.model.aerogrid['N'][:] * dUjf_dt[self.model.aerogrid['set_j'][:,(0,1,2)]],axis=1) / self.Vtas * -1 
        
        flf = self.q_dyn * self.model.aerogrid['N'].T*self.model.aerogrid['A']*np.dot(Qjj, wjf_1 + wjf_2)        
        Plf = np.zeros(np.shape(dUjf_dt))
        Plf[self.model.aerogrid['set_l'][:,0]] = flf[0,:]
        Plf[self.model.aerogrid['set_l'][:,1]] = flf[1,:]
        Plf[self.model.aerogrid['set_l'][:,2]] = flf[2,:]
        
        Pk_f = self.model.Dlk.T.dot(Plf) * self.k_flex

        # ------------   
        # --- Gust ---   
        # ------------ 
        if self.simcase and self.simcase['gust']:
            # Eintauchtiefe in die Boe berechnen
            s_gust = (X[0] - self.model.aerogrid['offset_j'][:,0] - self.x0)
            # downwash der 1-cos Boe auf ein jedes Panel berechnen
            wj_gust = self.WG_TAS * 0.5 * (1-np.cos(np.pi * s_gust / self.simcase['gust_gradient']))
            wj_gust[np.where(s_gust <= 0.0)] = 0.0
            wj_gust[np.where(s_gust > 2*self.simcase['gust_gradient'])] = 0.0
            
#             if t > 0.02 :
#                 wj_gust = 20.0 * np.ones(wj_gust.shape) / self.Vtas
            
#             wj_gust = wj_gust * 0.0
#             wj_gust += 4.0/self.Vtas*(1-np.cos(t*28*np.pi))
                
            # Ausrichtung der Boe fehlt noch
            gust_direction_vector = np.sum(self.model.aerogrid['N'] * np.dot(np.array([0,0,1]), calc_drehmatrix( my=self.simcase['gust_orientation']/180.0*np.pi, alpha=0.0, beta=0.0 )), axis=1)
            wj_gust = wj_gust *  gust_direction_vector
            flgust = self.q_dyn * self.model.aerogrid['N'].T*self.model.aerogrid['A']*np.dot(Qjj, wj_gust)
            Plgust = np.zeros((6*self.model.aerogrid['n']))
            Plgust[self.model.aerogrid['set_l'][:,0]] = flgust[0,:]
            Plgust[self.model.aerogrid['set_l'][:,1]] = flgust[1,:]
            Plgust[self.model.aerogrid['set_l'][:,2]] = flgust[2,:]
            
            Pk_gust = self.model.Dlk.T.dot(Plgust)
        else:
            Pk_gust = np.zeros(Pk_rbm.shape)
            wj_gust = np.zeros(wjx1.shape)
        
        # ----------------   
        # --- Unsteady ---   
        # ---------------- 
        
        # gather data
        wj = wjx1 + wj_cam + wjx2 + wjf_1 + wjf_2 + wj_gust
        # wj =  wj_gust
        lag_states = X[12+n_modes*2+3:12+n_modes*2+3+n_j*n_poles].reshape((n_j,n_poles))
        c_over_Vtas = (0.5*c_ref)/self.Vtas
        if t <= 0.0: # initial step
            self.t_old  = np.copy(t) 
            self.wj_old = np.copy(wj) 
            self.dwj_dt_old = np.zeros(n_j)
            self.dlag_states_dt_old = np.zeros(n_j*n_poles)
            
        dt = t - self.t_old

        # dwj_dt mittels "backward differences" berechnen
        if dt > 0.0: # solver läuft vorwärts
            dwj_dt = (wj - self.wj_old) / dt
            self.dwj_dt_old = np.copy(dwj_dt)
        else: # solver bleibt stehen oder läuft zurück
            dwj_dt = self.dwj_dt_old
            
#         dwj_dt = (wj - self.wj_old) / dt
#         # guard for NaNs and Infs as we divide by dt, which might be zero...
#         dwj_dt[np.isnan(dwj_dt)] = 0.0 
#         dwj_dt[np.isinf(dwj_dt)] = 0.0
         
        # save for next step
        self.t_old  = np.copy(t)
        self.wj_old = np.copy(wj)
        
        # B - Dämfungsterm
        cp_unsteady = ABCD[1,:,:].dot(dwj_dt) * c_over_Vtas 
        flunsteady = self.q_dyn * self.model.aerogrid['N'].T*self.model.aerogrid['A']*cp_unsteady
        Plunsteady = np.zeros(np.shape(Ujx1))
        Plunsteady[self.model.aerogrid['set_l'][:,0]] = flunsteady[0,:]
        Plunsteady[self.model.aerogrid['set_l'][:,1]] = flunsteady[1,:]
        Plunsteady[self.model.aerogrid['set_l'][:,2]] = flunsteady[2,:]
        Pk_unsteady_B = self.model.Dlk.T.dot(Plunsteady)
        
        # C - Beschleunigungsterm -entfällt -
        
        # D1-Dn - lag states
        dlag_states_dt = dwj_dt.repeat(n_poles).reshape((n_j, n_poles)) - betas*lag_states/c_over_Vtas
        dlag_states_dt = dlag_states_dt.reshape((-1))
#         if dlag_states_dt.sum() != 0.0:
#             print 't: ' + str(t) + ' / dt: ' + str(dt) + ' / sum dwj_dt: ' + str(dwj_dt.sum()) +  ' / sum dlag: ' + str(dlag_states_dt.sum()) + ' / sum lag: ' + str(lag_states.sum())
        
        D_dot_lag = np.zeros(n_j)
        for i_pole in np.arange(0,self.model.aero['n_poles']):
            D_dot_lag += ABCD[3+i_pole,:,:].dot(lag_states[:,i_pole])
        cp_unsteady = D_dot_lag 
        flunsteady = self.q_dyn * self.model.aerogrid['N'].T*self.model.aerogrid['A']*cp_unsteady
        Plunsteady = np.zeros(np.shape(Ujx1))
        Plunsteady[self.model.aerogrid['set_l'][:,0]] = flunsteady[0,:]
        Plunsteady[self.model.aerogrid['set_l'][:,1]] = flunsteady[1,:]
        Plunsteady[self.model.aerogrid['set_l'][:,2]] = flunsteady[2,:]
        Pk_unsteady_D = self.model.Dlk.T.dot(Plunsteady)
        
        Pk_unsteady = Pk_unsteady_D + Pk_unsteady_B 
        # --------------------------------   
        # --- summation of forces, EoM ---   
        # --------------------------------
        Pk_aero = Pk_rbm + Pk_cam + Pk_cs + Pk_f + Pk_cfd + Pk_gust + Pk_unsteady
        Pmac = np.dot(Dkx1.T, Pk_aero)
        Pb = np.dot(PHImac_cg.T, Pmac)

        g = np.array([0.0, 0.0, 9.8066]) # erdfest, geodetic
        g_cg = np.dot(PHInorm_cg[0:3,0:3], np.dot(Tgeo2body[0:3,0:3],g)) # bodyfixed
               
        # non-linear EoM, bodyfixed
        d2Ucg_dt2 = np.zeros(dUcg_dt.shape)
        #d2Ucg_dt2[0:3] = np.cross(dUcg_dt[0:3], dUcg_dt[3:6]) + np.dot(np.linalg.inv(Mb)[0:3,0:3], Pb[0:3]) + g_cg
        #d2Ucg_dt2[3:6] = np.dot(np.linalg.inv(Mb[3:6,3:6]) , np.dot( np.cross(-Mb[3:6,3:6], dUcg_dt[3:6]), dUcg_dt[3:6]) + Pb[3:6] )
        # Nastran
        d2Ucg_dt2[0:3] = np.dot(np.linalg.inv(Mb)[0:3,0:3], Pb[0:3]) + g_cg
        d2Ucg_dt2[3:6] = np.dot(np.linalg.inv(Mb)[3:6,3:6], Pb[3:6] )

        Pf = np.dot(PHIkf.T, Pk_aero) + Mfcg.dot( np.hstack((d2Ucg_dt2[0:3] - g_cg, d2Ucg_dt2[3:6])) ) # viel schneller!

        # linear, flexible EoM
        d2Uf_dt2 = np.dot( -np.linalg.inv(Mff),  ( np.dot(Dff, dUf_dt) + np.dot(Kff, Uf) - Pf  ) )

        # --------------   
        # --- output ---   
        # -------------- 
        # loadfactor im Sinne von Beschleunigung der Masse, Gravitation und Richtungsaenderung muessen abgezogen werden! 
        #Nxyz = (d2Ucg_dt2[0:3] - g_cg - np.cross(dUcg_dt[0:3], dUcg_dt[3:6]) )/9.8066  
        # Nastran
        Nxyz = (d2Ucg_dt2[0:3] - g_cg) /9.8066 
                
        Y = np.hstack((X[6:12], np.dot(PHIcg_norm, np.dot(Tgeo2body.T, d2Ucg_dt2)), dUf_dt, d2Uf_dt2, dcommand, dlag_states_dt, Nxyz[2] ))    
            
        if type == 'small_output':
            return Y
        elif type == 'full_output':
            response = {'X': X, 
                        'Y': Y,
                        't': np.array([t]),
                        'Pk_rbm': Pk_rbm,
                        'Pk_cam': Pk_cam,
                        'Pk_aero': Pk_aero,
                        'Pk_cs': Pk_cs,
                        'Pk_f': Pk_f,
                        'Pk_cfd': Pk_rbm * 0.0,
                        'Pk_gust': Pk_gust,
                        'Pk_unsteady': Pk_unsteady,
                        'Pk_unsteady_B': Pk_unsteady_B,
                        'Pk_unsteady_D': Pk_unsteady_D,
                        'q_dyn': np.array([self.q_dyn]),
                        'Pb': Pb,
                        'Pmac': Pmac,
                        'Pf': Pf,
                        'alpha': np.array([alpha]),
                        'beta': np.array([beta]),
                        #'Pg_aero': np.dot(PHIk_strc.T, Pk_aero),
                        'Ux2': Ux2,
                        'd2Ucg_dt2': d2Ucg_dt2,
                        'd2Uf_dt2': d2Uf_dt2,
                        'Uf': Uf,
                        'Nxyz': Nxyz,
                        'g_cg': g_cg,
                       }
            return response
    
    def ode_arg_sorter(self, t, X):
        return self.eval_equations(X, t, 'sim')
            
    def eval_equations(self, X_free, time, type='sim_full_output'):
        # this is a wrapper for the model equations 'eqn_basic'

        if type in[ 'sim', 'sim_full_output']:
            X = X_free
        
        # evaluate model equations
        if type=='sim':
            Y = self.equations(X, time, 'small_output')
            return Y[:-1] # Nz ist eine Rechengroesse und keine Simulationsgroesse!
            
        elif type=='sim_full_output':
            response = self.equations(X, time, 'full_output')
            return response
        
        