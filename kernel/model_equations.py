# -*- coding: utf-8 -*-
"""
Created on Thu Nov 27 15:43:35 2014

@author: voss_ar
"""

import numpy as np
from trim_tools import * 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class rigid:
    def __init__(self, model, trimcase, trimcond_X, trimcond_Y):
        self.model = model
        self.trimcase = trimcase
        self.trimcond_X = trimcond_X
        self.trimcond_Y = trimcond_Y
        self.counter = 0
        
    def equations(self, X, type):
        self.counter += 1
        #print ' # of eval: ' + str(self.counter)
        
        # Trim-spezifische Modelldaten holen    
        i_atmo     = self.model.atmo['key'].index(self.trimcase['altitude'])
        
        i_aero     = self.model.aero['key'].index(self.trimcase['aero'])
        Qjj        = self.model.aero['Qjj'][i_aero]    
        
        i_mass     = self.model.mass['key'].index(self.trimcase['mass'])
        PHIstrc_cg = self.model.mass['PHIstrc_cg'][i_mass]
        PHImac_cg  = self.model.mass['PHImac_cg'][i_mass]
        Mb         = self.model.mass['Mb'][i_mass]
        cggrid     = self.model.mass['cggrid'][i_mass]
        Mff        = self.model.mass['Mff'][i_mass]
        Kff        = self.model.mass['Kff'][i_mass]
        Dff        = self.model.mass['Dff'][i_mass]
        PHIf_strc  = self.model.mass['PHIf_strc'][i_mass]
        PHIjf      = self.model.mass['PHIjf'][i_mass]
        n_modes    = self.model.mass['n_modes'][i_mass] 
        
        
        PHIk_strc  = self.model.PHIk_strc
        Djx1       = self.model.Djx1
        Dkx1       = self.model.Dkx1
        
        # recover states
        Ucg      = np.array(X[0:6]) # x, y, z, phi, theta, psi earthfixed
        dUcg_dt  = np.array(X[6:12]) # u v w p q r bodyfixed
        Uf = np.array(X[12:12+n_modes])
        dUf_dt = np.array(X[12+n_modes:12+n_modes*2])
        
        dUmac_dt = np.dot(PHImac_cg, dUcg_dt) # auch noch bodyfixed
        Vtas = sum(dUmac_dt[0:3]**2)**0.5
        rho = self.model.atmo['rho'][i_atmo]
        q_dyn = rho/2.0*Vtas**2
        
        # ------------------------------    
        # --- aero rigid body motion ---   
        # ------------------------------ 
        alpha = X[4] - np.arctan(X[8]/X[6]) # alpha = theta - gamma
        beta  = X[5] - np.arctan(X[7]/X[6])
        my    = 0.0
        T_eb = calc_drehmatrix(my, alpha, beta) 
        Ux1 = np.hstack((np.dot(T_eb, dUmac_dt[0:3]), np.dot(T_eb, dUmac_dt[3:6]))) # jetzt im Aero-Koordinatensystem
        #Ux1 = np.array([dUmac_dt[0], dUmac_dt[1] + dUmac_dt[0] * np.tan(beta), dUmac_dt[2] + dUmac_dt[0] * np.tan(alpha), dUmac_dt[3], dUmac_dt[4], dUmac_dt[5]])
        
        Ujx1 = np.dot(Djx1,Ux1)
        # der downwash wj ist nur die Komponente von Uj, welche senkrecht zum Panel steht! 
        # --> mit N multiplizieren und danach die Norm bilden
        wjx1 = np.sum(self.model.aerogrid['N'][:] * Ujx1[self.model.aerogrid['set_j'][:,(0,1,2)]],axis=1) / Vtas
        flx1 = q_dyn * self.model.aerogrid['N'].T*self.model.aerogrid['A']*np.dot(Qjj, wjx1)
        # Bemerkung: greifen die Luftkraefte bei j,l oder k an?
        # Dies würde das Cmy beeinflussen!
        # Gewaehlt: l
        Plx1 = np.zeros(np.shape(Ujx1))
        Plx1[self.model.aerogrid['set_l'][:,0]] = flx1[0,:]
        Plx1[self.model.aerogrid['set_l'][:,1]] = flx1[1,:]
        Plx1[self.model.aerogrid['set_l'][:,2]] = flx1[2,:]
        
        Pk_rbm = np.dot(self.model.Dlk.T, Plx1)
        
        # -----------------------------   
        # --- aero camber and twist ---   
        # -----------------------------
        
        wj_cam = np.sin(self.model.camber_twist['cam_rad'] )
        flcam = q_dyn * self.model.aerogrid['N'].T*self.model.aerogrid['A']*np.dot(Qjj, wj_cam)
        Plcam = np.zeros(np.shape(Ujx1))
        Plcam[self.model.aerogrid['set_l'][:,0]] = flcam[0,:]
        Plcam[self.model.aerogrid['set_l'][:,1]] = flcam[1,:]
        Plcam[self.model.aerogrid['set_l'][:,2]] = flcam[2,:]
        
        Pk_cam = np.dot(self.model.Dlk.T, Plcam)
        
        
        
        # -----------------------------   
        # --- aero control surfaces ---   
        # -----------------------------    
    
        #Ux2 = np.hstack((X[np.where(self.trimcond_X[:,0]=='AIL-S1')[0][0]], X[np.where(self.trimcond_X[:,0]=='AIL-S1')[0][0]], X[np.where(self.trimcond_X[:,0]=='AIL-S1')[0][0]], X[np.where(self.trimcond_X[:,0]=='AIL-S1')[0][0]])) #X[12:14]
        from efcs import mephisto
        efcs = mephisto()
        Ux2 = efcs.efcs(X[np.where(self.trimcond_X[:,0]=='command_xi')[0][0]], X[np.where(self.trimcond_X[:,0]=='command_eta')[0][0]], X[np.where(self.trimcond_X[:,0]=='command_zeta')[0][0]])
        Ujx2 = np.zeros(np.shape(Ujx1))
        Plx2 = np.zeros(np.shape(Plx1))
        for i_x2 in range(len(Ux2)):
            Ujx2 = np.dot(self.model.Djx2[i_x2],[0,0,0,0,Ux2[i_x2],0])
            # Drehung der Normalenvektoren
            #drehmatrix_N = np.dot(self.model.coord['dircos'][1], calc_drehmatrix(0,Ux2[i_x2],0))
            #N_rot = []
            #for i_N in self.model.aerogrid['N']:
            #    N_rot.append(np.dot(drehmatrix_N, i_N))
            #N_rot = np.array(N_rot)
            wjx2 = np.sin(Ujx2[self.model.aerogrid['set_j'][:,(4)]])  #* Vtas/Vtas
            flx2 = q_dyn * self.model.aerogrid['N'].T*self.model.aerogrid['A']*np.dot(self.model.aero['Qjj'][i_aero], wjx2)
            #fjx2 = q_dyn * N_rot.T*self.model.aerogrid['A']*np.dot(self.model.aero['Qjj'][i_aero], wjx2)
            
            Plx2[self.model.aerogrid['set_l'][:,0]] += flx2[0,:]
            Plx2[self.model.aerogrid['set_l'][:,1]] += flx2[1,:]
            Plx2[self.model.aerogrid['set_l'][:,2]] += flx2[2,:]
        
        Pk_cs = np.dot(self.model.Dlk.T, Plx2)
        
        # ---------------------   
        # --- aero flexible ---   
        # ---------------------  
        # modale Beschleunigungen
        #dUf_dt_test = np.array([ 1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.])
                
        #dUjf_dt =  np.dot(self.model.PHIk_strc_tps, np.dot(PHIf_strc.T, dUf_dt )))
        dUjf_dt = np.dot(PHIjf, dUf_dt ) # viel schneller!
        wjf_2 = np.sum(self.model.aerogrid['N'][:] * dUjf_dt[self.model.aerogrid['set_j'][:,(0,1,2)]],axis=1) / Vtas
        flf_2 = q_dyn * self.model.aerogrid['N'].T*self.model.aerogrid['A']*np.dot(Qjj, wjf_2)        
        
        # modale Verformung
        #Uf_test = np.array([ 1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.])
        #Ujf = np.dot(self.model.Djk, np.dot(self.model.PHIk_strc_tps, np.dot(PHIf_strc.T, Uf )))
        Ujf = np.dot(PHIjf, Uf )
        wjf_1 = np.sin(Ujf[self.model.aerogrid['set_j'][:,(4)]])  #* Vtas/Vtas
        flf_1 = q_dyn * self.model.aerogrid['N'].T*self.model.aerogrid['A']*np.dot(Qjj, wjf_1)            
        
        Plf = np.zeros(np.shape(dUjf_dt))
        Plf[self.model.aerogrid['set_l'][:,0]] = flf_1[0,:] + flf_2[0,:]
        Plf[self.model.aerogrid['set_l'][:,1]] = flf_1[1,:] + flf_2[1,:]
        Plf[self.model.aerogrid['set_l'][:,2]] = flf_1[2,:] + flf_2[2,:]
        
        Pk_f = np.dot(self.model.Dlk.T, Plf)
        
        # --------------------------------   
        # --- summation of forces, EoM ---   
        # --------------------------------
        Pk_ges = Pk_rbm + Pk_cam + Pk_cs + Pk_f
        Pmac = np.dot(Dkx1.T, Pk_ges)
        Pb = np.dot(PHImac_cg.T, Pmac)

        Pg = np.dot(PHIk_strc.T, Pk_ges)
        Pf = np.dot(PHIf_strc, Pg )
        
        # Bemerkung: 
        # Die AIC liefert Druecke auf einem Panel, daher stehen die Kraefte senkrecht 
        # zur Oberflaeche, sodass die Kraefte gleich im koerperfesten Koordinatensystem sind.
        # Anregung im aero coord., Reaktion im body-fixed coord.  

        g = np.array([0.0, 0.0, -9.8066]) # erdfest, geodetic
        T_bg = calc_drehmatrix(Ucg[3], Ucg[4], Ucg[5])  #np.array([-np.sin(Ucg[4]), np.sin(Ucg[3])*np.cos(Ucg[4]),  np.cos(Ucg[3])*np.cos(Ucg[4])])
        g_rot = np.dot(T_bg, g) # bodyfixed
        
        
        # SPC 126
        if np.any(Pb[[0,1,5]] != 0):
            print str(Pb)
            print 'enforcing SPC 126'
            Pb[0] = 0.0
            Pb[1] = 0.0
            Pb[5] = 0.0
        
        # non-linear EoM, bodyfixed
        d2Ucg_dt2 = np.zeros(dUcg_dt.shape)
        d2Ucg_dt2[0:3] = np.cross(dUcg_dt[0:3], dUcg_dt[3:6]) + np.dot(np.linalg.inv(Mb)[0:3,0:3], Pb[0:3]) + g_rot
        d2Ucg_dt2[3:6] = np.dot(np.linalg.inv(Mb[3:6,3:6]) , np.dot( np.cross(-Mb[3:6,3:6], dUcg_dt[3:6]), dUcg_dt[3:6]) + Pb[3:6] )
        
        # linear, flexible EoM
        d2Uf_dt2 = np.dot( -np.linalg.inv(Mff),  ( np.dot(Dff, dUf_dt) + np.dot(Kff, Uf) - Pf  ) )


        # --------------   
        # --- output ---   
        # -------------- 
        Nxyz = (d2Ucg_dt2[0:3]-g_rot)/9.8066  
        # geodetic
        d2Ucg_dt2_geo = np.hstack((np.dot(T_bg.T, d2Ucg_dt2[0:3]), np.dot(T_bg.T, d2Ucg_dt2[3:6]))) 
        
        Y = np.hstack((dUcg_dt, d2Ucg_dt2_geo, dUf_dt, d2Uf_dt2, Nxyz[2]))    
            
        if type == 'trim':
            return Y
        elif type == 'full_output':
            response = {'X': X, 
                        'Y': Y,
                        'Pk_rbm': Pk_rbm,
                        'Pk_cam': Pk_cam,
                        'Pk_ges': Pk_ges,
                        'Pk_cs': Pk_cs,
                        'Pk_f': Pk_f,
                        'q_dyn': q_dyn,
                        'Pb': Pb,
                        'Pmac': Pmac,
                        'Pf': Pf,
                        'alpha': alpha,
                        'beta': beta,
                        'Pg': Pg,
                        'Ux2': Ux2,
                       }
            return response
    
            
    def eval_equations(self, X_free, type='full_output'):
        # this is a wrapper for the model equations 'eqn_basic'
        
        # get inputs from trimcond and apply inputs from fsolve 
        X = np.array(self.trimcond_X[:,2], dtype='float')
        X[np.where((self.trimcond_X[:,1] == 'free'))[0]] = X_free
        
        # evaluate model equations
        if type=='trim':
            Y = self.equations(X, 'trim')
            # get the current values from Y and substract tamlab.figure()
            # fsolve only finds the roots; Y = 0
            Y_target_ist = Y[np.where((self.trimcond_Y[:,1] == 'target'))[0]]
            Y_target_soll = np.array(self.trimcond_Y[:,2], dtype='float')[np.where((self.trimcond_Y[:,1] == 'target'))[0]]
            out = Y_target_ist - Y_target_soll
            return out
            
        elif type=='full_output':
            response = self.equations(X, 'full_output')
            # do something with this output, e.g. plotting, animations, saving, etc.            
            print ''            
            print 'Y: '         
            print '--------------------' 
            for i_Y in range(len(response['Y'])):
                print self.trimcond_Y[:,0][i_Y] + ': %.4f' % float(response['Y'][i_Y])


            fz_rbm =response['Pk_rbm'][self.model.aerogrid['set_k'][:,2]]
            fz_cam =response['Pk_cam'][self.model.aerogrid['set_k'][:,2]]
            fz_cs =response['Pk_cs'][self.model.aerogrid['set_k'][:,2]]
            fz_flex =response['Pk_f'][self.model.aerogrid['set_k'][:,2]]
            
            A = sum(self.model.aerogrid['A'][:])
            Pmac_c = response['Pmac']/response['q_dyn']/A
            print ''
            print 'aero derivatives:'
            print '--------------------' 
            print 'Cz_rbm: %.4f' % float(sum(fz_rbm)/response['q_dyn']/A)
            print 'Cz_cam: %.4f' % float(sum(fz_cam)/response['q_dyn']/A)
            print 'Cz_cs: %.4f' % float(sum(fz_cs)/response['q_dyn']/A)
            print 'Cz_f: %.4f' % float(sum(fz_flex)/response['q_dyn']/A)
            print '--------------'
            print 'Cz: %.4f' % float(Pmac_c[2])
            print 'Cmy: %.4f' % float(Pmac_c[4])
            print 'alpha: %.4f [deg]' % float(response['alpha']/np.pi*180)
            print 'command_xi: %.4f' % float( response['X'][np.where(self.trimcond_X[:,0]=='command_xi')[0][0]])
            print 'command_eta: %.4f' % float( response['X'][np.where(self.trimcond_X[:,0]=='command_eta')[0][0]])
            print 'command_zeta: %.4f' % float( response['X'][np.where(self.trimcond_X[:,0]=='command_zeta')[0][0]])
            print 'CS deflections [deg]: ' + str(response['Ux2']/np.pi*180)
            print 'dCz_da: %.4f' % float(Pmac_c[2]/response['alpha'])
            print '--------------------' 
            

            plotting = False
            if plotting:
                
                x = self.model.aerogrid['offset_k'][:,0]
                y = self.model.aerogrid['offset_k'][:,1]
                z = self.model.aerogrid['offset_k'][:,2]
                fx, fy, fz = response['Pk_rbm'][self.model.aerogrid['set_k'][:,0]],response['Pk_rbm'][self.model.aerogrid['set_k'][:,1]], response['Pk_rbm'][self.model.aerogrid['set_k'][:,2]]

                from mayavi import mlab
                mlab.figure()
                mlab.points3d(x, y, z, scale_factor=0.1)
                mlab.quiver3d(x, y, z, fx*0.01, fy*0.01, fz*0.01 , color=(0,1,0),  mode='2ddash', opacity=0.4,  scale_mode='vector', scale_factor=1.0)
                mlab.quiver3d(x+fx*0.01, y+fy*0.01, z+fz*0.01,fx*0.01, fy*0.01, fz*0.01 , color=(0,1,0),  mode='cone', scale_mode='scalar', scale_factor=0.5, resolution=16)
                mlab.title('Pk_rbm', size=0.2, height=0.95)
                
                mlab.figure() 
                mlab.points3d(x, y, z, scale_factor=0.1)
                mlab.quiver3d(x, y, z, response['Pk_cam'][self.model.aerogrid['set_k'][:,0]], response['Pk_cam'][self.model.aerogrid['set_k'][:,1]], response['Pk_cam'][self.model.aerogrid['set_k'][:,2]], color=(0,1,1), scale_factor=0.01)            
                mlab.title('Pk_camber_twist', size=0.2, height=0.95)
                
                mlab.figure()        
                mlab.points3d(x, y, z, scale_factor=0.1)
                mlab.quiver3d(x, y, z, response['Pk_cs'][self.model.aerogrid['set_k'][:,0]], response['Pk_cs'][self.model.aerogrid['set_k'][:,1]], response['Pk_cs'][self.model.aerogrid['set_k'][:,2]], color=(1,0,0), scale_factor=0.01)
                mlab.title('Pk_cs', size=0.2, height=0.95)
                
                mlab.figure()   
                mlab.points3d(x, y, z, scale_factor=0.1)
                mlab.quiver3d(x, y, z, response['Pk_f'][self.model.aerogrid['set_k'][:,0]], response['Pk_f'][self.model.aerogrid['set_k'][:,1]], response['Pk_f'][self.model.aerogrid['set_k'][:,2]], color=(1,0,1), scale_factor=0.01)
                mlab.title('Pk_flex', size=0.2, height=0.95)
                
                Uf = X[12:22]
                Ug = np.dot(self.model.mass['PHIf_strc'][0].T, Uf.T).T * 100.0
                x_r = self.model.strcgrid['offset'][:,0]
                y_r = self.model.strcgrid['offset'][:,1]
                z_r = self.model.strcgrid['offset'][:,2]
                x_f = self.model.strcgrid['offset'][:,0] + Ug[self.model.strcgrid['set'][:,0]]
                y_f = self.model.strcgrid['offset'][:,1] + Ug[self.model.strcgrid['set'][:,1]]
                z_f = self.model.strcgrid['offset'][:,2] + Ug[self.model.strcgrid['set'][:,2]]
                
                mlab.figure()
                mlab.points3d(x_r, y_r, z_r,  scale_factor=0.1)
                mlab.points3d(x_f, y_f, z_f, color=(0,0,1), scale_factor=0.1)
                mlab.title('flexible deformation', size=0.2, height=0.95)
                mlab.show()
            
            return response