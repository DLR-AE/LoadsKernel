# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 14:36:41 2015

@author: voss_ar
"""
import numpy as np
import csv, logging, copy
import PID, filter

class mephisto:
    def __init__(self):
        self.keys = ['AIL-S1', 'AIL-S2', 'AIL-S3', 'AIL-S4']
        self.Ux2_0 = np.array([0.0, 0.0, 0.0, 0.0])
        self.Ux2_lower = np.array([-15.0, -15.0,-15.0,-15.0])/180*np.pi
        self.Ux2_upper = np.array([ 15.0,  15.0, 15.0, 15.0])/180*np.pi
        
        self.alpha_lower = -5.0/180*np.pi
        self.alpha_upper = 10.0/180*np.pi
        
    def efcs(self, command_xi, command_eta, command_zeta):

        # Ausgangsposition
        delta_AILS1 = self.Ux2_0[0]
        delta_AILS2 = self.Ux2_0[1]
        delta_AILS3 = self.Ux2_0[2]
        delta_AILS4 = self.Ux2_0[3]
        
        # xi - Rollachse
        delta_AILS1 -= command_xi
        delta_AILS2 -= command_xi
        delta_AILS3 += command_xi
        delta_AILS4 += command_xi
        
        # eta - Nickachse
        delta_AILS1 -= command_eta
        delta_AILS2 -= command_eta
        delta_AILS3 -= command_eta
        delta_AILS4 -= command_eta
        
        # zeta - Gierachse
        if command_zeta < 0.0:
            # Rechtskurve -> rechts "bremsen"
            delta_AILS1 -= command_zeta
            delta_AILS2 += command_zeta
        else:
            delta_AILS3 -= command_zeta
            delta_AILS4 += command_zeta
        
        Ux2 = np.array([delta_AILS1, delta_AILS2, delta_AILS3, delta_AILS4])
        
        violation_lower = Ux2 < self.Ux2_lower
        if np.any(violation_lower):
            logging.warning( 'Commanded CS deflection not possible, violation of lower Ux2 bounds!')
            Ux2[violation_lower] = self.Ux2_lower[violation_lower]
            
        violation_upper = Ux2 > self.Ux2_upper
        if np.any(violation_upper):
            logging.warning( 'Commanded CS deflection not possible, violation of upper Ux2 bounds!')
            Ux2[violation_upper] = self.Ux2_upper[violation_upper]
            
        return Ux2
        
    def alpha_protetcion(self, alpha):
        if alpha < self.alpha_lower:
            logging.warning( 'Commanded alpha not possible, violation of lower alpha bounds!')
            alpha = self.alpha_lower
        if alpha > self.alpha_upper:
            logging.warning( 'Commanded alpha not possible, violation of upper alpha bounds!')
            alpha = self.alpha_upper
        return alpha
    
    def controller_init(self, command_0, setpoint_q):
        self.command_0 = command_0
        # set up dampfer
        self.damper = PID.PID_standart(Kp = 0.12, Ti = 0.08, Td = 0.0, t=0.0)
        self.damper.SetPoint=setpoint_q
        self.damper.sample_time=0.0
        #self.damper.windup_guard=0.01
        
        # set up actuator
        self.actuator = PID.PID_ideal(Kp = 100.0, Ki = 0.0, Kd = 0.0, t=0.0)
        self.actuator.SetPoint=0.0
        self.actuator.sample_time=0.0
        self.max_actuator_speed = 40.0/180.0*np.pi
        
    def controller(self, t, feedback_q, feedback_eta):
        # Daempfer
        self.damper.update(t=t, feedback_value=feedback_q) # q
        command_eta = self.command_0[1] + self.damper.output # eta
        
        # Aktuator
        self.actuator.setSetPoint(command_eta)
        self.actuator.update(t=t, feedback_value=feedback_eta) # eta
        command_deta = self.actuator.output # deta
        if command_deta > self.max_actuator_speed:
            command_deta = self.max_actuator_speed
        elif command_deta < -self.max_actuator_speed:
            command_deta = -self.max_actuator_speed
            
        # commands for xi and zeta remain untouched
        dcommand = np.array([0.0, command_deta, 0.0])
        return dcommand
        
class allegra:
    def __init__(self):
        self.keys = ['ELEV1', 'ELEV2', 'RUDDER']
        self.Ux2_0 = np.array([0.0, 0.0, 0.0])
        self.Ux2_lower = np.array([-30.0, -30.0, -30.0])/180*np.pi
        self.Ux2_upper = np.array([ 30.0,  30.0, 30.0])/180*np.pi
        
        self.alpha_lower = -4.0/180*np.pi
        self.alpha_upper = 6.0/180*np.pi
                
    def efcs(self, command_xi, command_eta, command_zeta):

        # Ausgangsposition
        delta_ELEV1 = self.Ux2_0[0]
        delta_ELEV2 = self.Ux2_0[1]
        delta_RUDDER = self.Ux2_0[2]              
        
        # eta - Nickachse
        delta_ELEV1 -= command_eta
        delta_ELEV2 -= command_eta
        
        # zeta - Gierachse
        delta_RUDDER = command_zeta
        
        Ux2 = np.array([delta_ELEV1, delta_ELEV2, delta_RUDDER])
        
        violation_lower = Ux2 < self.Ux2_lower
        if np.any(violation_lower):
            logging.warning( 'Commanded CS deflection not possible, violation of lower Ux2 bounds!')
            Ux2[violation_lower] = self.Ux2_lower[violation_lower]
            
        violation_upper = Ux2 > self.Ux2_upper
        if np.any(violation_upper):
            logging.warning( 'Commanded CS deflection not possible, violation of upper Ux2 bounds!')
            Ux2[violation_upper] = self.Ux2_upper[violation_upper]
            
        return Ux2

    def alpha_protetcion(self, alpha):
        if alpha < self.alpha_lower:
            logging.warning( 'Commanded alpha not possible, violation of lower alpha bounds!')
            alpha = self.alpha_lower
        if alpha > self.alpha_upper:
            logging.warning( 'Commanded alpha not possible, violation of upper alpha bounds!')
            alpha = self.alpha_upper
        return alpha

class XRF1:
    def __init__(self):
        self.keys = ['L11FLP', #0
                     'L12FLP', #1
                     'L13AIL', #2
                     'L14FLP', #3
                     'R11FLP', #4
                     'R12FLP', #5
                     'R13AIL', #6
                     'R14FLP', #7
                     'L21ELV', #8
                     'R21ELV', #9
                     'L31RUB', #10
                     ]
        self.Ux2_0 = np.array([0.0]*11)
        self.Ux2_lower = np.array([-30.0]*11)/180*np.pi
        self.Ux2_upper = np.array([ 30.0]*11)/180*np.pi
        
        self.alpha_lower = -10.0/180*np.pi
        self.alpha_upper =  10.0/180*np.pi
                
    def efcs(self, command_xi, command_eta, command_zeta):
        
        # Ausgangsposition
        Ux2 = copy.deepcopy(self.Ux2_0)            
        
        # xi - Rollachse
        Ux2[6] -= command_xi
        Ux2[2]  += command_xi
        
        # eta - Nickachse
        Ux2[8] -= command_eta
        Ux2[9] -= command_eta
        
        # zeta - Gierachse
        Ux2[10] -= command_zeta
        
        violation_lower = Ux2 < self.Ux2_lower
        if np.any(violation_lower):
            logging.warning( 'Commanded CS deflection not possible, violation of lower Ux2 bounds!')
            Ux2[violation_lower] = self.Ux2_lower[violation_lower]
            
        violation_upper = Ux2 > self.Ux2_upper
        if np.any(violation_upper):
            logging.warning( 'Commanded CS deflection not possible, violation of upper Ux2 bounds!')
            Ux2[violation_upper] = self.Ux2_upper[violation_upper]
            
        return Ux2

    def alpha_protetcion(self, alpha):
        if alpha < self.alpha_lower:
            logging.warning( 'Commanded alpha not possible, violation of lower alpha bounds!')
            alpha = self.alpha_lower
        if alpha > self.alpha_upper:
            logging.warning( 'Commanded alpha not possible, violation of upper alpha bounds!')
            alpha = self.alpha_upper
        return alpha
        
class discus2c:
    def __init__(self):
        self.keys = ['AIL_Rin', 'AIL_Rout', 'AIL_Lin', 'AIL_Lout', 'ELEV_R', 'ELEV_L', 'RUDD' ]
        self.Ux2_0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.Ux2_lower = np.array([-30.0, -30.0, -30.0, -30.0, -30.0, -30.0, -30.0]) / 180 * np.pi
        self.Ux2_upper = np.array([ 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0]) / 180 * np.pi
        
        self.alpha_lower = -10.0 / 180 * np.pi
        self.alpha_upper = 10.0 / 180 * np.pi
    
    def cs_signal_init(self, case_desc):
        csv.register_dialect('ohme1', delimiter='\t', skipinitialspace=True)
        csv.register_dialect('ohme2', delimiter=' ', skipinitialspace=True)
        cases =    ['FT-P1', 'FT-P2', 'FT-P3', 'Elev3211_A', 'Elev3211_B', 'B2B_1-10A', 'B2B_1-10B', '21_M5', '21_M11', '22_M6']
        tstart =   [ 80.0,    190.0,   168.0,   912.0,        933.0,        1236.5,      1255.0,      763.0,   1022.5,   348.0]
        dialects = ['ohme1', 'ohme1', 'ohme1', 'ohme1',      'ohme1',      'ohme1',     'ohme1',     'ohme2', 'ohme2',  'ohme2']
        files = ['/scratch/Discus2c_Data+Info/FT-P1_Pilot1.txt',
                 '/scratch/Discus2c_Data+Info/FT-P2_Pilot2.txt',
                 '/scratch/Discus2c_Data+Info/FT-P3_Pilot1.txt',
                 '/scratch/Discus2c_Data+Info/Elev3211_2-15A_Pilot1.txt',
                 '/scratch/Discus2c_Data+Info/Elev3211_2-15B_Pilot1.txt',
                 '/scratch/Discus2c_Data+Info/B2B_1-10A_Pilot2.txt',
                 '/scratch/Discus2c_Data+Info/B2B_1-10B_Pilot2.txt',
                 '/scratch/Discus2c_Data+Info/21_M5_Pilot2.txt',
                 '/scratch/Discus2c_Data+Info/21_M11_Pilot2.txt',
                 '/scratch/Discus2c_Data+Info/22_M6_Pilot2.txt']

        if case_desc not in cases:
            logging.error('No time signal for cs deflections found!')
            
        with open(files[cases.index(case_desc)]) as csvfile:
            reader = csv.DictReader(csvfile, dialect=dialects[cases.index(case_desc)])
            reader.next()  # skip second line of file
            reader.fieldnames = [name.strip() for name in reader.fieldnames] # remove spaces from column names
            data = []
            for row in reader:
                data.append([float(row['Time']), float(row['xi_l_corr'])/180.0*np.pi, float(row['xi_r_corr'])/180.0*np.pi, float(row['eta_corr'])/180.0*np.pi, float(row['zeta_corr'])/180.0*np.pi])
        self.data = np.array(data)
        self.tstart = tstart[cases.index(case_desc)]
        
    def cs_signal(self, t):
        line0 = np.argmin(np.abs(self.data[:,0] - t - self.tstart))
        line1 = line0 + 1
        
        t0 = self.data[line0,0]
        t1 = self.data[line1,0]
        
        dxi = + 0.5 * (self.data[line1,1] - self.data[line0,1]) / (t1 - t0) \
              - 0.5 * (self.data[line1,2] - self.data[line0,2]) / (t1 - t0)
        deta = - (self.data[line1,3] - self.data[line0,3]) / (t1 - t0)
        dzeta = (self.data[line1,4] - self.data[line0,4]) / (t1 - t0)

        return [dxi, deta, dzeta]
    
    def controller_init(self, sollwerte, mode='angular accelerations'):
        self.sollwerte = sollwerte
        if mode=='angular velocities':
            self.k = np.array([-10.0, 10.0, 10.0])
        elif mode=='angular accelerations':
            self.k = np.array([-10.0, 10.0, 10.0])
        else:
            logging.error('Mode {} for controller not implemented'.format(str(mode)))
                   
    def controller(self, ist_werte ):
        dcommand = self.k * (self.sollwerte - ist_werte)
        return dcommand

    def efcs(self, command_xi, command_eta, command_zeta):

        # Ausgangsposition
        delta_AIL_Rin = self.Ux2_0[0]
        delta_AIL_Rout = self.Ux2_0[1]
        delta_AIL_Lin = self.Ux2_0[2]
        delta_AIL_Lout = self.Ux2_0[3]
        delta_ELEV_R = self.Ux2_0[4]
        delta_ELEV_L = self.Ux2_0[5]
        delta_RUDD = self.Ux2_0[6]
        
        # xi - Rollachse
        if command_xi < 0.0: # Rolle nach links
            delta_AIL_Rin  -= command_xi*0.6
            delta_AIL_Rout -= command_xi*0.6
            delta_AIL_Lin  += command_xi*1.4
            delta_AIL_Lout += command_xi*1.4
        else:
            delta_AIL_Rin  -= command_xi*1.4
            delta_AIL_Rout -= command_xi*1.4
            delta_AIL_Lin  += command_xi*0.6
            delta_AIL_Lout += command_xi*0.6
        
        # eta - Nickachse
        delta_ELEV_R -= command_eta
        delta_ELEV_L -= command_eta
        
        # zeta - Gierachse
        delta_RUDD -= command_zeta # bei negativem zeta (rechts treten) soll das Ruder nach rechts ausschlagen, siehe CORD 321
        
        Ux2 = np.array([delta_AIL_Rin, delta_AIL_Rout, delta_AIL_Lin, delta_AIL_Lout, delta_ELEV_R, delta_ELEV_L, delta_RUDD])
        
        violation_lower = Ux2 < self.Ux2_lower
        if np.any(violation_lower):
            #print 'Warning: commanded CS deflection not possible, violation of lower Ux2 bounds!'
            Ux2[violation_lower] = self.Ux2_lower[violation_lower]
            
        violation_upper = Ux2 > self.Ux2_upper
        if np.any(violation_upper):
            #print 'Warning: commanded CS deflection not possible, violation of upper Ux2 bounds!'
            Ux2[violation_upper] = self.Ux2_upper[violation_upper]
            
        return Ux2

    def alpha_protetcion(self, alpha):
        if alpha < self.alpha_lower:
            logging.warning( 'Commanded alpha not possible, violation of lower alpha bounds!')
            alpha = self.alpha_lower
        if alpha > self.alpha_upper:
            logging.warning( 'Commanded alpha not possible, violation of upper alpha bounds!')
            alpha = self.alpha_upper
        return alpha

class flexop:
    def __init__(self):
        self.keys = ['ELEV-R1', 'ELEV-R2', 'ELEV-L1', 'ELEV-L2']
        self.Ux2_0 = np.array([0.0, 0.0, 0.0, 0.0])
        self.Ux2_lower = np.array([-30.0, -30.0, -30.0, -30.0])/180*np.pi
        self.Ux2_upper = np.array([ 30.0,  30.0, 30.0, 30.0])/180*np.pi
        
        self.alpha_lower = -10.0/180*np.pi
        self.alpha_upper =  10.0/180*np.pi
                
    def efcs(self, command_xi, command_eta, command_zeta):

        # Ausgangsposition
        delta_ELEVR1 = self.Ux2_0[0]
        delta_ELEVR2 = self.Ux2_0[1]
        delta_ELEVL1 = self.Ux2_0[2]
        delta_ELEVL2 = self.Ux2_0[3]
        
        # eta - Nickachse
        delta_ELEVR1 -= command_eta
        delta_ELEVR2 -= command_eta
        delta_ELEVL1 -= command_eta
        delta_ELEVL2 -= command_eta

        Ux2 = np.array([delta_ELEVR1, delta_ELEVR2, delta_ELEVL1, delta_ELEVL2])
        
        violation_lower = Ux2 < self.Ux2_lower
        if np.any(violation_lower):
            logging.warning( 'Commanded CS deflection not possible, violation of lower Ux2 bounds!')
            Ux2[violation_lower] = self.Ux2_lower[violation_lower]
            
        violation_upper = Ux2 > self.Ux2_upper
        if np.any(violation_upper):
            logging.warning( 'Commanded CS deflection not possible, violation of upper Ux2 bounds!')
            Ux2[violation_upper] = self.Ux2_upper[violation_upper]
            
        return Ux2
    
    def alpha_protetcion(self, alpha):
        if alpha < self.alpha_lower:
            logging.warning( 'Commanded alpha not possible, violation of lower alpha bounds!')
            alpha = self.alpha_lower
        if alpha > self.alpha_upper:
            logging.warning( 'Commanded alpha not possible, violation of upper alpha bounds!')
            alpha = self.alpha_upper
        return alpha
    
class halo:
    def __init__(self):
        self.keys = ['RUDDER', 'AIL_R', 'SPOIL_R', 'AIL_L', 'SPOIL_L', 'STAB1', 'STAB2', 'ELEV1', 'ELEV2']
        self.Ux2_0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.Ux2_lower = np.array([-30.0, -30.0, -90.0, -30.0, -90.0, -30.0, -30.0, -30.0, -30.0])/180*np.pi
        self.Ux2_upper = np.array([ 30.0,  30.0,   0.0,  30.0,   0.0,  30.0,  30.0,  30.0,  30.0])/180*np.pi
                
    def efcs(self, command_xi, command_eta, command_zeta):

        # Ausgangsposition
        delta_RUDDER    = self.Ux2_0[0]
        delta_AIL_R     = self.Ux2_0[1]
        delta_SPOIL_R   = self.Ux2_0[2]
        delta_AIL_L     = self.Ux2_0[3]
        delta_SPOIL_L   = self.Ux2_0[4]
        delta_STAB1     = self.Ux2_0[5]
        delta_STAB2     = self.Ux2_0[6]
        delta_ELEV1     = self.Ux2_0[7]
        delta_ELEV2     = self.Ux2_0[8]
        
        # xi - Rollachse
        # positives xi -> rollen nach rechts
        delta_AIL_R     -= command_xi
        delta_SPOIL_R   -= command_xi*2.0
        delta_AIL_L     += command_xi
        delta_SPOIL_L   += command_xi*2.0
        
        # eta - Nickachse
        # positives eta -> nicken nach oben
        delta_ELEV1 -= command_eta
        delta_ELEV2 -= command_eta
        
        # zeta - Gierachse
        # negatives zeta -> gieren nach rechts
        delta_RUDDER -= command_zeta
               
        Ux2 = np.array([delta_RUDDER,
                        delta_AIL_R, delta_SPOIL_R,
                        delta_AIL_L, delta_SPOIL_L,
                        delta_STAB1, delta_STAB2, 
                        delta_ELEV1, delta_ELEV2])
        
        violation_lower = Ux2 < self.Ux2_lower
        if np.any(violation_lower):
            #logging.warning( 'Commanded CS deflection not possible, violation of lower Ux2 bounds!')
            Ux2[violation_lower] = self.Ux2_lower[violation_lower]
            
        violation_upper = Ux2 > self.Ux2_upper
        if np.any(violation_upper):
            #logging.warning( 'Commanded CS deflection not possible, violation of upper Ux2 bounds!')
            Ux2[violation_upper] = self.Ux2_upper[violation_upper]
            
        return Ux2
    
    def cs_signal_init(self, case_desc):
        cases =    ['PU-1',  'PU-2',  'PU-3',  'R-1',   'R-2',   'R-3'] #PU:Pull-up, R:Roll
        tstart =   [ 49418.5, 49549.2, 49765.5, 49360.1, 49517.1, 49731.9]
        files = ['/scratch/HALO_Messdaten/HALOPullup1.txt',
                 '/scratch/HALO_Messdaten/HALOPullup2.txt',
                 '/scratch/HALO_Messdaten/HALOPullup3.txt',
                 '/scratch/HALO_Messdaten/HALORolle1.txt',
                 '/scratch/HALO_Messdaten/HALORolle2.txt',
                 '/scratch/HALO_Messdaten/HALORolle3.txt']

        if case_desc not in cases:
            logging.error('No time signal for cs deflections found!')
            
        with open(files[cases.index(case_desc)]) as csvfile:
            reader = csv.DictReader(csvfile, delimiter=' ')
            data = []
            for row in reader:
                data.append([float(row['time']), float(row['ail'])/180.0*np.pi, float(row['elv'])/180.0*np.pi, float(row['rud'])/180.0*np.pi])
        self.data = np.array(data)
        self.tstart = tstart[cases.index(case_desc)]
        
    def cs_signal(self, t):
        line0 = np.argmin(np.abs(self.data[:,0] - t - self.tstart)) #argmin: Minimum von Werten
        line1 = line0 + 1
        
        t0 = self.data[line0,0]
        t1 = self.data[line1,0]
        
        dxi   =-(self.data[line1,1] - self.data[line0,1]) / (t1 - t0)
        deta  =-(self.data[line1,2] - self.data[line0,2]) / (t1 - t0)
        dzeta =-(self.data[line1,3] - self.data[line0,3]) / (t1 - t0)

        return [dxi, deta, dzeta]
    
class dummy:
    def __init__(self):
        self.keys = ['dummy']
        self.Ux2 = np.array([0.0])
                
    def efcs(self, command_xi, command_eta, command_zeta):
        return self.Ux2
    
class fs35:
    def __init__(self):
        self.keys = ['ELEV1', 'ELEV2',]
        self.Ux2_0 = np.array([0.0, 0.0])
        self.Ux2_lower = np.array([-30.0, -30.0,])/180*np.pi
        self.Ux2_upper = np.array([ 30.0,  30.0,])/180*np.pi
                
    def efcs(self, command_xi, command_eta, command_zeta):

        # Ausgangsposition
        delta_ELEV1 = self.Ux2_0[0]
        delta_ELEV2 = self.Ux2_0[1]
        
        # eta - Nickachse
        delta_ELEV1 -= command_eta
        delta_ELEV2 -= command_eta
        
        Ux2 = np.array([delta_ELEV1, delta_ELEV2,])
        
        violation_lower = Ux2 < self.Ux2_lower
        if np.any(violation_lower):
            logging.warning( 'Commanded CS deflection not possible, violation of lower Ux2 bounds!')
            Ux2[violation_lower] = self.Ux2_lower[violation_lower]
            
        violation_upper = Ux2 > self.Ux2_upper
        if np.any(violation_upper):
            logging.warning( 'Commanded CS deflection not possible, violation of upper Ux2 bounds!')
            Ux2[violation_upper] = self.Ux2_upper[violation_upper]
            
        return Ux2
