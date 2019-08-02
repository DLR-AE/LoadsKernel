'''
Created on Aug 2, 2019

@author: voss_ar
'''
import numpy as np
import csv, logging

class Efcs:
    def __init__(self):
        self.keys = ['RUDDER', 'AIL_R', 'SPOIL_R', 'AIL_L', 'SPOIL_L', 'STAB1', 'STAB2', 'ELEV1', 'ELEV2']
        self.Ux2_0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.Ux2_lower = np.array([-30.0, -30.0, -90.0, -30.0, -90.0, -30.0, -30.0, -30.0, -30.0])/180*np.pi
        self.Ux2_upper = np.array([ 30.0,  30.0,   0.0,  30.0,   0.0,  30.0,  30.0,  30.0,  30.0])/180*np.pi
                
    def cs_mapping(self, command_xi, command_eta, command_zeta):

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