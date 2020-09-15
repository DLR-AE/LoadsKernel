'''
Created on Aug 2, 2019

@author: voss_ar
'''
import numpy as np
import csv, logging

class Efcs:
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

    def cs_mapping(self, commands):
        
        command_xi = commands[0] 
        command_eta = commands[1]
        command_zeta = commands[2]

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