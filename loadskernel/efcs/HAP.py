'''
Created on Aug 2, 2019

@author: voss_ar
'''
import numpy as np
import logging, pyfmi

import loadskernel.PID as PID

class Efcs:
    def __init__(self):
        self.keys = ['RUDD', 'ELEV1', 'ELEV2', 'AIL-P-A', 'AIL-P-B', 'AIL-P-C', 'AIL-P-D', 'AIL-S-A', 'AIL-S-B', 'AIL-S-C', 'AIL-S-D']
        self.Ux2_0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.Ux2_lower = np.array([-30.0, -30.0, -30.0, -30.0, -30.0, -30.0, -30.0, -30.0, -30.0, -30.0, -30.0])/180*np.pi
        self.Ux2_upper = np.array([ 30.0,  30.0,  30.0,  30.0,  30.0,  30.0,  30.0,  30.0,  30.0,  30.0,  30.0])/180*np.pi
                
    def cs_mapping(self, command_xi, command_eta, command_zeta):

        # Ausgangsposition
        dRUDD  = self.Ux2_0[0]
        dELEV1 = self.Ux2_0[1]
        dELEV2 = self.Ux2_0[2]
        dAILPA = self.Ux2_0[3]
        dAILPB = self.Ux2_0[4]
        dAILPC = self.Ux2_0[5]
        dAILPD = self.Ux2_0[6]
        dAILSA = self.Ux2_0[7]
        dAILSB = self.Ux2_0[8]
        dAILSC = self.Ux2_0[9]
        dAILSD = self.Ux2_0[10]
        
        # xi - Rollachse
        dAILPA += command_xi # bei positivem xi (Knueppel nach rechts) sollen die linken Querruder nach unten ausschlagen
        dAILPB += command_xi
        dAILPC += command_xi
        dAILPD += command_xi
        dAILSA -= command_xi # bei positivem xi (Knueppel nach rechts) sollen die rechten Querruder nach oben ausschlagen
        dAILSB -= command_xi
        dAILSC -= command_xi
        dAILSD -= command_xi
        
        # eta - Nickachse
        dELEV1 -= command_eta
        dELEV2 -= command_eta
        
        # zeta - Gierachse
        dRUDD -= command_zeta # bei negativem zeta (rechts treten) soll das Ruder nach rechts ausschlagen
        
        Ux2 = np.array([dRUDD, dELEV1, dELEV2, dAILPA, dAILPB, dAILPC, dAILPD, dAILSA, dAILSB, dAILSC, dAILSD])
        
#         violation_lower = Ux2 < self.Ux2_lower
#         if np.any(violation_lower):
#             logging.warning( 'Commanded CS deflection not possible, violation of lower Ux2 bounds!')
#             Ux2[violation_lower] = self.Ux2_lower[violation_lower]
#             
#         violation_upper = Ux2 > self.Ux2_upper
#         if np.any(violation_upper):
#             logging.warning( 'Commanded CS deflection not possible, violation of upper Ux2 bounds!')
#             Ux2[violation_upper] = self.Ux2_upper[violation_upper]
            
        return Ux2

    def controller_init(self, command_0, setpoint_q, setpoint_r ):
        self.command_0 = command_0
        # set up damper
        self.pitch_damper = PID.PID_standart(Kp = 3.0, Ti = 1.0, Td = 0.0, t=0.0)
        self.pitch_damper.SetPoint=setpoint_q
        self.pitch_damper.sample_time=0.0
  
        self.yaw_damper = PID.PID_standart(Kp = 3.0, Ti = 1.0, Td = 0.0, t=0.0)
        self.yaw_damper.SetPoint=setpoint_r
        self.yaw_damper.sample_time=0.0
          
        # set up actuator
        self.eta_actuator = PID.PID_ideal(Kp = 10.0, Ki = 0.0, Kd = 0.0, t=0.0)
        self.eta_actuator.SetPoint=0.0
        self.eta_actuator.sample_time=0.0
        self.max_actuator_speed = 40.0/180.0*np.pi
        self.max_eta = +20.0/180.0*np.pi
        self.min_eta = -10.0/180.0*np.pi
          
        self.zeta_actuator = PID.PID_ideal(Kp = 10.0, Ki = 0.0, Kd = 0.0, t=0.0)
        self.zeta_actuator.SetPoint=0.0
        self.zeta_actuator.sample_time=0.0
        self.max_zeta = +10.0/180.0*np.pi
        self.min_zeta = -10.0/180.0*np.pi
        
    def controller(self, t, 
                   feedback_p, feedback_q, feedback_r,
                   feedback_phi, feedback_theta, feedback_psi,
                   feedback_v, feedback_gamma,
                   feedback_eta, feedback_zeta, feedback_thrust):
        # Pitch
        self.pitch_damper.update(t=t, feedback_value=feedback_q) # q
        command_eta = self.command_0[1] + self.pitch_damper.output # eta
        if command_eta > self.max_eta:
            command_eta = self.max_eta
        elif command_eta < self.min_eta:
            command_eta = self.min_eta        
  
        self.eta_actuator.setSetPoint(command_eta)
        self.eta_actuator.update(t=t, feedback_value=feedback_eta) # eta
        command_deta = self.eta_actuator.output # deta
        if command_deta > self.max_actuator_speed:
            command_deta = self.max_actuator_speed
        elif command_deta < -self.max_actuator_speed:
            command_deta = -self.max_actuator_speed
              
        # Yaw
        self.yaw_damper.update(t=t, feedback_value=feedback_r) # r
        command_zeta = self.command_0[2] + self.yaw_damper.output # zeta
        if command_zeta > self.max_zeta:
            command_zeta = self.max_zeta
        elif command_zeta < self.min_zeta:
            command_zeta = self.min_zeta        
  
        self.zeta_actuator.setSetPoint(command_zeta)
        self.zeta_actuator.update(t=t, feedback_value=feedback_zeta) # zeta
        command_dzeta = self.zeta_actuator.output # dzeta
        if command_dzeta > self.max_actuator_speed:
            command_dzeta = self.max_actuator_speed
        elif command_dzeta < -self.max_actuator_speed:
            command_dzeta = -self.max_actuator_speed
              
        # commands for xi remains untouched
        dcommand = np.array([0.0, command_deta, command_dzeta, 0.0])
        return dcommand

    