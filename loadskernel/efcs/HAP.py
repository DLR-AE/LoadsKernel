'''
Created on Aug 2, 2019

@author: voss_ar
'''
import numpy as np
import logging, pyfmi

import loadskernel.PID as PID
from loadskernel.units import tas2eas

class Efcs:
    def __init__(self):
        self.keys = ['RUDD1', 'RUDD2', 'ELEV1', 'ELEV2', 'AIL-P-A1', 'AIL-P-A2', 'AIL-P-B', 'AIL-S-A1', 'AIL-S-A2', 'AIL-S-B']
        self.Ux2_0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                
    def cs_mapping(self, command_xi, command_eta, command_zeta):

        # Ausgangsposition
        dRUDD1  = self.Ux2_0[0]
        dRUDD2  = self.Ux2_0[1]
        dELEV1  = self.Ux2_0[2]
        dELEV2  = self.Ux2_0[3]
        dAILPA1 = self.Ux2_0[4]
        dAILPA2 = self.Ux2_0[5]
        dAILPB  = self.Ux2_0[6]
        dAILSA1 = self.Ux2_0[7]
        dAILSA2 = self.Ux2_0[8]
        dAILSB  = self.Ux2_0[9]
        
        # xi - Rollachse
        dAILPA1 += command_xi # bei positivem xi (Knueppel nach rechts) sollen die linken Querruder nach unten ausschlagen
        dAILPA2 += command_xi
        dAILPB  += command_xi
        dAILSA1 -= command_xi # bei positivem xi (Knueppel nach rechts) sollen die rechten Querruder nach oben ausschlagen
        dAILSA2 -= command_xi
        dAILSB  -= command_xi
        
        # eta - Nickachse
        dELEV1 -= command_eta
        dELEV2 -= command_eta
        
        # zeta - Gierachse
        dRUDD1 -= command_zeta # bei negativem zeta (rechts treten) soll das Ruder nach rechts ausschlagen
        dRUDD2 -= command_zeta 
        
        Ux2 = np.array([dRUDD1, dRUDD2, dELEV1, dELEV2, dAILPA1, dAILPA2, dAILPB, dAILSA1, dAILSA2, dAILSB])

        return Ux2

    def controller_init(self, command_0, setpoint_v, setpoint_h ):
        self.command_0 = command_0
        # set up damper
        self.roll_damper = PID.PID_standart(Kp = 1.0, Ti = 1000.0, Td = 0.0, t=0.0)
        self.roll_damper.SetPoint=0.0
        
        self.pitch_damper = PID.PID_standart(Kp = 3.0, Ti = 1.0, Td = 0.0, t=0.0)
        self.pitch_damper.SetPoint=0.0
  
        self.yaw_damper = PID.PID_standart(Kp = 1.0, Ti = 1000.0, Td = 0.0, t=0.0)
        self.yaw_damper.SetPoint=0.0
          
        # set up actuator
        self.xi_actuator = PID.PID_ideal(Kp = 10.0, Ki = 0.0, Kd = 0.0, t=0.0)
        self.xi_actuator.SetPoint=0.0
        
        self.eta_actuator = PID.PID_ideal(Kp = 10.0, Ki = 0.0, Kd = 0.0, t=0.0)
        self.eta_actuator.SetPoint=0.0
          
        self.zeta_actuator = PID.PID_ideal(Kp = 10.0, Ki = 0.0, Kd = 0.0, t=0.0)
        self.zeta_actuator.SetPoint=0.0
        
        # set up limits
        self.max_eta  = +20.0/180.0*np.pi
        self.min_eta  = -10.0/180.0*np.pi
        self.max_zeta = +15.0/180.0*np.pi
        self.min_zeta = -15.0/180.0*np.pi
        if tas2eas(setpoint_v, setpoint_h) > 15.0:
            # limit xi to 1/3 for VNE
            self.max_xi   = +5.0/180.0*np.pi
            self.min_xi   = -5.0/180.0*np.pi
        else:
            self.max_xi   = +15.0/180.0*np.pi
            self.min_xi   = -15.0/180.0*np.pi
        
        # for all actuators except thrust
        self.max_actuator_speed = 40.0/180.0*np.pi
        
    def controller(self, t, feedback):
        # Roll
        self.roll_damper.update(t=t, feedback_value=feedback['pqr'][0]) # q
        command_xi = self.apply_limit(self.command_0[0] + self.roll_damper.output, self.max_xi, self.min_xi) # xi
        # dXi
        self.xi_actuator.setSetPoint(command_xi)
        self.xi_actuator.update(t=t, feedback_value=feedback['XiEtaZetaThrust'][0]) # xi
        command_dxi = self.apply_limit(self.xi_actuator.output, self.max_actuator_speed, -self.max_actuator_speed)
        
        # Pitch
        self.pitch_damper.update(t=t, feedback_value=feedback['pqr'][1]) # q
        command_eta = self.apply_limit(self.command_0[1] + self.pitch_damper.output, self.max_eta, self.min_eta) # eta      
        # dEta
        self.eta_actuator.setSetPoint(command_eta)
        self.eta_actuator.update(t=t, feedback_value=feedback['XiEtaZetaThrust'][1]) # eta
        command_deta = self.apply_limit(self.eta_actuator.output, self.max_actuator_speed, -self.max_actuator_speed)
              
        # Yaw
        self.yaw_damper.update(t=t, feedback_value=feedback['pqr'][2]) # r
        command_zeta = self.apply_limit(self.command_0[2] - self.yaw_damper.output, self.max_zeta, self.min_zeta) # zeta      
        # dZeta
        self.zeta_actuator.setSetPoint(command_zeta)
        self.zeta_actuator.update(t=t, feedback_value=feedback['XiEtaZetaThrust'][2]) 
        command_dzeta = self.apply_limit(self.zeta_actuator.output, self.max_actuator_speed, -self.max_actuator_speed)
              
        # commands for xi remains untouched
        dcommand = np.array([command_dxi, command_deta, command_dzeta, 0.0])
        return dcommand

    def apply_limit(self, value, upper, lower):
        if value > upper:
            value = upper
        elif value < lower:
            value = lower
        return value  