'''
Created on Aug 2, 2019

@author: voss_ar
'''
import numpy as np
import csv, logging, copy

import loadskernel.PID as PID
import loadskernel.filter as filter

class Efcs:
    def __init__(self):
        self.keys = ['AIL-S1', 'AIL-S2', 'AIL-S3', 'AIL-S4']
        self.Ux2_0 = np.array([0.0, 0.0, 0.0, 0.0])
        self.Ux2_lower = np.array([-15.0, -15.0,-15.0,-15.0])/180*np.pi
        self.Ux2_upper = np.array([ 15.0,  15.0, 15.0, 15.0])/180*np.pi
        
        self.alpha_lower = -5.0/180*np.pi
        self.alpha_upper = 10.0/180*np.pi
        
    def cs_mapping(self, command_xi, command_eta, command_zeta):

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