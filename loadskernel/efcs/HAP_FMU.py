'''
Created on Aug 2, 2019

@author: voss_ar
'''
import numpy as np
import logging, pyfmi
from pyfmi.fmi import FMUModelCS2

import loadskernel.PID as PID
from loadskernel.efcs import HAP

class Efcs(HAP.Efcs):
    
    def fmi_init(self, filename_fmu):
        # Load the FMU
        # log_levels: Warnings=3, All=7, Nothing=0
        self.fmi = FMUModelCS2(filename_fmu, log_level=3)
        logging.info('init FMU {}, Version {}, {}'.format(self.fmi.get_name(), self.fmi.get_model_version(), self.fmi.get_generation_date_and_time()))
        inputs = self.fmi.get_input_list()
        outputs = self.fmi.get_output_list()
        self.reference_values = {}
        logging.debug('Found the following inputs:')
        for k, v in inputs.items():
            logging.debug(' - '+k)
            self.reference_values[k] = v.value_reference
        logging.debug('Found the following outputs:')
        for k, v in outputs.items():
            logging.debug(' - '+k)
            self.reference_values[k] = v.value_reference
            
    def fmu_init(self, filename, command_0, setpoint_v, setpoint_theta ):
        self.command_0 = command_0
        # set up fmu interface
        self.fmi_init(filename)
        """
        Die meisten Signale sollten aus dem xml selbsterkl��rend sein, die Outputs sind [elevator, rudder, throttle_l, throttle_r]
        TECS_OnOFF ist ein boolean, wenn off dann ist NUR Theta CMD aktiv in der L��ngsachse, wenn on dann ist gamma_cmd und v_cmd (EAS) aktiv.
        CL_OnOFF muss true  sein, sonst kommt nur 0.
        In der Seitenbewegung ist gerade immer nur Psi_dot / r_cmd aktiv.
        """
        # set up fmu 
        self.fmi.set_real([self.reference_values['Gamma_cmd']], [0.0])
        self.fmi.set_real([self.reference_values['V_cmd']],     [setpoint_v])
        self.fmi.set_real([self.reference_values['LonMan']], [0.0]) # Theta_cmd
        self.fmi.set_real([self.reference_values['LatMan']], [0.0]) # Phi_cmd
        #self.fmi.set_real([self.reference_values['Psi_dot_cmd']],[0.0])
        self.fmi.set_real([self.reference_values['TECS_OnOff']],[False])
        self.fmi.set_real([self.reference_values['CL_OnOff']],  [True])
        
        self.fmi.set_real([self.reference_values['p_B'], self.reference_values['q_B'], self.reference_values['r_B']],[0.0, 0.0, 0.0])
        self.fmi.set_real([self.reference_values['Phi'], self.reference_values['Theta'], self.reference_values['Psi']],[0.0, setpoint_theta, 0.0])
        self.fmi.set_real([self.reference_values['V_EAS']],[setpoint_v])
        self.fmi.set_real([self.reference_values['gamma']],[0.0])
        self.fmi.set_real([self.reference_values['alpha']],[0.0])
        self.fmi.set_real([self.reference_values['beta']],[0.0])
        
        #self.fmi.set_real([self.reference_values['y_out[1]'], self.reference_values['y_out[2]']], [command_0[1], command_0[2]])
        #self.fmi.set_real([self.reference_values['y_out[3]'], self.reference_values['y_out[4]']], [command_0[3], command_0[3]])

        self.fmi.initialize()
        self.last_time = 0.0
        
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
        
        self.xi_actuator = PID.PID_ideal(Kp = 10.0, Ki = 0.0, Kd = 0.0, t=0.0)
        self.xi_actuator.SetPoint=0.0
        self.xi_actuator.sample_time=0.0
        self.max_xi = +10.0/180.0*np.pi
        self.min_xi = -10.0/180.0*np.pi
    
    def controller(self, t, 
                   feedback_p, feedback_q, feedback_r,
                   feedback_phi, feedback_theta, feedback_psi,
                   feedback_v, feedback_gamma,
                   feedback_alpha, feedback_beta,
                   feedback_xi, feedback_eta, feedback_zeta, feedback_thrust):
          
        dt_fmu = t - self.last_time
        if dt_fmu > 0.01:
            self.fmi.set_real([self.reference_values['p_B'], self.reference_values['q_B'], self.reference_values['r_B']],[feedback_p, feedback_q, feedback_r])
            self.fmi.set_real([self.reference_values['Phi'], self.reference_values['Theta'], self.reference_values['Psi']],[feedback_phi, feedback_theta, feedback_psi])
            self.fmi.set_real([self.reference_values['V_EAS']],[feedback_v])
            self.fmi.set_real([self.reference_values['gamma']],[feedback_gamma])
            self.fmi.set_real([self.reference_values['alpha']],[feedback_alpha])
            self.fmi.set_real([self.reference_values['beta']],[feedback_beta])
            self.fmi.do_step(t-dt_fmu, dt_fmu, True)
            self.last_time = t
  
        command_eta, command_zeta, command_xi = self.fmi.get_real([self.reference_values['y_out[1]'], self.reference_values['y_out[2]'], self.reference_values['y_out[5]']])
        thrust_l, thrust_r        = self.fmi.get_real([self.reference_values['y_out[3]'], self.reference_values['y_out[4]']])
         
        command_eta = -command_eta + self.command_0[1]
        if command_eta > self.max_eta:
            command_eta = self.max_eta
        elif command_eta < self.min_eta:
            command_eta = self.min_eta        
         
        command_zeta = command_zeta + self.command_0[2] 
        if command_zeta > self.max_zeta:
            command_zeta = self.max_zeta
        elif command_zeta < self.min_zeta:
            command_zeta = self.min_zeta 
        
        command_xi = -command_xi + self.command_0[0] 
        if command_xi > self.max_xi:
            command_xi = self.max_xi
        elif command_xi < self.min_xi:
            command_xi = self.min_xi 
         
        # Aktuator
        self.eta_actuator.setSetPoint(command_eta)
        self.eta_actuator.update(t=t, feedback_value=feedback_eta) # eta
        command_deta = self.eta_actuator.output # deta
        if command_deta > self.max_actuator_speed:
            command_deta = self.max_actuator_speed
        elif command_deta < -self.max_actuator_speed:
            command_deta = -self.max_actuator_speed
          
        self.zeta_actuator.setSetPoint(command_zeta)
        self.zeta_actuator.update(t=t, feedback_value=feedback_zeta) # zeta
        command_dzeta = self.zeta_actuator.output # dzeta
        if command_dzeta > self.max_actuator_speed:
            command_dzeta = self.max_actuator_speed
        elif command_dzeta < -self.max_actuator_speed:
            command_dzeta = -self.max_actuator_speed
        
        self.xi_actuator.setSetPoint(command_xi)
        self.xi_actuator.update(t=t, feedback_value=feedback_xi) # xi
        command_dxi = self.xi_actuator.output # dxi
        if command_dxi > self.max_actuator_speed:
            command_dxi = self.max_actuator_speed
        elif command_dxi < -self.max_actuator_speed:
            command_dxi = -self.max_actuator_speed
         
        # commands for xi remains untouched
        dcommand = np.array([command_dxi, command_deta, command_dzeta, 0.0])
        return dcommand
    