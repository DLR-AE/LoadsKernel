'''
Created on Aug 2, 2019

@author: voss_ar
'''
import numpy as np
import logging, pyfmi
from pyfmi.fmi import FMUModelCS2

import loadskernel.PID as PID
from loadskernel.efcs import HAP
from loadskernel.units import tas2eas

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
            
    def fmu_init(self, filename, command_0, setpoint_v, setpoint_h, ):
        """
        Dokumentation der FMU in [1].
        [1] Weiser, C. and Looye, G., “P4.1 EFCS Simulink description & release notes.” DLR Institute of System Dynamics and Control, 30-Jun-2020.
        https://teamsites.dlr.de/dlr/HAP/Dateien/3_HAP/1_Plattform/P4_Flugmanagement_u_Regelung/P4.1_Flugregelung(SR)/FCSW_export/HAP_P41_SDD_L3.pdf
        """
        # store initial trim commands from loads kernel for late use
        self.command_0 = command_0
        # set up fmu interface, then set up the fmu
        self.fmi_init(filename)

        # AP_cmd, see Table 3.1 in [1]
        self.fmi.set_real([self.reference_values['AP_cmd[1]']], [1])            # EFCS on
        self.fmi.set_real([self.reference_values['AP_cmd[2]']], [0])            # Autopilot on
        self.fmi.set_real([self.reference_values['AP_cmd[3]']], [0])            # Autothrust on
        self.fmi.set_real([self.reference_values['AP_cmd[4]']], [0])            # Vertical navigation mode set to altitude (options: altitude=1, gamma=2)
        self.fmi.set_real([self.reference_values['AP_cmd[5]']], [0])            # Lateral navigation on
        self.fmi.set_real([self.reference_values['AP_cmd[6]']], [0])            # Speed management on
        self.fmi.set_real([self.reference_values['AP_cmd[7]']], [setpoint_h])   # Altitude command [m]
        self.fmi.set_real([self.reference_values['AP_cmd[9]']], [0.0])          # Gamma command [rad]
        self.fmi.set_real([self.reference_values['AP_cmd[11]']], [0.0])         # dVcas command [m/s], 0.0 = hold speed
        
        # Pilot_cmd, see Table 3.2 in [1]
        self.fmi.set_real([self.reference_values['Pilot_cmd[1]']], [0.0])       # dTheta command [rad]
        self.fmi.set_real([self.reference_values['Pilot_cmd[2]']], [0.0])       # dPhi command [rad]

        self.fmi.initialize()
        self.last_time = 0.0
        
        # set up actuator
        self.eta_actuator    = PID.PID_ideal(Kp = 100.0, Ki = 0.0, Kd = 0.0, t=0.0)
        self.zeta_actuator   = PID.PID_ideal(Kp = 0.5, Ki = 0.0, Kd = 0.0, t=0.0)
        self.xi_actuator     = PID.PID_ideal(Kp = 100.0, Ki = 0.0, Kd = 0.0, t=0.0)
        self.thrust_actuator = PID.PID_ideal(Kp = 0.1, Ki = 0.0, Kd = 0.0, t=0.0)
        
        # set up limits
        self.max_eta  = +20.0/180.0*np.pi
        self.min_eta  = -10.0/180.0*np.pi
        self.max_zeta = +15.0/180.0*np.pi
        self.min_zeta = -15.0/180.0*np.pi
        self.max_xi   = +15.0/180.0*np.pi
        self.min_xi   = -15.0/180.0*np.pi
        # for all actuators except thrust
        self.max_actuator_speed = 30.0/180.0*np.pi # 30 rad/s entspricht ca. 5 Hz, Wert von Christian Weiser, 01.07.2020
    
    def controller(self, t, feedback):
          
        dt_fmu = t - self.last_time
        if dt_fmu > 0.0:
            # If the time increment is positive, then perform a time step in the FMU.
            # Update Sensors, see Table 3.3 in [1]
            self.fmi.set_real([self.reference_values['Sensors[1]']],[tas2eas(feedback['Vtas'], feedback['h'])])     # Veas
            self.fmi.set_real([self.reference_values['Sensors[2]']],[feedback['Vtas']])                             # Vtas
            self.fmi.set_real([self.reference_values['Sensors[3]']],[feedback['h']])                                # baro altitude [m]
            self.fmi.set_real([self.reference_values['Sensors[4]'], self.reference_values['Sensors[5]']],
                              [feedback['alpha'], feedback['beta']])                                                # alpha, beta [rad]
            self.fmi.set_real([self.reference_values['Sensors[6]'], self.reference_values['Sensors[7]'], self.reference_values['Sensors[8]']],
                              feedback['pqr'])                                                                      # pqr [rad/s]
            self.fmi.set_real([self.reference_values['Sensors[9]'], self.reference_values['Sensors[10]'], self.reference_values['Sensors[11]']],
                              feedback['PhiThetaPsi'])                                                              # PhiThetaPsi [rad]
            self.fmi.set_real([self.reference_values['Sensors[12]']],[feedback['gamma']])                           # flight path angle Gamma [rad]
            self.fmi.set_real([self.reference_values['Sensors[16]']],[feedback['PhiThetaPsi'][2]])                  # course angle Chi [rad], without wind Chi = Psi
            # Perform time step.
            self.fmi.do_step(t-dt_fmu, dt_fmu, True)
            self.last_time = t
        
        # Get outputs from FMU
        command_eta, command_zeta, thrust_l, thrust_r, command_xi = self.fmi.get_real([self.reference_values['y_out[1]'], self.reference_values['y_out[2]'],
                                                                                       self.reference_values['y_out[3]'], self.reference_values['y_out[4]'],
                                                                                       self.reference_values['y_out[5]']])       
        # Add the trim commands (unknown to the FMU), adjust sign conventions, apply limits.
        command_xi   = self.apply_limit(-command_xi + self.command_0[0], self.max_xi, self.min_xi) 
        command_eta  = self.apply_limit(-command_eta  + self.command_0[1], self.max_eta, self.min_eta)
        command_zeta = self.apply_limit( command_zeta + self.command_0[2], self.max_zeta, self.min_zeta) 
        # The FMU distinguishes left and right engine but there is (currently) only one thrust command in loads kernel.
        # For now, take a mean value for both engines.
        thrust_l, thrust_r = self.fmi.get_real([self.reference_values['y_out[3]'], self.reference_values['y_out[4]']])
        command_thrust = 0.5*(thrust_l + thrust_r) + self.command_0[3]
        
        # Perform time step of actuators and apply limits.
        self.xi_actuator.setSetPoint(command_xi)
        self.xi_actuator.update(t=t, feedback_value=feedback['XiEtaZetaThrust'][0]) # xi actuator
        command_dxi = self.apply_limit(self.xi_actuator.output, self.max_actuator_speed, -self.max_actuator_speed)
        
        self.eta_actuator.setSetPoint(command_eta)
        self.eta_actuator.update(t=t, feedback_value=feedback['XiEtaZetaThrust'][1]) # eta actuator
        command_deta = self.apply_limit(self.eta_actuator.output, self.max_actuator_speed, -self.max_actuator_speed)
           
        self.zeta_actuator.setSetPoint(command_zeta)
        self.zeta_actuator.update(t=t, feedback_value=feedback['XiEtaZetaThrust'][2]) # zeta actuator
        command_dzeta = self.apply_limit(self.zeta_actuator.output, self.max_actuator_speed, -self.max_actuator_speed)
 
        self.thrust_actuator.setSetPoint(command_thrust)
        self.thrust_actuator.update(t=t, feedback_value=feedback['XiEtaZetaThrust'][3]) # thrust actuator/ engine control unit
        command_dthrust = self.thrust_actuator.output 
 
        # assemble command derivatives in correct order for loads kernel
        dcommand = np.array([command_dxi, command_deta, command_dzeta, command_dthrust])

#         # Actuator Rates available on LogOut [1:4] = {ddE, ddR, ddT, ddA}  
#         command_deta, command_dzeta, command_dthrust, command_dxi = self.fmi.get_real([self.reference_values['Log_out[1]'], self.reference_values['Log_out[2]'], 
#                                                                                        self.reference_values['Log_out[3]'],  self.reference_values['Log_out[4]']])       
# 
#         # assemble command derivatives in correct order for loads kernel
#         dcommand = np.array([-command_dxi*0.0, -command_deta, command_dzeta, command_dthrust])
        return dcommand
    
    def apply_limit(self, value, upper, lower):
        if value > upper:
            value = upper
        elif value < lower:
            value = lower
        return value  
    