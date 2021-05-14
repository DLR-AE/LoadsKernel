'''
Created on Aug 2, 2019

@author: voss_ar
'''
import numpy as np
import logging, pyfmi
from pyfmi.fmi import FMUModelCS2

from loadskernel.efcs import HAP
from loadskernel.units import tas2eas, tas2cas

class Efcs(HAP.Efcs):
    
    def fmi_init(self, filename_fmu):
        # Load the FMU
        # log_levels: Warnings=3, All=7, Nothing=0
        self.fmi = FMUModelCS2(filename_fmu, log_level=3)
        logging.info('init FMU {}, Version {}, {}'.format(self.fmi.get_name(), self.fmi.get_model_version() , self.fmi.get_generation_date_and_time() ))
        # all inputs, outputs and other modal parameters
        modelvariables = self.fmi.get_model_variables()
        self.ref_values = {}
        logging.debug('Found the following model variables:')
        for k, v in modelvariables.items():
            logging.debug(' - '+k)
            self.ref_values[k] = v.value_reference
            
    def fmu_init(self, filename_fmu, filename_actuator, setpoint ):
        """
        Dokumentation der FMU in [1].
        [1] Weiser, C., “HAP-P41-00-RN-L3 Flight Control Software Release Notes,” DLR Institute of System Dynamics and Control, 
        Version 2021.05, Jun. 2020, https://teamsites.dlr.de/dlr/HAP/Dateien/3_HAP/1_Plattform/P4_Flugmanagement_u_Regelung/P4.1_Flugregelung(SR)/04_FCSW_export.
        """
        # set up fmu interface, then set up the fmu
        self.fmi_init(filename_fmu)

        # Set flight management mode, see Table 3.1 in [1]
        # Turn EFCS on, Leave autopilot turned off / at default value
        self.fmi.set_integer([self.ref_values['FMS.MODE.EFCS_LON']], [1])
        self.fmi.set_integer([self.ref_values['FMS.MODE.EFCS_LAT']], [1])
        
        # Set pilot commands, see Table 3.1 in [1]
        # all commands are delta-commands, stick neutral means attitude hold if EFCS_LON/LAT is on
        self.fmi.set_real([self.ref_values['Pilot_cmd.stick_lon'], self.ref_values['Pilot_cmd.stick_lat'], self.ref_values['Pilot_cmd.stick_yaw']], 
                          [0.0, 0.0, 0.0]) 
        self.fmi.set_real([self.ref_values['Pilot_cmd.eng_left'], self.ref_values['Pilot_cmd.eng_right']], 
                          [0.0, 0.0]) 
        
        # Set Sensors, see Table 3.2 in [1]
        # airspeed (with Vcas ~ Vtas)
        self.fmi.set_real([self.ref_values['Sensors.AIRDATA.V_CAS'], self.ref_values['Sensors.AIRDATA.V_TAS']],
                          [tas2cas(setpoint['Vtas'], -setpoint['z']), setpoint['Vtas']])
        # barometric altitude [m], vertical speed [m/s]
        self.fmi.set_real([self.ref_values['Sensors.AIRDATA.h_baro'], self.ref_values['Sensors.AIRDATA.h_dot_baro']],
                          [-setpoint['z'], -setpoint['dz']])
        # attitude / euler angles PhiThetaPsi [rad]
        self.fmi.set_real([self.ref_values['Sensors.IRS.phi'], self.ref_values['Sensors.IRS.theta'], self.ref_values['Sensors.IRS.psi']],
                          setpoint['PhiThetaPsi'])
        # body rates pqr [rad/s]
        self.fmi.set_real([self.ref_values['Sensors.IRS.p_B'], self.ref_values['Sensors.IRS.q_B'], self.ref_values['Sensors.IRS.r_B']],
                          setpoint['pqr'])
        # load factor Nxyz [-]                                        
        self.fmi.set_real([self.ref_values['Sensors.IRS.nx'], self.ref_values['Sensors.IRS.ny'], self.ref_values['Sensors.IRS.nz']],
                          [setpoint['Nxyz']])
        
        self.fmi.initialize()
        self.last_time = 0.0
        
        # set up actuator        
        self.actuator = Actuator()
        self.actuator.fmu_init(filename_actuator)
        
        # set up limits
        self.max_eta  = +10.0/180.0*np.pi
        self.min_eta  = -10.0/180.0*np.pi
        self.max_zeta = +15.0/180.0*np.pi
        self.min_zeta = -15.0/180.0*np.pi
        if tas2eas(setpoint['Vtas'], -setpoint['z']) > 15.0:
            # limit xi to 1/3 for VNE
            self.max_xi   = +5.0/180.0*np.pi
            self.min_xi   = -5.0/180.0*np.pi
        else:
            self.max_xi   = +15.0/180.0*np.pi
            self.min_xi   = -15.0/180.0*np.pi
        # for all actuators except thrust
#         self.max_actuator_speed = 3000.0/180.0*np.pi # 30 rad/s entspricht ca. 5 Hz, Wert von Christian Weiser, 01.07.2020

        # apply limits to the FMU, careful with sign conventions!
        self.fmi.set_real([self.ref_values['surf_limit[1]'], self.ref_values['surf_limit[2]']], [-self.max_eta, -self.min_eta ])
        self.fmi.set_real([self.ref_values['surf_limit[3]'], self.ref_values['surf_limit[4]']], [ self.min_xi,   self.max_xi  ])
        self.fmi.set_real([self.ref_values['surf_limit[5]'], self.ref_values['surf_limit[6]']], [-self.max_zeta,-self.min_zeta])

    def controller(self, t, feedback):
        # If the time increment is positive, then perform a time step in the FMU.
        dt_fmu = t - self.last_time
        """
        Because the fmu uses an integrator with a discrete sampling rate (100 Hz, optimized for hardware), 
        the fmu time step must be dt_fmu = 0.01 s correspondingly. The aeroelastic simulation is running with a variable time step size, 
        but at least 10 000 Hz, so we are usually very close to 0.01 s. 
        """
        if dt_fmu > 0.01:
            # Update Sensors, see Table 3.2 in [1]
            # airspeed (with Vcas ~ Vtas)
            self.fmi.set_real([self.ref_values['Sensors.AIRDATA.V_CAS'], self.ref_values['Sensors.AIRDATA.V_TAS']],
                              [tas2cas(feedback['Vtas'], -feedback['z']) , feedback['Vtas']])
            # barometric altitude [m], vertical speed [m/s]
            self.fmi.set_real([self.ref_values['Sensors.AIRDATA.h_baro'], self.ref_values['Sensors.AIRDATA.h_dot_baro']],
                              [-feedback['z'], -feedback['dz']])
            # attitude / euler angles PhiThetaPsi [rad]
            self.fmi.set_real([self.ref_values['Sensors.IRS.phi'], self.ref_values['Sensors.IRS.theta'], self.ref_values['Sensors.IRS.psi']],
                              feedback['PhiThetaPsi'])
            # body rates pqr [rad/s]
            self.fmi.set_real([self.ref_values['Sensors.IRS.p_B'], self.ref_values['Sensors.IRS.q_B'], self.ref_values['Sensors.IRS.r_B']],
                              feedback['pqr'])
            # load factor Nxyz [-]                                        
            self.fmi.set_real([self.ref_values['Sensors.IRS.nx'], self.ref_values['Sensors.IRS.ny'], self.ref_values['Sensors.IRS.nz']],
                              [feedback['Nxyz']])
            # Perform time step.
            self.fmi.do_step(t-dt_fmu, dt_fmu, True)
            self.last_time = t
        
        # Get outputs from FMU
        commands = self.fmi.get_real([self.ref_values['Ux3_cmddpmdqmdrm[1]'], 
                                      self.ref_values['Ux3_cmddpmdqmdrm[2]'],
                                      self.ref_values['Ux3_cmddpmdqmdrm[3]']])       
        # Run the actuator to get the control surface rates
        d_pos = self.actuator.run(t, commands)
        # adjust sign conventions following section 4.3.3 Definition of control surface commands in [1]
        d_pos[1] *= -1.0
        d_pos[2] *= -1.0
        # assemble command derivatives in correct order for loads kernel
        dcommand = np.zeros(feedback['commands'].shape)
        dcommand[0:3] = d_pos

        return dcommand

class Actuator(Efcs):
            
    def fmu_init(self, filename):

        # set up fmu interface, then set up the fmu
        self.fmi_init(filename)

        # Set control surface rate limits
        self.fmi.set_real([self.ref_values['act.rate_limit_max[1]'], self.ref_values['act.rate_limit_max[2]'], self.ref_values['act.rate_limit_max[2]']], 
                             [+60.0/180.0*np.pi, +60.0/180.0*np.pi, +60.0/180.0*np.pi])
        self.fmi.set_real([self.ref_values['act.rate_limit_min[1]'], self.ref_values['act.rate_limit_min[2]'], self.ref_values['act.rate_limit_min[2]']], 
                             [-60.0/180.0*np.pi, -60.0/180.0*np.pi, -60.0/180.0*np.pi])

        self.fmi.initialize()
        self.last_time = 0.0

    def run(self, t, commands):
        # If the time increment is positive, then perform a time step in the FMU.
        dt_fmu = t - self.last_time
        if dt_fmu > 0.0:
            # Update commands
            self.fmi.set_real([self.ref_values['cmd[1]'], self.ref_values['cmd[2]'], self.ref_values['cmd[3]']],
                              commands)
            # Perform time step.
            self.fmi.do_step(t-dt_fmu, dt_fmu, True)
            self.last_time = t
        
        # Get control surface rates from FMU
        d_pos = self.fmi.get_real([self.ref_values['d_pos[1]'], 
                                   self.ref_values['d_pos[2]'],
                                   self.ref_values['d_pos[3]']])
        # Get control surface positions from FMU for double-checking
        pos   = self.fmi.get_real([self.ref_values['pos[1]'], 
                                   self.ref_values['pos[2]'],
                                   self.ref_values['pos[3]']])
        
        return d_pos

    