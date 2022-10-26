# -*- coding: utf-8 -*-
"""
Created on Thu Nov 27 15:35:22 2014

@author: voss_ar
"""
import numpy as np
import scipy.optimize as so
import logging, copy
from scipy.integrate import ode

from loadskernel.integrate import RungeKutta4, ExplicitEuler
import loadskernel.io_functions.specific_functions as specific_io

from loadskernel.equations.steady     import Steady
from loadskernel.equations.cfd_steady import CfdSteady
from loadskernel.equations.nonlin_steady import NonlinSteady
from loadskernel.equations.unsteady   import Unsteady
from loadskernel.equations.landing    import Landing
from loadskernel.equations.common     import ConvergenceError
from loadskernel.equations.frequency_domain import GustExcitation
from loadskernel.equations.frequency_domain import TurbulenceExcitation, LimitTurbulence
from loadskernel.equations.frequency_domain import KMethod
from loadskernel.equations.frequency_domain import KEMethod
from loadskernel.equations.frequency_domain import PKMethod
from loadskernel.equations.state_space import StateSpaceAnalysis
from loadskernel.equations.state_space import JacobiAnalysis

from loadskernel.trim_conditions import TrimConditions

from loadskernel.cfd_interfaces.tau_interface import TauError

class SolutionSequences(TrimConditions):
    
    def approx_jacobian(self, X0,func,epsilon,dt):
        """
        Approximate the Jacobian matrix of callable function func
        x       - The state vector at which the Jacobian matrix is desired
        func    - A vector-valued function of the form f(x,*args)
        epsilon - The peturbation used to determine the partial derivatives
        """
        X0 = np.asfarray(X0)
        jac = np.zeros([len(func(*(X0,0.0, 'sim'))),len(X0)])
        dX = np.zeros(len(X0))
        for i in range(len(X0)):
            f0 = func(*(X0,0.0, 'sim'))
            dX[i] = epsilon
            fi = func(*(X0+dX,0.0+dt, 'sim'))
            jac[:,i] = (fi - f0)/epsilon
            dX[i] = 0.0
        return jac
    
    def calc_jacobian(self):
        """
        The Jacobian matrix is computed about the trimmed flight condition. 
        Alternatively, it may be computed about the trim condition specified in the JCL with: X0 = np.array(self.trimcond_X[:,2], dtype='float')
        """
        
        if self.jcl.aero['method'] in ['mona_steady']:
            equations = Steady(self)
        else:
            logging.error('Unknown aero method: ' + str(self.jcl.aero['method']))

        # flight condition
        X0 = self.response['X'][0,:]
        #X0 = np.array(self.trimcond_X[:,2], dtype='float')
        logging.info('Calculating jacobian for ' + str(len(X0)) + ' variables...')
        jac = self.approx_jacobian(X0=X0, func=equations.equations, epsilon=0.01, dt=1.0) # epsilon sollte klein sein, dt sollte 1.0s sein
        self.response['X0'] = X0 # Linearisierungspunkt
        self.response['Y0'] = equations.equations(X0, t=0.0, modus='trim')
        self.response['jac'] = jac
        self.response['states'] = self.states[:,0].tolist()
        self.response['state_derivatives'] = self.state_derivatives[:,0].tolist()
        self.response['inputs'] = self.inputs[:,0].tolist()
        self.response['outputs'] = self.outputs[:,0].tolist()
        # States need to be reordered into ABCD matrices!
        # X = [ rbm,  flex,  command_cs,  lag_states ]
        # Y = [drbm, dflex, dcommand_cs, dlag_states, outputs]
        # [Y] = [A B] * [X]
        #       [C D]   
        idx_9dof = self.idx_states[3:12]
        idx_A = self.idx_states
        idx_B = self.idx_inputs
        idx_C = self.idx_outputs
        self.response['9DOF'] = jac[idx_9dof,:][:,idx_9dof] # rigid body motion only
        self.response['A'] = jac[idx_A,:][:,idx_A] # aircraft itself, including elastic states
        self.response['B'] = jac[idx_A,:][:,idx_B] # reaction of aircraft on external excitation
        self.response['C'] = jac[idx_C,:][:,idx_A] # sensors
        self.response['D'] = jac[idx_C,:][:,idx_B] # reaction of sensors on external excitation
        self.response['idx_A'] = idx_A
        self.response['idx_B'] = idx_B
        self.response['idx_C'] = idx_C
        self.response['desc'] = self.trimcase['desc']
        
        # perform analysis on jacobian matrix
        equations = JacobiAnalysis(self.response)
        equations.eval_equations()
        
    def calc_derivatives(self):
        self.calc_flexible_derivatives()
        self.calc_rigid_derivatives()
        self.calc_additional_derivatives('rigid')
        self.calc_additional_derivatives('flexible')
        self.print_derivatives('rigid')
        self.print_derivatives('flexible')
        self.calc_NP()
        self.calc_cs_effectiveness()
        logging.info('--------------------------------------------------------------------------------------')
        
    def calc_rigid_derivatives(self):        
        if self.jcl.aero['method'] in [ 'mona_steady', 'mona_unsteady', 'hybrid']:
            equations = Steady(self)
        elif self.jcl.aero['method'] in [ 'nonlin_steady']:
            equations = NonlinSteady(self)
        else:
            logging.error('Unknown aero method: ' + str(self.jcl.aero['method']))
        
        A = self.jcl.general['A_ref'] #sum(self.model.aerogrid['A'][:])
        delta = 0.01   
            
        X0 = np.array(self.trimcond_X[:,2], dtype='float')
        response0 = equations.equations(X0, 0.0, 'trim_full_output')
        derivatives = []
        logging.info('Calculating rigid derivatives...')
        for i in range(len(X0)):
            xi = copy.deepcopy(X0)
            xi[i] += delta
            response = equations.equations(xi, 0.0, 'trim_full_output')
            Pmac_c = (response['Pmac']-response0['Pmac'])/response['q_dyn']/A/delta
            derivatives.append([Pmac_c[0], Pmac_c[1], Pmac_c[2], Pmac_c[3]/self.model.macgrid['b_ref'], Pmac_c[4]/self.model.macgrid['c_ref'], Pmac_c[5]/self.model.macgrid['b_ref']])
        # write back original response and store results
        self.response['rigid_parameters'] = self.trimcond_X[:,0].tolist()
        self.response['rigid_derivatives'] = derivatives
    
    def calc_flexible_derivatives(self):
        """
        The calculation of flexible derivatives is based on adding an increment (delta) to selected trim parameters 
        in the trim condition. Then, the trim solution is calculated for the modified parameters and subtracted
        from a baseline calculation (response0), leading to the flexible derivatives with respect to the modified parameter.
        """
        
        if not self.trimcase['maneuver'] == 'derivatives':
            logging.warning("Please set 'maneuver' to 'derivatives' in your trimcase.")
        # save response a baseline
        response0 = self.response
        trimcond_X0 = copy.deepcopy(self.trimcond_X)
        
        i_atmo = self.model.atmo['key'].index(self.trimcase['altitude'])
        vtas = self.trimcase['Ma'] * self.model.atmo['a'][i_atmo]
        A = self.jcl.general['A_ref']
        delta = 0.01   
        parameters = ['theta', 'psi', 'p', 'q', 'r', 'command_xi', 'command_eta', 'command_zeta']
        derivatives = []
        logging.info('Calculating flexible derivatives...')
        for parameter in parameters:
            # modify selected parameter in trim conditions
            self.trimcond_X[np.where((np.vstack((self.states , self.inputs))[:,0] == parameter))[0][0],2] += delta
            if parameter == 'theta':
                theta = self.trimcond_X[np.where((np.vstack((self.states , self.inputs))[:,0] == parameter))[0][0],2]
                self.trimcond_X[np.where((np.vstack((self.states , self.inputs))[:,0] == 'u'))[0][0],2] = vtas*np.cos(theta)
                self.trimcond_X[np.where((np.vstack((self.states , self.inputs))[:,0] == 'w'))[0][0],2] = vtas*np.sin(theta)
            elif parameter == 'psi':
                psi = self.trimcond_X[np.where((np.vstack((self.states , self.inputs))[:,0] == parameter))[0][0],2]
                self.trimcond_X[np.where((np.vstack((self.states , self.inputs))[:,0] == 'u'))[0][0],2] = vtas*np.cos(psi)
                self.trimcond_X[np.where((np.vstack((self.states , self.inputs))[:,0] == 'v'))[0][0],2] = vtas*np.sin(psi)
            # re-calculate new trim
            self.exec_trim()
            Pmac_c = (self.response['Pmac']-response0['Pmac'])/response0['q_dyn']/A/delta
            derivatives.append([Pmac_c[0,0], Pmac_c[0,1], Pmac_c[0,2], Pmac_c[0,3]/self.model.macgrid['b_ref'], Pmac_c[0,4]/self.model.macgrid['c_ref'], Pmac_c[0,5]/self.model.macgrid['b_ref']])
            # restore trim condition for next loop 
            self.trimcond_X = copy.deepcopy(trimcond_X0)
        # write back original response and store results
        self.response = response0
        self.response['flexible_parameters'] = parameters
        self.response['flexible_derivatives'] = derivatives
        
    def calc_NP(self):
        pos = self.response['flexible_parameters'].index('theta')
        self.response['NP_flex'] = np.zeros(3)
        self.response['NP_flex'][0] = self.model.macgrid['offset'][0,0] - self.jcl.general['c_ref'] * self.response['flexible_derivatives'][pos][4] / self.response['flexible_derivatives'][pos][2]
        self.response['NP_flex'][1] = self.model.macgrid['offset'][0,1] + self.jcl.general['b_ref'] * self.response['flexible_derivatives'][pos][3] / self.response['flexible_derivatives'][pos][2] 
        logging.info('--------------------------------------------------------------------------------------')
        logging.info('Aeroelastic neutral point / aerodynamic center:')
        logging.info('NP_flex (x,y) = {:0.4g},{:0.4g}'.format(self.response['NP_flex'][0], self.response['NP_flex'][1]))
    
    def calc_cs_effectiveness(self):
        logging.info('--------------------------------------------------------------------------------------')
        logging.info('Aeroelastic control surface effectiveness:')
        logging.info('                     Cx         Cy         Cz         Cmx        Cmy        Cmz')
        for p in ['command_xi', 'command_eta', 'command_zeta']:
            pos_rigid = self.response['rigid_parameters'].index(p)
            pos_flex = self.response['flexible_parameters'].index(p)
            d = np.array(self.response['flexible_derivatives'][pos_flex]) / np.array(self.response['rigid_derivatives'][pos_rigid])
            tmp = '{:>20} {:< 10.4g} {:< 10.4g} {:< 10.4g} {:< 10.4g} {:< 10.4g} {:< 10.4g}'.format( p, d[0], d[1], d[2], d[3], d[4], d[5] )
            logging.info(tmp)

    def calc_additional_derivatives(self, key):
        # key: 'rigid' or 'flexible'
        """
        Achtung beim Vergleichen mit Nastran: Bei Nick-, Roll- und Gierderivativa ist die Skalierung der Raten 
        von Nastran sehr gewöhnungsbedürftig! Zum Beispiel:
        q * c_ref / (2 * V) = PITCH
        p * b_ref / (2 * V) = ROLL
        r * b_ref / (2 * V) = YAW
        """
        i_atmo = self.model.atmo['key'].index(self.trimcase['altitude'])
        vtas = self.trimcase['Ma'] * self.model.atmo['a'][i_atmo]

        self.response[key+'_parameters'] += ['p*', 'q*', 'r*']
        self.response[key+'_derivatives'].append(list(np.array(self.response[key+'_derivatives'][self.response[key+'_parameters'].index('p')]) / self.jcl.general['b_ref'] * 2.0 * vtas))
        self.response[key+'_derivatives'].append(list(np.array(self.response[key+'_derivatives'][self.response[key+'_parameters'].index('q')]) / self.jcl.general['c_ref'] * 2.0 * vtas))
        self.response[key+'_derivatives'].append(list(np.array(self.response[key+'_derivatives'][self.response[key+'_parameters'].index('r')]) / self.jcl.general['b_ref'] * 2.0 * vtas))
 
    def print_derivatives(self, key):
        # print some information into log file
        # key: 'rigid' or 'flexible'
        parameters = self.response[key+'_parameters']
        derivatives = self.response[key+'_derivatives']
        logging.info('--------------------------------------------------------------------------------------')
        logging.info('Calculated ' + key +' derivatives for ' + str(len(parameters)) + ' variables.')
        logging.info('MAC_ref = {}'.format(self.jcl.general['MAC_ref']))
        logging.info('A_ref = {}'.format(self.jcl.general['A_ref']))
        logging.info('b_ref = {}'.format(self.jcl.general['b_ref']))
        logging.info('c_ref = {}'.format(self.jcl.general['c_ref']))
        logging.info('q_dyn = {}'.format(self.response['q_dyn'][0]))
        logging.info('Derivatives given in body axis (aft-right-up):')
        logging.info('                     Cx         Cy         Cz         Cmx        Cmy        Cmz')
        for p, d in zip(parameters, derivatives):
            tmp = '{:>20} {:< 10.4g} {:< 10.4g} {:< 10.4g} {:< 10.4g} {:< 10.4g} {:< 10.4g}'.format( p, d[0], d[1], d[2], d[3], d[4], d[5] )
            logging.info(tmp)
                  
    def exec_trim(self):
        if self.jcl.aero['method'] in [ 'mona_steady', 'mona_unsteady', 'hybrid', 'nonlin_steady', 'freq_dom']:
            self.direct_trim()
        elif self.jcl.aero['method'] in [ 'cfd_steady']:
            self.iterative_trim()
        else:
            logging.error('Unknown aero method: ' + str(self.jcl.aero['method']))
        if self.successful:
            # To align the trim results with the time/frequency simulations, we expand the response by one dimension. 
            # Notation: (n_timesteps, n_dof) --> the trim results can be considered as the solution at time step zero. 
            # This saves a significant amount of lines of additional code in the post processing.
            for key in self.response.keys():
                self.response[key] = np.expand_dims(self.response[key], axis=0)
        
    def direct_trim(self):
        # The purpose of HYBRD is to find a zero of a system of N non-
        # linear functions in N variables by a modification of the Powell
        # hybrid method.  The user must provide a subroutine which calcu-
        # lates the functions.  The Jacobian is then calculated by a for-
        # ward-difference approximation.
        # http://www.math.utah.edu/software/minpack/minpack/hybrd.html
            
        if self.jcl.aero['method'] in [ 'mona_steady', 'mona_unsteady', 'hybrid', 'freq_dom'] and not hasattr(self.jcl, 'landinggear'):
            equations = Steady(self)
        elif self.jcl.aero['method'] in [ 'nonlin_steady']:
            equations = NonlinSteady(self)
        elif self.simcase['landinggear'] and self.jcl.landinggear['method'] in ['generic', 'skid']:
            equations = Landing(self)
        else:
            logging.error('Unknown aero method: ' + str(self.jcl.aero['method']))
        
        xfree_0 = np.array(self.trimcond_X[:,2], dtype='float')[np.where((self.trimcond_X[:,1] == 'free'))[0]]
        
        if self.trimcase['maneuver'] == 'bypass':
            logging.info('Running bypass...')
            self.response = equations.eval_equations(xfree_0, time=0.0, modus='trim_full_output')
            self.successful = True
        else:
            logging.info('Running trim for ' + str(len(xfree_0)) + ' variables...')
            xfree, info, status, msg= so.fsolve(equations.eval_equations, xfree_0, args=(0.0, 'trim'), full_output=True)
            logging.info(msg)
            logging.debug('Function evaluations: ' + str(info['nfev']))
            
            # no errors, check trim status for success
            if status == 1:
                # if trim was successful, then do one last evaluation with the final parameters.
                self.response = equations.eval_equations(xfree, time=0.0, modus='trim_full_output')
                self.successful = True
            else:
                self.response = {}
                self.successful = False
                logging.warning('SolutionSequences failed for subcase {}. The SolutionSequences solver reports: {}'.format(self.trimcase['subcase'], msg))
        equations.finalize()
        return

    def iterative_trim(self):
        if self.jcl.aero['method'] in [ 'mona_steady', 'mona_unsteady', 'hybrid']:
            equations = Steady(self)
        elif self.jcl.aero['method'] in [ 'cfd_steady']:
            specific_io.check_para_path(self.jcl)
            specific_io.copy_para_file(self.jcl, self.trimcase)
            specific_io.check_tau_folders(self.jcl)
            equations = CfdSteady(self)
        else:
            logging.error('Unknown aero method: ' + str(self.jcl.aero['method']))
        
        self.set_modal_states_fix()
        xfree_0 = np.array(self.trimcond_X[:,2], dtype='float')[np.where((self.trimcond_X[:,1] == 'free'))[0]] # start trim from scratch
        
        if self.trimcase['maneuver'] == 'bypass':
            logging.info('running bypass...')
            self.response = equations.eval_equations(xfree_0, time=0.0, modus='trim_full_output')
            self.successful = True
        else:
            logging.info('running trim for ' + str(len(xfree_0)) + ' variables...')
            try:
                """
                Because the iterative trim is typically used in combination with CFD, some solver settings need to be modified.
                - The jacobian matrix is constructed using finite differences. With CFD, a sufficiently large step size should 
                be used to obtain meaningful gradients (signal-to-noise ratio). This is controlled with parameter 'epsfcn=1.0e-3'.
                - Because both the aerodynamic solution and the aero-structural coupling are iterative procedures, the residuals 
                add up and the tolerance of the trim solution has to be increased. This is controlled with parameter 'xtol=1.0e-3'.
                - Approaching the trim point in small steps improves the robustness of the CFD solution. This is controlled with 
                parameter 'factor=0.1'.
                """
                xfree, info, status, msg= so.fsolve(equations.eval_equations_iteratively, xfree_0, args=(0.0, 'trim'), 
                                                    full_output=True, epsfcn=1.0e-3, xtol=1.0e-3, factor=0.1 )
            except TauError as e:
                self.response = {}
                self.successful = False
                logging.warning('SolutionSequences failed for subcase {} due to CFDError: {}'.format(self.trimcase['subcase'], e))
            except ConvergenceError as e:
                self.response = {}
                self.successful = False
                logging.warning('SolutionSequences failed for subcase {} due to ConvergenceError: {}'.format(self.trimcase['subcase'], e))
            else:
                logging.info(msg)
                logging.info('function evaluations: ' + str(info['nfev']))
                # no errors, check trim status for success
                if status == 1:
                    # if trim was successful, then do one last evaluation with the final parameters.
                    self.response = equations.eval_equations_iteratively(xfree, time=0.0, modus='trim_full_output')
                    self.successful = True
                else:
                    self.response = {}
                    self.successful = False
                    logging.warning('SolutionSequences failed for subcase {}. The SolutionSequences solver reports: {}'.format(self.trimcase['subcase'], msg))
        equations.finalize()
        return

    def exec_sim(self):
        # decrease dimension to simplify indexing (undo what has been done in the last lines of exec_trim() ) 
        self.response['X'] = self.response['X'][0,:]
        self.response['Y'] = self.response['Y'][0,:]
        # select solution sequence
        if self.jcl.aero['method'] in [ 'mona_steady', 'mona_unsteady', 'hybrid', 'nonlin_steady']:
            self.exec_sim_time_dom()
        elif self.jcl.aero['method'] in ['freq_dom']:
            self.exec_sim_freq_dom()
        else:
            logging.error('Unknown aero method: ' + str(self.jcl.aero['method']))
        
    def exec_sim_time_dom(self):
        if self.jcl.aero['method'] in [ 'mona_steady', 'hybrid'] and not hasattr(self.jcl, 'landinggear'):
            equations = Steady(self, X0=self.response['X'], simcase=self.simcase)
        elif self.jcl.aero['method'] in [ 'nonlin_steady']:
            equations = NonlinSteady(self, X0=self.response['X'], simcase=self.simcase)
        elif self.simcase['landinggear'] and self.jcl.landinggear['method'] in ['generic', 'skid']:
            self.add_landinggear() # add landing gear to system
            equations = Landing(self, X0=self.response['X'], simcase=self.simcase)
        elif self.jcl.aero['method'] in [ 'mona_unsteady']:
            if 'disturbance' in self.simcase.keys():
                logging.info('Adding disturbance of {} to state(s) '.format(self.simcase['disturbance']))
                self.response['X'][11+self.simcase['disturbance_mode']] += self.simcase['disturbance']
            self.add_lagstates() # add lag states to system
            equations = Unsteady(self, X0=self.response['X'], simcase=self.simcase)
        else:
            logging.error('Unknown aero method: ' + str(self.jcl.aero['method']))

        X0 = self.response['X']
        dt = self.simcase['dt']
        t_final = self.simcase['t_final']
        logging.info('Running time simulation for ' + str(t_final) + ' sec...')
        integrator = self.select_integrator(equations, 'Adams-Bashforth')
        integrator.set_initial_value(X0, 0.0)
        xt = []
        t = []
        while integrator.successful() and integrator.t < t_final:  
            integrator.integrate(integrator.t+dt)
            xt.append(integrator.y)
            t.append(integrator.t)
            
        if integrator.successful():
            logging.info('Simulation finished.')
            logging.info('Running (again) with full outputs at selected time steps...')
            equations.eval_equations(self.response['X'], 0.0, modus='sim_full_output')
            for i_step in np.arange(0,len(t)):
                response_step = equations.eval_equations(xt[i_step], t[i_step], modus='sim_full_output')
                for key in response_step.keys():
                    self.response[key] = np.vstack((self.response[key],response_step[key]))
            self.successful = True
        else:
            self.response = {}
            self.successful = False
            logging.warning('Integration failed!')
            return
    
    def select_integrator(self, equations, integration_scheme='Adams-Bashforth'):
        """
        Select an ode integration scheme:
        - two methods from scipy.integrate.ode (Adams-Bashforth and RK45) with variable time step size and 
        - two own implementations (RK4 and Euler) with fixed time step size 
        are available.
        Recommended: 'Adams-Bashforth'
        """
        if integration_scheme == 'RK4_FixedTimeStep':
            integrator = RungeKutta4(equations.ode_arg_sorter).set_integrator(stepwidth=1e-4)
        elif integration_scheme == 'Euler_FixedTimeStep':
            integrator = ExplicitEuler(equations.ode_arg_sorter).set_integrator(stepwidth=1e-4)
        elif integration_scheme == 'Adams-Bashforth':
            integrator = ode(equations.ode_arg_sorter).set_integrator('vode', method='adams', nsteps=2000, rtol=1e-4, atol=1e-4, max_step=5e-4) # non-stiff: 'adams', stiff: 'bdf'
        elif integration_scheme == 'RK45':
            integrator = ode(equations.ode_arg_sorter).set_integrator('dopri5', nsteps=2000, rtol=1e-2, atol=1e-8, max_step=1e-4)
        return integrator
            
    def exec_sim_freq_dom(self):
        if self.simcase['gust']:
            equations = GustExcitation(self, X0=self.response['X'], simcase=self.simcase)
        elif self.simcase['turbulence']:
            equations = TurbulenceExcitation(self, X0=self.response['X'], simcase=self.simcase)
        elif self.simcase['limit_turbulence']:
            equations = LimitTurbulence(self, X0=self.response['X'], simcase=self.simcase)
            self.response['Pmon_turb'] = 0.0
            self.response['correlations'] = 0.0
            self.response['X'] = np.expand_dims(self.response['X'], axis=0)
            self.response['Y'] = np.expand_dims(self.response['Y'], axis=0)
        response_sim = equations.eval_equations()
        for key in response_sim.keys():
            self.response[key] = response_sim[key] + self.response[key]
        logging.info('Frequency domain simulation finished.')
        self.successful = True
    
    def exec_flutter(self):
        # select solution sequence
        if self.simcase['flutter_para']['method'] == 'k':
            equations = KMethod(self, X0=self.response['X'][0,:], simcase=self.simcase)
        elif self.simcase['flutter_para']['method'] == 'ke':
            equations = KEMethod(self, X0=self.response['X'][0,:], simcase=self.simcase)
        elif self.simcase['flutter_para']['method'] == 'pk':
            equations = PKMethod(self, X0=self.response['X'][0,:], simcase=self.simcase)
        elif self.simcase['flutter_para']['method'] == 'statespace':
            equations = StateSpaceAnalysis(self, X0=self.response['X'][0,:], simcase=self.simcase)
        response_flutter = equations.eval_equations()
        logging.info('Flutter analysis finished.')
        for key in response_flutter.keys():
            self.response[key] = response_flutter[key]
        self.successful = True
        