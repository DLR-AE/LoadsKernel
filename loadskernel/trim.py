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

from loadskernel.model_equations.steady     import Steady
from loadskernel.model_equations.cfd_steady import CfdSteady
from loadskernel.model_equations.nonlin_steady import NonlinSteady
from loadskernel.model_equations.unsteady   import Unsteady
from loadskernel.model_equations.landing    import Landing
from loadskernel.model_equations.frequency_domain import GustExcitation

from loadskernel.trim_conditions import TrimConditions

class Trim(TrimConditions):
    def calc_jacobian(self):
        if self.jcl.aero['method'] in [ 'mona_steady', 'hybrid']:
            equations = model_equations.Steady(self)
            """
            The Jacobian matrix is computed about the trimmed flight condition. 
            Alternatively, it may be computed about the trim condition specified in the JCL with: X0 = np.array(self.trimcond_X[:,2], dtype='float')
            """
            X0 = self.response['X']
            self.idx_lag_states = []
        elif self.jcl.aero['method'] in [ 'mona_unsteady']:
            self.add_lagstates()
            equations = model_equations.Unsteady(self)
        else:
            logging.error('Unknown aero method: ' + str(self.jcl.aero['method']))
            
        
        def approx_jacobian(X0,func,epsilon,dt):
            """Approximate the Jacobian matrix of callable function func
               * Parameters
                 x       - The state vector at which the Jacobian matrix is desired
                 func    - A vector-valued function of the form f(x,*args)
                 epsilon - The peturbation used to determine the partial derivatives
                 *args   - Additional arguments passed to func
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
        
        logging.info('Calculating jacobian for ' + str(len(X0)) + ' variables...')
        jac = approx_jacobian(X0=X0, func=equations.equations, epsilon=0.001, dt=1.0) # epsilon sollte klein sein, dt sollte 1.0s sein
        X = self.response['X']
        Y = self.response['Y']
        self.response.clear()
        self.response['X'] = X
        self.response['Y'] = Y
        self.response['X0'] = X0 # Linearisierungspunkt
        self.response['Y0'] = equations.equations(X0, t=0.0, modus='trim')
        self.response['jac'] = jac
        self.response['states'] = self.states[:,0]
        self.response['state_derivativess'] = self.state_derivatives[:,0]
        self.response['inputs'] = self.inputs[:,0]
        self.response['outputs'] = self.outputs[:,0]
        # States need to be reordered into ABCD matrices!
        # X = [ rbm,  flex,  command_cs,  lag_states ]
        # Y = [drbm, dflex, dcommand_cs, dlag_states, outputs]
        # [Y] = [A B] * [X]
        #       [C D]   
        idx_A = self.idx_states+self.idx_lag_states
        idx_B = self.idx_inputs
        idx_C = self.idx_outputs
        self.response['A'] = jac[idx_A,:][:,idx_A] # aircraft itself
        self.response['B'] = jac[idx_A,:][:,idx_B] # reaction of aircraft on external excitation
        self.response['C'] = jac[idx_C,:][:,idx_A] # sensors
        self.response['D'] = jac[idx_C,:][:,idx_B] # reaction of sensors on external excitation
        self.response['idx_A'] = idx_A
        self.response['idx_B'] = idx_B
        self.response['idx_C'] = idx_C
            
    def calc_derivatives(self):        
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
        logging.info('Calculating derivatives for ' + str(len(X0)) + ' variables...')
        logging.info('MAC_ref = {}'.format(self.jcl.general['MAC_ref']))
        logging.info('A_ref = {}'.format(self.jcl.general['A_ref']))
        logging.info('b_ref = {}'.format(self.jcl.general['b_ref']))
        logging.info('c_ref = {}'.format(self.jcl.general['c_ref']))
        logging.info('')
        logging.info('Derivatives given in body axis (aft-right-up):')
        logging.info('--------------------------------------------------------------------------------------')
        logging.info('                     Cx         Cy         Cz         Cmx        Cmy        Cmz')
        for i in range(len(X0)):
            xi = copy.deepcopy(X0)
            xi[i] += delta
            response = equations.equations(xi, 0.0, 'trim_full_output')
            Pmac_c = (response['Pmac']-response0['Pmac'])/response['q_dyn']/A/delta
            tmp = '{:>20} {:< 10.4g} {:< 10.4g} {:< 10.4g} {:< 10.4g} {:< 10.4g} {:< 10.4g}'.format( self.trimcond_X[i,0], Pmac_c[0], Pmac_c[1], Pmac_c[2], Pmac_c[3]/self.model.macgrid['b_ref'], Pmac_c[4]/self.model.macgrid['c_ref'], Pmac_c[5]/self.model.macgrid['b_ref'] )
            logging.info(tmp)
        logging.info('--------------------------------------------------------------------------------------')
                            
    def exec_trim(self):
        if self.jcl.aero['method'] in [ 'mona_steady', 'mona_unsteady', 'hybrid', 'nonlin_steady', 'freq_dom']:
            self.direct_trim()
        elif self.jcl.aero['method'] in [ 'cfd_steady']:
            self.iterative_trim()
        else:
            logging.error('Unknown aero method: ' + str(self.jcl.aero['method']))
        
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
        elif self.simcase['landinggear'] and self.jcl.landinggear['method'] == 'generic':
            equations = Landing(self)
        else:
            logging.error('Unknown aero method: ' + str(self.jcl.aero['method']))
        
        xfree_0 = np.array(self.trimcond_X[:,2], dtype='float')[np.where((self.trimcond_X[:,1] == 'free'))[0]]
        
        if self.trimcase['maneuver'] == 'bypass':
            logging.info('running bypass...')
            self.response = equations.eval_equations(xfree_0, time=0.0, modus='trim_full_output')
            self.successful = True
        else:
            logging.info('running trim for ' + str(len(xfree_0)) + ' variables...')
            xfree, info, status, msg= so.fsolve(equations.eval_equations, xfree_0, args=(0.0, 'trim'), full_output=True)
            logging.info(msg)
            logging.info('function evaluations: ' + str(info['nfev']))
            
            # no errors, check trim status for success
            if status == 1:
                # if trim was successful, then do one last evaluation with the final parameters.
                self.response = equations.eval_equations(xfree, time=0.0, modus='trim_full_output')
                self.successful = True
            else:
                self.response = {}
                self.successful = False
                logging.warning('Trim failed for subcase {}. The Trim solver reports: {}'.format(self.trimcase['subcase'], msg))
                return

    def iterative_trim(self):
        if self.jcl.aero['method'] in [ 'mona_steady', 'mona_unsteady', 'hybrid']:
            equations = Steady(self)
        elif self.jcl.aero['method'] in [ 'cfd_steady']:
            equations = CfdSteady(self)
            specific_io.check_para_path(self.jcl)
            specific_io.copy_para_file(self.jcl, self.trimcase)
            specific_io.check_tau_folders(self.jcl)
        else:
            logging.error('Unknown aero method: ' + str(self.jcl.aero['method']))
        
        # remove modes from trimcond_Y and _Y
        i_mass = self.model.mass['key'].index(self.trimcase['mass'])
        n_modes = self.model.mass['n_modes'][i_mass]
        
        for i_mode in range(n_modes):
            self.trimcond_X[np.where((self.trimcond_X[:,0] == 'Uf'+str(i_mode)))[0][0],1] = 'fix'
            self.trimcond_X[np.where((self.trimcond_X[:,0] == 'dUf_dt'+str(i_mode)))[0][0],1] = 'fix'
            self.trimcond_Y[np.where((self.trimcond_Y[:,0] == 'dUf_dt'+str(i_mode)))[0][0],1] = 'fix'
            self.trimcond_Y[np.where((self.trimcond_Y[:,0] == 'd2Uf_d2t'+str(i_mode)))[0][0],1] = 'fix'
        xfree_0 = np.array(self.trimcond_X[:,2], dtype='float')[np.where((self.trimcond_X[:,1] == 'free'))[0]] # start trim from scratch
        
        if self.trimcase['maneuver'] == 'bypass':
            logging.info('running bypass...')
            self.response = equations.eval_equations(xfree_0, time=0.0, modus='trim_full_output')
        else:
            logging.info('running trim for ' + str(len(xfree_0)) + ' variables...')
            try:
                xfree, info, status, msg= so.fsolve(equations.eval_equations_iteratively, xfree_0, args=(0.0, 'trim'), full_output=True, epsfcn=1.0e-3, xtol=1.0e-3 )
            except Common.TauError as e:
                self.response = {}
                self.successful = False
                logging.warning('Trim failed for subcase {} due to TauError: {}'.format(self.trimcase['subcase'], e))
                return
            except Common.ConvergenceError as e:
                self.response = {}
                self.successful = False
                logging.warning('Trim failed for subcase {} due to ConvergenceError: {}'.format(self.trimcase['subcase'], e))
                return
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
                    logging.warning('Trim failed for subcase {}. The Trim solver reports: {}'.format(self.trimcase['subcase'], msg))
                    return

    def exec_sim(self):
        if self.jcl.aero['method'] in [ 'mona_steady', 'mona_unsteady', 'hybrid', 'nonlin_steady']:
            self.exec_sim_time_dom()
        elif self.jcl.aero['method'] in [ 'freq_dom']:
            self.exec_sim_freq_dom()
        else:
            logging.error('Unknown aero method: ' + str(self.jcl.aero['method']))
        
        
    def exec_sim_time_dom(self):
        if self.jcl.aero['method'] in [ 'mona_steady', 'hybrid'] and not hasattr(self.jcl, 'landinggear'):
            equations = Steady(self, X0=self.response['X'], simcase=self.simcase)
        elif self.jcl.aero['method'] in [ 'nonlin_steady']:
            equations = NonlinSteady(self, X0=self.response['X'], simcase=self.simcase)
        elif self.simcase['landinggear'] and self.jcl.landinggear['method'] == 'generic':
            self.add_landinggear() # add landing gear to system
            equations = Landing(self, X0=self.response['X'], simcase=self.simcase)
        elif self.jcl.aero['method'] in [ 'mona_unsteady']:
            if 'disturbance' in self.simcase.keys():
                logging.info('adding disturbance of {} to state(s) '.format(self.simcase['disturbance']))
                self.response['X'][11+self.simcase['disturbance_mode']] += self.simcase['disturbance']
            self.add_lagstates() # add lag states to system
            equations = Unsteady(self, X0=self.response['X'], simcase=self.simcase)
        else:
            logging.error('Unknown aero method: ' + str(self.jcl.aero['method']))

        X0 = self.response['X']
        dt = self.simcase['dt']
        t_final = self.simcase['t_final']
        logging.info('running time simulation for ' + str(t_final) + ' sec...')
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
            logging.info('running (again) with full outputs at selected time steps...')
            equations.eval_equations(self.response['X'], 0.0, modus='sim_full_output')
            for i_step in np.arange(0,len(t)):
                response_step = equations.eval_equations(xt[i_step], t[i_step], modus='sim_full_output')
                for key in self.response.keys():
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
        equations = GustExcitation(self, X0=self.response['X'], simcase=self.simcase)
        response_sim = equations.eval_equations()
        for key in response_sim.keys():
            response_sim[key] += self.response[key]
        self.response = response_sim
        logging.info('Frequency domain simulation finished.')
        self.successful = True
        