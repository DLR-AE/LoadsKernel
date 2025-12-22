import copy
import logging
import numpy as np
import scipy.optimize as so
from scipy.integrate import ode
from scipy.fftpack import fft, fftfreq

from loadskernel.integrate import RungeKutta4, ExplicitEuler, AdamsBashforth
from loadskernel.equations.mona_time_domain import Steady, Unsteady, NonlinSteady, Landing
from loadskernel.equations.cfd_time_domain import CfdSteady, CfdUnsteady
from loadskernel.equations.common import ConvergenceError
from loadskernel.equations import mona_frequency_domain, cfd_frequency_domain, mona_state_space
from loadskernel.trim_conditions import TrimConditions
from loadskernel.cfd_interfaces.tau_interface import TauError
from loadskernel.io_functions.data_handling import load_hdf5_sparse_matrix, load_hdf5_dict
from loadskernel.solution_tools import polynomial_pulse, one_m_cosine_pulse


class SolutionSequences(TrimConditions):

    def approx_jacobian(self, X0, func, epsilon, dt):
        """
        Approximate the Jacobian matrix of callable function func
        x       - The state vector at which the Jacobian matrix is desired
        func    - A vector-valued function of the form f(x,*args)
        epsilon - The peturbation used to determine the partial derivatives
        """
        X0 = np.asarray(X0, dtype=np.double)
        jac = np.zeros([len(func(*(X0, 0.0, 'sim'))), len(X0)])
        dX = np.zeros(len(X0))
        for i in range(len(X0)):
            f0 = func(*(X0, 0.0, 'sim'))
            dX[i] = epsilon
            fi = func(*(X0 + dX, 0.0 + dt, 'sim'))
            jac[:, i] = (fi - f0) / epsilon
            dX[i] = 0.0
        return jac

    def calc_jacobian(self):
        """
        The Jacobian matrix is computed about the trimmed flight condition.
        Alternatively, it may be computed about the trim condition specified in the JCL with
        X0 = np.array(self.trimcond_X[:,2], dtype='float')
        """

        if self.jcl.aero['method'] in ['mona_steady']:
            equations = Steady(self)
        else:
            logging.error('Unknown aero method: %s', self.jcl.aero['method'])
            equations = None

        X0 = self.response['X'][0, :]
        logging.info('Calculating jacobian for %d variables...', len(X0))
        # epsilon sollte klein sein, dt sollte 1.0s sein
        jac = self.approx_jacobian(X0=X0, func=equations.equations, epsilon=0.01, dt=1.0)
        self.response['X0'] = X0  # Linearisierungspunkt
        self.response['Y0'] = equations.equations(X0, t=0.0, modus='trim')
        self.response['jac'] = jac
        self.response['states'] = self.states[:, 0].tolist()
        self.response['state_derivatives'] = self.state_derivatives[:, 0].tolist()
        self.response['inputs'] = self.inputs[:, 0].tolist()
        self.response['outputs'] = self.outputs[:, 0].tolist()
        # States need to be reordered into ABCD matrices!
        # X = [rbm,  flex,  command_cs,  lag_states ]
        # Y = [drbm, dflex, dcommand_cs, dlag_states, outputs]
        # [Y] = [A B] * [X]
        #       [C D]
        idx_9dof = self.idx_states[3:12]
        idx_A = self.idx_states
        idx_B = self.idx_inputs
        idx_C = self.idx_outputs
        self.response['9DOF'] = jac[idx_9dof, :][:, idx_9dof]  # rigid body motion only
        self.response['A'] = jac[idx_A, :][:, idx_A]  # aircraft itself, including elastic states
        self.response['B'] = jac[idx_A, :][:, idx_B]  # reaction of aircraft on external excitation
        self.response['C'] = jac[idx_C, :][:, idx_A]  # sensors
        self.response['D'] = jac[idx_C, :][:, idx_B]  # reaction of sensors on external excitation
        self.response['idx_A'] = idx_A
        self.response['idx_B'] = idx_B
        self.response['idx_C'] = idx_C
        self.response['desc'] = self.trimcase['desc']

        # perform analysis on jacobian matrix
        equations = mona_state_space.JacobiAnalysis(self.response)
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
        if self.jcl.aero['method'] in ['mona_steady', 'mona_unsteady']:
            equations = Steady(self)
        elif self.jcl.aero['method'] in ['nonlin_steady']:
            equations = NonlinSteady(self)
        else:
            logging.error('Unknown aero method: %s', self.jcl.aero['method'])
            equations = None

        A = self.jcl.general['A_ref']
        delta = 0.01

        X0 = np.array(self.trimcond_X[:, 2], dtype='float')
        response0 = equations.equations(X0, 0.0, 'trim_full_output')
        derivatives = []
        logging.info('Calculating rigid derivatives...')
        for i in range(len(X0)):
            xi = copy.deepcopy(X0)
            xi[i] += delta
            response = equations.equations(xi, 0.0, 'trim_full_output')
            Pmac_c = (response['Pmac'] - response0['Pmac']) / response['q_dyn'] / A / delta
            derivatives.append([Pmac_c[0], Pmac_c[1], Pmac_c[2], Pmac_c[3] / self.model['macgrid']['b_ref'],
                                Pmac_c[4] / self.model['macgrid']['c_ref'], Pmac_c[5] / self.model['macgrid']['b_ref']])
        # write back original response and store results
        self.response['rigid_parameters'] = self.trimcond_X[:, 0].tolist()
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

        vtas = self.trimcase['Ma'] * self.model['atmo'][self.trimcase['altitude']]['a'][()]
        A = self.jcl.general['A_ref']

        delta = 0.01
        parameters = ['theta', 'psi', 'p', 'q', 'r', 'command_xi', 'command_eta', 'command_zeta']
        derivatives = []
        logging.info('Calculating flexible derivatives...')
        for parameter in parameters:
            # modify selected parameter in trim conditions
            self.trimcond_X[np.where((np.vstack((self.states, self.inputs))[:, 0] == parameter))[0][0], 2] += delta
            if parameter == 'theta':
                theta = self.trimcond_X[np.where((np.vstack((self.states, self.inputs))[:, 0] == parameter))[0][0], 2]
                self.trimcond_X[np.where((np.vstack((self.states, self.inputs))[:, 0] == 'u'))[0][0], 2] = vtas * np.cos(theta)
                self.trimcond_X[np.where((np.vstack((self.states, self.inputs))[:, 0] == 'w'))[0][0], 2] = vtas * np.sin(theta)
            elif parameter == 'psi':
                psi = self.trimcond_X[np.where((np.vstack((self.states, self.inputs))[:, 0] == parameter))[0][0], 2]
                self.trimcond_X[np.where((np.vstack((self.states, self.inputs))[:, 0] == 'u'))[0][0], 2] = vtas * np.cos(psi)
                self.trimcond_X[np.where((np.vstack((self.states, self.inputs))[:, 0] == 'v'))[0][0], 2] = vtas * np.sin(psi)
            # re-calculate new trim
            self.exec_trim()
            Pmac_c = (self.response['Pmac'] - response0['Pmac']) / response0['q_dyn'] / A / delta
            derivatives.append([Pmac_c[0, 0], Pmac_c[0, 1], Pmac_c[0, 2], Pmac_c[0, 3] / self.model['macgrid']['b_ref'],
                                Pmac_c[0, 4] / self.model['macgrid']['c_ref'], Pmac_c[0, 5] / self.model['macgrid']['b_ref']])
            # restore trim condition for next loop
            self.trimcond_X = copy.deepcopy(trimcond_X0)
        # write back original response and store results
        self.response = response0
        self.response['flexible_parameters'] = parameters
        self.response['flexible_derivatives'] = derivatives

    def calc_NP(self):
        pos = self.response['flexible_parameters'].index('theta')
        self.response['NP_flex'] = np.zeros(3)
        self.response['NP_flex'][0] = self.model['macgrid']['offset'][0, 0] - self.jcl.general['c_ref'] \
            * self.response['flexible_derivatives'][pos][4] / self.response['flexible_derivatives'][pos][2]
        self.response['NP_flex'][1] = self.model['macgrid']['offset'][0, 1] + self.jcl.general['b_ref'] \
            * self.response['flexible_derivatives'][pos][3] / self.response['flexible_derivatives'][pos][2]
        logging.info('--------------------------------------------------------------------------------------')
        logging.info('Aeroelastic neutral point / aerodynamic center:')
        logging.info('NP_flex (x,y) = %0.4g,%0.4g', self.response['NP_flex'][0], self.response['NP_flex'][1])

    def calc_cs_effectiveness(self):
        logging.info('--------------------------------------------------------------------------------------')
        logging.info('Aeroelastic control surface effectiveness:')
        logging.info('                     Cx         Cy         Cz         Cmx        Cmy        Cmz')
        for p in ['command_xi', 'command_eta', 'command_zeta']:
            pos_rigid = self.response['rigid_parameters'].index(p)
            pos_flex = self.response['flexible_parameters'].index(p)
            d = np.array(self.response['flexible_derivatives'][pos_flex]) \
                / np.array(self.response['rigid_derivatives'][pos_rigid])
            tmp = f'{p:>20} {d[0]:< 10.4g} {d[1]:< 10.4g} {d[2]:< 10.4g} {d[3]:< 10.4g} {d[4]:< 10.4g} {d[5]:< 10.4g}'
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
        vtas = self.trimcase['Ma'] * self.model['atmo'][self.trimcase['altitude']]['a'][()]

        self.response[key + '_parameters'] += ['p*', 'q*', 'r*']
        self.response[key + '_derivatives'].append(list(np.array(
            self.response[key + '_derivatives'][self.response[key + '_parameters'].index('p')])
            / self.jcl.general['b_ref'] * 2.0 * vtas))
        self.response[key + '_derivatives'].append(list(np.array(
            self.response[key + '_derivatives'][self.response[key + '_parameters'].index('q')])
            / self.jcl.general['c_ref'] * 2.0 * vtas))
        self.response[key + '_derivatives'].append(list(np.array(
            self.response[key + '_derivatives'][self.response[key + '_parameters'].index('r')])
            / self.jcl.general['b_ref'] * 2.0 * vtas))

    def print_derivatives(self, key):
        # print some information into log file
        # key: 'rigid' or 'flexible'
        parameters = self.response[key + '_parameters']
        derivatives = self.response[key + '_derivatives']
        logging.info('--------------------------------------------------------------------------------------')
        logging.info('Calculated %s derivatives for %d variables.', key, len(parameters))
        logging.info('MAC_ref = %s', self.jcl.general['MAC_ref'])
        logging.info('A_ref = %s', self.jcl.general['A_ref'])
        logging.info('b_ref = %s', self.jcl.general['b_ref'])
        logging.info('c_ref = %s', self.jcl.general['c_ref'])
        logging.info('q_dyn = %s', self.response['q_dyn'][0])
        logging.info('Derivatives given in body axis (aft-right-up):')
        logging.info('                     Cx         Cy         Cz         Cmx        Cmy        Cmz')
        for p, d in zip(parameters, derivatives):
            tmp = f'{p:>20} {d[0]:< 10.4g} {d[1]:< 10.4g} {d[2]:< 10.4g} {d[3]:< 10.4g} {d[4]:< 10.4g} {d[5]:< 10.4g}'
            logging.info(tmp)

    def exec_trim(self):
        if self.jcl.aero['method'] in ['mona_steady', 'mona_unsteady', 'nonlin_steady', 'freq_dom', 'mona_freq_dom']:
            self.direct_trim()
        elif self.jcl.aero['method'] in ['cfd_steady', 'cfd_unsteady']:
            self.iterative_trim()
        elif self.jcl.aero['method'] in ['cfd_freq_dom']:
            logging.info('Using response / trim data form CFD-based GAF computation.')
            # Fetch data from the GAF computation and fill the response.
            # The assumption is that the gust is superposed with the linearization point used in the GAF computation.
            key = '.'.join(self.trimcase['desc'].split('.')[:-1])
            if key in self.model['GAFs']:
                self.response = load_hdf5_dict(self.model['GAFs'][key]['response'])
                self.successful = True
            else:
                logging.error('No response / trim data found for "%s" in model!', key)
                self.successful = False
        else:
            logging.error('Unknown aero method: %s', str(self.jcl.aero['method']))
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

        if self.jcl.aero['method'] in ['mona_steady', 'mona_unsteady',
                                       'freq_dom', 'mona_freq_dom'] and not hasattr(self.jcl, 'landinggear'):
            equations = Steady(self)
        elif self.jcl.aero['method'] in ['nonlin_steady']:
            equations = NonlinSteady(self)
        elif self.simcase['landinggear'] and self.jcl.landinggear['method'] in ['generic', 'skid']:
            equations = Landing(self)
        else:
            logging.error('Unknown aero method: %s', self.jcl.aero['method'])
            equations = None

        xfree_0 = np.array(self.trimcond_X[:, 2], dtype='float')[np.where((self.trimcond_X[:, 1] == 'free'))[0]]

        if self.trimcase['maneuver'] == 'bypass':
            logging.info('Bypassing trim.')
            self.response = equations.eval_equations(xfree_0, time=0.0, modus='trim_full_output')
            self.successful = True
        else:
            logging.info('Running trim for %d variables...', len(xfree_0))
            xfree, info, status, msg = so.fsolve(equations.eval_equations, xfree_0, args=(0.0, 'trim'), full_output=True)
            logging.info('%s', msg)
            logging.debug('Function evaluations: %d', info['nfev'])

            # no errors, check trim status for success
            if status == 1:
                # if trim was successful, then do one last evaluation with the final parameters.
                self.response = equations.eval_equations(xfree, time=0.0, modus='trim_full_output')
                self.successful = True
            else:
                self.response = {}
                self.successful = False
                logging.warning('SolutionSequences failed for subcase %s. The SolutionSequences solver reports: %s',
                                self.trimcase['subcase'], msg)
        equations.finalize()
        return

    def iterative_trim(self):
        if self.jcl.aero['method'] in ['mona_steady', 'mona_unsteady']:
            equations = Steady(self)
        elif self.jcl.aero['method'] in ['cfd_steady', 'cfd_unsteady', 'cfd_freq_dom']:
            equations = CfdSteady(self)
        else:
            logging.error('Unknown aero method: %s', self.jcl.aero['method'])
            equations = None

        self.set_modal_states_fix()
        # start trim from scratch
        xfree_0 = np.array(self.trimcond_X[:, 2], dtype='float')[np.where((self.trimcond_X[:, 1] == 'free'))[0]]

        if self.trimcase['maneuver'] == 'bypass':
            logging.info('Bypassing trim.')
            self.response = equations.eval_equations(xfree_0, time=0.0, modus='trim_full_output')
            self.successful = True
        else:
            logging.info('Running trim for %d variables...', len(xfree_0))
            """
            Because the iterative trim is typically used in combination with CFD, some solver settings need to be modified.
            - The jacobian matrix is constructed using finite differences. With CFD, a sufficiently large step size should
            be used to obtain meaningful gradients (signal-to-noise ratio). This is controlled with parameter 'epsfcn=1.0e-3'.
            - Because both the aerodynamic solution and the aero-structural coupling are iterative procedures, the residuals
            add up and the tolerance of the trim solution has to be increased. This is controlled with parameter 'xtol=1.0e-3'.
            - Approaching the trim point in small steps improves the robustness of the CFD solution. This is controlled with
            parameter 'factor=0.1'.
            """
            try:
                xfree, info, status, msg = so.fsolve(
                    equations.eval_equations_iteratively, xfree_0, args=(0.0, 'trim'),
                    full_output=True, epsfcn=1.0e-3, xtol=1.0e-3, factor=0.1)
            except TauError as e:
                self.response = {}
                self.successful = False
                logging.warning('SolutionSequences failed for subcase %s due to CFDError: %s',
                                self.trimcase['subcase'], e)
            except ConvergenceError as e:
                self.response = {}
                self.successful = False
                logging.warning('SolutionSequences failed for subcase %s due to ConvergenceError: %s',
                                self.trimcase['subcase'], e)
            else:
                logging.info('%s', msg)
                logging.info('function evaluations: %d', info['nfev'])
                if status == 1:
                    self.response = equations.eval_equations_iteratively(
                        xfree, time=0.0, modus='trim_full_output')
                    self.successful = True
                else:
                    self.response = {}
                    self.successful = False
                    logging.warning('SolutionSequences failed for subcase %s. The SolutionSequences solver reports: %s',
                                    self.trimcase['subcase'], msg)
        equations.finalize()
        return

    def exec_sim(self):
        # select solution sequence
        if self.jcl.aero['method'] in ['mona_steady', 'mona_unsteady', 'nonlin_steady', 'cfd_unsteady']:
            self.exec_sim_time_dom()
        elif self.jcl.aero['method'] in ['freq_dom', 'mona_freq_dom', 'cfd_freq_dom']:
            self.exec_sim_freq_dom()
        else:
            logging.error('Unknown aero method: %s', str(self.jcl.aero['method']))

    def exec_sim_time_dom(self):
        """
        Select the right set of equations.
        If required, add new states, e.g. for the landing gear or unsteady aerodynamics.
        """
        # get initial solution from trim
        X0 = self.response['X'][0, :]
        # select solution sequence
        if self.jcl.aero['method'] in ['mona_steady'] and not hasattr(self.jcl, 'landinggear'):
            equations = Steady(self, X0)
        elif self.jcl.aero['method'] in ['nonlin_steady']:
            equations = NonlinSteady(self, X0)
        elif self.simcase['landinggear'] and self.jcl.landinggear['method'] in ['generic', 'skid']:
            # add landing gear to system
            self.add_landinggear()
            # reset initial solution including new states
            X0 = self.response['X'][0, :]
            equations = Landing(self, X0)
        elif self.jcl.aero['method'] in ['mona_unsteady']:
            if 'disturbance' in self.simcase.keys():
                logging.info('Adding disturbance of %s to state(s) ', self.simcase['disturbance'])
                self.response['X'][0, 11 + self.simcase['disturbance_mode']] += self.simcase['disturbance']
            # add lag states to system
            self.add_lagstates()
            # reset initial solution including new states
            X0 = self.response['X'][0, :]
            equations = Unsteady(self, X0)
        elif self.jcl.aero['method'] in ['cfd_unsteady']:
            equations = CfdUnsteady(self, X0)
        else:
            logging.error('Unknown aero method: %s', self.jcl.aero['method'])
            equations = None

        """
        There are two ways of time intergartion.

        In most cases, the Adams Bashforth method provided by scipy.integrate.ode is used:
        Advantages
        + Good accuracy controll by adaptive time step size
        + Tested by continuous integration chain with a long history of numerically equivalet results
        Disadvantages
        - Accepts only the derivative 'dy' and doesn't handle any additional outputs (like the response at the given time
          step). This requires a second run at the selected time steps to obatin the full outputs / response dictionary.
        - Adaptive step size not suitable for CFD applications

        Self-implemented Adams Bashforth integration sheme:
        Advantages
        + Fixed time step size
        + Handles dictionary outputs of the ode functions
        Disadvantages
        - Not fully tested
        """

        if 'dt_integration' in self.simcase:
            dt_integration = self.simcase['dt_integration']
        else:
            dt_integration = self.simcase['dt']
        dt = self.simcase['dt']
        t_final = self.simcase['t_final']
        xt = []
        t = []

        logging.info('Running time simulation for %g sec...', t_final)
        if self.jcl.aero['method'] in ['cfd_unsteady']:
            integrator = self.select_integrator(equations, 'AdamsBashforth_FixedTimeStep', dt_integration)
            integrator.set_initial_value(X0, 0.0)

            while integrator.successful() and integrator.t < t_final:
                integrator.integrate(integrator.t + dt)
                xt.append(integrator.y)
                t.append(integrator.t)
                # To avoid an excessive amount of data, e.g. during unsteady cfd simulations,
                # keep only the response data on the first mpi process (id = 0).
                if self.myid == 0:
                    for key in integrator.output_dict.keys():
                        self.response[key] = np.vstack((self.response[key], integrator.output_dict[key]))

        else:
            integrator = self.select_integrator(equations, 'AdamsBashforth')
            integrator.set_initial_value(X0, 0.0)

            while integrator.successful() and integrator.t < t_final:
                integrator.integrate(integrator.t + dt)
                xt.append(integrator.y)
                t.append(integrator.t)

            if integrator.successful():
                logging.info('Simulation finished. Running (again) with full outputs at selected time steps...')
                equations.eval_equations(X0, 0.0, modus='sim_full_output')
                for i_step in np.arange(0, len(t)):
                    response_step = equations.eval_equations(xt[i_step], t[i_step], modus='sim_full_output')
                    for key in response_step.keys():
                        self.response[key] = np.vstack((self.response[key], response_step[key]))

        # Handle unsucessful time integration
        if integrator.successful():
            self.successful = True
        else:
            self.response = {}
            self.successful = False
            logging.warning('Integration failed!')
            return

    def select_integrator(self, equations, integration_scheme='AdamsBashforth', stepwidth=1e-4):
        """
        Select an ode integration scheme:
        - two methods from scipy.integrate.ode (Adams-Bashforth and RK45) with variable time step size and
        - three own implementations (RK4, Euler and AdamsBashforth) with fixed time step size
        are available.
        Recommended: 'Adams-Bashforth'
        """
        if integration_scheme == 'RK4_FixedTimeStep':
            integrator = RungeKutta4(equations.ode_arg_sorter).set_integrator(stepwidth)
        elif integration_scheme == 'Euler_FixedTimeStep':
            integrator = ExplicitEuler(equations.ode_arg_sorter).set_integrator(stepwidth)
        elif integration_scheme == 'AdamsBashforth_FixedTimeStep':
            integrator = AdamsBashforth(equations.ode_arg_sorter).set_integrator(stepwidth)
        elif integration_scheme == 'AdamsBashforth':
            integrator = ode(equations.ode_arg_sorter).set_integrator('vode', method='adams', nsteps=2000,
                                                                      rtol=1e-4, atol=1e-4, max_step=5e-4)
        elif integration_scheme == 'RK45':
            integrator = ode(equations.ode_arg_sorter).set_integrator('dopri5', nsteps=2000,
                                                                      rtol=1e-2, atol=1e-8, max_step=1e-4)
        else:
            logging.error('Unknown integration scheme: %s.', integration_scheme)
            integrator = None
        return integrator

    def exec_sim_freq_dom(self):
        # get initial solution from trim
        X0 = self.response['X'][0, :]
        if self.jcl.aero['method'] in ['freq_dom', 'mona_freq_dom']:
            # select solution sequence
            if self.simcase['gust']:
                equations = mona_frequency_domain.GustExcitation(self, X0)
            elif self.simcase['turbulence']:
                equations = mona_frequency_domain.TurbulenceExcitation(self, X0)
            elif self.simcase['limit_turbulence']:
                equations = mona_frequency_domain.LimitTurbulence(self, X0)
                self.response['Pmon_turb'] = 0.0
                self.response['correlations'] = 0.0
            else:
                logging.error('Unknown frequency domain simulation type.')
                equations = None
        elif self.jcl.aero['method'] in ['cfd_freq_dom']:
            # select solution sequence
            if self.simcase['gust']:
                equations = cfd_frequency_domain.GustExcitation(self, X0)
            else:
                logging.error('Unknown CFD-based simulation type.')
                equations = None
        else:
            logging.error('Unknown aero method: %s', self.jcl.aero['method'])
            equations = None
        response_sim = equations.eval_equations()
        for key, item in response_sim.items():
            self.response[key] = item + self.response[key]
        logging.info('Frequency domain simulation finished.')
        self.successful = True

    def exec_flutter(self):
        if self.jcl.aero['method'] in ['freq_dom', 'mona_freq_dom']:
            # Get initial solution from trim
            X0 = self.response['X'][0, :]
            # Select mona-based solution sequence
            if self.simcase['flutter_para']['method'] == 'k':
                equations = mona_frequency_domain.KMethod(self, X0)
            elif self.simcase['flutter_para']['method'] == 'ke':
                equations = mona_frequency_domain.KEMethod(self, X0)
            elif self.simcase['flutter_para']['method'] in ['pk', 'pk_schwochow']:
                equations = mona_frequency_domain.PKMethodSchwochow(self, X0)
            elif self.simcase['flutter_para']['method'] in ['pk_rodden']:
                equations = mona_frequency_domain.PKMethodRodden(self, X0)
            else:
                logging.error('Unknown mona-based flutter method: %s', self.simcase['flutter_para']['method'])
                equations = None
        elif (self.jcl.aero['method'] in ['mona_steady', 'mona_unsteady']
                and self.simcase['flutter_para']['method'] == 'statespace'):
            # Get initial solution from trim
            X0 = self.response['X'][0, :]
            equations = mona_state_space.StateSpaceAnalysis(self, X0)
        elif self.jcl.aero['method'] in ['cfd_freq_dom']:
            # Select cfd-based solution sequence
            if self.simcase['flutter_para']['method'] == 'k':
                equations = cfd_frequency_domain.KMethod(self)
            elif self.simcase['flutter_para']['method'] == 'ke':
                equations = cfd_frequency_domain.KEMethod(self)
            elif self.simcase['flutter_para']['method'] in ['pk_rodden']:
                equations = cfd_frequency_domain.PKMethodRodden(self)
            else:
                logging.error('Unknown CFD-based flutter method: %s', self.simcase['flutter_para']['method'])
                equations = None
        else:
            logging.error('Unknown aero method: %s', self.jcl.aero['method'])
            equations = None
        response_flutter = equations.eval_equations()
        logging.info('Flutter analysis finished.')
        for key, item in response_flutter.items():
            self.response[key] = item
        self.successful = True

    def exec_pulse(self):
        # Get initial solution from trim
        X0 = self.response['X'][0, :]
        Vtas = sum(X0[6:9] ** 2) ** 0.5
        # In case I decide to scale the GAFs with the dynamic pressure, I can use q_dyn from here
        # q_dyn = self.response['q_dyn'][0]

        # Inline function to calculate reduced frequencies, Nastran definition
        def f2k(f):
            return 2.0 * np.pi * f * self.jcl.general['c_ref'] / 2.0 / Vtas

        # Get number of modes
        n_modes_rbm = 5
        n_modes_flex = self.model['mass'][self.trimcase['mass']]['n_modes'][()]
        PHIkh = self.model['mass'][self.trimcase['mass']]['PHIkh'][()]
        n_modes = n_modes_rbm + n_modes_flex
        # This is the index of each mode in the state vector X
        idx_modes = list(range(1, n_modes_rbm + 1)) + list(range(1 + n_modes_rbm + 6, 1 + n_modes_rbm + 6 + n_modes_flex))
        logging.info('Calculating GAFs via pulse excitation for %d rigid body modes and %d flexible modes...',
                     n_modes_rbm, n_modes_flex)
        # Load matrices
        PHIk_cfd = load_hdf5_sparse_matrix(self.model['PHIk_cfd'])
        PHIcfd_cg = self.model['mass'][self.trimcase['mass']]['PHIcfd_cg'][()]

        # Step 1: set-up frequency parameters, generate pulse signal, and init storage
        n_freqs = int(self.simcase['pulse_para']['fmax'] / self.simcase['pulse_para']['df'])
        if n_freqs % 2 != 0:  # n_freq is odd
            n_freqs += 1  # make even
        # Calculate all parameters from the number of freqs
        fmax = n_freqs * self.simcase['pulse_para']['df']
        dt = 1.0 / fmax
        t_final = 1.0 / self.simcase['pulse_para']['df']
        # Update simcase for time domain simulation
        self.simcase['dt'] = dt
        self.simcase['t_final'] = t_final
        # Whole frequency space including negative frequencies
        fftfreqs = fftfreq(n_freqs, dt)
        # Positive only frequencies where we need to calculate the TFs and excitations
        positiv_fftfreqs = np.abs(fftfreqs[:n_freqs // 2 + 1])
        # Only reduced frequencies < 3.0 are of interest
        k = f2k(positiv_fftfreqs)
        idx_k = np.where(k < 3.0)[0]
        k_red = k[idx_k]
        # Generate small-amplitude pulse signal
        t, unit_pulse = polynomial_pulse(dt, t_final, eps=1.0)
        # Scale the unit pulse for each mode such that the aplitudes are small.
        # Right now the scaling is hard-codes based on test with the DC3, but might need to be adjusted
        # for different configurations. On the other hand, I'm no fan of too many user-defined parameters...
        pulse_factor = [1e-3, 1e-3, 1e-4, 1e-4, 1e-4] + [1e-3] * n_modes_flex
        # The pulse's sign is used to align the aicraft rigid body motion with the nastran coordinate
        # system (compatibility with DLM-based solutions).
        pulse_sign = [1.0, -1.0, -1.0, 1.0, -1.0] + [1.0] * n_modes_flex
        pulse_signal = [unit_pulse * factor for factor in pulse_factor]
        pulse_signal = np.array(pulse_signal)
        gust_f = fft(pulse_signal)
        # Init storage for CFD forces
        # To avoid an excessive amount of data, e.g. during unsteady cfd simulations,
        # keep only the response data on the first mpi process (id = 0).
        if self.myid == 0:
            n_cfd = len(self.response['Pcfd'].squeeze())
            Pcfd_ref = np.zeros((n_cfd, len(t)))
            Pcfd_pulse = np.zeros((n_cfd, len(t)))
            Pcfd_gust = np.zeros((n_cfd, len(t)))
            Pb_pulse = np.zeros((6, n_modes, len(t)))
            Qhk = np.zeros((self.model['aerogrid']['n'][()] * 6, n_modes, len(k_red)), dtype=complex)
            Qhh = np.zeros((n_modes, n_modes, len(k_red)), dtype=complex)
            Qh_gust = np.zeros((n_modes, len(k_red)), dtype=complex)

        # Step 2: Run reference simulation without pulse
        # Select CFD solution sequence and initialize
        equations = CfdUnsteady(self, X0)
        logging.info('Running reference time simulation for %g sec...', t_final)
        # Loop over time steps
        for i_step, t_step in enumerate(t):
            X = copy.deepcopy(X0)
            output_dict = equations.eval_equations(X, t_step, modus='sim_full_output')
            if self.myid == 0:
                Pcfd_ref[:, i_step] = output_dict['Pcfd']
        equations.finalize()

        # Step 3a: Run pulse simulations for all modes
        for i_mode, idx_mode in zip(range(n_modes), idx_modes):
            # Re-initialze CFD solution sequence for each mode
            equations = CfdUnsteady(self, X0)
            logging.info('Running small-amplitude pulse simulation for mode %d for %g sec...', i_mode + 2, t_final)
            # Loop over time steps
            for i_step, t_step in enumerate(t):
                X = copy.deepcopy(X0)
                X[idx_mode] += pulse_signal[i_mode, i_step] * pulse_sign[i_mode]
                output_dict = equations.eval_equations(X, t_step, modus='sim_full_output')
                if self.myid == 0:
                    Pcfd_pulse[:, i_step] = output_dict['Pcfd']
            equations.finalize()

            # Step 3b: Calculate TF for current mode
            logging.info('Calculating transfer functions...')
            if self.myid == 0:
                # Compensate for initial condition and drift over time, transfer to aero grid 'k'
                Pcfd = Pcfd_pulse - Pcfd_ref
                Pk = PHIk_cfd.T.dot(Pcfd)
                # Calculate transfer functions
                Pk_f = fft(Pk, axis=1)
                TF = Pk_f / (gust_f[i_mode, :])
                # Store
                Qhk[:, i_mode, :] = TF[:, idx_k]
                Pb_pulse[:, i_mode, :] = np.dot(PHIcfd_cg.T, Pcfd)

        # Step 4a: Run pulse simulation for gust mode in z-direction (orientation = 0 degrees)
        # Set-up small-amplitude 1-cosine gust with amplitude of 0.003 * Vtas
        WG_TAS = 3e-3
        # Add a lead time so that the initialization of the gust happens ahead of the aircraft.
        # This avoids a jump / wiggle in the first time steps of the CFD solution.
        T1 = 0.1  # seconds
        # Select a short gust gradient (shorter than the 9-107m prescribed in CS-25.341)
        half_length = 4.0  # meters
        t, gust_signal = one_m_cosine_pulse(dt, t_final, Vtas, eps=WG_TAS * Vtas, half_length=half_length, T1=T1)
        gust_f = fft(gust_signal)
        self.simcase['gust'] = True
        self.simcase['gust_orientation'] = 0
        self.simcase['gust_gradient'] = half_length
        self.simcase['WG_TAS'] = WG_TAS
        self.simcase['gust_para'] = {}
        self.simcase['gust_para']['T1'] = T1
        # Select CFD solution sequence and initialize
        equations = CfdUnsteady(self, X0)
        logging.info('Running small-amplitude gust simulation for %g sec...', t_final)
        # Loop over time steps
        for i_step, t_step in enumerate(t):
            X = copy.deepcopy(X0)
            output_dict = equations.eval_equations(X, t_step, modus='sim_full_output')
            if self.myid == 0:
                Pcfd_gust[:, i_step] = output_dict['Pcfd']
        equations.finalize()

        # Step 4b: Calculate TF for gust mode
        logging.info('Calculating transfer functions...')
        if self.myid == 0:
            # Compensate for initial condition and drift over time, transfer to aero grid 'k'
            Pcfd = Pcfd_gust - Pcfd_ref
            Pk = PHIk_cfd.T.dot(Pcfd)
            # Calculate transfer functions
            Pk_f = fft(Pk, axis=1)
            TF = Pk_f / gust_f
            # Store
            Qk_gust = TF[:, idx_k]
            Pb_gust = np.dot(PHIcfd_cg.T, Pcfd)

        if self.myid == 0:
            # Because the CFD-based GAFs are calculated on the VLM/DLM aerogrid 'k',
            # project also the initial trim solution on the k-set.
            self.response['Pk_aero'] = PHIk_cfd.T.dot(self.response['Pcfd'].squeeze())
            # Apply modal transformation per frequency k_red to obtain Qhh
            for i, _ in enumerate(k_red):
                Qhh[:, :, i] = PHIkh.T.dot(Qhk[:, :, i])
                Qh_gust[:, i] = PHIkh.T.dot(Qk_gust[:, i])
            # Store results in response dictionary
            self.response['desc'] = self.trimcase['desc']
            self.response['k_red'] = k_red
            self.response['Qhk'] = Qhk
            self.response['Qhh'] = Qhh
            self.response['Qk_gust'] = Qk_gust
            # The time signals are only saved for plotting / plausibility checking
            self.response['pulse_signal'] = pulse_signal
            self.response['gust_signal'] = gust_signal
            self.response['t_pulse'] = t
            self.response['Pb_pulse'] = Pb_pulse
            self.response['Pb_gust'] = Pb_gust

        self.successful = True
