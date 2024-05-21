import logging
import numpy as np
from scipy import linalg

from loadskernel.equations.common import Common, ConvergenceError
from loadskernel.solution_tools import gravitation_on_earth


class Steady(Common):

    def equations(self, X, t, modus):
        self.counter += 1
        # recover states
        Tgeo2body, Tbody2geo = self.geo2body(X)
        dUcg_dt, Uf, dUf_dt = self.recover_states(X)
        Vtas, q_dyn = self.recover_Vtas(X)
        onflow = self.recover_onflow(X)
        alpha, beta, gamma = self.windsensor(X, Vtas, Uf, dUf_dt)
        Ux2 = self.get_Ux2(X)

        # aerodynamics
        Pk_rbm, wj_rbm = self.rbm(onflow, q_dyn, Vtas)
        Pk_cam, wj_cam = self.camber_twist(q_dyn)
        Pk_cs, wj_cs = self.cs(X, Ux2, q_dyn)
        Pk_f, wj_f = self.flexible(Uf, dUf_dt, onflow, q_dyn, Vtas)
        Pk_gust, wj_gust = self.gust(X, q_dyn)

        wj = wj_rbm + wj_cam + wj_cs + wj_f + wj_gust
        Pk_idrag = self.idrag(wj, q_dyn)
        Pk_unsteady = Pk_rbm * 0.0

        Pextra, Pb_ext, Pf_ext = self.engine(X, Vtas, q_dyn, Uf, dUf_dt, t)

        # correction coefficients
        Pb_corr = self.correctioon_coefficients(alpha, beta, q_dyn)

        # summation of forces
        Pk_aero = Pk_rbm + Pk_cam + Pk_cs + Pk_f + Pk_gust + Pk_idrag + Pk_unsteady
        Pmac = np.dot(self.Dkx1.T, Pk_aero)
        Pb = np.dot(self.PHImac_cg.T, Pmac) + Pb_corr + Pb_ext

        g_cg = gravitation_on_earth(self.PHInorm_cg, Tgeo2body)

        # EoM
        d2Ucg_dt2, Nxyz = self.rigid_EoM(dUcg_dt, Pb, g_cg, modus)
        Pf = np.dot(self.PHIkf.T, Pk_aero) + self.Mfcg.dot(np.hstack((d2Ucg_dt2[0:3] - g_cg, d2Ucg_dt2[3:6]))) + Pf_ext
        d2Uf_dt2 = self.flexible_EoM(dUf_dt, Uf, Pf)

        # CS derivatives
        dcommand = self.get_command_derivatives(t, X, Vtas, gamma, alpha, beta, Nxyz, np.dot(Tbody2geo, X[6:12])[0:3])

        # output
        Y = np.hstack((np.dot(Tbody2geo, X[6:12]),
                       np.dot(self.PHIcg_norm, d2Ucg_dt2),
                       dUf_dt,
                       d2Uf_dt2,
                       dcommand,
                       Nxyz[2],
                       Vtas,
                       beta,
                       ))

        if modus in ['trim', 'sim']:
            return Y
        elif modus in ['trim_full_output']:
            response = {'X': X,
                        'Y': Y,
                        't': np.array([t]),
                        'Pk_rbm': Pk_rbm,
                        'Pk_cam': Pk_cam,
                        'Pk_aero': Pk_aero,
                        'Pk_cs': Pk_cs,
                        'Pk_f': Pk_f,
                        'Pk_gust': Pk_gust,
                        'Pk_unsteady': Pk_unsteady,
                        'Pk_idrag': Pk_idrag,
                        'q_dyn': np.array([q_dyn]),
                        'Pb': Pb,
                        'Pmac': Pmac,
                        'Pf': Pf,
                        'alpha': np.array([alpha]),
                        'beta': np.array([beta]),
                        'Ux2': Ux2,
                        'dUcg_dt': dUcg_dt,
                        'd2Ucg_dt2': d2Ucg_dt2,
                        'Uf': Uf,
                        'dUf_dt': dUf_dt,
                        'd2Uf_dt2': d2Uf_dt2,
                        'Nxyz': Nxyz,
                        'g_cg': g_cg,
                        'Pextra': Pextra,
                        }
            return response
        elif modus in ['sim_full_output']:
            # For time domain simulations, typically not all results are required. To reduce the amount of data while
            # maintaining compatibility with the trim, empty arrays are used.
            response = {'X': X,
                        'Y': Y,
                        't': np.array([t]),
                        'Pk_aero': Pk_aero,
                        'q_dyn': np.array([q_dyn]),
                        'Pb': Pb,
                        'Pmac': Pmac,
                        'alpha': np.array([alpha]),
                        'beta': np.array([beta]),
                        'Ux2': Ux2,
                        'dUcg_dt': dUcg_dt,
                        'd2Ucg_dt2': d2Ucg_dt2,
                        'Uf': Uf,
                        'dUf_dt': dUf_dt,
                        'd2Uf_dt2': d2Uf_dt2,
                        'Nxyz': Nxyz,
                        'g_cg': g_cg,
                        'Pextra': Pextra,
                        }
            return response

    def eval_equations(self, X_free, time, modus='trim_full_output'):
        # this is a wrapper for the model equations 'eqn_basic'
        if modus in ['trim', 'trim_full_output']:
            # get inputs from trimcond and apply inputs from fsolve
            X = np.array(self.trimcond_X[:, 2], dtype='float')
            X[np.where((self.trimcond_X[:, 1] == 'free'))[0]] = X_free
        elif modus in ['sim', 'sim_full_output']:
            X = X_free

        # evaluate model equations
        if modus == 'trim':
            Y = self.equations(X, time, 'trim')
            # get the current values from Y and substract tamlab.figure()
            # fsolve only finds the roots; Y = 0
            Y_target_ist = Y[np.where((self.trimcond_Y[:, 1] == 'target'))[0]]
            Y_target_soll = np.array(self.trimcond_Y[:, 2], dtype='float')[np.where((self.trimcond_Y[:, 1] == 'target'))[0]]
            out = Y_target_ist - Y_target_soll
            return out

        elif modus == 'sim':
            Y = self.equations(X, time, 'sim')
            return Y[self.solution.idx_state_derivatives + self.solution.idx_input_derivatives]

        elif modus == 'sim_full_output':
            response = self.equations(X, time, 'sim_full_output')
            return response

        elif modus == 'trim_full_output':
            response = self.equations(X, time, 'trim_full_output')
            return response

    def eval_equations_iteratively(self, X_free, time, modus='trim_full_output'):
        # this is a wrapper for the model equations

        # get inputs from trimcond and apply inputs from fsolve
        X = np.array(self.trimcond_X[:, 2], dtype='float')
        X[np.where((self.trimcond_X[:, 1] == 'free'))[0]] = X_free
        logging.debug('X_free: {}'.format(X_free))
        converged = False
        inner_loops = 0
        while not converged:
            inner_loops += 1
            response = self.equations(X, time, 'trim_full_output')
            logging.info('Inner iteration {}, calculate structural deformations.'.format(self.counter))
            Uf_new = linalg.solve(self.Kff, response['Pf'])

            # Add a relaxation factor between each loop to reduce overshoot and/or oscillations.
            # In case the solution hasn't converged in a reasonable time (say 8 loops), increase the relaxation.
            # A low relaxation factor is slower but more robust, f_relax = 1.0 implies no relaxation.
            # After 16 inner loops, decrease the convergence criterion epsilon.
            if inner_loops < 8:
                f_relax = 0.8
            else:
                f_relax = 0.5
            if inner_loops < 16:
                epsilon = self.jcl.general['b_ref'] * 1.0e-5
            else:
                epsilon = self.jcl.general['b_ref'] * 1.0e-4
            logging.info(' - Relaxation factor: {}'.format(f_relax))
            logging.info(' - Epsilon: {:0.6g}'.format(epsilon))

            # recover Uf_old from last step and blend with Uf_now
            Uf_old = [self.trimcond_X[np.where((self.trimcond_X[:, 0] == 'Uf' + str(i_mode)))[0][0], 2] for i_mode in
                      range(1, self.n_modes + 1)]
            Uf_old = np.array(Uf_old, dtype='float')
            Uf_new = Uf_new * f_relax + Uf_old * (1.0 - f_relax)

            # set new values for Uf in trimcond for next loop and store in response
            for i_mode in range(self.n_modes):
                self.trimcond_X[np.where((self.trimcond_X[:, 0] == 'Uf' + str(i_mode + 1)))[0][0], 2] = '{:g}'.format(
                    Uf_new[i_mode])
                response['X'][12 + i_mode] = Uf_new[i_mode]

            # convergence parameter for iterative evaluation
            Ug_f_body = np.dot(self.PHIf_strc.T, Uf_new.T).T
            defo_new = Ug_f_body[self.strcgrid['set'][:, :3]].max()  # Groesste Verformung, meistens Fluegelspitze
            ddefo = defo_new - self.defo_old
            self.defo_old = np.copy(defo_new)
            if np.abs(ddefo) < epsilon:
                converged = True
                logging.info(' - Max. deformation: {:0.6g}, delta: {:0.6g} smaller than epsilon, converged.'.format(
                    defo_new, ddefo))
            else:
                logging.info(' - Max. deformation: {:0.6g}, delta: {:0.6g}'.format(defo_new, ddefo))
            if inner_loops > 20:
                raise ConvergenceError('No convergence of structural deformation achieved after {} inner loops. \
                    Check convergence of CFD solution and/or convergence criterion "delta".'.format(inner_loops))
        # get the current values from Y and substract tamlab.figure()
        # fsolve only finds the roots; Y = 0
        Y_target_ist = response['Y'][np.where((self.trimcond_Y[:, 1] == 'target'))[0]]
        Y_target_soll = np.array(self.trimcond_Y[:, 2], dtype='float')[np.where((self.trimcond_Y[:, 1] == 'target'))[0]]
        out = Y_target_ist - Y_target_soll

        if modus in ['trim']:
            return out
        elif modus == 'trim_full_output':
            return response
