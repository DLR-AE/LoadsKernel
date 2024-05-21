import numpy as np

from loadskernel.equations.steady import Steady
from loadskernel.solution_tools import gravitation_on_earth


class CfdSteady(Steady):

    def equations(self, X, t, modus):
        self.counter += 1
        # recover states
        Tgeo2body, Tbody2geo = self.geo2body(X)
        dUcg_dt, Uf, dUf_dt = self.recover_states(X)
        Vtas, q_dyn = self.recover_Vtas(X)
        alpha, beta, gamma = self.windsensor(X, Vtas, Uf, dUf_dt)
        Ux2 = self.get_Ux2(X)
        delta_XYZ = np.array([0.0, 0.0, 0.0])
        PhiThetaPsi = X[self.solution.idx_states[3:6]]
        # aerodynamics
        self.cfd_interface.update_general_para()
        self.cfd_interface.init_solver()
        self.cfd_interface.set_grid_velocities(X[6:12])
        self.cfd_interface.set_euler_transformation(delta_XYZ, PhiThetaPsi)
        self.cfd_interface.prepare_meshdefo(Uf, Ux2)
        self.cfd_interface.run_solver()
        Pcfd = self.cfd_interface.get_last_solution()

        Pk_rbm = np.zeros(6 * self.aerogrid['n'])
        Pk_cam = Pk_rbm * 0.0
        Pk_cs = Pk_rbm * 0.0
        Pk_f = Pk_rbm * 0.0
        Pk_gust = Pk_rbm * 0.0
        Pk_idrag = Pk_rbm * 0.0
        Pk_unsteady = Pk_rbm * 0.0

        Pextra, Pb_ext, Pf_ext = self.engine(X, Vtas, q_dyn, Uf, dUf_dt, t)

        # correction coefficients
        Pb_corr = self.correctioon_coefficients(alpha, beta, q_dyn)

        # summation of forces
        Pk_aero = Pk_rbm + Pk_cam + Pk_cs + Pk_f + Pk_gust + Pk_idrag + Pk_unsteady
        Pmac = np.dot(self.Dkx1.T, Pk_aero)
        Pb = np.dot(self.PHImac_cg.T, Pmac) + Pb_corr + Pb_ext + np.dot(self.PHIcfd_cg.T, Pcfd)

        g_cg = gravitation_on_earth(self.PHInorm_cg, Tgeo2body)

        # EoM
        d2Ucg_dt2, Nxyz = self.rigid_EoM(dUcg_dt, Pb, g_cg, modus)
        Pf = np.dot(self.PHIkf.T, Pk_aero) + self.Mfcg.dot(np.hstack((d2Ucg_dt2[0:3] - g_cg, d2Ucg_dt2[3:6]))) \
            + Pf_ext + np.dot(self.PHIcfd_f.T, Pcfd)
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
        dy = np.hstack((np.dot(Tbody2geo, X[6:12]),
                        np.dot(self.PHIcg_norm, d2Ucg_dt2),
                        dUf_dt,
                        d2Uf_dt2,
                        dcommand,
                        ))
        if modus in ['trim']:
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
                        'Pmac': self.PHIcg_mac.T.dot(self.PHIcfd_cg.T).dot(Pcfd),
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
                        'Pcfd': Pcfd,
                        'dy': dy,
                        }
            return response

    def finalize(self):
        self.cfd_interface.release_memory()


class CfdUnsteady(CfdSteady):

    def ode_arg_sorter(self, t, X):
        return self.eval_equations(X, t, 'sim_full_output')

    def equations(self, X, t, modus):
        self.counter += 1
        # recover states
        Tgeo2body, Tbody2geo = self.geo2body(X)
        dUcg_dt, Uf, dUf_dt = self.recover_states(X)
        Vtas, q_dyn = self.recover_Vtas(X)
        alpha, beta, gamma = self.windsensor(X, Vtas, Uf, dUf_dt)
        Ux2 = self.get_Ux2(X)
        delta_XYZ = X[self.solution.idx_states[0:3]] - self.X0[self.solution.idx_states[0:3]]
        PhiThetaPsi = X[self.solution.idx_states[3:6]]

        # aerodynamics
        self.cfd_interface.update_general_para()
        self.cfd_interface.update_timedom_para()
        if self.simcase['gust']:
            self.cfd_interface.update_gust_para(Vtas, self.WG_TAS * Vtas)
        self.cfd_interface.init_solver()
        self.cfd_interface.set_euler_transformation(delta_XYZ, PhiThetaPsi)
        self.cfd_interface.prepare_meshdefo(Uf, Ux2)
        # Remember to start SU2 at time step 2, because steps 0 and 1 are taken up by the steady restart solution.
        # To establish the current time step, we can reuse the existing counter.
        self.cfd_interface.run_solver(i_timestep=self.counter + 1)
        Pcfd = self.cfd_interface.get_last_solution()

        Pk_rbm = np.zeros(6 * self.aerogrid['n'])
        Pk_cam = Pk_rbm * 0.0
        Pk_cs = Pk_rbm * 0.0
        Pk_f = Pk_rbm * 0.0
        Pk_gust = Pk_rbm * 0.0
        Pk_idrag = Pk_rbm * 0.0
        Pk_unsteady = Pk_rbm * 0.0

        Pextra, Pb_ext, Pf_ext = self.engine(X, Vtas, q_dyn, Uf, dUf_dt, t)

        # correction coefficients
        Pb_corr = self.correctioon_coefficients(alpha, beta, q_dyn)

        # summation of forces
        Pk_aero = Pk_rbm + Pk_cam + Pk_cs + Pk_f + Pk_gust + Pk_idrag + Pk_unsteady
        Pmac = np.dot(self.Dkx1.T, Pk_aero)
        Pb = np.dot(self.PHImac_cg.T, Pmac) + Pb_corr + Pb_ext + np.dot(self.PHIcfd_cg.T, Pcfd)

        g_cg = gravitation_on_earth(self.PHInorm_cg, Tgeo2body)

        # EoM
        d2Ucg_dt2, Nxyz = self.rigid_EoM(dUcg_dt, Pb, g_cg, modus)
        Pf = np.dot(self.PHIkf.T, Pk_aero) + self.Mfcg.dot(np.hstack((d2Ucg_dt2[0:3] - g_cg, d2Ucg_dt2[3:6]))) \
            + Pf_ext + np.dot(self.PHIcfd_f.T, Pcfd)
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
        dy = np.hstack((np.dot(Tbody2geo, X[6:12]),
                        np.dot(self.PHIcg_norm, d2Ucg_dt2),
                        dUf_dt,
                        d2Uf_dt2,
                        dcommand,
                        ))

        if modus in ['sim']:
            return Y

        elif modus in ['sim_full_output']:
            # For time domain simulations, typically not all results are required. To reduce the amount of data while
            # maintaining compatibility with the trim, empty arrays are used.
            response = {'X': X,
                        'Y': Y,
                        't': np.array([t]),
                        'Pk_aero': Pk_aero,
                        'q_dyn': np.array([q_dyn]),
                        'Pb': Pb,
                        'Pmac': self.PHIcg_mac.T.dot(self.PHIcfd_cg.T).dot(Pcfd),
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
                        'Pcfd': Pcfd,
                        'dy': dy,
                        }
            return response
