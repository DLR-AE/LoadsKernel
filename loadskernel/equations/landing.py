import numpy as np
from loadskernel.equations.common import Common
from loadskernel.solution_tools import gravitation_on_earth


class Landing(Common):

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

        # correction coefficients
        Pb_corr = self.correctioon_coefficients(alpha, beta, q_dyn)

        # landing gear
        Pextra, _, dp2, ddp2, F1, F2 = self.landinggear(X, Tbody2geo)

        # summation of forces
        Pk_aero = Pk_rbm + Pk_cam + Pk_cs + Pk_f + Pk_gust + Pk_idrag + Pk_unsteady
        Pmac = np.dot(self.Dkx1.T, Pk_aero)
        Pb = np.dot(self.PHImac_cg.T, Pmac) + Pb_corr + np.dot(self.PHIextra_cg.T, Pextra)

        g_cg = gravitation_on_earth(self.PHInorm_cg, Tgeo2body)

        # EoM
        d2Ucg_dt2, Nxyz = self.rigid_EoM(dUcg_dt, Pb, g_cg, modus)
        Pf = np.dot(self.PHIkf.T, Pk_aero) + self.Mfcg.dot(
            np.hstack((d2Ucg_dt2[0:3] - g_cg, d2Ucg_dt2[3:6]))) + np.dot(self.PHIf_extra, Pextra)  # viel schneller!
        d2Uf_dt2 = self.flexible_EoM(dUf_dt, Uf, Pf)

        # CS derivatives
        dcommand = self.get_command_derivatives(t, X, Vtas, gamma, alpha, beta, Nxyz, np.dot(Tbody2geo, X[6:12])[0:3])

        # output
        if modus in ['trim', 'trim_full_output']:
            Y = np.hstack((np.dot(Tbody2geo, X[6:12]),
                           np.dot(self.PHIcg_norm, d2Ucg_dt2),
                           dUf_dt,
                           d2Uf_dt2,
                           dcommand,
                           Nxyz[2],
                           Vtas,
                           beta,
                           ))
        elif modus in ['sim', 'sim_full_output']:
            Y = np.hstack((np.dot(Tbody2geo, X[6:12]),
                           np.dot(self.PHIcg_norm, d2Ucg_dt2),
                           dUf_dt,
                           d2Uf_dt2,
                           dcommand,
                           np.hstack((dp2, ddp2)),
                           Nxyz[2],
                           Vtas,
                           beta,
                           ))
        if modus in ['trim', 'sim']:
            return Y
        elif modus in ['trim_full_output', 'sim_full_output']:
            # calculate translations, velocities and accelerations of some additional points
            # (might also be used for sensors in a closed-loop system
            # position LG attachment point over ground
            p1 = -self.cggrid['offset'][:, 2] + self.extragrid['offset'][:, 2] \
                + self.PHIextra_cg.dot(np.dot(self.PHInorm_cg, X[0:6]))[self.extragrid['set'][:, 2]] \
                + self.PHIf_extra.T.dot(X[12:12 + self.n_modes])[self.extragrid['set'][:, 2]]
            # velocity LG attachment point
            dp1 = self.PHIextra_cg.dot(np.dot(self.PHInorm_cg, np.dot(Tbody2geo, X[6:12])))[self.extragrid['set'][:, 2]] \
                + self.PHIf_extra.T.dot(X[12 + self.n_modes:12 + self.n_modes * 2])[self.extragrid['set'][:, 2]]
            # acceleration LG attachment point
            ddp1 = self.PHIextra_cg.dot(np.dot(self.PHInorm_cg, np.dot(Tbody2geo, Y[6:12])))[self.extragrid['set'][:, 2]] \
                + self.PHIf_extra.T.dot(Y[12 + self.n_modes:12 + self.n_modes * 2])[self.extragrid['set'][:, 2]]

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
                        'p1': p1,
                        'dp1': dp1,
                        'ddp1': ddp1,
                        'F1': F1,
                        'F2': F2,
                        }
            return response

    def eval_equations(self, X_free, time, modus='trim_full_output'):

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
            return Y[self.solution.idx_state_derivatives + self.solution.idx_input_derivatives
                     + self.solution.idx_lg_derivatives]  # Nz ist eine Rechengroesse und keine Simulationsgroesse!

        elif modus == 'sim_full_output':
            response = self.equations(X, time, 'sim_full_output')
            return response

        elif modus == 'trim_full_output':
            response = self.equations(X, time, 'trim_full_output')
            return response
