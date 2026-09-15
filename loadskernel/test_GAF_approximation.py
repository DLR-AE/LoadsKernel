import numpy as np
import scipy as sp
from matplotlib import pyplot as plt

from loadskernel import build_aero_functions, bary_rational
from loadskernel.io_functions import data_handling

from maethods.systems.systems import gilbert_realization
from maethods.systems import tdaaa

model = data_handling.load_hdf5('/data/FFD_LoadsKernel/model_jcl_FFD_loop9_gust_subsonic_su2_freqdom.hdf5')
GAFs = model['GAFs']['M2']['M14']['FL200']
Vtas = 450.0
c_over_Vtas = (0.5 * model['macgrid']['c_ref'][()]) / Vtas

Qhh = np.moveaxis(GAFs['Qhh'], -1, 0)
Qhk = np.moveaxis(GAFs['Qhk'], -1, 0)
Qgusth = np.expand_dims(np.moveaxis(GAFs['Qgusth'], -1, 0), axis=-1)
k = GAFs['k_red'][()]
k_interp = np.arange(0.0, (k.max()), 0.001)

dZ = k.max()
dt = 1 / dZ

Q = Qhh[:, 1, 3]
# Q = Qgusth[:, 0, 0]

# r = bary_rational.AAA(x=1j*k, y=Q, rtol=1e-3)
# poles = r.poles()


# :param Y: Output data in frequnecy domain
# :param U: Input data in frequnecy domain
# :param Z: Sample frequnecies
# :return z: Support points
# :return y: Output support values
# :return u: Input support values
# :return w: barycentric weights
# :return err: total error history
# :return r: tranfer function handle
# :return pol: Poles of transfer function
z, y, u, w, err, r1, pol = tdaaa.AAA(Y=Q, U=np.ones_like(Q), Z=1j * k, tol=1e-3, mmax=100)

# desired_poles = -abs(pol.real) + 1j * pol.imag

# idx = np.argwhere(desired_poles.imag == 0)
# desired_poles = np.append(desired_poles[idx], desired_poles[np.setdiff1d(np.arange(0, len(desired_poles)), idx)])
# while np.any(abs(desired_poles.imag) > dZ / 2):
#     mask = abs(desired_poles.imag) > dZ / 2
#     desired_poles[mask] = desired_poles[mask] - 1j * np.sign(desired_poles[mask].imag) * (dZ / 2)
# kr = np.log(1e-6) / dt
# desired_poles[desired_poles.real < kr] = kr + 1j * desired_poles[desired_poles.real < kr].imag

# r2, z, y, u, w = tdaaa._shiftpol_tdaaa(z, y, u, desired_poles)

Ar, Br, Cr, Dr, Er = tdaaa.buildsys_tdaaa(z, w, y, u)
SSS = sp.signal.StateSpace(Ar.real, Br.real, Cr, Dr)

plt.figure()
# Plot the original data
plt.plot(Q.real, Q.imag, 'o', label='Original GAF Data')
plt.plot(r1(1j*k_interp).real, r1(1j*k_interp).imag, '-', label='AAA Approximation')
plt.plot(r1(z).real, r1(z).imag, 'sk', label='Support points')
# plt.plot(r2(1j*k_interp).real, r2(1j*k_interp).imag, '-', label='AAA Approximation, poles stabilized')

TF = SSS.to_tf()
w, H = sp.signal.freqresp(TF, k_interp)
plt.plot(H.real, H.imag, '--', label='Scipy SS/TF')

# Step response: amplitude 1.0 starting at t = 0.2 s over a total time of 10 s.
t_step = np.arange(0.0, 30.0, 1e-4)
u_step = np.where(t_step >= 2.0, 1e-3, 0.0)

t_out, y_step, x_step = sp.signal.lsim(SSS, U=u_step, T=t_step)

plt.figure()
plt.plot(t_out, y_step, label='Step response')
plt.grid()

plt.show()


# # Compute residues via formula for res of quotient of analytic functions
# with np.errstate(divide="ignore", invalid="ignore"):
#     N = (1 / (np.subtract.outer(desired_poles, y))) @ (y * w)
#     Ddiff = (-((1 / np.subtract.outer(desired_poles, y))**2) @ w)
#     residues = N / Ddiff
# residuals = np.expand_dims(np.expand_dims(residues, axis=0), axis=0)
# A, B, C, D = gilbert_realization(residuals=residuals, poles=desired_poles, tol_rank=1e-6)
# SSS = sp.signal.StateSpace(A.real, B.real, C, D)

# Q_exp = np.expand_dims(np.expand_dims(Q, axis=-1), axis=-1)
# ABCD, n_poles, betas, RMSE = build_aero_functions.rfa(Q_exp, k=k, poles=z)
# ABCD, n_poles, betas, RMSE = build_aero_functions.rfa(Q_exp, k=k, n_poles=6, poles=None)

rfa = build_aero_functions.RFArevisted(Y=Q, k=k, max_poles=10, rtol=1e-2)
rfa.perform_rfa_iteratively()
rfa.plot_approximation()

SS = rfa.to_ss()
SS = sp.signal.StateSpace(rfa.A, rfa.C.T, rfa.B.T, rfa.D.T)

residue = np.expand_dims(np.expand_dims(rfa.x, axis=0), axis=0)
A, B, C, D = gilbert_realization(residue, rfa.poles, tol_rank=1e-6)
SS = sp.signal.StateSpace(A, B, C, D)

TF = SS.to_tf()
w, H = sp.signal.freqresp(TF, k_interp)

plt.plot(H.real, H.imag, '--', label='Scipy SS/TF')



print('Done.')
