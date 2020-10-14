import numpy as np
from scipy import linalg
import logging, copy
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import itertools

from loadskernel.units import * 
from loadskernel.model_equations.frequency_domain import PKMethod



class StateSpaceAnalysis(PKMethod):      
        
    def eval_equations(self):
        self.setup_frequence_parameters()
        
        logging.info('building systems') 
        self.build_AICs()
        states = ['y', 'z', 'phi', 'alpha', 'beta',]
        for i_mode in range(1, self.n_modes_f+1):
            states += ['Uf'+str(i_mode)]
        states += ['v', 'w', 'p', 'q', 'r']
        for i_mode in range(1, self.n_modes_f+1):
            states += ['dUf_dt'+str(i_mode)]


        eigenvalue, eigenvector = linalg.eig(self.system(self.Vvec[0]))
        
        bandbreite = eigenvalue.__abs__().max() - eigenvalue.__abs__().min()
        idx_pos = np.where(np.logical_and(eigenvalue.__abs__() / bandbreite >= 1e-6, eigenvalue.imag >= 0.0))[0]  # no zero eigenvalues
        idx_sort = np.argsort(np.abs(eigenvalue.imag[idx_pos]))  # sort result by eigenvalue
        eigenvalues0 = eigenvalue[idx_pos][idx_sort]
        eigenvectors0 = eigenvector[:, idx_pos][:, idx_sort]
        
        eigenvalues = []; eigenvectors = []; freqs = []; damping = []; Vtas = []
        eigenvectors_old = copy.deepcopy(eigenvectors0)
        # loop over Vtas
        for i_V in range(len(self.Vvec)):
            Vtas_i = self.Vvec[i_V]
            eigenvalues_i, eigenvectors_i = self.calc_eigenvalues(self.system(Vtas_i), eigenvectors_old)
            
            # store 
            eigenvalues.append(eigenvalues_i)
            eigenvectors.append(eigenvectors_i)
            freqs.append(eigenvalues_i.imag /2.0/np.pi)
            damping.append(2.0 * eigenvalues_i.real / eigenvalues_i.imag)
            Vtas.append([Vtas_i]*len(eigenvalues_i))
            
            eigenvectors_old = eigenvectors_i
           
        response = {'eigenvalues':np.array(eigenvalues),
                    'eigenvectors':np.array(eigenvectors),
                    'freqs':np.array(freqs),
                    'damping':np.array(damping),
                    'Vtas':np.array(Vtas),
                    'states': states,
                   }
        return response 
        
#         colors = itertools.cycle(( plt.cm.tab20c(np.linspace(0, 1, 20)) ))
#         markers = itertools.cycle(('+', 'o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'x', 'D',))
#         desc = [str(mode) for mode in range(n_eigenvalues)]
#         label_states = ['y', 'z', 'phi', 'theta', 'psi', 'Uf1', 'Uf2', 'Uf3', 'Uf4',
#                         'v', 'w', 'p', 'q', 'r', 'dUf1', 'dUf2','dUf3', 'dUf4']
#         
# 
#         fig4 = plt.figure()
#         ax4 = fig4.add_axes([0.15, 0.15, 0.5, 0.75]) # List is [left, bottom, width, height]
#         
#         fig5, ax5 = plt.subplots()
#         im = ax5.imshow(eigenvector.__abs__(), cmap='hot_r', aspect='auto', origin='upper', vmin=0.0, vmax=1.0)
#         
#         for j in range(n_eigenvalues): 
#             marker = next(markers)
#             color = next(colors)
#             ax4.plot(eigenvalue[j].real, eigenvalue[j].imag, marker=marker, c=color, linewidth=2.0, label=desc[j])
#             ax5.plot(j,label_states.__len__(), marker=marker, c=color,)
#         
#         ax4.set_xlabel('real')
#         ax4.set_ylabel('imag')
#         
#         for ax in [ax4]:
#             yax = ax.get_yaxis()
#             yax.set_label_coords(x=-0.18, y=0.5)
#             ax.grid(b=True, which='both', axis='both')
#             #ax3.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
#             #ax3.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
#             lgd = ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., ncol=2, fontsize=10)
# 
#         ax5.yaxis.set_ticks(np.arange(0,label_states.__len__(),1))
#         ax5.yaxis.set_ticklabels(label_states)
#         ax5.yaxis.set_tick_params(rotation=0)
#         ax5.xaxis.set_ticks(np.arange(0,n_eigenvalues,1))
#         ax5.xaxis.set_ticklabels(np.arange(0,n_eigenvalues,1))
#         
#         ax_divider = make_axes_locatable(ax5)
#         cax = ax_divider.append_axes("top", size="7%", pad="1%")
#         cb = fig5.colorbar(im, cax=cax, orientation="horizontal")
#         # change tick position to top. Tick position defaults to bottom and overlaps the image.
#         cax.xaxis.set_ticks_position("top")
#         plt.show()
#         
#         
#         response = {'freqs':np.array(freqs).T,
#                     'damping':np.array(damping).T,
#                     'Vtas':np.array(Vtas).T,
#                    }
#         return response    
            

    
    def calc_Qhh_1(self, Qjj):
        return self.PHIlh.T.dot(self.model.aerogrid['Nmat'].T.dot(self.model.aerogrid['Amat'].dot(Qjj).dot(self.Djh_1)))
    
    def calc_Qhh_2(self, Qjj):
        return self.PHIlh.T.dot(self.model.aerogrid['Nmat'].T.dot(self.model.aerogrid['Amat'].dot(Qjj).dot(self.Djh_2)))
    
    def build_AICs(self):
        # do some pre-multiplications first, then the interpolation
        if self.jcl.aero['method'] in ['mona_steady']:
            self.Qhh_1 = self.calc_Qhh_1(self.model.aero['Qjj'][self.i_aero])
            self.Qhh_2 = self.calc_Qhh_2(self.model.aero['Qjj'][self.i_aero])
#         elif self.jcl.aero['method'] in ['mona_unsteady']:
#             ABCD = self.model.aero['ABCD'][self.i_aero]
#             for k_red in self.model.aero['k_red']:
#                 D = np.zeros((self.model.aerogrid['n'], self.model.aerogrid['n']), dtype='complex')
#                 j = 1j # imaginary number
#                 for i_pole, beta in zip(np.arange(0,self.model.aero['n_poles']), self.model.aero['betas']):                
#                     D += ABCD[3+i_pole,:,:] * j*k_red / (j*k_red + beta)
#                 Qjj_unsteady = ABCD[0,:,:] + ABCD[1,:,:]*j*k_red + ABCD[2,:,:]*(j*k_red)**2 + D
#                 Qhh_1.append(self.calc_Qhh_1(Qjj_unsteady))
#                 Qhh_2.append(self.calc_Qhh_2(Qjj_unsteady))
    
    def system(self, Vtas):
        rho = self.model.atmo['rho'][self.i_atmo]
        Mhh_inv = np.linalg.inv(self.Mhh)
        
        upper_part = np.concatenate((np.zeros((self.n_modes, self.n_modes), dtype='float'), np.eye(self.n_modes, dtype='float')), axis=1)
        lower_part = np.concatenate(( -Mhh_inv.dot(self.Khh - rho/2.0*Vtas**2.0*self.Qhh_1), -Mhh_inv.dot(self.Dhh - rho/2.0*Vtas*self.Qhh_2)), axis=1)
        A = np.concatenate((upper_part, lower_part))
        
        return A 