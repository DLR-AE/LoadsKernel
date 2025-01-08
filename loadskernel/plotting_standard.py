# -*- coding: utf-8 -*-
import logging
import itertools
import os
import numpy as np
from scipy.spatial import ConvexHull

from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

from loadskernel.units import tas2eas, eas2tas

plt.rcParams.update({'font.size': 16,
                     'svg.fonttype': 'none',
                     'savefig.dpi': 300, })


class LoadPlots():

    def __init__(self, jcl, model):
        self.jcl = jcl
        self.model = model
        self.responses = None
        self.monstations = None
        self.potatos_fz_mx = []  # Wing, HTP
        self.potatos_mx_my = []  # Wing, HTP, VTP
        self.potatos_fz_my = []
        self.potatos_fy_mx = []  # VTP
        self.potatos_mx_mz = []  # VTP
        self.potatos_my_mz = []  # FUS
        self.cuttingforces_wing = []
        self.cuttingforces_fuselage = []
        self.im = plt.imread(os.path.join(
            os.path.dirname(__file__), 'graphics', 'LK_logo2.png'))

        if hasattr(self.jcl, 'loadplots'):
            if 'potatos_fz_mx' in self.jcl.loadplots:
                self.potatos_fz_mx = self.jcl.loadplots['potatos_fz_mx']
            if 'potatos_mx_my' in self.jcl.loadplots:
                self.potatos_mx_my = self.jcl.loadplots['potatos_mx_my']
            if 'potatos_fz_my' in self.jcl.loadplots:
                self.potatos_fz_my = self.jcl.loadplots['potatos_fz_my']
            if 'potatos_fy_mx' in self.jcl.loadplots:
                self.potatos_fy_mx = self.jcl.loadplots['potatos_fy_mx']
            if 'potatos_mx_mz' in self.jcl.loadplots:
                self.potatos_mx_mz = self.jcl.loadplots['potatos_mx_mz']
            if 'potatos_my_mz' in self.jcl.loadplots:
                self.potatos_my_mz = self.jcl.loadplots['potatos_my_mz']
            if 'cuttingforces_wing' in self.jcl.loadplots:
                self.cuttingforces_wing = self.jcl.loadplots['cuttingforces_wing']
            if 'cuttingforces_fuselage' in self.jcl.loadplots:
                self.cuttingforces_fuselage = self.jcl.loadplots['cuttingforces_fuselage']
        else:
            logging.info('jcl.loadplots not specified in the JCL - no automatic plotting of load envelopes possible.')

    def add_responses(self, responses):
        self.responses = responses

    def add_monstations(self, monstations):
        self.monstations = monstations

    def create_axes(self, logo=True):
        fig = plt.figure()
        # List is [left, bottom, width, height]
        ax = fig.add_axes([0.2, 0.15, 0.7, 0.75])
        if logo:
            newax = fig.add_axes([0.04, 0.02, 0.10, 0.08])
            newax.imshow(self.im, interpolation='hanning', zorder=-2)
            newax.axis('off')
            newax.set_rasterization_zorder(-1)
        return ax

    def plot_monstations(self, filename_pdf):
        # launch plotting
        self.pp = PdfPages(filename_pdf)
        self.potato_plots()
        if self.cuttingforces_wing:
            self.cuttingforces_along_axis_plots(monstations=self.cuttingforces_wing, axis=1)
        if self.cuttingforces_fuselage:
            self.cuttingforces_along_axis_plots(monstations=self.cuttingforces_fuselage, axis=0)
        self.pp.close()
        logging.info('Plots saved as ' + filename_pdf)

    def potato_plot(self, station, desc, color, dof_xaxis, dof_yaxis, show_hull=True, show_labels=False, show_minmax=False):
        loads = np.array(self.monstations[station]['loads'])
        if isinstance(self.monstations[station]['subcases'], list):
            # This is an exception if source is not a hdf5 file.
            # For example, the monstations have been pre-processed by a merge script and are lists already.
            subcases = self.monstations[station]['subcases']
        else:
            # make sure this is a list of strings
            subcases = list(self.monstations[station]['subcases'].asstr()[:])

        points = np.vstack((loads[:, dof_xaxis], loads[:, dof_yaxis])).T
        self.subplot.scatter(points[:, 0], points[:, 1], color=color, label=desc, zorder=-2)  # plot points

        if show_hull and points.shape[0] >= 3:
            try:
                hull = ConvexHull(points)  # calculated convex hull from scattered points
                for simplex in hull.simplices:  # plot convex hull
                    self.subplot.plot(points[simplex, 0], points[simplex, 1], color=color, linewidth=2.0, linestyle='--')
                crit_trimcases = [subcases[i] for i in hull.vertices]
                if show_labels:
                    for i_case in range(crit_trimcases.__len__()):
                        self.subplot.text(points[hull.vertices[i_case], 0], points[hull.vertices[i_case], 1],
                                          str(subcases[hull.vertices[i_case]]), fontsize=8)
            except Exception:
                crit_trimcases = []

        elif show_minmax:
            pos_max_loads = np.argmax(points, 0)
            pos_min_loads = np.argmin(points, 0)
            pos_minmax_loads = np.concatenate((pos_min_loads, pos_max_loads))
            # plot points
            self.subplot.scatter(points[pos_minmax_loads, 0], points[pos_minmax_loads, 1], color=(1, 0, 0), zorder=-2)
            crit_trimcases = [subcases[i] for i in pos_minmax_loads]

        else:
            crit_trimcases = subcases[:]

        if show_labels:
            for crit_trimcase in crit_trimcases:
                pos = subcases.index(crit_trimcase)
                self.subplot.text(points[pos, 0], points[pos, 1], str(subcases[pos]), fontsize=8)

        self.crit_trimcases += crit_trimcases

    def potato_plot_nicely(self, station, desc, dof_xaxis, dof_yaxis, var_xaxis, var_yaxis):
        self.subplot.cla()
        self.potato_plot(station,
                         desc=station,
                         color='cornflowerblue',
                         dof_xaxis=dof_xaxis,
                         dof_yaxis=dof_yaxis,
                         show_hull=True,
                         show_labels=True,
                         show_minmax=False)

        self.subplot.legend(loc='best')
        self.subplot.ticklabel_format(style='sci', axis='x', scilimits=(-2, 2))
        self.subplot.ticklabel_format(style='sci', axis='y', scilimits=(-2, 2))
        self.subplot.grid(visible=True, which='major', axis='both')
        self.subplot.minorticks_on()
        yax = self.subplot.get_yaxis()
        yax.set_label_coords(x=-0.18, y=0.5)
        self.subplot.set_xlabel(var_xaxis)
        self.subplot.set_ylabel(var_yaxis)
        self.subplot.set_rasterization_zorder(-1)
        self.pp.savefig()

    def potato_plots(self):
        logging.info('Start potato-plotting...')
        self.subplot = self.create_axes()

        potato = np.sort(np.unique(self.potatos_fz_mx + self.potatos_mx_my + self.potatos_fz_my + self.potatos_fy_mx
                                   + self.potatos_mx_mz + self.potatos_my_mz))
        self.crit_trimcases = []
        for station in potato:
            if station in self.potatos_fz_mx:
                var_xaxis = 'Fz [N]'
                var_yaxis = 'Mx [Nm]'
                dof_xaxis = 2
                dof_yaxis = 3
                self.potato_plot_nicely(station, station, dof_xaxis, dof_yaxis, var_xaxis, var_yaxis)
            if station in self.potatos_mx_my:
                var_xaxis = 'Mx [N]'
                var_yaxis = 'My [Nm]'
                dof_xaxis = 3
                dof_yaxis = 4
                self.potato_plot_nicely(station, station, dof_xaxis, dof_yaxis, var_xaxis, var_yaxis)
            if station in self.potatos_fz_my:
                var_xaxis = 'Fz [N]'
                var_yaxis = 'My [Nm]'
                dof_xaxis = 2
                dof_yaxis = 4
                self.potato_plot_nicely(station, station, dof_xaxis, dof_yaxis, var_xaxis, var_yaxis)
            if station in self.potatos_fy_mx:
                var_xaxis = 'Fy [N]'
                var_yaxis = 'Mx [Nm]'
                dof_xaxis = 1
                dof_yaxis = 3
                self.potato_plot_nicely(station, station, dof_xaxis, dof_yaxis, var_xaxis, var_yaxis)
            if station in self.potatos_mx_mz:
                var_xaxis = 'Mx [Nm]'
                var_yaxis = 'Mz [Nm]'
                dof_xaxis = 3
                dof_yaxis = 5
                self.potato_plot_nicely(station, station, dof_xaxis, dof_yaxis, var_xaxis, var_yaxis)
            if station in self.potatos_my_mz:
                var_xaxis = 'My [Nm]'
                var_yaxis = 'Mz [Nm]'
                dof_xaxis = 4
                dof_yaxis = 5
                self.potato_plot_nicely(station, station, dof_xaxis, dof_yaxis, var_xaxis, var_yaxis)
        plt.close()

    def cuttingforces_along_axis_plots(self, monstations, axis):
        assert axis in [0, 1, 2], 'Plotting along an axis only supported for axis 0, 1 or 2!'
        logging.info('Start plotting cutting forces along axis {}...'.format(axis))
        # Read the data required for plotting.
        loads = []
        offsets = []
        subcases = []
        for station in monstations:
            # trigger to read the data now with [:]
            loads.append(list(self.monstations[station]['loads'][:]))
            offsets.append(list(self.monstations[station]['offset'][:]))
            # Get subcase description at monitoring station
            if isinstance(self.monstations[station]['subcases'], list):
                # This is an exception if source is not a hdf5 file.
                # For example, the monstations have been pre-processed by a merge script and are lists already.
                subcases.append(self.monstations[station]['subcases'])
            else:
                # make sure this is a list of strings
                subcases.append(list(self.monstations[station]['subcases'].asstr()[:]))
        loads = np.array(loads)
        offsets = np.array(offsets)

        # Loop over all six components of the cutting loads.
        label_loads = ['Fx [N]', 'Fy [N]', 'Fz [N]', 'Mx [Nm]', 'My [Nm]', 'Mz [Nm]']
        subplot = self.create_axes()
        for idx_load in range(6):
            # Recycle the existing subplot.
            subplot.cla()

            # Plot loads.
            if loads.shape[1] > 50:
                logging.debug('Plotting of every load case skipped due to large number (>50) of cases')
            else:
                subplot.plot(offsets[:, axis], loads[:, :, idx_load], color='cornflowerblue',
                             linestyle='-', marker='.', zorder=-2)

            # Loop over all stations and label min and max loads.
            for idx_station in range(len(monstations)):
                # Identifiy min and max loads
                idx_max_loads = np.argmax(loads[idx_station, :, idx_load])
                idx_min_loads = np.argmin(loads[idx_station, :, idx_load])

                # Highlight max loads with red dot and print subcase description.
                subplot.scatter(offsets[idx_station, axis],
                                loads[idx_station, idx_max_loads, idx_load],
                                color='r')
                subplot.text(offsets[idx_station, axis],
                             loads[idx_station, idx_max_loads, idx_load],
                             str(subcases[idx_station][idx_max_loads]),
                             fontsize=4, verticalalignment='bottom')

                # Highlight min loads with red dot and print subcase description.
                subplot.scatter(offsets[idx_station, axis],
                                loads[idx_station, idx_min_loads, idx_load],
                                color='r')
                subplot.text(offsets[idx_station, axis],
                             loads[idx_station, idx_min_loads, idx_load],
                             str(subcases[idx_station][idx_min_loads]),
                             fontsize=4, verticalalignment='top')

            subplot.ticklabel_format(style='sci', axis='y', scilimits=(-2, 2))
            subplot.grid(visible=True, which='major', axis='both')
            subplot.minorticks_on()
            yax = subplot.get_yaxis()
            yax.set_label_coords(x=-0.18, y=0.5)
            if axis == 0:
                subplot.set_xlabel('x [m]')
            elif axis == 1:
                subplot.set_xlabel('y [m]')
            elif axis == 3:
                subplot.set_xlabel('z [m]')
            subplot.set_ylabel(label_loads[idx_load])
            subplot.set_rasterization_zorder(-1)
            self.pp.savefig()
        plt.close()

    def plot_monstations_time(self, filename_pdf):
        logging.info('start plotting cutting forces over time ...')
        pp = PdfPages(filename_pdf)
        potato = np.sort(np.unique(self.potatos_fz_mx + self.potatos_mx_my + self.potatos_fz_my + self.potatos_fy_mx
                                   + self.potatos_mx_mz + self.potatos_my_mz))
        for station in potato:
            monstation = self.monstations[station]
            _, ax = plt.subplots(6, sharex=True, figsize=(8, 10))
            # Plot all load cases for which time series are stored in the monstation
            for i_case in [key for key in monstation.keys() if key.isnumeric()]:
                loads = monstation[i_case]['loads']
                t = monstation[i_case]['t']
                ax[0].plot(t, loads[:, 0], 'k', zorder=-2)
                ax[1].plot(t, loads[:, 1], 'k', zorder=-2)
                ax[2].plot(t, loads[:, 2], 'k', zorder=-2)
                ax[3].plot(t, loads[:, 3], 'k', zorder=-2)
                ax[4].plot(t, loads[:, 4], 'k', zorder=-2)
                ax[5].plot(t, loads[:, 5], 'k', zorder=-2)
            # make plots nice
            ax[0].set_position([0.2, 0.83, 0.7, 0.12])
            ax[0].title.set_text(station)
            ax[0].set_ylabel('Fx [N]')
            ax[0].get_yaxis().set_label_coords(x=-0.18, y=0.5)
            ax[0].grid(visible=True, which='major', axis='both')
            ax[0].minorticks_on()
            ax[0].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

            ax[1].set_position([0.2, 0.68, 0.7, 0.12])
            ax[1].set_ylabel('Fy [N]')
            ax[1].get_yaxis().set_label_coords(x=-0.18, y=0.5)
            ax[1].grid(visible=True, which='major', axis='both')
            ax[1].minorticks_on()
            ax[1].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

            ax[2].set_position([0.2, 0.53, 0.7, 0.12])
            ax[2].set_ylabel('Fz [N]')
            ax[2].get_yaxis().set_label_coords(x=-0.18, y=0.5)
            ax[2].grid(visible=True, which='major', axis='both')
            ax[2].minorticks_on()
            ax[2].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

            ax[3].set_position([0.2, 0.38, 0.7, 0.12])
            ax[3].set_ylabel('Mx [Nm]')
            ax[3].get_yaxis().set_label_coords(x=-0.18, y=0.5)
            ax[3].grid(visible=True, which='major', axis='both')
            ax[3].minorticks_on()
            ax[3].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

            ax[4].set_position([0.2, 0.23, 0.7, 0.12])
            ax[4].set_ylabel('My [Nm]')
            ax[4].get_yaxis().set_label_coords(x=-0.18, y=0.5)
            ax[4].grid(visible=True, which='major', axis='both')
            ax[4].minorticks_on()
            ax[4].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

            ax[5].set_position([0.2, 0.08, 0.7, 0.12])
            ax[5].set_ylabel('Mz [Nm]')
            ax[5].get_yaxis().set_label_coords(x=-0.18, y=0.5)
            ax[5].grid(visible=True, which='major', axis='both')
            ax[5].minorticks_on()
            ax[5].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

            ax[5].set_xlabel('t [sec]')

            ax[0].set_rasterization_zorder(-1)
            ax[1].set_rasterization_zorder(-1)
            ax[2].set_rasterization_zorder(-1)
            ax[3].set_rasterization_zorder(-1)
            ax[4].set_rasterization_zorder(-1)
            ax[5].set_rasterization_zorder(-1)

            pp.savefig()
            plt.close()
        pp.close()
        logging.info('plots saved as ' + filename_pdf)


class FlutterPlots(LoadPlots):

    def plot_fluttercurves(self):
        logging.info('start plotting flutter curves...')
        fig, ax = plt.subplots(3, sharex=True, figsize=(8, 10))
        ax_vtas = ax[2].twiny()
        for response in self.responses:
            trimcase = self.jcl.trimcase[response['i'][()]]
            h = self.model['atmo'][trimcase['altitude']]['h'][()]
            # Plot boundaries
            fmin = 2 * np.floor(response['freqs'][:].min() / 2)
            if fmin < -50.0 or np.isnan(fmin):
                fmin = -50.0
            fmax = 2 * np.ceil(response['freqs'][:].max() / 2)
            if fmax > 50.0 or np.isnan(fmax):
                fmax = 50.0
            Vmin = 0
            Vmax = 2 * np.ceil(tas2eas(response['Vtas'][:].max(), h) / 2)
            if Vmax > 500.0 or np.isnan(Vmax):
                Vmax = 500.0
            gmin = -0.11
            gmax = 0.11

            colors = itertools.cycle((plt.cm.tab20c(np.linspace(0, 1, 20))))
            markers = itertools.cycle(('+', 'o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'x', 'D',))

            ax[0].cla()
            ax[1].cla()
            ax[2].cla()
            ax_vtas.cla()

            for j in range(response['freqs'].shape[1]):
                marker = next(markers)
                color = next(colors)
                ax[0].plot(tas2eas(response['Vtas'][:, j], h), response['freqs'][:, j],
                           marker=marker, markersize=4.0, linewidth=1.0, color=color)
                ax[1].plot(tas2eas(response['Vtas'][:, j], h), response['damping'][:, j],
                           marker=marker, markersize=4.0, linewidth=1.0, color=color)
                ax[2].plot(tas2eas(response['Vtas'][:, j], h), response['damping'][:, j],
                           marker=marker, markersize=4.0, linewidth=1.0, color=color)

            # make plots nice
            fig.suptitle(trimcase['desc'], fontsize=16)

            ax[0].set_position([0.15, 0.55, 0.75, 0.35])
            ax[0].set_ylabel('Frequency [Hz]')
            ax[0].get_yaxis().set_label_coords(x=-0.13, y=0.5)
            ax[0].grid(visible=True, which='major', axis='both')
            ax[0].minorticks_on()
            ax[0].axis([Vmin, Vmax, fmin, fmax])

            ax[1].set_position([0.15, 0.35, 0.75, 0.18])
            ax[1].set_ylabel('Damping (zoom)')
            ax[1].get_yaxis().set_label_coords(x=-0.13, y=0.5)
            ax[1].grid(visible=True, which='major', axis='both')
            ax[1].minorticks_on()
            ax[1].axis([Vmin, Vmax, gmin, gmax])

            ax[2].set_position([0.15, 0.15, 0.75, 0.18])
            ax[2].set_ylabel('Damping')
            ax[2].get_yaxis().set_label_coords(x=-0.13, y=0.5)
            ax[2].grid(visible=True, which='major', axis='both')
            ax[2].minorticks_on()
            ax[2].axis([Vmin, Vmax, -1.1, 1.1])
            ax[2].set_xlabel('$V_{eas} [m/s]$')

            # additional axis for Vtas

            ax_vtas.set_position([0.15, 0.15, 0.75, 0.18])
            # set the position of the second x-axis to bottom
            ax_vtas.xaxis.set_ticks_position('bottom')
            ax_vtas.xaxis.set_label_position('bottom')
            ax_vtas.spines['bottom'].set_position(('outward', 60))
            x1, x2 = ax[1].get_xlim()
            ax_vtas.set_xlim((eas2tas(x1, h), eas2tas(x2, h)))
            ax_vtas.minorticks_on()
            ax_vtas.set_xlabel('$V_{tas} [m/s]$')

            self.pp.savefig()

    def plot_eigenvalues(self):
        logging.info('start plotting eigenvalues and -vectors...')
        fig, ax = plt.subplots(1, 3, figsize=(16, 9))
        ax_freq = ax[0].twinx()
        ax_divider = make_axes_locatable(ax[2])
        ax_cbar = ax_divider.append_axes("top", size="4%", pad="1%")
        for response in self.responses:
            trimcase = self.jcl.trimcase[response['i'][()]]
            simcase = self.jcl.simcase[response['i'][()]]
            h = self.model['atmo'][trimcase['altitude']]['h'][()]

            # this kind of plot is only feasible for methods which iterate over Vtas, e.g. not the K- or KE-methods
            if 'flutter' in simcase and simcase['flutter_para']['method'] not in ['pk_rodden', 'pk_schwochow',
                                                                                  'pk', 'statespace']:
                logging.info('skip plotting of eigenvalues and -vectors for {}'.format(trimcase['desc']))
                continue

            # Plot boundaries
            rmax = np.ceil(response['eigenvalues'][:].real.max())
            rmin = np.floor(response['eigenvalues'][:].real.min())
            imax = np.ceil(response['eigenvalues'][:].imag.max())
            imin = np.floor(response['eigenvalues'][:].imag.min())

            for i in range(response['Vtas'].shape[0]):
                colors = itertools.cycle((plt.cm.tab20c(np.linspace(0, 1, 20))))
                markers = itertools.cycle(('+', 'o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'x', 'D',))
                desc = [str(mode) for mode in range(response['eigenvalues'].shape[1])]

                ax[0].cla()
                ax[1].cla()
                ax[2].cla()
                ax_cbar.cla(), ax_freq.cla()  # clear all axes for next plot
                # plot eigenvector
                im_eig = ax[2].imshow(response['eigenvectors'][i].__abs__(), cmap='hot_r', aspect='auto',
                                      origin='upper', vmin=0.0, vmax=1.0)
                # add colorbar to plot
                fig.colorbar(im_eig, cax=ax_cbar, orientation="horizontal")
                # plot eigenvalues
                for j in range(response['eigenvalues'].shape[1]):
                    marker = next(markers)
                    color = next(colors)
                    ax[0].plot(response['eigenvalues'][:, j].real, response['eigenvalues'][:, j].imag,
                               color=color, linestyle='--')
                    ax[0].plot(response['eigenvalues'][i, j].real, response['eigenvalues'][i, j].imag,
                               marker=marker, markersize=8.0, color=color, label=desc[j])
                    ax[1].plot(response['eigenvalues'][:, j].real, response['eigenvalues'][:, j].imag,
                               color=color, linestyle='--')
                    ax[1].plot(response['eigenvalues'][i, j].real, response['eigenvalues'][i, j].imag,
                               marker=marker, markersize=8.0, color=color, label=desc[j])
                    ax[2].plot(j, response['states'].__len__(),
                               marker=marker, markersize=8.0, c=color)

                # make plots nice
                fig.suptitle(t='{}, Veas={:.2f} m/s, Vtas={:.2f} m/s'.format(
                    trimcase['desc'], tas2eas(response['Vtas'][i, 0], h), response['Vtas'][i, 0]), fontsize=16)
                ax[0].set_position([0.12, 0.1, 0.25, 0.8])
                ax[0].set_xlabel('Real')
                ax[0].set_ylabel('Imag')
                ax[0].get_yaxis().set_label_coords(x=-0.13, y=0.5)
                ax[0].grid(visible=True, which='major', axis='both')
                ax[0].minorticks_on()
                ax[0].axis([rmin, rmax, imin, imax])

                # additional axis for frequency
                # set the position of the second y-axis to left
                ax_freq.yaxis.set_ticks_position('left')
                ax_freq.yaxis.set_label_position('left')
                ax_freq.spines['left'].set_position(('outward', 60))
                y1, y2 = ax[0].get_ylim()
                ax_freq.set_ylim((y1 / 2.0 / np.pi, y2 / 2.0 / np.pi))
                ax_freq.minorticks_on()
                ax_freq.set_ylabel('Frequency [Hz]')

                ax[1].set_position([0.40, 0.1, 0.1, 0.8])
                ax[1].set_xlabel('Real (zoom)')
                ax[1].grid(visible=True, which='major', axis='both')
                ax[1].minorticks_on()
                ax[1].axis([-1.0, 1.0, imin, imax])
                # connect with y-axis from left hand plot
                ax[0].sharey(ax[1])
                ax[1].yaxis.set_tick_params(which='both', labelleft=False, labelright=False)
                ax[1].yaxis.offsetText.set_visible(False)
                # add legend
                ax[1].legend(bbox_to_anchor=(1.10, 1), loc='upper left', borderaxespad=0.0, fontsize=10)

                ax[2].set_position([0.60, 0.1, 0.35, 0.8])
                ax[2].yaxis.set_ticks(np.arange(0, response['states'].__len__(), 1))
                ax[2].yaxis.set_ticklabels(response['states'].asstr(), fontsize=10)
                ax[2].yaxis.set_tick_params(rotation=0)
                ax[2].xaxis.set_ticks(np.arange(0, response['eigenvalues'].shape[1], 1))
                ax[2].xaxis.set_ticklabels(np.arange(0, response['eigenvalues'].shape[1], 1), fontsize=10)
                ax[2].grid(visible=True, which='major', axis='both')

                # change tick position to top. Tick position defaults to bottom and overlaps the image.
                ax_cbar.xaxis.set_ticks_position("top")

                self.pp.savefig()

    def plot_fluttercurves_to_pdf(self, filename_pdf):
        self.pp = PdfPages(filename_pdf)
        self.plot_fluttercurves()
        self.pp.close()
        logging.info('plots saved as ' + filename_pdf)

    def plot_eigenvalues_to_pdf(self, filename_pdf):
        self.pp = PdfPages(filename_pdf)
        self.plot_eigenvalues()
        self.pp.close()
        logging.info('plots saved as ' + filename_pdf)


class TurbulencePlots(LoadPlots):

    def plot_monstations(self, filename_pdf):

        # launch plotting
        self.pp = PdfPages(filename_pdf)
        self.potato_plots()
        self.pp.close()
        logging.info('plots saved as ' + filename_pdf)

    def potato_plot(self, station, desc, color, dof_xaxis, dof_yaxis, show_hull=True, show_labels=False, show_minmax=False):
        loads = np.array(self.monstations[station]['loads'])
        subcases = list(self.monstations[station]['subcases'].asstr()[:])  # make sure this is a list
        turbulence_loads = np.array(self.monstations[station]['turbulence_loads'])
        correlations = np.array(self.monstations[station]['correlations'])

        X0 = loads[:, dof_xaxis]
        Y0 = loads[:, dof_yaxis]

        tangent_pos = loads.T + turbulence_loads.T * correlations.T
        tangent_neg = loads.T - turbulence_loads.T * correlations.T

        AB_pos = loads.T + turbulence_loads.T * ((1.0 - correlations.T) / 2.0) ** 0.5
        AB_neg = loads.T - turbulence_loads.T * ((1.0 - correlations.T) / 2.0) ** 0.5

        CD_pos = loads.T + turbulence_loads.T * ((1.0 + correlations.T) / 2.0) ** 0.5
        CD_neg = loads.T - turbulence_loads.T * ((1.0 + correlations.T) / 2.0) ** 0.5

        X = np.vstack((tangent_pos[dof_xaxis, dof_xaxis, :],
                       tangent_neg[dof_xaxis, dof_xaxis, :],
                       tangent_pos[dof_yaxis, dof_xaxis, :],
                       tangent_neg[dof_yaxis, dof_xaxis, :],
                       AB_pos[dof_yaxis, dof_xaxis, :],
                       AB_neg[dof_yaxis, dof_xaxis, :],
                       CD_pos[dof_yaxis, dof_xaxis, :],
                       CD_neg[dof_yaxis, dof_xaxis, :],
                       )).T
        Y = np.vstack((tangent_pos[dof_xaxis, dof_yaxis, :],
                       tangent_neg[dof_xaxis, dof_yaxis, :],
                       tangent_pos[dof_yaxis, dof_yaxis, :],
                       tangent_neg[dof_yaxis, dof_yaxis, :],
                       AB_neg[dof_xaxis, dof_yaxis, :],
                       AB_pos[dof_xaxis, dof_yaxis, :],
                       CD_pos[dof_xaxis, dof_yaxis, :],
                       CD_neg[dof_xaxis, dof_yaxis, :],
                       )).T
        self.subplot.scatter(X0, Y0, color=color, label=desc, zorder=-2)
        self.subplot.scatter(X.ravel(), Y.ravel(), color=color, zorder=-2)

        if show_hull:
            for i_subcase in range(len(subcases)):
                self.fit_ellipse(X0[i_subcase], Y0[i_subcase], X[i_subcase, :], Y[i_subcase, :], color)

        if show_labels:
            labels = []
            for subcase in subcases:
                for point in ['_T1', '_T2', '_T3', '_T4', '_AB', '_EF', '_CD', '_GH']:
                    labels.append(subcase + point)
            for x, y, label in zip(X.ravel(), Y.ravel(), labels):
                self.subplot.text(x, y, label, fontsize=8)

    def fit_ellipse(self, X0, Y0, X, Y, color):
        # Formulate and solve the least squares problem ||Ax - b ||^2
        A = np.vstack([(X - X0) ** 2, 2.0 * (X - X0) * (Y - Y0), (Y - Y0) ** 2]).T
        b = np.ones_like(X)
        x = np.linalg.lstsq(A, b, rcond=None)[0].squeeze()

        # Print the equation of the ellipse in standard form
        logging.debug(
            'The ellipse is given by {0:.3}x^2 + {1:.3}*2xy+{2:.3}y^2 = 1'.format(x[0], x[1], x[2]))

        # Calculate the parameters of the ellipse
        alpha = -0.5 * np.arctan(2 * x[1] / (x[2] - x[0]))
        eta = x[0] + x[2]
        zeta = (x[2] - x[0]) / np.cos(2 * alpha)
        major = (2.0 / (eta - zeta)) ** 0.5
        minor = (2.0 / (eta + zeta)) ** 0.5

        logging.debug('Major axis = {:.3f}'.format(major))
        logging.debug('Minor axis = {:.3f}'.format(minor))
        logging.debug('Rotation = {:.3f} deg'.format(alpha / np.pi * 180.0))

        # Plot the given samples
        # self.subplot.scatter(X, Y, label='Data Points')

        X, Y = self.ellipse_polar(major, minor, alpha)
        self.subplot.plot(X + X0, Y + Y0, color=color, linewidth=2.0, linestyle='--')

    def ellipse_polar(self, major, minor, alpha, phi=np.linspace(0, 2.0 * np.pi, 360)):
        X = 0.0 + major * np.cos(phi) * np.cos(alpha) - minor * np.sin(phi) * np.sin(alpha)
        Y = 0.0 + major * np.cos(phi) * np.sin(alpha) + minor * np.sin(phi) * np.cos(alpha)
        return X, Y
