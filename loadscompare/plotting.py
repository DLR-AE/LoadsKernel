# -*- coding: utf-8 -*-
import os

import matplotlib.pyplot as plt

from loadskernel import plotting_standard


class Plotting(plotting_standard.LoadPlots):

    def __init__(self, fig):
        plt.rcParams.update({'font.size': 16,
                             'svg.fonttype': 'none'})
        self.fig = fig
        self.subplot = None
        self.crit_trimcases = []

    def clear_figure(self):
        self.fig.clf()
        self.subplot = None
        # Add the logo in the bottom left corner.
        im = plt.imread(os.path.dirname(__file__) + '/../graphics/LK_logo2.png')
        newax = self.fig.add_axes([0.04, 0.02, 0.10, 0.08])
        newax.imshow(im, interpolation='hanning')
        newax.axis('off')

    def potato_plots(self, dataset_sel, station, descs, colors,
                     dof_xaxis, dof_yaxis, var_xaxis, var_yaxis,
                     show_hull, show_labels, show_minmax):
        if self.subplot is None:
            # Create a single axes that fills most of the figure, leaving space for the logo and labels.
            # Store the axes in the class so that it can be used by potato_plot.
            self.clear_figure()
            self.subplot = self.fig.add_axes([0.2, 0.15, 0.7, 0.75])  # List is [left, bottom, width, height]
        else:
            self.subplot.cla()
        # This function relies on the potato plotting function in LK imported above to avoid code duplications.
        for i, dataset in enumerate(dataset_sel):
            self.crit_trimcases = []
            self.add_monstations(dataset)
            self.potato_plot(station, descs[i], colors[i], dof_xaxis, dof_yaxis,
                             show_hull, show_labels, show_minmax)
        # The labels, margins, etc. are adjusted in this function to fit the window space.
        a = self.subplot
        a.legend(loc='best')
        a.ticklabel_format(style='sci', axis='x', scilimits=(-2, 2))
        a.ticklabel_format(style='sci', axis='y', scilimits=(-2, 2))
        a.grid(True)
        a.get_yaxis().set_label_coords(x=-0.18, y=0.5)
        a.set_xlabel(var_xaxis)
        a.set_ylabel(var_yaxis)

    def timehistories(self, monstation, subcases, dofs_idx, dofs_text):
        # Clear the figure, then create subplots for each state to plot, sharing the x-axis (time).
        # This is necessary as the number of axes can change, depending on the selection by the user.
        self.clear_figure()
        ax = self.fig.subplots(len(dofs_idx), sharex=True)
        # Make sure that ax is always iterable, even if there is only one state to plot.
        if len(dofs_idx) == 1:
            ax = [ax]
        # Plotting the time histories for each dof and subcase.
        for a, dof_idx, dof_text in zip(ax, dofs_idx, dofs_text):
            for subcase in subcases:
                time = monstation[subcase]['t'][()]
                data = monstation[subcase]['loads'][:, dof_idx]
                a.plot(time, data, label=subcase)
            # Format current axis.
            a.ticklabel_format(style='sci', axis='y', scilimits=(-2, 2))
            a.grid(True)
            a.set_ylabel(dof_text)
            # Push left bound to the right to make space for the ylabel.
            left, bottom, width, height = a.get_position().bounds
            a.set_position([left + 0.05, bottom, width - 0.05, height])
            a.get_yaxis().set_label_coords(x=-0.18, y=0.5)
            # Show legend per plot
            a.legend(loc='upper right')
        ax[-1].set_xlabel('Time [s]')
