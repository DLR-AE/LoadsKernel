# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 09:24:10 2017

@author: voss_ar
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

from loadskernel import plotting_standard

class Plotting(plotting_standard.StandardPlots):
    def __init__(self, fig):
        plt.rcParams.update({'font.size': 16,
                             'svg.fonttype':'none'})
        self.subplot = fig.add_axes([0.2, 0.15, 0.7, 0.75]) # List is [left, bottom, width, height]
        
    def plot_nothing(self):
        self.subplot.cla()
        
    def potato_plots(self, dataset_sel, station, descs, colors, dof_xaxis, dof_yaxis, var_xaxis, var_yaxis, show_hull, show_labels):
        # This function relies on the potato plotting function in LK imported above to avoid code duplications.
        # The labels, margins, etc. are adjusted in this function to fit the window space.
        self.subplot.cla()
        for i_dataset in range(len(dataset_sel)):
            self.crit_trimcases = []
            self.add_monstations(dataset_sel[i_dataset])
            self.potato_plot(station, descs[i_dataset], colors[i_dataset], dof_xaxis, dof_yaxis, show_hull, show_labels)
        self.subplot.legend(loc='best')
        self.subplot.ticklabel_format(style='sci', axis='x', scilimits=(-2,2))
        self.subplot.ticklabel_format(style='sci', axis='y', scilimits=(-2,2))
        self.subplot.grid(True)
        yax = self.subplot.get_yaxis()
        yax.set_label_coords(x=-0.18, y=0.5)
        self.subplot.set_xlabel(var_xaxis)
        self.subplot.set_ylabel(var_yaxis)
        