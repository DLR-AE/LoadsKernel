# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 09:24:10 2017

@author: voss_ar
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull


class Plotting:
    def __init__(self, subplot):
        self.subplot = subplot
        pass
    
    def plot_sin(self):
        x = np.arange(0,10,0.01)
        self.subplot.cla()
        self.subplot.plot(x,np.sin(x))
        
    def plot_nothing(self):
        self.subplot.cla()
        
    def potato_plots(self, monstations, station, descs, colors, dof_xaxis, dof_yaxis, var_xaxis, var_yaxis, show_hull, show_labels):
        self.subplot.cla()
        for i_dataset in range(len(monstations)):
            self.potato_plot(monstations[i_dataset], station, descs[i_dataset], colors[i_dataset], dof_xaxis, dof_yaxis, var_xaxis, var_yaxis, show_hull, show_labels)
        self.subplot.legend(loc='best')
        
    def potato_plot(self, monstations, station, desc, color, dof_xaxis, dof_yaxis, var_xaxis, var_yaxis, show_hull, show_labels):
        
        if (monstations[station]['loads_dyn2stat'] == []) or ('loads_dyn2stat' not in monstations[station].keys()) :
            loads_string = 'loads'
            subcase_string = 'subcase'
        else:
            loads_string = 'loads_dyn2stat'
            subcase_string = 'subcases_dyn2stat'

        loads   = np.array(monstations[station][loads_string])
        points = np.vstack((loads[:,dof_xaxis], loads[:,dof_yaxis])).T
        crit_trimcases = []
        self.subplot.scatter(points[:,0], points[:,1], color=color, label=desc) # plot points
        
        if show_hull and points.shape[0] >= 3:
            try:
                hull = ConvexHull(points) # calculated convex hull from scattered points
                for simplex in hull.simplices:                   # plot convex hull
                    self.subplot.plot(points[simplex,0], points[simplex,1], color=color, linewidth=2.0, linestyle='--')
                for i_case in range(hull.nsimplex):
                    crit_trimcases.append(monstations[station][subcase_string][hull.vertices[i_case]])
                    if show_labels:                    
                        self.subplot.text(points[hull.vertices[i_case],0], points[hull.vertices[i_case],1], str(monstations[station][subcase_string][hull.vertices[i_case]]), fontsize=8)
            except:
                pass
        else:
            crit_trimcases += monstations[station][subcase_string][:]
        
        self.subplot.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        self.subplot.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        self.subplot.grid('on')
        self.subplot.set_xlabel(var_xaxis)
        self.subplot.set_ylabel(var_yaxis)