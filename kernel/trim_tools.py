# -*- coding: utf-8 -*-
"""
Created on Thu Nov 27 17:44:58 2014

@author: voss_ar
"""
import numpy as np

def calc_drehmatrix( my=0.0, alpha=0.0, beta=0.0 ):
    # Alle Winkel in [rad] !
    # alpha: Anstellwinkel
    # beta: Schiebewinkel
    # my: Haengewinkel (wird oft nicht betrachtet)
    drehmatrix_my = np.array(([1., 0. , 0.], [0., np.cos(my), np.sin(my)], [0., -np.sin(my), np.cos(my)]))
    drehmatrix_a  = np.array(([np.cos(alpha), 0, -np.sin(alpha)], [0, 1, 0], [np.sin(alpha), 0, np.cos(alpha)]))
    drehematrix_b = np.array(([np.cos(beta), -np.sin(beta), 0], [np.sin(beta), np.cos(beta), 0],[0, 0, 1]))
    drehmatrix = np.dot(np.dot(drehmatrix_my, drehmatrix_a),drehematrix_b)
    
    return drehmatrix
    
