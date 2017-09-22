# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 09:24:10 2017

@author: voss_ar
"""


import numpy as np
from mayavi import mlab
from tvtk.api import tvtk

class Plotting:
    def __init__(self):
        self.update_only=False
        pass

    def plot_nothing(self):
        mlab.clf(self.fig)
        self.update_only=False
        #mlab.orientation_axes()
        # init Matplotlib Plot
        #self.fig = mlab.figure(bgcolor=(1,1,1))
        
    def add_figure(self, fig):
        self.fig = fig
        self.fig.scene.background = (1.,1.,1.)
        self.plot_nothing()
        pass
    
    def add_strcgrid(self, strcgrid):
        self.strcgrid = strcgrid
    

    def plot_masses(self, MGG, Mb, cggrid):
        # get nodal masses
        m_cg = Mb[0,0]
        m = MGG.diagonal()[0::6]
        
        radius_mass_cg = ((m_cg*3.)/(4.*2700.0*np.pi))**(1./3.) 
        radius_masses = ((m*3.)/(4.*2700.0*np.pi))**(1./3.) #/ radius_mass_cg
        #radius_masses = radius_masses/radius_masses.max()
        #self.plot_nothing()
        #self.setup_mass_display(radius_masses, radius_mass_cg, cggrid)
        if self.update_only:
            self.update_mass_display(radius_masses, radius_mass_cg, cggrid)
        else:
            self.setup_mass_display(radius_masses, radius_mass_cg, cggrid)
            self.update_only=True
        mlab.draw(self.fig)


    def setup_mass_display(self, radius_masses, radius_mass_cg, cggrid):
        ug1 = tvtk.UnstructuredGrid(points=self.strcgrid['offset'])
        ug1.point_data.scalars = radius_masses
        # plot points as glyphs
        self.src_masses = mlab.pipeline.add_dataset(ug1)
        points = mlab.pipeline.glyph(self.src_masses, scale_mode='scalar', scale_factor = 1.0, color=(1,0.7,0))
        #points.glyph.glyph.scale_mode = 'scale_by_scalar'
        points.glyph.glyph.range = np.array([0.0, 1.0])
        
        ug2 = tvtk.UnstructuredGrid(points=cggrid['offset'])
        ug2.point_data.scalars = np.array([radius_mass_cg])
        # plot points as glyphs
        self.src_mass_cg = mlab.pipeline.add_dataset(ug2)
        points = mlab.pipeline.glyph(self.src_mass_cg, scale_mode='scalar', scale_factor = 1.0, color=(1,1,0), opacity=0.3, resolution=64)
        #points.glyph.glyph.scale_mode = 'scale_by_scalar'  
        points.glyph.glyph.range = np.array([0.0, 1.0])      

    def update_mass_display(self, radius_masses, radius_mass_cg, cggrid):
        self.src_masses.outputs[0].points.from_array(self.strcgrid['offset'])
        self.src_masses.outputs[0].point_data.scalars.from_array(radius_masses)
        self.src_mass_cg.outputs[0].points.from_array(cggrid['offset'])
        self.src_mass_cg.outputs[0].point_data.scalars.from_array(np.array([radius_mass_cg]))

