# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 09:24:10 2017

@author: voss_ar
"""


import numpy as np
from mayavi import mlab
from tvtk.api import tvtk
from mayavi.sources.utils import has_attributes

class Plotting:
    def __init__(self):
        self.pscale = 0.1
        pass

    def plot_nothing(self):
        mlab.clf(self.fig)
        self.show_masses=False
        self.show_strc=False
        self.show_mode=False
        self.show_aero=False
        self.show_coupling=False
        
    def add_figure(self, fig):
        self.fig = fig
        self.fig.scene.background = (1.,1.,1.)
        self.plot_nothing()
    
    def add_model(self, model):
        self.model = model
        self.strcgrid = model.strcgrid
        self.calc_distance()
        self.calc_focalpoint()

    def calc_distance(self):
        self.distance = 1.5*(  (self.strcgrid['offset'][:,0].max()-self.strcgrid['offset'][:,0].min())**2 \
                          + (self.strcgrid['offset'][:,1].max()-self.strcgrid['offset'][:,1].min())**2 \
                          + (self.strcgrid['offset'][:,2].max()-self.strcgrid['offset'][:,2].min())**2 )**0.5
    def calc_focalpoint(self):
        self.focalpoint = (self.strcgrid['offset'].min(axis=0) + self.strcgrid['offset'].max(axis=0))/2.0
        
    def set_view_left_above(self):
        self.azimuth   =  60.0
        self.elevation = -65.0
        self.roll      =  55.0
        self.set_view()
        
    def set_view_right_above(self):
        self.azimuth   = 120.0
        self.elevation = -65.0
        self.roll      = -55.0
        self.set_view()
        
    def set_view_back(self):
        self.azimuth   = 180.0
        self.elevation = -90.0
        self.roll      = -90.0
        self.set_view()
        
    def set_view_side(self):
        self.azimuth   =  90.0
        self.elevation = -90.0
        self.roll      =   0.0
        self.set_view()

    def set_view(self):
        mlab.view(azimuth=self.azimuth, elevation=self.elevation, roll=self.roll,  distance=self.distance, focalpoint=self.focalpoint)
        #mlab.orientation_axes()

    # ------------
    # --- mass ---
    #-------------
    def hide_masses(self):
        self.src_masses.remove()
        self.src_mass_cg.remove()
        self.show_masses=False
        mlab.draw(self.fig)
        
    def plot_masses(self, MGG, Mb, cggrid, rho=2700.0):
        # get nodal masses
        m_cg = Mb[0,0]
        m = MGG.diagonal()[0::6]
        
        radius_mass_cg = ((m_cg*3.)/(4.*rho*np.pi))**(1./3.) 
        radius_masses = ((m*3.)/(4.*rho*np.pi))**(1./3.) 

        if self.show_masses:
            self.update_mass_display(radius_masses, radius_mass_cg, cggrid)
        else:
            self.setup_mass_display(radius_masses, radius_mass_cg, cggrid)
            self.show_masses=True
        mlab.draw(self.fig)

    def setup_mass_display(self, radius_masses, radius_mass_cg, cggrid):
        ug1 = tvtk.UnstructuredGrid(points=self.strcgrid['offset'])
        ug1.point_data.scalars = radius_masses
        # plot points as glyphs
        self.src_masses = mlab.pipeline.add_dataset(ug1)
        points = mlab.pipeline.glyph(self.src_masses, scale_mode='scalar', scale_factor = 1.0, color=(1,0.7,0))
        points.glyph.glyph.clamping = False
        #points.glyph.glyph.range = np.array([0.0, 1.0])
        
        ug2 = tvtk.UnstructuredGrid(points=cggrid['offset'])
        ug2.point_data.scalars = np.array([radius_mass_cg])
        # plot points as glyphs
        self.src_mass_cg = mlab.pipeline.add_dataset(ug2)
        points = mlab.pipeline.glyph(self.src_mass_cg, scale_mode='scalar', scale_factor = 1.0, color=(1,1,0), opacity=0.3, resolution=64)
        points.glyph.glyph.clamping = False
        #points.glyph.glyph.range = np.array([0.0, 1.0])      

    def update_mass_display(self, radius_masses, radius_mass_cg, cggrid):
        self.src_masses.outputs[0].points.from_array(self.strcgrid['offset'])
        self.src_masses.outputs[0].point_data.scalars.from_array(radius_masses)
        self.src_mass_cg.outputs[0].points.from_array(cggrid['offset'])
        self.src_mass_cg.outputs[0].point_data.scalars.from_array(np.array([radius_mass_cg]))

    # ------------
    # --- strc ---
    #-------------
    def hide_strc(self):
        self.src_strc.remove()
        self.show_strc=False
        mlab.draw(self.fig)
        
    def plot_strc(self):
        self.src_strc = self.setup_strc_display(offsets=self.strcgrid['offset'], color=(0,0,1), p_scale=self.pscale)
        self.show_strc=True
        mlab.draw(self.fig)
    
    def hide_mode(self):
        self.src_mode.remove()
        self.show_mode=False
        mlab.draw(self.fig)
        
    def plot_mode(self, offsets):
        if self.show_mode:
            self.update_mode_display(offsets=offsets)
        else:
            self.src_mode = self.setup_strc_display(offsets=offsets, color=(0,1,0), p_scale=self.pscale)
            self.show_mode=True
        mlab.draw(self.fig)
        
    def setup_strc_display(self, offsets, color, p_scale):
        ug = tvtk.UnstructuredGrid(points=offsets)
        #ug.point_data.scalars = scalars
        if hasattr(self.model, 'strcshell'):
            # plot shell as surface
            shells = []
            for shell in self.model.strcshell['cornerpoints']: 
                shells.append([np.where(self.strcgrid['ID']==id)[0][0] for id in shell])
            shell_type = tvtk.Polygon().cell_type
            ug.set_cells(shell_type, shells)
            src_strc = mlab.pipeline.add_dataset(ug)
            points  = mlab.pipeline.glyph(src_strc, color=color, scale_factor=p_scale) 
            surface = mlab.pipeline.surface(src_strc, opacity=0.4, color=color)
        else: 
            # plot points as glyphs
            src_strc = mlab.pipeline.add_dataset(ug)
            points = mlab.pipeline.glyph(src_strc, color=color, scale_factor=p_scale)
        points.glyph.glyph.scale_mode = 'data_scaling_off'
        return src_strc
        
        
    def update_mode_display(self, offsets):
        self.src_mode.outputs[0].points.from_array(offsets)
        #self.src_mode.outputs[0].point_data.scalars.from_array(scalars)
        
    # ------------
    # --- aero ---
    #-------------
    def hide_aero(self):
        self.src_aerogrid.remove()
        self.show_aero=False
        mlab.draw(self.fig)
        
    def plot_aero(self):
        self.setup_aero_display(color=(1,1,1), p_scale=self.pscale)
        self.show_aero=True
        mlab.draw(self.fig)
        
    def setup_aero_display(self, color, p_scale):
        ug = tvtk.UnstructuredGrid(points=self.model.aerogrid['cornerpoint_grids'][:,(1,2,3)])
        shells = []
        for shell in self.model.aerogrid['cornerpoint_panels']: 
            shells.append([np.where(self.model.aerogrid['cornerpoint_grids'][:,0]==id)[0][0] for id in shell])
        shell_type = tvtk.Polygon().cell_type
        ug.set_cells(shell_type, shells)
        #ug.cell_data.scalars = scalars
        self.src_aerogrid = mlab.pipeline.add_dataset(ug)
        
        points = mlab.pipeline.glyph(self.src_aerogrid, color=color, scale_factor=p_scale)
        points.glyph.glyph.scale_mode = 'data_scaling_off'
        
        surface = mlab.pipeline.surface(self.src_aerogrid, color=color)
        surface.actor.mapper.scalar_visibility=False
        surface.actor.property.edge_visibility=True
        #surface.actor.property.edge_color=(0.9,0.9,0.9)
        surface.actor.property.edge_color=(0,0,0)
        surface.actor.property.line_width=0.5
    
    def update_aero_display(self, scalars): # currently unused
        self.src_aerogrid.outputs[0].cell_data.scalars.from_array(scalars)
        self.src_aerogrid.update()

    # --------------
    # --- coupling ---
    #---------------
    def hide_aero_strc_coupling(self):
        self.src_grid_i.remove()
        self.src_grid_d.remove()
        if  hasattr(self.model, 'coupling_rules'):
            self.src_splinerules.remove()
        self.show_coupling=False
        mlab.draw(self.fig)
            
    def plot_aero_strc_coupling(self):
        if  hasattr(self.model, 'coupling_rules'):
            self.src_grid_i, self.src_grid_d, self.src_splinerules \
            = self.plot_splinerules(self.model.splinegrid, '', self.model.aerogrid, '_k', self.model.coupling_rules, self.model.coord)
        else:
            self.src_grid_i, self.src_grid_d \
            = self.plot_splinegrids(self.model.splinegrid, '', self.model.aerogrid, '_k')
        self.show_coupling=True
            
            
    def plot_splinegrids(self, grid_i,  set_i,  grid_d, set_d):
        p_scale = self.pscale # points
        src_grid_i = mlab.points3d(grid_i['offset'+set_i][:,0], grid_i['offset'+set_i][:,1], grid_i['offset'+set_i][:,2], scale_factor=p_scale*2, color=(0,1,0))
        src_grid_d = mlab.points3d(grid_d['offset'+set_d][:,0], grid_d['offset'+set_d][:,1], grid_d['offset'+set_d][:,2], scale_factor=p_scale, color=(1,0,0))
        return src_grid_i, src_grid_d
        
    def plot_splinerules(self, grid_i,  set_i,  grid_d, set_d, splinerules, coord):

        # transfer points into common coord
        offset_dest_i = []
        for i_point in range(len(grid_i['ID'])):
            pos_coord = coord['ID'].index(grid_i['CP'][i_point])
            offset_dest_i.append(np.dot(coord['dircos'][pos_coord],grid_i['offset'+set_i][i_point])+coord['offset'][pos_coord])
        offset_dest_i = np.array(offset_dest_i)
        
        offset_dest_d = []
        for i_point in range(len(grid_d['ID'])):
            pos_coord = coord['ID'].index(grid_d['CP'][i_point])
            offset_dest_d.append(np.dot(coord['dircos'][pos_coord],grid_d['offset'+set_d][i_point])+coord['offset'][pos_coord])
        offset_dest_d = np.array(offset_dest_d)
        
        position_d = []
        position_i = []
        for i_i in range(len(splinerules['ID_i'])):
            for i_d in range(len(splinerules['ID_d'][i_i])): 
                position_d.append( np.where(grid_d['ID']==splinerules['ID_d'][i_i][i_d])[0][0] )
                position_i.append( np.where(grid_i['ID']==splinerules['ID_i'][i_i])[0][0] )
       
        x = offset_dest_i[position_i,0]
        y = offset_dest_i[position_i,1]
        z = offset_dest_i[position_i,2]
        u = offset_dest_d[position_d,0] - x
        v = offset_dest_d[position_d,1] - y
        w = offset_dest_d[position_d,2] - z
        p_scale = 0.05 # points
    
        src_grid_i = mlab.points3d(grid_i['offset'+set_i][:,0], grid_i['offset'+set_i][:,1], grid_i['offset'+set_i][:,2], scale_factor=p_scale*2, color=(0,1,0))
        src_grid_d = mlab.points3d(grid_d['offset'+set_d][:,0], grid_d['offset'+set_d][:,1], grid_d['offset'+set_d][:,2], scale_factor=p_scale, color=(1,0,0))
        src_spilerules = mlab.quiver3d(x,y,z,u,v,w, mode='2ddash', scale_factor=1.0, color=(0,0,0), opacity=0.4)
        return src_grid_i, src_grid_d, src_spilerules
       
    # --------------
    # --- monstations ---
    #---------------
    def hide_monstations(self):
        self.src_mongrid_i.remove()
        self.src_mongrid_d.remove()
        self.src_mongrid_rules.remove()
        self.show_monstations=False
        mlab.draw(self.fig)
            
    def plot_monstations(self):
        self.src_mongrid_i, self.src_mongrid_d, self.src_mongrid_rules \
        = self.plot_splinerules(self.model.mongrid, '', self.model.strcgrid, '', self.model.mongrid_rules, self.model.coord)
        self.show_monstations=True

            