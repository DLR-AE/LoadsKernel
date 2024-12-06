import numpy as np
from mayavi import mlab
from tvtk.api import tvtk

from loadskernel.io_functions.data_handling import load_hdf5_dict


class Plotting:

    def __init__(self):
        pass

    def plot_nothing(self):
        mlab.clf(self.fig)
        self.show_masses = False
        self.show_strc = False
        self.show_mode = False
        self.show_aero = False
        self.show_panel_normal_vectors = False
        self.show_cfdgrids = False
        self.show_coupling = False
        self.show_cs = False
        self.show_cell = False
        self.show_monstations = False
        self.show_iges = False

    def add_figure(self, fig):
        self.fig = fig
        self.fig.scene.background = (1., 1., 1.)
        self.plot_nothing()

    def add_model(self, model):
        self.model = model
        self.strcgrid = load_hdf5_dict(self.model['strcgrid'])
        self.splinegrid = load_hdf5_dict(self.model['splinegrid'])
        self.aerogrid = load_hdf5_dict(self.model['aerogrid'])
        self.x2grid = load_hdf5_dict(self.model['x2grid'])
        self.mongrid = load_hdf5_dict(self.model['mongrid'])
        self.coord = load_hdf5_dict(self.model['coord'])
        self.Djx2 = self.model['Djx2'][()]
        self.calc_parameters_from_model_size()
        self.calc_focalpoint()

    def add_cfdgrids(self, cfdgrids):
        self.cfdgrids = cfdgrids

    def add_iges_meshes(self, meshes):
        self.iges_meshes = meshes

    def calc_parameters_from_model_size(self):
        # Calculate the overall size of the model.
        model_size = ((self.strcgrid['offset'][:, 0].max() - self.strcgrid['offset'][:, 0].min()) ** 2
                      + (self.strcgrid['offset'][:, 1].max() - self.strcgrid['offset'][:, 1].min()) ** 2
                      + (self.strcgrid['offset'][:, 2].max() - self.strcgrid['offset'][:, 2].min()) ** 2) ** 0.5
        # Set some parameters which typically give a good view.
        self.model_size = model_size
        self.distance = model_size * 1.5
        self.pscale = np.min([model_size / 400.0, 0.1])
        self.macscale = np.min([model_size / 10.0, 1.0])

    def calc_focalpoint(self):
        self.focalpoint = (self.strcgrid['offset'].min(
            axis=0) + self.strcgrid['offset'].max(axis=0)) / 2.0

    def set_view_left_above(self):
        self.azimuth = 60.0
        self.elevation = -65.0
        self.roll = 55.0
        self.set_view()

    def set_view_back(self):
        self.azimuth = 180.0
        self.elevation = -90.0
        self.roll = -90.0
        self.set_view()

    def set_view_side(self):
        self.azimuth = 90.0
        self.elevation = -90.0
        self.roll = 0.0
        self.set_view()

    def set_view_top(self):
        self.azimuth = 180.0
        self.elevation = 0.0
        self.roll = 0.0
        self.distance *= 1.5  # zoom out more
        self.set_view()
        self.calc_parameters_from_model_size()  # rest zoom

    def set_view(self):
        mlab.view(azimuth=self.azimuth, elevation=self.elevation, roll=self.roll, distance=self.distance,
                  focalpoint=self.focalpoint)
        # mlab.orientation_axes()

    def hide_masses(self):
        self.src_masses.remove()
        self.src_mass_cg.remove()
        self.show_masses = False
        mlab.draw(self.fig)

    def plot_masses(self, MGG, Mb, cggrid, rho=2700.0):
        # get nodal masses
        m_cg = Mb[0, 0]
        m = MGG.diagonal()[0::6]

        radius_mass_cg = ((m_cg * 3.) / (4. * rho * np.pi)) ** (1. / 3.)
        radius_masses = ((m * 3.) / (4. * rho * np.pi)) ** (1. / 3.)

        if self.show_masses:
            self.update_mass_display(radius_masses, radius_mass_cg, cggrid)
        else:
            self.setup_mass_display(radius_masses, radius_mass_cg, cggrid)
            self.show_masses = True
        mlab.draw(self.fig)

    def setup_mass_display(self, radius_masses, radius_mass_cg, cggrid):
        self.ug1_mass = tvtk.UnstructuredGrid(points=self.strcgrid['offset'])
        self.ug1_mass.point_data.scalars = radius_masses
        # plot points as glyphs
        self.src_masses = mlab.pipeline.add_dataset(self.ug1_mass)
        points = mlab.pipeline.glyph(
            self.src_masses, scale_mode='scalar', scale_factor=1.0, color=(1, 0.7, 0))
        points.glyph.glyph.clamping = False

        self.ug2_mass = tvtk.UnstructuredGrid(points=cggrid['offset'])
        self.ug2_mass.point_data.scalars = np.array([radius_mass_cg])
        # plot points as glyphs
        self.src_mass_cg = mlab.pipeline.add_dataset(self.ug2_mass)
        points = mlab.pipeline.glyph(self.src_mass_cg, scale_mode='scalar', scale_factor=1.0, color=(1, 1, 0),
                                     opacity=0.3, resolution=64)
        points.glyph.glyph.clamping = False

    def update_mass_display(self, radius_masses, radius_mass_cg, cggrid):
        self.ug1_mass.points.from_array(self.strcgrid['offset'])
        self.ug1_mass.point_data.scalars.from_array(radius_masses)
        self.ug1_mass.modified()
        self.ug2_mass.points.from_array(cggrid['offset'])
        self.ug2_mass.point_data.scalars.from_array(np.array([radius_mass_cg]))
        self.ug2_mass.modified()

    def hide_strc(self):
        self.src_strc.remove()
        self.show_strc = False
        mlab.draw(self.fig)

    def plot_strc(self):
        self.src_strc = self.setup_strc_display(
            offsets=self.strcgrid['offset'], color=(0, 0, 1), p_scale=self.pscale)
        self.show_strc = True
        mlab.draw(self.fig)

    def hide_mode(self):
        self.src_mode.remove()
        self.show_mode = False
        mlab.draw(self.fig)

    def plot_mode(self, offsets):
        if self.show_mode:
            self.update_mode_display(offsets=offsets)
        else:
            self.src_mode = self.setup_strc_display(
                offsets=offsets, color=(0, 1, 0), p_scale=self.pscale)
            self.show_mode = True
        mlab.draw(self.fig)

    def setup_strc_display(self, offsets, color, p_scale):
        self.ug_strc = tvtk.UnstructuredGrid(points=offsets)
        if 'strcshell' in self.model:
            # plot shell as surface
            shells = []
            for shell in self.model['strcshell']['cornerpoints'][()]:
                shells.append([np.where(self.strcgrid['ID'] == id)[0][0]
                              for id in shell[np.isfinite(shell)]])
            shell_type = tvtk.Polygon().cell_type
            self.ug_strc.set_cells(shell_type, shells)
            src_strc = mlab.pipeline.add_dataset(self.ug_strc)
            points = mlab.pipeline.glyph(
                src_strc, color=color, scale_factor=p_scale)
            surface = mlab.pipeline.surface(src_strc, opacity=0.4, color=color)
            surface.actor.property.edge_visibility = True
            surface.actor.property.line_width = 0.5
        else:
            # plot points as glyphs
            src_strc = mlab.pipeline.add_dataset(self.ug_strc)
            points = mlab.pipeline.glyph(
                src_strc, color=color, scale_factor=p_scale)
        points.glyph.glyph.scale_mode = 'data_scaling_off'
        return src_strc

    def update_mode_display(self, offsets):
        self.ug_strc.points.from_array(offsets)
        self.ug_strc.modified()

    def hide_aero(self):
        if self.show_aero:
            self.src_aerogrid.remove()
            self.src_MAC.remove()
            self.show_aero = False
            mlab.draw(self.fig)
        if self.show_panel_normal_vectors:
            self.src_panel_normal_vectors.remove()
            self.show_panel_normal_vectors = False

    def plot_aero(self, scalars=None, colormap='coolwarm', vminmax=[-10.0, 10.0]):
        self.setup_aero_display(scalars, colormap, vminmax)
        self.setup_mac_display()
        self.show_aero = True
        mlab.draw(self.fig)

    def plot_panel_normal_vectors(self):
        # This function plots the normal vectors on each aerodynamic panel to identify the orientation visually.
        x, y, z = self.aerogrid['offset_k'][:, 0], self.aerogrid['offset_k'][:, 1], self.aerogrid['offset_k'][:, 2]
        Nx, Ny, Nz, = self.aerogrid['N'][:, 0], self.aerogrid['N'][:, 1], self.aerogrid['N'][:, 2]
        self.src_panel_normal_vectors = mlab.quiver3d(x, y, z, Nx, Ny, Nz, color=(0, 1, 0), opacity=0.4,
                                                      scale_mode='vector', scale_factor=1.0)
        self.show_panel_normal_vectors = True
        mlab.draw(self.fig)

    def setup_mac_display(self):
        ug2 = tvtk.UnstructuredGrid(points=np.array([self.MAC]))
        self.src_MAC = mlab.pipeline.add_dataset(ug2)
        points = mlab.pipeline.glyph(self.src_MAC, scale_mode='scalar', scale_factor=self.macscale, color=(1, 0, 0),
                                     opacity=0.4, resolution=64)
        points.glyph.glyph.clamping = False

    def setup_aero_display(self, scalars, colormap, vminmax):
        # Generate an unstructured grid from the aerodynamic panels and connect them to shell elements
        ug1 = tvtk.UnstructuredGrid(points=self.aerogrid['cornerpoint_grids'][:, (1, 2, 3)])
        shells = []
        for shell in self.aerogrid['cornerpoint_panels']:
            shells.append([np.where(self.aerogrid['cornerpoint_grids'][:, 0] == id)[0][0] for id in shell])
        shell_type = tvtk.Polygon().cell_type
        ug1.set_cells(shell_type, shells)

        if scalars is not None:
            # Add unstructured grid with scalar data per shell element
            ug1.cell_data.scalars = scalars
            self.src_aerogrid = mlab.pipeline.add_dataset(ug1)
            # Generate surface plot from that data
            surface = mlab.pipeline.surface(self.src_aerogrid, colormap=colormap, vmin=vminmax[0], vmax=vminmax[1])
            surface.module_manager.scalar_lut_manager.show_legend = True
            surface.module_manager.scalar_lut_manager.label_text_property.color = (
                0, 0, 0)
            surface.module_manager.scalar_lut_manager.label_text_property.font_family = 'courier'
            surface.module_manager.scalar_lut_manager.label_text_property.bold = False
            surface.module_manager.scalar_lut_manager.label_text_property.italic = False
            surface.module_manager.scalar_lut_manager.number_of_labels = 5

        else:
            # Add unstructured grid without any scalar data
            self.src_aerogrid = mlab.pipeline.add_dataset(ug1)
            # Generate surface plot
            surface = mlab.pipeline.surface(self.src_aerogrid, color=(1, 1, 1))
        surface.actor.property.edge_visibility = True
        surface.actor.property.edge_color = (0, 0, 0)
        surface.actor.property.line_width = 0.5

    def hide_cfdgrids(self):
        for src in self.src_cfdgrids:
            src.remove()
        self.show_cfdgrids = False
        mlab.draw(self.fig)

    def plot_cfdgrids(self, markers):
        self.src_cfdgrids = []
        for marker in self.cfdgrids:
            if marker in markers:
                self.setup_cfdgrid_display(
                    grid=self.cfdgrids[marker], color=(1, 1, 1), scalars=None)
        self.show_cfdgrids = True
        mlab.draw(self.fig)

    def setup_cfdgrid_display(self, grid, color, scalars):
        ug = tvtk.UnstructuredGrid(points=grid['offset'])
        # ug.point_data.scalars = scalars
        shells = []
        for shell in grid['points_of_surface']:
            shells.append([np.where(grid['ID'] == id)[0][0] for id in shell])
        shell_type = tvtk.Polygon().cell_type
        ug.set_cells(shell_type, shells)
        src_cfdgrid = mlab.pipeline.add_dataset(ug)
        self.src_cfdgrids.append(src_cfdgrid)

        surface = mlab.pipeline.surface(
            src_cfdgrid, opacity=1.0, line_width=0.5, color=color)
        surface.actor.property.edge_visibility = True

    def hide_aero_strc_coupling(self):
        self.src_grid_i.remove()
        self.src_grid_d.remove()
        if 'coupling_rules' in self.model:
            self.src_splinerules.remove()
        self.show_coupling = False
        mlab.draw(self.fig)

    def plot_aero_strc_coupling(self):
        if 'coupling_rules' in self.model:
            coupling_rules = load_hdf5_dict(self.model['coupling_rules'])
            self.src_grid_i, self.src_grid_d, self.src_splinerules = self.plot_splinerules(self.splinegrid, '',
                                                                                           self.aerogrid, '_k',
                                                                                           coupling_rules, self.coord)
        else:
            self.src_grid_i, self.src_grid_d = self.plot_splinegrids(
                self.splinegrid, '', self.aerogrid, '_k')
        self.show_coupling = True

    def plot_splinegrids(self, grid_i, set_i, grid_d, set_d):
        src_grid_i = mlab.points3d(grid_i['offset' + set_i][:, 0],
                                   grid_i['offset' + set_i][:, 1],
                                   grid_i['offset' + set_i][:, 2],
                                   scale_factor=self.pscale * 2, color=(0, 1, 0))
        src_grid_d = mlab.points3d(grid_d['offset' + set_d][:, 0],
                                   grid_d['offset' + set_d][:, 1],
                                   grid_d['offset' + set_d][:, 2],
                                   scale_factor=self.pscale, color=(1, 0, 0))
        return src_grid_i, src_grid_d

    def plot_splinerules(self, grid_i, set_i, grid_d, set_d, splinerules, coord):

        # transfer points into common coord
        offset_dest_i = []
        for i_point in range(len(grid_i['ID'])):
            pos_coord = np.where(coord['ID'] == grid_i['CP'][i_point])[0][0]
            offset_dest_i.append(np.dot(coord['dircos'][pos_coord], grid_i['offset' + set_i][i_point])
                                 + coord['offset'][pos_coord])
        offset_dest_i = np.array(offset_dest_i)

        offset_dest_d = []
        for i_point in range(len(grid_d['ID'])):
            pos_coord = np.where(coord['ID'] == grid_d['CP'][i_point])[0][0]
            offset_dest_d.append(np.dot(coord['dircos'][pos_coord], grid_d['offset' + set_d][i_point])
                                 + coord['offset'][pos_coord])
        offset_dest_d = np.array(offset_dest_d)

        position_i = []
        position_d = []

        for ID_i in splinerules:
            for ID_d in splinerules[ID_i]:
                position_i.append(np.where(grid_i['ID'] == int(ID_i))[0][0])
                position_d.append(np.where(grid_d['ID'] == int(ID_d))[0][0])

        x = offset_dest_i[position_i, 0]
        y = offset_dest_i[position_i, 1]
        z = offset_dest_i[position_i, 2]
        u = offset_dest_d[position_d, 0] - x
        v = offset_dest_d[position_d, 1] - y
        w = offset_dest_d[position_d, 2] - z

        src_grid_i = mlab.points3d(grid_i['offset' + set_i][:, 0],
                                   grid_i['offset' + set_i][:, 1],
                                   grid_i['offset' + set_i][:, 2],
                                   scale_factor=self.pscale * 2, color=(0, 1, 0))
        src_grid_d = mlab.points3d(grid_d['offset' + set_d][:, 0],
                                   grid_d['offset' + set_d][:, 1],
                                   grid_d['offset' + set_d][:, 2],
                                   scale_factor=self.pscale, color=(1, 0, 0))
        src_spilerules = mlab.quiver3d(
            x, y, z, u, v, w, mode='2ddash', scale_factor=1.0, color=(0, 0, 0), opacity=0.4)
        return src_grid_i, src_grid_d, src_spilerules

    def hide_monstations(self):
        self.src_mongrid_i.remove()
        self.src_mongrid_d.remove()
        self.src_mongrid_rules.remove()
        self.show_monstations = False
        mlab.draw(self.fig)

    def plot_monstations(self, monstation_id):
        if self.show_monstations:
            self.hide_monstations()
        # create a sub-set from all mongrid_rules
        rules = {
            monstation_id: self.model['mongrid_rules'][str(monstation_id)][()]}
        self.src_mongrid_i, self.src_mongrid_d, self.src_mongrid_rules = self.plot_splinerules(self.mongrid, '',
                                                                                               self.strcgrid, '',
                                                                                               rules, self.coord)
        self.show_monstations = True

    def hide_cs(self):
        self.src_cs.remove()
        self.show_cs = False
        mlab.draw(self.fig)

    def plot_cs(self, i_surf, axis, deg):
        # determine deflections
        if axis == 'y-axis':
            Uj = np.dot(self.Djx2[i_surf], [
                        0, 0, 0, 0, deg / 180.0 * np.pi, 0])
        elif axis == 'z-axis':
            Uj = np.dot(self.Djx2[i_surf], [
                        0, 0, 0, 0, 0, deg / 180.0 * np.pi])
        else:
            Uj = np.dot(self.Djx2[i_surf], [0, 0, 0, 0, 0, 0])
        # find those panels belonging to the current control surface i_surf
        members_of_i_surf = [np.where(self.aerogrid['ID'] == x)[
            0][0] for x in self.x2grid[str(i_surf)]['ID'][()]]
        points = self.aerogrid['offset_k'][members_of_i_surf, :] \
            + Uj[self.aerogrid['set_k'][members_of_i_surf, :][:, (0, 1, 2)]]

        if self.show_cs:
            self.update_cs_display(points)
        else:
            self.setup_cs_display(points)
            self.show_cs = True
        mlab.draw(self.fig)

    def setup_cs_display(self, points):
        self.ug1_cs = tvtk.UnstructuredGrid(points=points)
        # plot points as glyphs
        self.src_cs = mlab.pipeline.add_dataset(self.ug1_cs)
        points = mlab.pipeline.glyph(
            self.src_cs, scale_mode='scalar', scale_factor=self.pscale, color=(1, 0, 0))
        points.glyph.glyph.scale_mode = 'data_scaling_off'

    def update_cs_display(self, points):
        self.ug1_cs.points.from_array(points)
        self.ug1_cs.modified()

    def hide_cell(self):
        self.src_cell.remove()
        self.show_cell = False
        mlab.draw(self.fig)

    def plot_cell(self, cell_data, show_cells):
        if self.show_cell:
            # self.update_cell_display(cell_data=cell_data)
            # The pure update doesn't work in case different shells are selected.
            self.hide_cell()
        self.setup_cell_display(offsets=self.strcgrid['offset'], color=(0, 0, 1), p_scale=self.pscale,
                                cell_data=cell_data, show_cells=show_cells)
        self.show_cell = True
        mlab.draw(self.fig)

    def setup_cell_display(self, offsets, color, p_scale, cell_data, show_cells):
        ug = tvtk.UnstructuredGrid(points=offsets)
        # ug.point_data.scalars = scalars
        if 'strcshell' in self.model:
            # plot shell as surface
            shells = []
            data = []
            for i_shell in range(self.model['strcshell']['n'][()]):
                shell = self.model['strcshell']['cornerpoints'][i_shell]
                if shell in show_cells:
                    data.append(cell_data[i_shell])
                    shells.append([np.where(self.strcgrid['ID'] == id)[0][0]
                                  for id in shell[np.isfinite(shell)]])
            shell_type = tvtk.Polygon().cell_type
            ug.set_cells(shell_type, shells)
            ug.cell_data.scalars = data
            self.src_cell = mlab.pipeline.add_dataset(ug)
            # points  = mlab.pipeline.glyph(self.src_cell, color=color, scale_factor=p_scale)
            surface = mlab.pipeline.surface(self.src_cell, opacity=1.0, line_width=0.5, colormap='plasma',
                                            vmin=cell_data.min(), vmax=cell_data.max())
            surface.actor.property.edge_visibility = True

            surface.module_manager.scalar_lut_manager.show_legend = True
            surface.module_manager.scalar_lut_manager.label_text_property.color = (
                0, 0, 0)
            surface.module_manager.scalar_lut_manager.label_text_property.font_family = 'times'
            surface.module_manager.scalar_lut_manager.label_text_property.bold = False
            surface.module_manager.scalar_lut_manager.label_text_property.italic = False
            surface.module_manager.scalar_lut_manager.number_of_labels = 5

        else:
            # plot points as glyphs
            self.src_cell = mlab.pipeline.add_dataset(ug)
            points = mlab.pipeline.glyph(
                self.src_cell, color=color, scale_factor=p_scale)
            points.glyph.glyph.scale_mode = 'data_scaling_off'

    def update_cell_display(self, cell_data):
        self.src_cell.outputs[0].cell_data.scalars.from_array(cell_data)
        self.src_cell.update()

    def hide_iges(self):
        for src in self.src_iges:
            src.remove()
        self.show_iges = False
        mlab.draw(self.fig)

    def plot_iges(self, selected_meshes):
        self.src_iges = []
        for mesh in self.iges_meshes:
            if mesh['desc'] in selected_meshes:
                self.setup_iges_display(mesh['vtk'])
        self.show_iges = True
        mlab.draw(self.fig)

    def setup_iges_display(self, vtk_object):
        src = mlab.pipeline.add_dataset(vtk_object)
        self.src_iges.append(src)
        mlab.pipeline.surface(
            src, opacity=0.4, line_width=0.5, color=(0.5, 0.5, 0.5))
