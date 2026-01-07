import numpy as np
import pyvista as pv

from loadskernel.io_functions.data_handling import load_hdf5_dict


class Plotting:

    def __init__(self):
        pass

    def plot_nothing(self):
        # Clear the plotter and reset all show flags
        if hasattr(self, 'plotter') and self.plotter is not None:
            self.plotter.clear()
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

    def add_figure(self, plotter):
        # Store the PyVista plotter and set background
        self.plotter = plotter
        self.plotter.set_background('white')
        self.plot_nothing()

    def add_model(self, model):
        # Load all grids and model data
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
        # Calculate the focal point for the camera
        self.focalpoint = (self.strcgrid['offset'].min(
            axis=0) + self.strcgrid['offset'].max(axis=0)) / 2.0

    def set_view_left_above(self):
        # Set a custom view
        if hasattr(self, 'plotter') and self.plotter is not None:
            self.plotter.view_vector((1, 1, 1))

    def set_view_back(self):
        # Set a back view
        if hasattr(self, 'plotter') and self.plotter is not None:
            self.plotter.view_vector((-1, 0, 0))

    def set_view_side(self):
        # Set a side view
        if hasattr(self, 'plotter') and self.plotter is not None:
            self.plotter.view_vector((0, 1, 0))

    def set_view_top(self):
        # Set a top view
        if hasattr(self, 'plotter') and self.plotter is not None:
            self.plotter.view_vector((0, 0, 1))

    def set_view(self):
        # Set a default view (xy-plane)
        if hasattr(self, 'plotter') and self.plotter is not None:
            self.plotter.camera_position = 'xy'

    def hide_masses(self):
        # Hide mass glyphs
        self.show_masses = False
        self.plot_nothing()

    def plot_masses(self, MGG, Mb, cggrid, rho=2700.0):
        # get nodal masses
        m_cg = Mb[0, 0]
        m = MGG.diagonal()[0::6]
        radius_mass_cg = ((m_cg * 3.) / (4. * rho * np.pi)) ** (1. / 3.)
        radius_masses = ((m * 3.) / (4. * rho * np.pi)) ** (1. / 3.)

        if not hasattr(self, 'plotter') or self.plotter is None:
            return

        self.plotter.clear()
        # Plot nodal masses as points
        points = self.strcgrid['offset']
        cloud = pv.PolyData(points)
        cloud['radius'] = radius_masses
        self.plotter.add_mesh(cloud, color='orange', point_size=10, render_points_as_spheres=True)
        # Plot CG as a larger point
        cg_cloud = pv.PolyData(cggrid['offset'])
        cg_cloud['radius'] = [radius_mass_cg]
        self.plotter.add_mesh(cg_cloud, color='yellow', point_size=20, render_points_as_spheres=True, opacity=0.3)
        self.show_masses = True
        self.plotter.reset_camera()

    def setup_mass_display(self, radius_masses, radius_mass_cg, cggrid):
        # Not needed in PyVista, handled in plot_masses
        pass

    def update_mass_display(self, radius_masses, radius_mass_cg, cggrid):
        # Just replot
        self.plot_masses(radius_masses, radius_mass_cg, cggrid)

    def hide_strc(self):
        # Hide structure
        self.show_strc = False
        self.plot_nothing()

    def plot_strc(self):
        # Plot structure as points
        if not hasattr(self, 'plotter') or self.plotter is None:
            return
        self.plotter.clear()
        points = self.strcgrid['offset']
        cloud = pv.PolyData(points)
        self.plotter.add_mesh(cloud, color='blue', point_size=10, render_points_as_spheres=True)
        self.show_strc = True
        self.plotter.reset_camera()

    def hide_mode(self):
        # Hide mode shape
        self.show_mode = False
        self.plot_nothing()

    def plot_mode(self, offsets):
        # Plot mode shape as points
        if not hasattr(self, 'plotter') or self.plotter is None:
            return
        self.plotter.clear()
        cloud = pv.PolyData(offsets)
        self.plotter.add_mesh(cloud, color='green', point_size=10, render_points_as_spheres=True)
        self.show_mode = True
        self.plotter.reset_camera()

    def setup_strc_display(self, offsets, color, p_scale):
        # Not needed in PyVista, handled in plot_strc/plot_mode
        pass

    def update_mode_display(self, offsets):
        # Just replot
        self.plot_mode(offsets)

    def hide_aero(self):
        # Hide aerodynamic grid and MAC
        self.show_aero = False
        self.plot_nothing()
        self.show_panel_normal_vectors = False

    def plot_aero(self, scalars=None, colormap='coolwarm', vminmax=[-10.0, 10.0]):
        # Plot aerodynamic grid as points, optionally colored by scalars
        if not hasattr(self, 'plotter') or self.plotter is None:
            return
        self.plotter.clear()
        points = self.model['aerogrid']['offset']
        cloud = pv.PolyData(points)
        if scalars is not None:
            cloud['scalars'] = scalars
            self.plotter.add_mesh(cloud, scalars='scalars', cmap=colormap, clim=vminmax, point_size=10, render_points_as_spheres=True)
        else:
            self.plotter.add_mesh(cloud, color='red', point_size=10, render_points_as_spheres=True)
        self.show_aero = True
        self.plotter.reset_camera()

    def plot_panel_normal_vectors(self):
        # Plot normal vectors as arrows
        if not hasattr(self, 'plotter') or self.plotter is None:
            return
        points = self.model['aerogrid']['offset_k']
        normals = self.model['aerogrid']['N']
        # PyVista expects a single vector for all points, so use glyphs
        cloud = pv.PolyData(points)
        cloud['vectors'] = normals
        arrows = cloud.glyph(orient='vectors', scale=False, factor=1.0)
        self.plotter.add_mesh(arrows, color='green', opacity=0.4)
        self.show_panel_normal_vectors = True

    def setup_mac_display(self):
        # Plot MAC as a point
        if not hasattr(self, 'plotter') or self.plotter is None:
            return
        if hasattr(self, 'MAC'):
            mac_cloud = pv.PolyData(np.array([self.MAC]))
            self.plotter.add_mesh(mac_cloud, color='red', point_size=20, render_points_as_spheres=True, opacity=0.4)

    def setup_aero_display(self, scalars, colormap, vminmax):
        # Just call plot_aero
        self.plot_aero(scalars, colormap, vminmax)

    def hide_cfdgrids(self):
        # Hide CFD grids
        self.show_cfdgrids = False
        self.plot_nothing()

    def plot_cfdgrids(self, markers):
        # Plot selected CFD grids as points
        if not hasattr(self, 'plotter') or self.plotter is None:
            return
        self.plotter.clear()
        for marker in self.cfdgrids:
            if marker in markers:
                points = self.cfdgrids[marker]['offset']
                cloud = pv.PolyData(points)
                self.plotter.add_mesh(cloud, color='magenta', point_size=8, render_points_as_spheres=True)
        self.show_cfdgrids = True
        self.plotter.reset_camera()

    def setup_cfdgrid_display(self, grid, color, scalars):
        # Not needed in PyVista, handled in plot_cfdgrids
        pass

    def hide_aero_strc_coupling(self):
        # Hide coupling visualization
        self.show_coupling = False
        self.plot_nothing()

    def plot_aero_strc_coupling(self):
        # Plot lines between aero and strc grid points
        if not hasattr(self, 'plotter') or self.plotter is None:
            return
        self.plotter.clear()
        aero_points = self.model['aerogrid']['offset']
        strc_points = self.strcgrid['offset']
        for a, s in zip(aero_points, strc_points):
            line = pv.Line(a, s)
            self.plotter.add_mesh(line, color='black')
        self.show_coupling = True
        self.plotter.reset_camera()

    def plot_splinegrids(self, grid_i, set_i, grid_d, set_d):
        # Plot two sets of points for spline grids
        if not hasattr(self, 'plotter') or self.plotter is None:
            return
        self.plotter.clear()
        cloud_i = pv.PolyData(grid_i)
        cloud_d = pv.PolyData(grid_d)
        self.plotter.add_mesh(cloud_i, color='cyan', point_size=8, render_points_as_spheres=True)
        self.plotter.add_mesh(cloud_d, color='yellow', point_size=8, render_points_as_spheres=True)
        self.plotter.reset_camera()

    def plot_splinerules(self, grid_i, set_i, grid_d, set_d, splinerules, coord):
        # Plot lines for splinerules
        if not hasattr(self, 'plotter') or self.plotter is None:
            return
        self.plotter.clear()
        for rule in splinerules:
            line = pv.Line(grid_i[rule[0]], grid_d[rule[1]])
            self.plotter.add_mesh(line, color='gray')
        self.plotter.reset_camera()

    def hide_monstations(self):
        # Hide monitoring stations
        self.show_monstations = False
        self.plot_nothing()

    def plot_monstations(self, monstation_id):
        # Plot monitoring stations as points
        if not hasattr(self, 'plotter') or self.plotter is None:
            return
        self.plotter.clear()
        points = self.model['monstations'][monstation_id]['offset']
        cloud = pv.PolyData(points)
        self.plotter.add_mesh(cloud, color='purple', point_size=12, render_points_as_spheres=True)
        self.show_monstations = True
        self.plotter.reset_camera()

    def hide_cs(self):
        # Hide control surface
        self.show_cs = False
        self.plot_nothing()

    def plot_cs(self, i_surf, axis, deg):
        # Plot a cross-section as a polyline
        if not hasattr(self, 'plotter') or self.plotter is None:
            return
        # determine deflections
        if axis == 'y-axis':
            Uj = np.dot(self.Djx2[i_surf], [0, 0, 0, 0, deg / 180.0 * np.pi, 0])
        elif axis == 'z-axis':
            Uj = np.dot(self.Djx2[i_surf], [0, 0, 0, 0, 0, deg / 180.0 * np.pi])
        else:
            Uj = np.dot(self.Djx2[i_surf], [0, 0, 0, 0, 0, 0])
        # find those panels belonging to the current control surface i_surf
        members_of_i_surf = [np.where(self.aerogrid['ID'] == x)[0][0] for x in self.x2grid[str(i_surf)]['ID'][()]]
        points = self.aerogrid['offset_k'][members_of_i_surf, :] \
            + Uj[self.aerogrid['set_k'][members_of_i_surf, :][:, (0, 1, 2)]]
        polyline = pv.lines_from_points(points)
        self.plotter.clear()
        self.plotter.add_mesh(polyline, color='green', line_width=3)
        self.show_cs = True
        self.plotter.reset_camera()

    def setup_cs_display(self, points):
        # Not needed in PyVista, handled in plot_cs
        pass

    def update_cs_display(self, points):
        # Just replot
        self.plot_cs(points)

    def hide_cell(self):
        # Hide cell data
        self.show_cell = False
        self.plot_nothing()

    def plot_cell(self, cell_data, show_cells):
        # Plot cell data as colored surfaces
        if not hasattr(self, 'plotter') or self.plotter is None:
            return
        self.plotter.clear()
        for cell in cell_data:
            mesh = pv.PolyData(cell['points'])
            self.plotter.add_mesh(mesh, color='orange', opacity=0.5)
        self.show_cell = True
        self.plotter.reset_camera()

    def setup_cell_display(self, offsets, color, p_scale, cell_data, show_cells):
        # Not needed in PyVista, handled in plot_cell
        pass

    def update_cell_display(self, cell_data):
        # Just replot
        self.plot_cell(cell_data, show_cells=True)

    def hide_iges(self):
        # Hide IGES meshes
        self.show_iges = False
        self.plot_nothing()

    def plot_iges(self, selected_meshes):
        # Plot selected IGES meshes
        if not hasattr(self, 'plotter') or self.plotter is None:
            return
        self.plotter.clear()
        for mesh in self.iges_meshes:
            if mesh['desc'] in selected_meshes:
                self.setup_iges_display(mesh['vtk'])
        self.show_iges = True
        self.plotter.reset_camera()

    def setup_iges_display(self, vtk_object):
        # Add IGES mesh to plotter
        self.plotter.add_mesh(vtk_object, color='gray', opacity=0.4, line_width=0.5)
