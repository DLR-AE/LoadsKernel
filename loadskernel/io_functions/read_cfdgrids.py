import h5py
import logging

import numpy as np
import scipy.io.netcdf as netcdf


class ReadCfdgrids:

    def __init__(self, jcl):
        self.jcl = jcl
        if 'surface' in jcl.meshdefo:
            self.filename_grid = self.jcl.meshdefo['surface']['filename_grid']
            self.markers = self.jcl.meshdefo['surface']['markers']
        else:
            logging.error('jcl.meshdefo has no key "surface"')

    def read_surface(self, merge_domains=False):
        if 'fileformat' in self.jcl.meshdefo['surface'] and self.jcl.meshdefo['surface']['fileformat'].lower() == 'cgns':
            self.read_cfdmesh_cgns(merge_domains)
        elif 'fileformat' in self.jcl.meshdefo['surface'] and self.jcl.meshdefo['surface']['fileformat'].lower() == 'netcdf':
            self.read_cfdmesh_netcdf(merge_domains)
        elif 'fileformat' in self.jcl.meshdefo['surface'] and self.jcl.meshdefo['surface']['fileformat'].lower() == 'su2':
            self.read_cfdmesh_su2(merge_domains)
        else:
            logging.error('jcl.meshdefo["surface"]["fileformat"] must be "netcdf", "cgns" or "su2"')
            return

    def read_cfdmesh_cgns(self, merge_domains=False):
        logging.info('Extracting all points from grid {}'.format(self.filename_grid))
        f = h5py.File(self.filename_grid, 'r')
        f_scale = 1.0 / 1000.0  # convert to SI units: 0.001 if mesh is given in [mm], 1.0 if given in [m]
        keys = f['Base'].keys()
        keys.sort()
        self.cfdgrids = []
        if merge_domains:
            x = np.array([])
            y = np.array([])
            z = np.array([])
            for key in keys:
                # loop over domains
                if key[:4] == 'dom-':
                    logging.info(' - {} included'.format(key))
                    domain = f['Base'][key]
                    x = np.concatenate((x, domain['GridCoordinates']['CoordinateX'][' data'][:].reshape(-1) * f_scale))
                    y = np.concatenate((y, domain['GridCoordinates']['CoordinateY'][' data'][:].reshape(-1) * f_scale))
                    z = np.concatenate((z, domain['GridCoordinates']['CoordinateZ'][' data'][:].reshape(-1) * f_scale))
                else:
                    logging.info(' - {} skipped'.format(key))
            f.close()
            n = len(x)
            # build cfdgrid
            cfdgrid = {}
            cfdgrid['ID'] = np.arange(n) + 1
            cfdgrid['CP'] = np.zeros(n)
            cfdgrid['CD'] = np.zeros(n)
            cfdgrid['n'] = n
            cfdgrid['offset'] = np.vstack((x, y, z)).T
            cfdgrid['set'] = np.arange(6 * cfdgrid['n']).reshape(-1, 6)
            cfdgrid['desc'] = 'all domains'
            self.cfdgrids.append(cfdgrid)
        else:
            for key in keys:
                # loop over domains
                if key[:4] == 'dom-':
                    logging.info(' - {} included'.format(key))
                    domain = f['Base'][key]
                    x = domain['GridCoordinates']['CoordinateX'][' data'][:].reshape(-1) * f_scale
                    y = domain['GridCoordinates']['CoordinateY'][' data'][:].reshape(-1) * f_scale
                    z = domain['GridCoordinates']['CoordinateZ'][' data'][:].reshape(-1) * f_scale
                    n = len(x)
                    # build cfdgrid
                    cfdgrid = {}
                    cfdgrid['ID'] = np.arange(n) + 1
                    cfdgrid['CP'] = np.zeros(n)
                    cfdgrid['CD'] = np.zeros(n)
                    cfdgrid['n'] = n
                    cfdgrid['offset'] = np.vstack((x, y, z)).T
                    cfdgrid['set'] = np.arange(6 * cfdgrid['n']).reshape(-1, 6)
                    cfdgrid['desc'] = key
                    self.cfdgrids.append(cfdgrid)
                else:
                    logging.info(' - {} skipped'.format(key))
        f.close()

    def read_cfdmesh_netcdf(self, merge_domains=False):
        markers = self.markers
        logging.info('Extracting points belonging to marker(s) {} from grid {}'.format(str(markers), self.filename_grid))
        # --- get all points on surfaces ---
        ncfile_grid = netcdf.NetCDFFile(self.filename_grid, 'r')
        boundarymarker_surfaces = ncfile_grid.variables['boundarymarker_of_surfaces'][:]
        points_of_surface = []
        # merge triangles with quadrilaterals
        if 'points_of_surfacetriangles' in ncfile_grid.variables:
            points_of_surface += ncfile_grid.variables['points_of_surfacetriangles'][:].tolist()
        if 'points_of_surfacequadrilaterals' in ncfile_grid.variables:
            points_of_surface += ncfile_grid.variables['points_of_surfacequadrilaterals'][:].tolist()

        if merge_domains:
            # --- get points on surfaces according to marker ---
            surfaces = np.array([], dtype=int)
            for marker in markers:
                surfaces = np.hstack((surfaces, np.where(boundarymarker_surfaces == marker)[0]))
            points = np.unique([points_of_surface[s] for s in surfaces])
            # build cfdgrid
            self.cfdgrid = {}
            self.cfdgrid['ID'] = points
            self.cfdgrid['CP'] = np.zeros(self.cfdgrid['ID'].shape)
            self.cfdgrid['CD'] = np.zeros(self.cfdgrid['ID'].shape)
            self.cfdgrid['n'] = len(self.cfdgrid['ID'])
            self.cfdgrid['offset'] = np.vstack((ncfile_grid.variables['points_xc'][:][points].copy(),
                                                ncfile_grid.variables['points_yc'][:][points].copy(),
                                                ncfile_grid.variables['points_zc'][:][points].copy())).T
            self.cfdgrid['set'] = np.arange(6 * self.cfdgrid['n']).reshape(-1, 6)
            self.cfdgrid['desc'] = markers
            self.cfdgrid['points_of_surface'] = [points_of_surface[s] for s in surfaces]
        else:
            self.cfdgrids = []
            for marker in markers:
                # --- get points on surfaces according to marker ---
                surfaces = np.where(boundarymarker_surfaces == marker)[0]
                points = np.unique([points_of_surface[s] for s in surfaces])
                # build cfdgrid
                cfdgrid = {}
                cfdgrid['ID'] = points
                cfdgrid['CP'] = np.zeros(cfdgrid['ID'].shape)
                cfdgrid['CD'] = np.zeros(cfdgrid['ID'].shape)
                cfdgrid['n'] = len(cfdgrid['ID'])
                cfdgrid['offset'] = np.vstack((ncfile_grid.variables['points_xc'][:][points].copy(),
                                               ncfile_grid.variables['points_yc'][:][points].copy(),
                                               ncfile_grid.variables['points_zc'][:][points].copy())).T
                cfdgrid['set'] = np.arange(6 * cfdgrid['n']).reshape(-1, 6)
                cfdgrid['desc'] = str(marker)
                cfdgrid['points_of_surface'] = [points_of_surface[s] for s in surfaces]
                self.cfdgrids.append(cfdgrid)
        ncfile_grid.close()

    def read_cfdmesh_su2(self, merge_domains=False):
        """
        The description of the SU2 mesh file format is given here: https://su2code.github.io/docs/Mesh-File/
        """
        logging.info('Extracting points belonging to surface marker(s) from grid {}'.format(self.filename_grid))
        # Open the ascii file and read all lines.
        with open(self.filename_grid, 'r') as fid:
            lines = fid.readlines()
        # Loop over all lines, if a keyword such as NELEM, NPOIN or MARKER_TAG is found,
        # this announces a new section in the file.
        surface_points = {}
        n_lines = len(lines)
        i = 0
        while i < n_lines:
            if str.find(lines[i], 'NELEM') != -1:
                n_elem = int(lines[i].split()[1])
                # There is nothing we need to do with the volume element connectivity, so we can skip this section.
                # Skipping all lines (at once) saves a lot of time, since there are many volume elements...
                i += n_elem
            elif str.find(lines[i], 'NPOIN') != -1:
                # New section found.
                # Here, the coordinates of all points are given, including surface and volume points.
                n_points = int(lines[i].split()[1])
                i += 1
                # Loop over next lines to read all points and their coordinates
                tmp = []
                for x in range(n_points):
                    tmp.append(lines[i + x].split())
                points = np.array(tmp, dtype=np.float)
                i += x
            elif str.find(lines[i], 'MARKER_TAG') != -1:
                # New section found.
                # Here, all points are listed that belong to one marker.
                # In addition, the connectivity is given, which we need e.g. for plotting.
                marker = lines[i].split()[1]
                i += 1
                n_elem = int(lines[i].split()[1])
                i += 1
                # Loop over next lines to read all surface points
                triangles = []
                quadrilaterals = []
                for x in range(n_elem):
                    split_line = lines[i + x].split()
                    if split_line[0] == '5':
                        # Triangle elements are identified with a 5
                        triangles.append([int(id) for id in split_line[1:]])
                    elif split_line[0] == '9':
                        # Quadrilateral elements are identified with a 9
                        quadrilaterals.append([int(id) for id in split_line[1:]])
                    else:
                        logging.error('Surface elements of type "{}" are not implemented!'.format(split_line[0]))

                i += x
                # Store everything
                surface_points[marker] = {}
                surface_points[marker]['points'] = np.unique(triangles + quadrilaterals)
                surface_points[marker]['points_of_surface'] = triangles + quadrilaterals
            i += 1

        if merge_domains:
            # First we need to do some merging...
            tmp_points = np.array([], dtype=int)
            tmp_surfaces = []
            for marker in self.markers:
                tmp_points = np.hstack((tmp_points, surface_points[marker]['points']))
                tmp_surfaces += surface_points[marker]['points_of_surface']

            # Assemble the cfdgrid
            self.cfdgrid = {}
            self.cfdgrid['ID'] = np.unique(tmp_points)
            self.cfdgrid['CP'] = np.zeros(self.cfdgrid['ID'].shape)
            self.cfdgrid['CD'] = np.zeros(self.cfdgrid['ID'].shape)
            self.cfdgrid['n'] = len(self.cfdgrid['ID'])
            self.cfdgrid['offset'] = points[self.cfdgrid['ID'], :3]
            self.cfdgrid['set'] = np.arange(6 * self.cfdgrid['n']).reshape(-1, 6)
            self.cfdgrid['desc'] = self.markers
            self.cfdgrid['points_of_surface'] = tmp_surfaces
        else:
            self.cfdgrids = {}
            for marker in surface_points.keys():
                # Assemble the cfdgrid
                cfdgrid = {}
                cfdgrid['ID'] = surface_points[marker]['points']
                cfdgrid['CP'] = np.zeros(cfdgrid['ID'].shape)
                cfdgrid['CD'] = np.zeros(cfdgrid['ID'].shape)
                cfdgrid['n'] = len(cfdgrid['ID'])
                cfdgrid['offset'] = points[cfdgrid['ID'], :3]
                cfdgrid['set'] = np.arange(6 * cfdgrid['n']).reshape(-1, 6)
                cfdgrid['desc'] = marker
                cfdgrid['points_of_surface'] = surface_points[marker]['points_of_surface']
                self.cfdgrids[marker] = cfdgrid
