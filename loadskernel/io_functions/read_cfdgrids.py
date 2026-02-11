from itertools import chain
import logging

import numpy as np
import scipy.io.netcdf as netcdf


class ReadCfdgrids:

    def __init__(self):
        self.cfdgrids = {}
        self.cfdgrid = {}

    def read_surface(self, jcl):
        # Pick up the filename of the grid and the markers from the jcl.
        # The markers are needed to extract the points that belong to the surface(s) of interest.
        if 'surface' in jcl.meshdefo:
            filename_grid = jcl.meshdefo['surface']['filename_grid']
            markers = jcl.meshdefo['surface']['markers']
            # Swich between different file formats as specified in the jcl.
            if 'fileformat' in jcl.meshdefo['surface'] and jcl.meshdefo['surface']['fileformat'].lower() == 'netcdf':
                self.read_netcdf(filename_grid, markers)
            elif 'fileformat' in jcl.meshdefo['surface'] and jcl.meshdefo['surface']['fileformat'].lower() == 'su2':
                self.read_su2(filename_grid, markers)
            else:
                logging.error('jcl.meshdefo["surface"]["fileformat"] must be "netcdf" or "su2"')
        else:
            logging.error('jcl.meshdefo has no key "surface"')

    def read_netcdf(self, filename_grid, markers):
        logging.info('Extracting points belonging to marker(s) %s from grid %s', markers, filename_grid)
        # get all points on surfaces
        ncfile_grid = netcdf.NetCDFFile(filename_grid, 'r')
        boundarymarker_surfaces = ncfile_grid.variables['boundarymarker_of_surfaces'][:]
        points_of_surface = []
        # merge triangles with quadrilaterals
        if 'points_of_surfacetriangles' in ncfile_grid.variables:
            points_of_surface += ncfile_grid.variables['points_of_surfacetriangles'][:].tolist()
        if 'points_of_surfacequadrilaterals' in ncfile_grid.variables:
            points_of_surface += ncfile_grid.variables['points_of_surfacequadrilaterals'][:].tolist()

        # Merge all markers into one cfdgrid
        surfaces = np.array([], dtype=int)
        for marker in markers:
            surfaces = np.hstack((surfaces, np.where(boundarymarker_surfaces == marker)[0]))
        points = np.unique([points_of_surface[s] for s in surfaces])
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

        # Assemble the cfdgrids, one grid for each marker
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
            self.cfdgrids[str(marker)] = cfdgrid
        ncfile_grid.close()

    def read_su2(self, filename_grid, markers=None):
        """
        The description of the SU2 mesh file format is given here: https://su2code.github.io/docs/Mesh-File/
        """
        logging.info('Extracting points belonging to surface marker(s) from grid %s', filename_grid)
        # Open the ascii file and read all lines.
        with open(filename_grid, 'r', encoding='utf-8') as fid:
            lines = fid.readlines()
        # Loop over all lines, if a keyword such as NELEM, NPOIN or MARKER_TAG is found,
        # this announces a new section in the file.
        surface_points = {}
        n_lines = len(lines)
        i = 0
        all_points = None
        while i < n_lines:
            # Splitting lines into parameters and values: According to the mesh specification (and like in all SU2 config
            # files), a '=' is always followed by a space, for example 'PARAMETER_XY= value'. Some mesh generators use a
            # more relaxed syntax like 'PARAMETER_XY=value', meaning that a line needs to be split at the '=' and not the
            # space.
            if str.find(lines[i], 'NELEM') != -1:
                # There is nothing we need to do with the volume element connectivity, so we can skip this section.
                # Skipping all lines (at once) saves a lot of time, since there are many volume elements...
                n_elem = int(lines[i].split('=')[1])
                i += n_elem
            elif str.find(lines[i], 'NPOIN') != -1:
                # Here, the coordinates of all points are given, including surface and volume points.
                n_points = int(lines[i].split('=')[1])
                i += 1
                # Loop over next lines to read all points and their coordinates
                tmp = []
                for x in range(n_points):
                    tmp.append(lines[i + x].split())
                all_points = np.array(tmp, dtype=float)
                i += x
            elif str.find(lines[i], 'MARKER_TAG') != -1:
                # Here, all points are listed that belong to one marker.
                # In addition, the connectivity is given, which we need e.g. for plotting.
                marker = lines[i].split('=')[1].strip()
                i += 1
                n_elem = int(lines[i].split('=')[1])
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
                        logging.error('Surface elements of type "%s" are not implemented!', split_line[0])
                points_of_surface = list(chain.from_iterable(triangles + quadrilaterals))
                i += x
                # Store everything
                surface_points[marker] = {}
                surface_points[marker]['triangles'] = triangles
                surface_points[marker]['quadrilaterals'] = quadrilaterals
                surface_points[marker]['points_of_surface'] = np.unique(points_of_surface)
            i += 1

        # Assemble the cfdgrids, one grid for each marker
        for marker, item in surface_points.items():
            cfdgrid = {}
            cfdgrid['ID'] = item['points_of_surface']
            cfdgrid['CP'] = np.zeros(cfdgrid['ID'].shape)
            cfdgrid['CD'] = np.zeros(cfdgrid['ID'].shape)
            cfdgrid['n'] = len(cfdgrid['ID'])
            cfdgrid['offset'] = all_points[cfdgrid['ID'], :3]
            cfdgrid['set'] = np.arange(6 * cfdgrid['n']).reshape(-1, 6)
            cfdgrid['desc'] = marker
            cfdgrid['triangles'] = item['triangles']
            cfdgrid['quadrilaterals'] = item['quadrilaterals']
            self.cfdgrids[marker] = cfdgrid

        # Merge all desired markers into one cfdgrid
        points_of_all_surfaces = []
        all_triangles = []
        all_quadrilaterals = []
        # If no markers are specified, all markers are merged into one cfdgrid.
        if markers is None:
            markers = surface_points.keys()
        for marker in markers:
            points_of_all_surfaces += surface_points[marker]['points_of_surface'].tolist()
            all_triangles += surface_points[marker]['triangles']
            all_quadrilaterals += surface_points[marker]['quadrilaterals']
        self.cfdgrid['ID'] = np.unique(points_of_all_surfaces)
        self.cfdgrid['CP'] = np.zeros(self.cfdgrid['ID'].shape)
        self.cfdgrid['CD'] = np.zeros(self.cfdgrid['ID'].shape)
        self.cfdgrid['n'] = len(self.cfdgrid['ID'])
        self.cfdgrid['offset'] = all_points[self.cfdgrid['ID'], :3]
        self.cfdgrid['set'] = np.arange(6 * self.cfdgrid['n']).reshape(-1, 6)
        self.cfdgrid['desc'] = markers
        self.cfdgrid['triangles'] = all_triangles
        self.cfdgrid['quadrilaterals'] = all_quadrilaterals
