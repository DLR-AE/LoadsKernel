# -*- coding: utf-8 -*-
import os
import numpy as np
# To be able to use PySide or PyQt4 and not run in conflicts with traits,
# we need to import QtGui and QtCore from pyface.qt
from pyface.qt import QtGui, QtCore

from traits.api import HasTraits, Instance, on_trait_change
from traitsui.api import View, Item
from mayavi.core.ui.api import MayaviScene, MlabSceneModel, SceneEditor

from loadskernel.io_functions import data_handling
from loadskernel.io_functions.data_handling import load_hdf5_sparse_matrix, load_hdf5_dict
from modelviewer.plotting import Plotting
from modelviewer.pytran import NastranSOL101
from modelviewer.cfdgrid import TauGrid, SU2Grid
from modelviewer.iges import IgesMesh


class Visualization(HasTraits):
    scene = Instance(MlabSceneModel, ())

    @on_trait_change('scene.activated')
    def update_plot(self):
        # This function is called when the view is opened. We don't
        # populate the scene when the view is not yet open, as some
        # VTK features require a GLContext.

        # We can do normal mlab calls on the embedded scene.
        return self.scene.mlab.gcf()

    # the layout of the dialog screated
    view = View(Item('scene', editor=SceneEditor(scene_class=MayaviScene),
                     height=600, width=600, show_label=False),
                resizable=True  # We need this to resize with the parent widget
                )


class MayaviQWidget(QtGui.QWidget):

    def __init__(self, parent=None):
        QtGui.QWidget.__init__(self, parent)
        layout = QtGui.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.visualization = Visualization()
        # The edit_traits call will generate the widget to embed.
        self.ui = self.visualization.edit_traits(parent=self,
                                                 kind='subpanel').control
        layout.addWidget(self.ui)
        self.ui.setParent(self)


class Modelviewer():

    def __init__(self):

        # define file options
        self.file_opt = {}
        self.file_opt['filters'] = "Loads Kernel files (model*.hdf5);;all HDF5 files (*.hdf5);;all files (*.*)"
        self.file_opt['initialdir'] = os.getcwd()
        self.file_opt['title'] = 'Open a Loads Kernel Model'

        # define file options
        self.hdf5_opt = {}
        self.hdf5_opt['filters'] = "Nastran HDF5 files (*.h5);;all files (*.*)"
        self.hdf5_opt['initialdir'] = os.getcwd()
        self.hdf5_opt['title'] = 'Open a Nastran HDF5 File'

        # define file options
        self.nc_opt = {}
        self.nc_opt['filters'] = "all files (*.*)"
        self.nc_opt['initialdir'] = os.getcwd()
        self.nc_opt['title'] = 'Open a Grid File'

        # define file options
        self.iges_opt = {}
        self.iges_opt['filters'] = "IGES files (*.igs *.iges);;all files (*.*)"
        self.iges_opt['initialdir'] = os.getcwd()
        self.iges_opt['title'] = 'Open an IGES File'

        self.plotting = Plotting()
        self.nastran = NastranSOL101()

        self.iges = IgesMesh()

    def run(self):
        # Don't create a new QApplication, it would unhook the Events
        # set by Traits on the existing QApplication. Simply use the
        # '.instance()' method to retrieve the existing one.
        app = QtGui.QApplication.instance()
        # Init the application's menues, tabs, etc.
        self.initGUI()
        # Start the main event loop.
        app.exec_()

    def test(self):
        """
        This function is intended for CI testing. To test at least some parts of the code, the app is initialized, but
        never started. Instead, all windows are closed again.
        """
        app = QtGui.QApplication.instance()
        self.initGUI()
        app.closeAllWindows()

    def initGUI(self):
        # Use one Widget as a main container.
        self.container = QtGui.QWidget()
        # Init all sub-widgets.
        self.initTabs()
        self.initMayaviFigure()
        self.initWindow()
        # Arrange the layout inside the container.
        layout = QtGui.QGridLayout(self.container)
        layout.addWidget(self.tabs_widget, 0, 0)
        layout.addWidget(self.mayavi_widget, 0, 1)

    def initTabs(self):
        # Configure tabs widget
        self.tabs_widget = QtGui.QTabWidget()
        sizePolicy = QtGui.QSizePolicy(
            QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Preferred)
        self.tabs_widget.setSizePolicy(sizePolicy)
        self.tabs_widget.setMinimumWidth(300)
        self.tabs_widget.setMaximumWidth(450)

        # Add tabs
        self.initStrcTab()
        self.initMassTab()
        self.initAeroTab()
        self.initCSTab()
        self.initCouplingTab()
        self.initMonstationsTab()
        self.initPytranTab()
        self.initIgesTab()

    def initStrcTab(self):
        tab_strc = QtGui.QWidget()
        self.tabs_widget.addTab(tab_strc, 'strc')
        # Elements of strc tab
        lb_undeformed = QtGui.QLabel('Undeformed')
        bt_strc_show = QtGui.QPushButton('Show')
        bt_strc_show.clicked.connect(self.plotting.plot_strc)
        bt_strc_hide = QtGui.QPushButton('Hide')
        bt_strc_hide.clicked.connect(self.plotting.hide_strc)
        # lists of mass case and mode number
        lb_modes_mass = QtGui.QLabel('Mass')
        lb_modes_number = QtGui.QLabel('Modes')
        self.list_modes_mass = QtGui.QListWidget()
        self.list_modes_mass.itemSelectionChanged.connect(self.update_modes)
        self.list_modes_number = QtGui.QListWidget()
        self.list_modes_number.itemSelectionChanged.connect(
            self.get_mode_data_for_plotting)
        self.lb_freq = QtGui.QLabel('Frequency: {:0.4f} Hz'.format(0.0))
        self.lb_uf = QtGui.QLabel('Scaling: 1.0')
        # slider for generalized coordinate magnification factor
        self.sl_uf = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.sl_uf.setMinimum(-50)
        self.sl_uf.setMaximum(+50)
        self.sl_uf.setSingleStep(1)
        self.sl_uf.setValue(5)
        self.sl_uf.setTickPosition(QtGui.QSlider.TicksBelow)
        self.sl_uf.setTickInterval(5)
        self.sl_uf.valueChanged.connect(self.get_mode_data_for_plotting)
        bt_mode_hide = QtGui.QPushButton('Hide')
        bt_mode_hide.clicked.connect(self.plotting.hide_mode)

        layout_strc = QtGui.QGridLayout(tab_strc)
        layout_strc.addWidget(lb_undeformed, 0, 0, 1, -1)
        layout_strc.addWidget(bt_strc_show, 1, 0, 1, -1)
        layout_strc.addWidget(bt_strc_hide, 2, 0, 1, -1)
        layout_strc.addWidget(lb_modes_mass, 3, 0, 1, 1)
        layout_strc.addWidget(lb_modes_number, 3, 1, 1, 1)
        layout_strc.addWidget(self.list_modes_mass, 4, 0, 1, 1)
        layout_strc.addWidget(self.list_modes_number, 4, 1, 1, 1)
        layout_strc.addWidget(self.lb_freq, 5, 0, 1, -1)
        layout_strc.addWidget(self.lb_uf, 6, 0, 1, -1)
        layout_strc.addWidget(self.sl_uf, 7, 0, 1, -1)
        layout_strc.addWidget(bt_mode_hide, 8, 0, 1, -1)

    def initMassTab(self):
        tab_mass = QtGui.QWidget()
        self.tabs_widget.addTab(tab_mass, "mass")
        # Elements of mass tab
        self.list_mass = QtGui.QListWidget()
        self.list_mass.itemSelectionChanged.connect(
            self.get_mass_data_for_plotting)
        self.lb_rho = QtGui.QLabel('Rho: 2700 kg/m^3')
        # slider for generalized coordinate magnification factor
        self.sl_rho = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.sl_rho.setMinimum(10)
        self.sl_rho.setMaximum(3000)
        self.sl_rho.setSingleStep(100)
        self.sl_rho.setValue(2700)
        self.sl_rho.setTickPosition(QtGui.QSlider.TicksBelow)
        self.sl_rho.setTickInterval(500)
        self.sl_rho.valueChanged.connect(self.get_mass_data_for_plotting)
        self.lb_cg = QtGui.QLabel(
            'CG: x={:0.4f}, y={:0.4f}, z={:0.4f} m'.format(0.0, 0.0, 0.0))
        self.lb_cg_mac = QtGui.QLabel('CG: x={:0.4f} % MAC'.format(0.0))
        self.lb_mass = QtGui.QLabel('Mass: {:0.2f} kg'.format(0.0))
        self.lb_Ixx = QtGui.QLabel('Ixx:  {:0.4g} kg m^2'.format(0.0))
        self.lb_Iyy = QtGui.QLabel('Iyy:  {:0.4g} kg m^2'.format(0.0))
        self.lb_Izz = QtGui.QLabel('Izz:  {:0.4g} kg m^2'.format(0.0))
        bt_mass_hide = QtGui.QPushButton('Hide')
        bt_mass_hide.clicked.connect(self.plotting.hide_masses)

        layout_mass = QtGui.QVBoxLayout(tab_mass)
        layout_mass.addWidget(self.list_mass)
        layout_mass.addWidget(self.lb_cg)
        layout_mass.addWidget(self.lb_cg_mac)
        layout_mass.addWidget(self.lb_mass)
        layout_mass.addWidget(self.lb_Ixx)
        layout_mass.addWidget(self.lb_Iyy)
        layout_mass.addWidget(self.lb_Izz)
        layout_mass.addWidget(self.lb_rho)
        layout_mass.addWidget(self.sl_rho)
        layout_mass.addWidget(bt_mass_hide)

    def initAeroTab(self):
        tab_aero = QtGui.QWidget()
        self.tabs_widget.addTab(tab_aero, "aero")
        # Elements of aero tab
        self.list_aero = QtGui.QListWidget()
        self.list_aero.itemSelectionChanged.connect(self.get_aero_for_plotting)
        self.cb_w2gj = QtGui.QCheckBox('Color by W2GJ [deg]')
        self.cb_w2gj.setChecked(False)
        self.cb_w2gj.stateChanged.connect(self.get_aero_for_plotting)
        self.cb_normal_vectors = QtGui.QCheckBox('Panel normal vectors')
        self.cb_normal_vectors.setChecked(False)
        self.cb_normal_vectors.stateChanged.connect(self.get_aero_for_plotting)
        bt_aero_hide = QtGui.QPushButton('Hide')
        bt_aero_hide.clicked.connect(self.plotting.hide_aero)
        self.lb_MAC = QtGui.QLabel(
            'MAC: x={:0.4f}, y={:0.4f} m'.format(0.0, 0.0))
        self.lb_MAC2 = QtGui.QLabel('')

        self.list_markers = QtGui.QListWidget()
        self.list_markers.setSelectionMode(
            QtGui.QAbstractItemView.ExtendedSelection)  # allow multiple selections
        self.list_markers.itemSelectionChanged.connect(
            self.get_new_markers_for_plotting)
        bt_cfdgrid_hide = QtGui.QPushButton('Hide CFD Grids')
        bt_cfdgrid_hide.clicked.connect(self.plotting.hide_cfdgrids)

        layout_aero = QtGui.QVBoxLayout(tab_aero)
        layout_aero.addWidget(self.list_aero)
        layout_aero.addWidget(self.lb_MAC)
        layout_aero.addWidget(self.lb_MAC2)
        layout_aero.addWidget(self.cb_w2gj)
        layout_aero.addWidget(self.cb_normal_vectors)
        layout_aero.addWidget(bt_aero_hide)
        layout_aero.addWidget(self.list_markers)
        layout_aero.addWidget(bt_cfdgrid_hide)

    def initCouplingTab(self):
        tab_coupling = QtGui.QWidget()
        self.tabs_widget.addTab(tab_coupling, "coupling")
        # Elements of coupling tab
        bt_coupling_show = QtGui.QPushButton('Show')
        bt_coupling_show.clicked.connect(self.plotting.plot_aero_strc_coupling)
        bt_coupling_hide = QtGui.QPushButton('Hide')
        bt_coupling_hide.clicked.connect(self.plotting.hide_aero_strc_coupling)

        layout_coupling = QtGui.QVBoxLayout(tab_coupling)
        layout_coupling.addWidget(bt_coupling_show)
        layout_coupling.addWidget(bt_coupling_hide)
        layout_coupling.addStretch(1)

    def initMonstationsTab(self):
        tab_monstations = QtGui.QWidget()
        self.tabs_widget.addTab(tab_monstations, "monstations")
        # Elements of monstations tab
        self.list_monstations = QtGui.QListWidget()
        self.list_monstations.itemSelectionChanged.connect(
            self.get_monstation_for_plotting)
        self.lb_monstation_coord = QtGui.QLabel('Coord:')
        bt_monstations_hide = QtGui.QPushButton('Hide')
        bt_monstations_hide.clicked.connect(self.plotting.hide_monstations)

        layout_monstations = QtGui.QVBoxLayout(tab_monstations)
        layout_monstations.addWidget(self.list_monstations)
        layout_monstations.addWidget(self.lb_monstation_coord)
        layout_monstations.addWidget(bt_monstations_hide)

    def initCSTab(self):
        tab_cs = QtGui.QWidget()
        self.tabs_widget.addTab(tab_cs, "cs")
        # Elements of cs tab
        self.list_cs = QtGui.QListWidget()
        self.list_cs.itemSelectionChanged.connect(self.get_new_cs_for_plotting)
        self.lb_deg = QtGui.QLabel('Deflection: 0 deg')
        # slider for generalized coordinate magnification factor
        self.sl_deg = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.sl_deg.setMinimum(-30)
        self.sl_deg.setMaximum(30)
        self.sl_deg.setSingleStep(10)
        self.sl_deg.setValue(0)
        self.sl_deg.setTickPosition(QtGui.QSlider.TicksBelow)
        self.sl_deg.setTickInterval(10)
        self.sl_deg.valueChanged.connect(self.get_cs_data_for_plotting)
        self.cb_axis = QtGui.QComboBox()
        self.cb_axis.addItems(['y-axis', 'z-axis'])
        self.cb_axis.currentIndexChanged.connect(self.get_cs_data_for_plotting)
        bt_cs_hide = QtGui.QPushButton('Hide')
        bt_cs_hide.clicked.connect(self.plotting.hide_cs)

        layout_cs = QtGui.QVBoxLayout(tab_cs)
        layout_cs.addWidget(self.list_cs)
        layout_cs.addWidget(self.lb_deg)
        layout_cs.addWidget(self.sl_deg)
        layout_cs.addWidget(self.cb_axis)
        layout_cs.addWidget(bt_cs_hide)

    def initPytranTab(self):
        tab_pytran = QtGui.QWidget()
        self.tabs_widget.addTab(tab_pytran, "pytran")
        # Elements of results tab
        self.list_celldata = QtGui.QListWidget()
        self.list_celldata.itemSelectionChanged.connect(
            self.get_new_cell_data_for_plotting)
        self.list_show_cells = QtGui.QListWidget()
        self.list_show_cells.setSelectionMode(
            QtGui.QAbstractItemView.ExtendedSelection)
        self.list_show_cells.itemSelectionChanged.connect(
            self.get_new_cell_data_for_plotting)

        bt_cell_hide = QtGui.QPushButton('Hide Nastran results')
        bt_cell_hide.clicked.connect(self.plotting.hide_cell)

        layout_pytran = QtGui.QGridLayout(tab_pytran)
        layout_pytran.addWidget(self.list_celldata, 0, 0, 1, 1)
        layout_pytran.addWidget(self.list_show_cells, 0, 1, 1, 1)
        layout_pytran.addWidget(bt_cell_hide, 1, 0, 1, -1)

    def initIgesTab(self):
        tab_iges = QtGui.QWidget()
        self.tabs_widget.addTab(tab_iges, "iges")

        self.list_iges = QtGui.QListWidget()
        self.list_iges.setSelectionMode(
            QtGui.QAbstractItemView.ExtendedSelection)  # allow multiple selections
        self.list_iges.itemSelectionChanged.connect(self.get_iges_for_plotting)
        bt_iges_hide = QtGui.QPushButton('Hide IGES')
        bt_iges_hide.clicked.connect(self.plotting.hide_iges)

        layout_iges = QtGui.QVBoxLayout(tab_iges)
        layout_iges.addWidget(self.list_iges)
        layout_iges.addWidget(bt_iges_hide)

    def initMayaviFigure(self):
        # ----------------------------
        # --- set up Mayavi Figure ---
        # ----------------------------
        self.mayavi_widget = MayaviQWidget(self.container)
        sizePolicy = QtGui.QSizePolicy(
            QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        self.mayavi_widget.setSizePolicy(sizePolicy)
        fig = self.mayavi_widget.visualization.update_plot()
        self.plotting.add_figure(fig)

    def initWindow(self):
        # ------------------------------
        # --- set up window and menu ---
        # ------------------------------
        self.window = QtGui.QMainWindow()
        mainMenu = self.window.menuBar()
        fileMenu = mainMenu.addMenu('File')
        # Add load button
        loadButtonModel = QtGui.QAction('Load model', self.window)
        loadButtonModel.setShortcut('Ctrl+L')
        loadButtonModel.triggered.connect(self.load_model)
        fileMenu.addAction(loadButtonModel)

        self.loadButtonNastran = QtGui.QAction(
            'Load Nastran results', self.window)
        self.loadButtonNastran.setShortcut('Ctrl+R')
        self.loadButtonNastran.setDisabled(True)
        self.loadButtonNastran.triggered.connect(self.load_nastran_results)
        fileMenu.addAction(self.loadButtonNastran)

        self.loadButtonTauGrid = QtGui.QAction('Load Tau Grid', self.window)
        self.loadButtonTauGrid.setShortcut('Ctrl+T')
        self.loadButtonTauGrid.triggered.connect(self.load_tau_grid)
        fileMenu.addAction(self.loadButtonTauGrid)

        self.loadButtonSU2Grid = QtGui.QAction('Load SU2 Grid', self.window)
        self.loadButtonSU2Grid.setShortcut('Ctrl+S')
        self.loadButtonSU2Grid.triggered.connect(self.load_su2_grid)
        fileMenu.addAction(self.loadButtonSU2Grid)

        self.loadButtonIges = QtGui.QAction('Load IGES', self.window)
        self.loadButtonIges.setShortcut('Ctrl+I')
        self.loadButtonIges.triggered.connect(self.load_iges)
        fileMenu.addAction(self.loadButtonIges)

        # Add exit button
        exitButton = QtGui.QAction('Exit', self.window)
        exitButton.setShortcut('Ctrl+Q')
        exitButton.triggered.connect(self.window.close)
        fileMenu.addAction(exitButton)

        self.viewMenu = mainMenu.addMenu('View')
        self.viewMenu.setEnabled(False)
        # Add view buttons

        bt_view_left_above = QtGui.QAction('Left Above', self.window)
        bt_view_left_above.triggered.connect(self.plotting.set_view_left_above)
        self.viewMenu.addAction(bt_view_left_above)

        bt_view_top = QtGui.QAction('Top', self.window)
        bt_view_top.triggered.connect(self.plotting.set_view_top)
        self.viewMenu.addAction(bt_view_top)

        bt_view_back = QtGui.QAction('Back', self.window)
        bt_view_back.triggered.connect(self.plotting.set_view_back)
        self.viewMenu.addAction(bt_view_back)

        bt_view_side = QtGui.QAction('Side', self.window)
        bt_view_side.triggered.connect(self.plotting.set_view_side)
        self.viewMenu.addAction(bt_view_side)

        self.window.setCentralWidget(self.container)
        self.window.setWindowTitle("Loads Kernel Model Viewer")
        self.window.show()

    def update_modes(self):
        if self.list_modes_mass.currentItem() is not None:
            key = self.list_modes_mass.currentItem().data(0)
            tmp = self.list_modes_number.currentItem()
            if tmp is not None:
                old_mode = tmp.data(0)
            self.list_modes_number.clear()
            for mode in range(1, self.model['mass'][key]['n_modes'][()] + 1):
                item = QtGui.QListWidgetItem(str(mode))
                self.list_modes_number.addItem(item)
                if tmp is not None and int(old_mode) == mode:
                    self.list_modes_number.setCurrentItem(item)
            self.get_mode_data_for_plotting()

    def get_mode_data_for_plotting(self):
        uf_i = np.sign(self.sl_uf.value()) * (self.sl_uf.value() / 5.0) ** 2.0
        self.lb_uf.setText('Scaling: {:0.2f}'.format(uf_i))
        if self.list_modes_mass.currentItem() is not None and self.list_modes_number.currentItem() is not None:
            key = self.list_modes_mass.currentItem().data(0)
            mass = self.model['mass'][key]
            i_mode = int(self.list_modes_number.currentItem().data(0)) - 1
            uf = np.zeros((mass['n_modes'][()], 1))
            uf[i_mode] = uf_i
            ug = mass['PHIf_strc'][()].T.dot(uf)
            offset_f = ug[self.model['strcgrid']['set'][:, :3]].squeeze()
            self.plotting.plot_mode(
                self.model['strcgrid']['offset'][()] + offset_f)
            # the eigenvalue directly corresponds to the generalized stiffness if Mass is scaled to 1.0
            eigenvalue = mass['Kff'][()].diagonal()[i_mode]
            freq = np.real(eigenvalue) ** 0.5 / 2 / np.pi
            self.lb_freq.setText('Frequency: {:0.4f} Hz'.format(freq))

    def get_mass_data_for_plotting(self, *args):
        rho = np.double(self.sl_rho.value())
        self.lb_rho.setText('Scaling: {:0.0f} kg/m^3'.format(rho))
        if self.list_mass.currentItem() is not None:
            key = self.list_mass.currentItem().data(0)
            Mgg = load_hdf5_sparse_matrix(self.model['mass'][key]['MGG'])
            Mb = self.model['mass'][key]['Mb'][()]
            cggrid = load_hdf5_dict(self.model['mass'][key]['cggrid'])
            self.plotting.plot_masses(Mgg, Mb, cggrid, rho)
            self.lb_cg.setText('CG: x={:0.4f}, y={:0.4f}, z={:0.4f} m'.format(cggrid['offset'][0, 0],
                                                                              cggrid['offset'][0, 1],
                                                                              cggrid['offset'][0, 2]))
            # cg_mac = (x_cg - x_mac)*c_ref * 100 [%]
            # negativ bedeutet Vorlage --> stabil
            cg_mac = (cggrid['offset'][0, 0] - self.MAC[0]) / \
                self.model['macgrid']['c_ref'][()] * 100.0
            if cg_mac == 0.0:
                rating = 'indifferent'
            elif cg_mac < 0.0:
                rating = 'stable'
            elif cg_mac > 0.0:
                rating = 'unstable'
            self.lb_cg_mac.setText(
                'CG: x={:0.4f} % MAC, {}'.format(cg_mac, rating))
            self.lb_mass.setText('Mass: {:0.2f} kg'.format(Mb[0, 0]))
            self.lb_Ixx.setText('Ixx: {:0.4g} kg m^2'.format(Mb[3, 3]))
            self.lb_Iyy.setText('Iyy: {:0.4g} kg m^2'.format(Mb[4, 4]))
            self.lb_Izz.setText('Izz: {:0.4g} kg m^2'.format(Mb[5, 5]))

    def get_monstation_for_plotting(self, *args):
        if self.list_monstations.currentItem() is not None:
            key = self.list_monstations.currentItem().data(0)
            pos = list(self.model['mongrid']['name'].asstr()).index(key)
            monstation_id = self.model['mongrid']['ID'][pos]
            self.plotting.plot_monstations(monstation_id)
            self.lb_monstation_coord.setText(
                'Coord: {}'.format(self.model['mongrid']['CD'][pos]))

    def calc_MAC(self, key):
        # The mean aerodynamic center is calculated from the aerodynamics.
        # This approach includes also the downwash from wing on HTP.
        Qjj = self.model['aero'][key]['Qjj'][()]
        PHIlk = load_hdf5_sparse_matrix(self.model['PHIlk'])
        Dkx1 = self.model['Dkx1'][()]
        aerogrid = load_hdf5_dict(self.model['aerogrid'])
        macgrid = load_hdf5_dict(self.model['macgrid'])
        # assume unit downwash
        Ujx1 = np.dot(self.model['Djx1'][()], [0, 0, 1.0, 0, 0, 0])
        wj = np.sum(aerogrid['N'][:]
                    * Ujx1[aerogrid['set_j'][:, (0, 1, 2)]], axis=1)
        fl = aerogrid['N'].T * aerogrid['A'] * np.dot(Qjj, wj)
        Pl = np.zeros(aerogrid['n'] * 6)
        Pl[aerogrid['set_l'][:, 0]] = fl[0, :]
        Pl[aerogrid['set_l'][:, 1]] = fl[1, :]
        Pl[aerogrid['set_l'][:, 2]] = fl[2, :]
        Pmac = Dkx1.T.dot(PHIlk.T.dot(Pl))
        self.MAC = np.zeros(3)
        self.MAC[0] = macgrid['offset'][0, 0] - Pmac[4] / Pmac[2]
        self.MAC[1] = macgrid['offset'][0, 1] + Pmac[3] / Pmac[2]
        self.plotting.MAC = self.MAC

    def get_aero_for_plotting(self):
        if self.list_aero.currentItem() is not None:
            key = self.list_aero.currentItem().data(0)
            self.calc_MAC(key)
            self.lb_MAC.setText('MAC: x={:0.4f}, y={:0.4f} m'.format(self.MAC[0], self.MAC[1]))
            self.lb_MAC2.setText('(based on AIC from "{}", rigid, subsonic)'.format(key))
            if self.plotting.show_aero:
                self.plotting.hide_aero()
            if self.cb_w2gj.isChecked():
                self.plotting.plot_aero(self.model['camber_twist']['cam_rad'][()] / np.pi * 180.0)
            else:
                self.plotting.plot_aero()
            if self.cb_normal_vectors.isChecked():
                self.plotting.plot_panel_normal_vectors()

    def get_new_cs_for_plotting(self, *args):
        # To show a different control surface, new points need to be created. Thus, remove last control surface from plot.
        if self.plotting.show_cs:
            self.plotting.hide_cs()
        self.sl_deg.setValue(0)
        self.lb_deg.setText('Deflection: {:0.0f} deg'.format(0.0))
        self.get_cs_data_for_plotting()

    def get_cs_data_for_plotting(self, *args):
        deg = np.double(self.sl_deg.value())
        self.lb_deg.setText('Deflection: {:0.0f} deg'.format(deg))
        if self.list_cs.currentItem() is not None:
            # determine cs
            key = self.list_cs.currentItem().data(0)
            i_surf = np.where(self.model['x2grid']
                              ['key'].asstr()[:] == key)[0][0]
            axis = self.cb_axis.currentText()
            # hand over for plotting
            self.plotting.plot_cs(i_surf, axis, deg)

    def get_new_cell_data_for_plotting(self, *args):
        if (self.list_show_cells.currentItem() is not None) and (self.list_celldata.currentItem() is not None):
            items = self.list_show_cells.selectedItems()
            show_cells = [int(item.text()) for item in items]
            key = self.list_celldata.currentItem().data(0)
            celldata = self.nastran.celldata[key]
            self.plotting.plot_cell(celldata, show_cells)

    def get_new_markers_for_plotting(self, *args):
        # To show a different control surface, new points need to be created. Thus, remove last control surface from plot.
        if self.plotting.show_cfdgrids:
            self.plotting.hide_cfdgrids()
        self.get_markers_for_plotting()

    def get_markers_for_plotting(self, *args):
        if self.list_markers.currentItem() is not None:
            # determine marker
            items = self.list_markers.selectedItems()
            selected_markers = [item.text() for item in items]
            self.plotting.plot_cfdgrids(selected_markers)

    def get_iges_for_plotting(self, *args):
        if self.list_iges.currentItem() is not None:
            # determine marker
            items = self.list_iges.selectedItems()
            selected_meshes = [item.text() for item in items]
            if self.plotting.show_iges:
                self.plotting.hide_iges()
            self.plotting.plot_iges(selected_meshes)

    def load_model(self):
        # open file dialog
        filename = QtGui.QFileDialog.getOpenFileName(self.window, self.file_opt['title'],
                                                     self.file_opt['initialdir'], self.file_opt['filters'])[0]
        if filename != '':
            # load model
            self.model = data_handling.load_hdf5(filename)
            # update fields
            self.update_fields()
            self.calc_MAC(list(self.model['aero'].keys())[0])
            self.plotting.add_model(self.model)
            self.file_opt['initialdir'] = os.path.split(filename)[0]
            self.viewMenu.setEnabled(True)
            self.loadButtonNastran.setEnabled(True)

    def update_fields(self):
        self.list_mass.clear()
        self.list_modes_mass.clear()
        for key in self.model['mass'].keys():
            self.list_mass.addItem(QtGui.QListWidgetItem(key))
            self.list_modes_mass.addItem(QtGui.QListWidgetItem(key))

        self.list_aero.clear()
        for key in self.model['aero'].keys():
            self.list_aero.addItem(QtGui.QListWidgetItem(key))

        self.list_cs.clear()
        for key in self.model['x2grid']['key'].asstr():
            self.list_cs.addItem(QtGui.QListWidgetItem(key))

        self.list_monstations.clear()
        if 'mongrid' in self.model:
            for name in self.model['mongrid']['name'].asstr():
                self.list_monstations.addItem(QtGui.QListWidgetItem(str(name)))

    def load_nastran_results(self):
        filename = QtGui.QFileDialog.getOpenFileName(self.window, self.hdf5_opt['title'],
                                                     self.hdf5_opt['initialdir'], self.hdf5_opt['filters'])[0]
        if filename != '':
            self.nastran.load_file(filename)
            self.nastran.add_model(self.model)
            self.nastran.prepare_celldata()
            self.update_celldata()

    def update_celldata(self):
        self.list_celldata.clear()
        for key in self.nastran.celldata.keys():
            self.list_celldata.addItem(QtGui.QListWidgetItem(key))
        self.list_show_cells.clear()
        if hasattr(self.model, 'strcshell'):
            for key in self.model.strcshell['ID']:
                self.list_show_cells.addItem(QtGui.QListWidgetItem(str(key)))

    def load_tau_grid(self):
        filename = QtGui.QFileDialog.getOpenFileName(self.window, self.nc_opt['title'],
                                                     self.nc_opt['initialdir'], self.nc_opt['filters'])[0]
        if filename != '':
            self.tabs_widget.setCurrentIndex(2)
            self.cfdgrid = TauGrid()
            self.cfdgrid.load_file(filename)
            self.plotting.add_cfdgrids(self.cfdgrid.cfdgrids)
            self.update_markers()
            self.nc_opt['initialdir'] = os.path.split(filename)[0]

    def load_su2_grid(self):
        filename = QtGui.QFileDialog.getOpenFileName(self.window, self.nc_opt['title'],
                                                     self.nc_opt['initialdir'], self.nc_opt['filters'])[0]
        if filename != '':
            self.tabs_widget.setCurrentIndex(2)
            self.cfdgrid = SU2Grid()
            self.cfdgrid.load_file(filename)
            self.plotting.add_cfdgrids(self.cfdgrid.cfdgrids)
            self.update_markers()
            self.nc_opt['initialdir'] = os.path.split(filename)[0]

    def update_markers(self):
        self.list_markers.clear()
        for marker in self.cfdgrid.cfdgrids:
            self.list_markers.addItem(QtGui.QListWidgetItem(marker))

    def load_iges(self):
        filename = QtGui.QFileDialog.getOpenFileName(self.window, self.iges_opt['title'],
                                                     self.iges_opt['initialdir'], self.iges_opt['filters'])[0]
        if filename != '':
            self.tabs_widget.setCurrentIndex(6)
            self.iges.load_file(filename)
            self.plotting.add_iges_meshes(self.iges.meshes)
            self.update_list_iges()
            self.iges_opt['initialdir'] = os.path.split(filename)[0]

    def update_list_iges(self):
        self.list_iges.clear()
        for mesh in self.iges.meshes:
            self.list_iges.addItem(QtGui.QListWidgetItem(mesh['desc']))


def command_line_interface():
    m = Modelviewer()
    m.run()


if __name__ == "__main__":
    command_line_interface()
