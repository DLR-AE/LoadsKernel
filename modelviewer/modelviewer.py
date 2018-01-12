# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 17:45:02 2017

@author: voss_ar
"""


import sys, os, copy
cwd = os.getcwd()
sys.path.append(cwd)
# Here you add the location of the Loads Kernel 
sys.path.append("../kernel")

import io_functions
from plotting import Plotting

# First, and before importing any Enthought packages, set the ETS_TOOLKIT
# environment variable to qt4, to tell Traits that we will use Qt.
os.environ['ETS_TOOLKIT'] = 'qt4'
# By default, the PySide binding will be used. If you want the PyQt bindings
# to be used, you need to set the QT_API environment variable to 'pyqt'
#os.environ['QT_API'] = 'pyqt'

# To be able to use PySide or PyQt4 and not run in conflicts with traits,
# we need to import QtGui and QtCore from pyface.qt
from pyface.qt import QtGui, QtCore
#from pyface.qt.QtGui import *
# Alternatively, you can bypass this line, but you need to make sure that
# the following lines are executed before the import of PyQT:
#   import sip
#   sip.setapi('QString', 2)

from traits.api import HasTraits, Instance, on_trait_change
from traitsui.api import View, Item
from mayavi.core.ui.api import MayaviScene, MlabSceneModel, \
        SceneEditor

import numpy as np

################################################################################
#The actual visualization
class Visualization(HasTraits):
    scene = Instance(MlabSceneModel, ())
    
    @on_trait_change('scene.activated')
    def update_plot(self):
        # This function is called when the view is opened. We don't
        # populate the scene when the view is not yet open, as some
        # VTK features require a GLContext.

        # We can do normal mlab calls on the embedded scene.
        #self.scene.mlab.test_points3d()
        return self.scene.mlab.gcf()

    # the layout of the dialog screated
    view = View(Item('scene', editor=SceneEditor(scene_class=MayaviScene),
                     height=600, width=600, show_label=False),
                resizable=True # We need this to resize with the parent widget
                )
    
# The QWidget containing the visualization, this is pure PyQt4 code.
class MayaviQWidget(QtGui.QWidget):
    def __init__(self, parent=None):
        QtGui.QWidget.__init__(self, parent)
        layout = QtGui.QVBoxLayout(self)
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(0)
        self.visualization = Visualization()

        # If you want to debug, beware that you need to remove the Qt
        # input hook.
        #QtCore.pyqtRemoveInputHook()
        #import pdb ; pdb.set_trace()
        #QtCore.pyqtRestoreInputHook()

        # The edit_traits call will generate the widget to embed.
        self.ui = self.visualization.edit_traits(parent=self,
                                                 kind='subpanel').control
        layout.addWidget(self.ui)
        self.ui.setParent(self)


class Modelviewer():
    
    def __init__(self):
    
        # define file options
        self.file_opt = {}
        self.file_opt['filters']  = "Loads Kernel files (model*.pickle);;all pickle files (*.pickle);;all files (*.*)"
        self.file_opt['initialdir'] = os.getcwd()
        self.file_opt['title']      = 'Open a Loads Kernel Model'
        
        self.plotting = Plotting()
        self.initGUI()
        pass
    
    def initGUI(self):
        # Don't create a new QApplication, it would unhook the Events
        # set by Traits on the existing QApplication. Simply use the
        # '.instance()' method to retrieve the existing one.
        app = QtGui.QApplication.instance()
        self.container = QtGui.QWidget()
        self.container.hide()
        
        # -------------------
        # --- set up tabs ---
        # -------------------        
        # Create Widgets
        tab_strc        = QtGui.QWidget() 
        tab_mass        = QtGui.QWidget()    
        tab_aero        = QtGui.QWidget()
        tab_coupling    = QtGui.QWidget()
        tab_monstations = QtGui.QWidget()
        tab_cs          = QtGui.QWidget()
        
        # Configure tabs
        tabs_widget = QtGui.QTabWidget()
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Preferred)
        tabs_widget.setSizePolicy(sizePolicy)
        tabs_widget.setMinimumWidth(300)
        
        # Add widgets to tabs
        tabs_widget.addTab(tab_strc, 'strc')
        tabs_widget.addTab(tab_mass,"mass")
        tabs_widget.addTab(tab_aero,"aero")
        tabs_widget.addTab(tab_cs,"cs")
        tabs_widget.addTab(tab_coupling,"coupling") 
        tabs_widget.addTab(tab_monstations,"monstations")
        
        # Elements of mass tab
        self.list_mass = QtGui.QListWidget()      
        self.list_mass.itemClicked.connect(self.get_mass_data_for_plotting)
        self.lb_rho = QtGui.QLabel('Rho: 2700 kg/m^3')
        # slider for generalized coordinate magnification factor
        self.sl_rho = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.sl_rho.setMinimum(100)
        self.sl_rho.setMaximum(3000)
        self.sl_rho.setSingleStep(100)
        self.sl_rho.setValue(2700)
        self.sl_rho.setTickPosition(QtGui.QSlider.TicksBelow)
        self.sl_rho.setTickInterval(500)
        self.sl_rho.valueChanged.connect(self.get_mass_data_for_plotting)
        self.lb_cg   = QtGui.QLabel('CG: x={:0.4f}, y={:0.4f}, z={:0.4f} m'.format(0.0, 0.0, 0.0))
        self.lb_cg_mac = QtGui.QLabel('CG: x={:0.4f} % MAC'.format(0.0))
        self.lb_mass = QtGui.QLabel('Mass: {:0.2f} kg'.format(0.0))
        self.lb_Ixx  = QtGui.QLabel('Ixx:  {:0.4g} kg m^2'.format(0.0))
        self.lb_Iyy  = QtGui.QLabel('Iyy:  {:0.4g} kg m^2'.format(0.0))
        self.lb_Izz  = QtGui.QLabel('Izz:  {:0.4g} kg m^2'.format(0.0))
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
        layout_mass.addStretch(1)
        
        # Elements of strc tab
        lb_undeformed = QtGui.QLabel('Undeformed')
        bt_strc_show = QtGui.QPushButton('Show')
        bt_strc_show.clicked.connect(self.plotting.plot_strc)
        bt_strc_hide = QtGui.QPushButton('Hide')
        bt_strc_hide.clicked.connect(self.plotting.hide_strc)
        # lists of mass case and mode number
        lb_modes_mass  = QtGui.QLabel('Mass')
        lb_modes_number = QtGui.QLabel('Modes')
        self.list_modes_mass = QtGui.QListWidget()      
        self.list_modes_mass.itemClicked.connect(self.update_modes)
        self.list_modes_number = QtGui.QListWidget()      
        self.list_modes_number.itemClicked.connect(self.get_mode_data_for_plotting)
        self.lb_freq = QtGui.QLabel('Frequency: {:0.4f} Hz'.format(0.0))
        self.lb_uf = QtGui.QLabel('Scaling: 1.0')
        # slider for generalized coordinate magnification factor
        self.sl_uf = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.sl_uf.setMinimum(0)
        self.sl_uf.setMaximum(30)
        self.sl_uf.setSingleStep(1)
        self.sl_uf.setValue(10)
        self.sl_uf.setTickPosition(QtGui.QSlider.TicksBelow)
        self.sl_uf.setTickInterval(5)
        self.sl_uf.valueChanged.connect(self.get_mode_data_for_plotting)
        bt_mode_hide = QtGui.QPushButton('Hide')
        bt_mode_hide.clicked.connect(self.plotting.hide_mode)
        
        layout_strc = QtGui.QGridLayout(tab_strc)
        layout_strc.addWidget(lb_undeformed,0,0,1,-1)
        layout_strc.addWidget(bt_strc_show, 1,0,1,-1)
        layout_strc.addWidget(bt_strc_hide, 2,0,1,-1)
        layout_strc.addWidget(lb_modes_mass,3,0,1,1)
        layout_strc.addWidget(lb_modes_number,3,1,1,1)
        layout_strc.addWidget(self.list_modes_mass,4,0,1,1)
        layout_strc.addWidget(self.list_modes_number,4,1,1,1)
        layout_strc.addWidget(self.lb_freq,5,0,1,-1) 
        layout_strc.addWidget(self.lb_uf,6,0,1,-1)    
        layout_strc.addWidget(self.sl_uf,7,0,1,-1)
        layout_strc.addWidget(bt_mode_hide,8,0,1,-1)
        #layout_strc.addStretch(1)
        
        # Elements of aero tab
        bt_aero_show = QtGui.QPushButton('Show')
        bt_aero_show.clicked.connect(self.plotting.plot_aero)
        bt_aero_hide = QtGui.QPushButton('Hide')
        bt_aero_hide.clicked.connect(self.plotting.hide_aero)
        
        layout_aero = QtGui.QVBoxLayout(tab_aero)
        layout_aero.addWidget(bt_aero_show)
        layout_aero.addWidget(bt_aero_hide)
        layout_aero.addStretch(1)
        
        # Elements of coupling tab
        bt_coupling_show = QtGui.QPushButton('Show')
        bt_coupling_show.clicked.connect(self.plotting.plot_aero_strc_coupling)
        bt_coupling_hide = QtGui.QPushButton('Hide')
        bt_coupling_hide.clicked.connect(self.plotting.hide_aero_strc_coupling)
        
        layout_coupling = QtGui.QVBoxLayout(tab_coupling)
        layout_coupling.addWidget(bt_coupling_show)
        layout_coupling.addWidget(bt_coupling_hide)
        layout_coupling.addStretch(1)
        
        # Elements of monstations tab
        bt_monstations_show = QtGui.QPushButton('Show')
        bt_monstations_show.clicked.connect(self.plotting.plot_monstations)
        bt_monstations_hide = QtGui.QPushButton('Hide')
        bt_monstations_hide.clicked.connect(self.plotting.hide_monstations)
        
        layout_monstations = QtGui.QVBoxLayout(tab_monstations)
        layout_monstations.addWidget(bt_monstations_show)
        layout_monstations.addWidget(bt_monstations_hide)
        layout_monstations.addStretch(1)
        
        # Elements of cs tab
        self.list_cs = QtGui.QListWidget()  
        self.list_cs.itemClicked.connect(self.get_new_cs_for_plotting)
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
        layout_cs.addStretch(1)
        
        # ----------------------------
        # --- set up Mayavi Figure ---
        # ----------------------------
        mayavi_widget = MayaviQWidget(self.container)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        mayavi_widget.setSizePolicy(sizePolicy)
        fig = mayavi_widget.visualization.update_plot()
        self.plotting.add_figure(fig)
        
        # ------------------------
        # --- layout container ---
        # ------------------------
        layout = QtGui.QGridLayout(self.container)
        layout.addWidget(tabs_widget, 0, 0)
        layout.addWidget(mayavi_widget, 0, 1)
                
        # ------------------------------
        # --- set up window and menu ---
        # ------------------------------
        self.window = QtGui.QMainWindow()
        mainMenu = self.window.menuBar()
        fileMenu = mainMenu.addMenu('File')
        # Add load button
        loadButton = QtGui.QAction('Load model', self.window)
        loadButton.setShortcut('Ctrl+L')
        loadButton.triggered.connect(self.load_model)
        fileMenu.addAction(loadButton)
        # Add exit button
        exitButton = QtGui.QAction('Exit', self.window)
        exitButton.setShortcut('Ctrl+Q')
        exitButton.triggered.connect(self.window.close)
        fileMenu.addAction(exitButton)
                
        fileMenu = mainMenu.addMenu('View')
        # Add view buttons
        bt_view_left_above = QtGui.QAction('Left Above', self.window)
        bt_view_left_above.triggered.connect(self.plotting.set_view_left_above)
        fileMenu.addAction(bt_view_left_above)
        
        bt_view_right_above = QtGui.QAction('Right Above', self.window)
        bt_view_right_above.triggered.connect(self.plotting.set_view_right_above)
        fileMenu.addAction(bt_view_right_above)
        
        bt_view_back = QtGui.QAction('Back', self.window)
        bt_view_back.triggered.connect(self.plotting.set_view_back)
        fileMenu.addAction(bt_view_back)
        
        bt_view_side = QtGui.QAction('Side', self.window)
        bt_view_side.triggered.connect(self.plotting.set_view_side)
        fileMenu.addAction(bt_view_side)
                
        self.window.setCentralWidget(self.container)
        self.window.setWindowTitle("Loads Kernel Model Viewer")
        self.window.show()
        
        # Start the main event loop.
        app.exec_()
    
    def update_modes(self):
        if self.list_modes_mass.currentItem() is not None:
            key = self.list_modes_mass.currentItem().data(0)
            i_mass = self.model.mass['key'].index(key)
            tmp = self.list_modes_number.currentItem()
            if tmp is not None:
                old_mode = tmp.data(0)
            self.list_modes_number.clear()
            for mode in range(self.model.mass['n_modes'][i_mass]):
                item = QtGui.QListWidgetItem(str(mode))
                self.list_modes_number.addItem(item)
                if tmp is not None and int(old_mode) == mode:
                    self.list_modes_number.setCurrentItem(item)
            self.get_mode_data_for_plotting()
    
    def get_mode_data_for_plotting(self):
        uf_i = 10.0**(np.double(self.sl_uf.value())/10.0)
        self.lb_uf.setText('Scaling: {:0.2f}'.format(uf_i))
        if self.list_modes_mass.currentItem() is not None and self.list_modes_number.currentItem() is not None:
            key = self.list_modes_mass.currentItem().data(0)
            i_mass = self.model.mass['key'].index(key)
            i_mode = int(self.list_modes_number.currentItem().data(0))
            uf = np.zeros((self.model.mass['n_modes'][i_mass],1))
            uf[i_mode] = uf_i
            ug = self.model.mass['PHIf_strc'][i_mass].T.dot(uf)
            offset_f = ug[self.model.strcgrid['set'][:,(0,1,2)]].squeeze()
            self.plotting.plot_mode(self.model.strcgrid['offset']+offset_f)
            # the eigenvalue directly corresponds to the generalized stiffness if Mass is scaled to 1.0
            eigenvalue = self.model.mass['Kff'][i_mass].diagonal()[i_mode]
            freq = np.real(eigenvalue)**0.5 /2/np.pi
            self.lb_freq.setText('Frequency: {:0.4f} Hz'.format(freq))
            
            
    def get_mass_data_for_plotting(self, *args):
        rho = np.double(self.sl_rho.value())
        self.lb_rho.setText('Scaling: {:0.0f} kg/m^3'.format(rho))
        if self.list_mass.currentItem() is not None:
            key = self.list_mass.currentItem().data(0)
            i_mass = self.model.mass['key'].index(key)
            self.plotting.plot_masses(self.model.mass['MGG'][i_mass], self.model.mass['Mb'][i_mass], self.model.mass['cggrid'][i_mass], rho)
            self.lb_cg.setText('CG: x={:0.4f}, y={:0.4f}, z={:0.4f} m'.format(self.model.mass['cggrid'][i_mass]['offset'][0,0],
                                                                                self.model.mass['cggrid'][i_mass]['offset'][0,1],
                                                                                self.model.mass['cggrid'][i_mass]['offset'][0,2]))
            # cg_mac = (x_cg - x_mac)*c_ref * 100 [%]
            # negativ bedeutet Vorlage --> stabil
            cg_mac = (self.model.mass['cggrid'][i_mass]['offset'][0,0]-self.model.macgrid['offset'][0,0])/self.model.macgrid['c_ref']*100.0
            if cg_mac == 0.0:
                rating = 'indifferent'
            elif cg_mac < 0.0:
                rating = 'stable'
            elif cg_mac > 0.0:
                rating = 'unstable'
            self.lb_cg_mac.setText('CG: x={:0.4f} % MAC, {}'.format(cg_mac, rating))
            self.lb_mass.setText('Mass: {:0.2f} kg'.format(self.model.mass['Mb'][i_mass][0,0]))
            self.lb_Ixx.setText('Ixx: {:0.4g} kg m^2'.format(self.model.mass['Mb'][i_mass][3,3]))
            self.lb_Iyy.setText('Iyy: {:0.4g} kg m^2'.format(self.model.mass['Mb'][i_mass][4,4]))
            self.lb_Izz.setText('Izz: {:0.4g} kg m^2'.format(self.model.mass['Mb'][i_mass][5,5]))
    
    
    def get_new_cs_for_plotting(self, *args):
        # To show a different control surface, new points need to be created. Thus, remove last control surface from plot.
        if self.plotting.show_cs:
            self.plotting.hide_cs()
        self.sl_deg.setValue(0.0)
        self.lb_deg.setText('Deflection: {:0.0f} deg'.format(0.0))
        self.get_cs_data_for_plotting()
        
    def get_cs_data_for_plotting(self, *args):
        deg = np.double(self.sl_deg.value())
        self.lb_deg.setText('Deflection: {:0.0f} deg'.format(deg))
        if self.list_cs.currentItem() is not None:
            # determine cs
            key = self.list_cs.currentItem().data(0)
            i_surf = self.model.x2grid['key'].index(key)
            axis = self.cb_axis.currentText()
            # hand over for plotting
            self.plotting.plot_cs(i_surf, axis, deg)
            
    def load_model(self):
        # open file dialog
        filename = QtGui.QFileDialog.getOpenFileName(self.window, self.file_opt['title'], self.file_opt['initialdir'], self.file_opt['filters'])[0]
        if filename != '':
            # load model
            io = io_functions.specific_functions()
            path_output = os.path.split(filename)[0]
            job_name = os.path.split(filename)[1]
            if job_name.endswith('.pickle'):
                job_name = job_name[:-7]
            if job_name.startswith('model_'):
                job_name = job_name[6:]
            path_output = io.check_path(path_output)
            self.model = io.load_model(job_name, path_output)
            # update fields
            self.update_fields()
            self.plotting.add_model(self.model)
            self.plotting.plot_nothing()
            self.file_opt['initialdir'] = os.path.split(filename)[0]
            self.container.show()

    def update_fields(self):
        self.list_mass.clear()
        self.list_modes_mass.clear()
        for key in self.model.mass['key']:
            self.list_mass.addItem(QtGui.QListWidgetItem(key))
            self.list_modes_mass.addItem(QtGui.QListWidgetItem(key))

        self.list_cs.clear()
        for key in self.model.x2grid['key']:
            self.list_cs.addItem(QtGui.QListWidgetItem(key))
            
    
if __name__ == "__main__":
    modelviewer = Modelviewer()
    