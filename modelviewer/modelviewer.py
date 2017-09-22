# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 17:45:02 2017

@author: voss_ar
"""




import sys, os
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
                     height=600, width=800, show_label=False),
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
        container = QtGui.QWidget()
        
        # Create tabs
        tab_strcgrid    = QtGui.QWidget() 
        tab_mass        = QtGui.QWidget()    
        tab_aero        = QtGui.QWidget()
        tab_spline      = QtGui.QWidget()
        tab_monstations = QtGui.QWidget()
        
        # Add tabs
        tabs_widget = QtGui.QTabWidget()
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Preferred)
        tabs_widget.setSizePolicy(sizePolicy)
        tabs_widget.addTab(tab_strcgrid, 'strcgrid')
        tabs_widget.addTab(tab_mass,"mass")
        tabs_widget.addTab(tab_aero,"aero")
        tabs_widget.addTab(tab_spline,"spline") 
        tabs_widget.addTab(tab_monstations,"monstations")
        
        # Selection elements
        self.list_mass = QtGui.QListWidget()      
        self.list_mass.itemClicked.connect(self.show_mass)
        # Fill tabs
        layout_mass = QtGui.QGridLayout(tab_mass)
        layout_mass.addWidget(self.list_mass)
        
        label = QtGui.QLabel(container)
        label.setText('left')
        
        mayavi_widget = MayaviQWidget(container)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        mayavi_widget.setSizePolicy(sizePolicy)
        fig = mayavi_widget.visualization.update_plot()
        self.plotting.add_figure(fig)
        
        layout = QtGui.QGridLayout(container)
        layout.addWidget(tabs_widget, 0, 0)
        layout.addWidget(label, 1, 0)
        layout.addWidget(mayavi_widget, 0, 1, rowSpan=2)
        
        container.show()
        
        self.window = QtGui.QMainWindow()
        mainMenu = self.window.menuBar()
        fileMenu = mainMenu.addMenu('File')
        # Add load button
        loadButton = QtGui.QAction(QtGui.QIcon('exit24.png'), 'Load model', self.window)
        loadButton.setShortcut('Ctrl+L')
        loadButton.triggered.connect(self.load_monstation)
        fileMenu.addAction(loadButton)
        # Add exit button
        exitButton = QtGui.QAction(QtGui.QIcon('exit24.png'), 'Exit', self.window)
        exitButton.setShortcut('Ctrl+Q')
        exitButton.triggered.connect(self.window.close)
        fileMenu.addAction(exitButton)
                
        self.window.setCentralWidget(container)
        self.window.setWindowTitle("Loads Kernel Model Viewer")
        self.window.show()
        
    
        # Start the main event loop.
        app.exec_()
            
    def show_mass(self, *args):
        if self.list_mass.currentItem() is not None:
            key = self.list_mass.currentItem().data(0)
            print key
            i_mass = self.model.mass['key'].index(key)
            self.plotting.add_strcgrid(self.model.strcgrid)
            self.plotting.plot_masses(self.model.mass['MGG'][i_mass], self.model.mass['Mb'][i_mass], self.model.mass['cggrid'][i_mass])
            
    def load_monstation(self):
        print 'open file'
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
            self.plotting.plot_nothing()
            self.file_opt['initialdir'] = os.path.split(filename)[0]

    def update_fields(self):
        self.list_mass.clear()
        for key in self.model.mass['key']:
            self.list_mass.addItem(QtGui.QListWidgetItem(key))
    
if __name__ == "__main__":
    modelviewer = Modelviewer()
    