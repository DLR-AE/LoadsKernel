# -*- coding: utf-8 -*-


from PyQt5 import QtCore
from PyQt5.QtWidgets import (QApplication, QWidget, QTabWidget, QSizePolicy, QGridLayout, QMainWindow, QAction, QListWidget, QListWidgetItem, 
                             QAbstractItemView, QFileDialog, QComboBox, QCheckBox, QLabel)

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.backend_bases import key_press_handler # implement the default mpl key bindings
from matplotlib.figure import Figure

import numpy as np
import os, copy

from loadscompare import plotting
import loadskernel.io_functions as io_functions
import loadskernel.io_functions.specific_functions
   

class Compare():
    def __init__(self):
        self.datasets = {   'ID':[], 
                            'dataset':[],
                            'desc': [],
                            'color':[],
                            'n': 0,
                        }
        self.common_monstations = np.array([])
        self.colors = ['cornflowerblue', 'limegreen', 'violet', 'darkviolet', 'turquoise', 'orange', 'tomato','darkgrey', 'black']
        self.dof = [ 'Fx [N]', 'Fy [N]', 'Fz [N]', 'Mx [Nm]', 'My [Nm]', 'Mz [Nm]']
            
        # define file options
        self.file_opt = {}
        self.file_opt['filters']  = "HDF5 monitoring station files (monstation*.hdf5);;Pickle monitoring station files (monstation*.pickle);;all files (*.*)"
        self.file_opt['initialdir'] = os.getcwd()
        self.file_opt['title']      = 'Load Monstations'
    
    def run(self):
        self.initApplication()
    
    def initApplication(self):
        app = QApplication([])
        self.container = QWidget()

        self.initMatplotlibFigure()
        self.initTabs()
        self.initWindow()

        # layout container 
        layout = QGridLayout(self.container)
        # Notation: layout.addWidget(widget, row, column, rowSpan, columnSpan)
        layout.addWidget(self.tabs_widget, 0, 0, 2, 1)
        layout.addWidget(self.canvas, 1, 1)
        layout.addWidget(self.toolbar, 0, 1)

        # Start the main event loop.
        app.exec_()
    
    def initTabs(self):
        # Configure tabs widget
        self.tabs_widget = QTabWidget()
        # configure sizing, limit width of tabs widget in favor of plotting area
        sizePolicy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.tabs_widget.setSizePolicy(sizePolicy)
        self.tabs_widget.setMinimumWidth(300)
        self.tabs_widget.setMaximumWidth(450)

        # Add tabs
        self.initLoadsTab()

    def initLoadsTab(self):
        tab_loads = QWidget()
        self.tabs_widget.addTab(tab_loads, 'Section Loads')
        # Elements of loads tab
        self.lb_dataset = QListWidget()
        self.lb_dataset.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.lb_dataset.itemSelectionChanged.connect(self.show_choice)
        self.lb_dataset.itemChanged.connect(self.update_desc)
        
        self.lb_mon = QListWidget()
        self.lb_mon.itemSelectionChanged.connect(self.show_choice)
        
        self.cb_color = QComboBox()
        self.cb_color.addItems(self.colors)
        self.cb_color.activated[str].connect(self.update_color)
        
        self.cb_xaxis = QComboBox()
        self.cb_xaxis.addItems(self.dof)
        self.cb_xaxis.setCurrentIndex(3)
        self.cb_xaxis.activated.connect(self.show_choice)
        
        self.cb_yaxis = QComboBox()
        self.cb_yaxis.addItems(self.dof)
        self.cb_yaxis.setCurrentIndex(4)
        self.cb_yaxis.activated.connect(self.show_choice)
        
        self.cb_hull = QCheckBox("show convex hull")
        self.cb_hull.setChecked(False)
        self.cb_hull.stateChanged.connect(self.show_choice)
        
        self.cb_labels = QCheckBox("show labels")
        self.cb_labels.setChecked(False)
        self.cb_labels.stateChanged.connect(self.show_choice)
        
        self.cb_minmax = QCheckBox("show min/max")
        self.cb_minmax.setChecked(False)
        self.cb_minmax.stateChanged.connect(self.show_choice)
        
        self.label_n_loadcases = QLabel()
        
        
        layout = QGridLayout(tab_loads)
        # Notation: layout.addWidget(widget, row, column, rowSpan, columnSpan)
        # side-by-side
        layout.addWidget(self.lb_dataset,   0,0,1,1)
        layout.addWidget(self.lb_mon,       0,1,1,1)
        # one after the other
        layout.addWidget(self.cb_color,     1,0,1,2) 
        layout.addWidget(self.cb_xaxis,     2,0,1,2)
        layout.addWidget(self.cb_yaxis,     3,0,1,2)
        layout.addWidget(self.cb_hull,      4,0,1,2)
        layout.addWidget(self.cb_labels,    5,0,1,2)
        layout.addWidget(self.cb_minmax,    6,0,1,2)
        layout.addWidget(self.label_n_loadcases, 7,0,1,2)
        
    
    def initMatplotlibFigure(self):
        # init Matplotlib Plot
        fig1 = Figure()
        # hand over subplot to plotting class
        self.plotting = plotting.Plotting(fig1)
        # embed figure
        self.canvas = FigureCanvasQTAgg(fig1)
        self.canvas.draw()
        # configure sizing, set minimum size
        sizePolicy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.canvas.setSizePolicy(sizePolicy)
        self.canvas.setMinimumWidth(800)
        self.canvas.setMinimumHeight(600)
        
        self.toolbar = NavigationToolbar2QT(self.canvas, self.container)
        self.toolbar.update()
    
    def initWindow(self):
        # Set up window and menu 
        self.window = QMainWindow()
        mainMenu = self.window.menuBar()
        # Add Menu to window
        fileMenu = mainMenu.addMenu('File')
        # Add load button
        action = QAction('Load Monstations', self.window)
        action.setShortcut('Ctrl+L')
        action.triggered.connect(self.load_monstation)
        fileMenu.addAction(action)

        # Add Action buttons
        action = QAction('Merge Monstations', self.window)
        action.setShortcut('Ctrl+M')
        action.triggered.connect(self.merge_monstation)
        fileMenu.addAction(action)
        
        action = QAction('Save Monstations', self.window)
        action.setShortcut('Ctrl+S')
        action.triggered.connect(self.save_monstation)
        fileMenu.addAction(action)
        
        # Add exit button
        action = QAction('Exit', self.window)
        action.setShortcut('Ctrl+Q')
        action.triggered.connect(self.window.close)
        fileMenu.addAction(action)

        self.window.setCentralWidget(self.container)
        self.window.setWindowTitle("Loads Compare")
        self.window.show()
    
        
    def show_choice(self, *args):
        # called on change in listbox, combobox, etc
        # discard extra variables
        if len(self.lb_dataset.selectedItems()) == 1:
            self.cb_color.setCurrentText(self.datasets['color'][self.lb_dataset.currentRow()])
            self.cb_color.setEnabled(True)
        else:
            self.cb_color.setDisabled(True)
        self.update_plot()
            
    def update_color(self, color):
        self.datasets['color'][self.lb_dataset.currentRow()] = color
        self.update_plot()
        
    def update_desc(self, *args):
        self.datasets['desc'][self.lb_dataset.currentRow()] = self.lb_dataset.currentItem().text()
        self.update_plot()
            
    def update_plot(self):
        if self.lb_dataset.currentItem() is not None and self.lb_mon.currentItem() is not None:
            # Reverse current selection of datasets for plotting. The dataset added/created last is plotted first.
            # This is useful for example after merging different datasets. The resulting dataset would obscure the view if plotted last. 
            current_selection = [item.row() for item in self.lb_dataset.selectedIndexes()]
            current_selection.reverse()
            dataset_sel = [self.datasets['dataset'][i] for i in current_selection]
            color_sel   = [self.datasets['color'][i] for i in current_selection]
            desc_sel    = [self.datasets['desc'][i] for i in current_selection]
            mon_sel     = self.common_monstations[self.lb_mon.currentRow()]
            n_subcases = [dataset[mon_sel]['subcases'].__len__() for dataset in dataset_sel ]
            try:
                n_subcases_dyn2stat = [dataset[mon_sel]['subcases_dyn2stat'].__len__() for dataset in dataset_sel ]
            except:
                n_subcases_dyn2stat = [0]
            
            self.plotting.potato_plots( dataset_sel, 
                                        mon_sel, 
                                        desc_sel, 
                                        color_sel, 
                                        self.cb_xaxis.currentIndex(), 
                                        self.cb_yaxis.currentIndex(),
                                        self.cb_xaxis.currentText(), 
                                        self.cb_yaxis.currentText(),
                                        self.cb_hull.isChecked(),
                                        self.cb_labels.isChecked(),
                                        self.cb_minmax.isChecked(),
                                      )
            self.label_n_loadcases.setText('Selected load case: {} \nDyn2Stat: {}'.format(np.sum(n_subcases), np.sum(n_subcases_dyn2stat)))
        else:    
            self.plotting.plot_nothing()
        self.canvas.draw()
    
    def get_loads_string(self, x, station):
        # Check for dynamic loads.
        if np.size(self.datasets['dataset'][x][station]['t'][0]) == 1:
            # Scenario 1: There are only static loads.
            print( '- {}: found static loads'.format(station))
            loads_string   = 'loads'
            subcase_string = 'subcases'
            t_string = 't'
        elif (np.size(self.datasets['dataset'][x][station]['t'][0]) > 1) and ('loads_dyn2stat' in self.datasets['dataset'][x][station].keys()) and (self.datasets['dataset'][x][station]['loads_dyn2stat'] != []):
            # Scenario 2: Dynamic loads have been converted to quasi-static time slices / snapshots.
            print( '- {}: found dyn2stat loads -> discarding dynamic loads'.format(station))
            loads_string   = 'loads_dyn2stat'
            subcase_string = 'subcases_dyn2stat'
            t_string = 't_dyn2stat'
        else:
            # Scenario 3: There are only dynamic loads. 
            print( '- {}: found dynamic loads -> please convert to static first (dyn2stat)'.format(station))
        return loads_string, subcase_string, t_string

    def merge_monstation(self):
        if len(self.lb_dataset.selectedItems()) > 1:
            # Init new dataset.
            new_dataset = {}
            for x in [item.row() for item in self.lb_dataset.selectedIndexes()]:
                print ('Working on {} ...'.format(self.datasets['desc'][x]))
                for station in self.common_monstations:
                    if station not in new_dataset.keys():
                        # create (empty) entries for new monstation
                        new_dataset[station] = {'CD': self.datasets['dataset'][x][station]['CD'][()],
                                                'CP': self.datasets['dataset'][x][station]['CP'][()],
                                                'offset': self.datasets['dataset'][x][station]['offset'][()],
                                                'subcases': [],
                                                'loads':[],
                                                't':[],
                                                }
                    loads_string, subcase_string, t_string = self.get_loads_string(x, station)
                    # Merge.   
                    new_dataset[station]['loads']           += list(self.datasets['dataset'][x][station][loads_string][()])
                    new_dataset[station]['subcases']        += list(self.datasets['dataset'][x][station][subcase_string][()])
                    new_dataset[station]['t']               += list(self.datasets['dataset'][x][station][t_string][()])
        
            # Save into data structure.
            self.datasets['ID'].append(self.datasets['n'])  
            self.datasets['dataset'].append(new_dataset)
            self.datasets['color'].append(self.colors[self.datasets['n']])
            self.datasets['desc'].append('dataset '+ str(self.datasets['n']))
            self.datasets['n'] += 1
            # Update fields.
            self.update_fields()
    
    def load_monstation(self):
        # open file dialog
        filename = QFileDialog.getOpenFileName(self.window, self.file_opt['title'], self.file_opt['initialdir'], self.file_opt['filters'])[0]
        if filename != '':
            if '.pickle' in filename:
                with open(filename, 'rb') as f:
                    dataset = io_functions.specific_functions.load_pickle(f)
            elif '.hdf5' in filename:
                dataset = io_functions.specific_functions.load_hdf5(filename)
            
            # save into data structure
            self.datasets['ID'].append(self.datasets['n'])  
            self.datasets['dataset'].append(dataset)
            self.datasets['color'].append(self.colors[self.datasets['n']])
            self.datasets['desc'].append('dataset '+ str(self.datasets['n']))
            self.datasets['n'] += 1
            # update fields
            self.update_fields()
            self.file_opt['initialdir'] = os.path.split(filename)[0]
    
    def save_monstation(self):
        """
        Saving an HDF5 file to HDF5 does not work (yet).
        """
        if self.lb_dataset.currentItem() is not None and len(self.lb_dataset.selectedItems()) == 1:
            dataset_sel = self.datasets['dataset'][self.lb_dataset.currentRow()]
            # open file dialog
            filename = QFileDialog.getSaveFileName(self.window, self.file_opt['title'], self.file_opt['initialdir'], self.file_opt['filters'])[0]
            if filename != '' and '.pickle' in filename:
                with open(filename, 'wb') as f:
                    io_functions.specific_functions.dump_pickle(dataset_sel, f)
            if filename != '' and '.hdf5' in filename:
                io_functions.specific_functions.dump_hdf5(filename, dataset_sel)

    def update_fields(self):
        self.lb_dataset.clear()
        for desc in self.datasets['desc']:
            item = QListWidgetItem(desc)
            item.setFlags(item.flags() | QtCore.Qt.ItemIsEditable)
            self.lb_dataset.addItem(item)

        keys = [list(dataset) for dataset in self.datasets['dataset']]
        self.common_monstations = np.unique(keys)
        self.lb_mon.clear()
        for x in self.common_monstations:
            self.lb_mon.addItem(QListWidgetItem(x))
        
if __name__ == "__main__":
    c = Compare()
    c.run()
    
    
    