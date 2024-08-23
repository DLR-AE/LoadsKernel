# -*- coding: utf-8 -*-
import os
import sys
from itertools import compress

from pyface.qt import QtCore
from pyface.qt.QtGui import (QApplication, QWidget, QTabWidget, QSizePolicy, QGridLayout, QMainWindow, QAction, QListWidget,
                             QListWidgetItem, QAbstractItemView, QFileDialog, QComboBox, QCheckBox, QLabel)
import matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure

import numpy as np

from loadscompare import plotting
from loadskernel.io_functions import data_handling


matplotlib.use('Qt5Agg')


class Compare():

    def __init__(self):
        self.datasets = {'ID': [],
                         'dataset': [],
                         'desc': [],
                         'color': [],
                         'n': 0,
                         }
        self.common_monstations = np.array([])
        self.colors = ['cornflowerblue', 'limegreen', 'violet', 'darkviolet', 'turquoise', 'orange', 'tomato', 'darkgrey',
                       'black']
        self.dof = ['Fx [N]', 'Fy [N]', 'Fz [N]', 'Mx [Nm]', 'My [Nm]', 'Mz [Nm]']

        # define file options
        self.file_opt = {}
        self.file_opt['filters'] = "HDF5 monitoring station files (monstation*.hdf5);;Pickle monitoring station files \
            (monstation*.pickle);;all files (*.*)"
        self.file_opt['initialdir'] = os.getcwd()
        self.file_opt['title'] = 'Load Monstations'

    def run(self):
        # Create the app.
        app = self.initApp()
        # Init the application's menues, tabs, etc.
        self.initGUI()
        # Start the main event loop.
        app.exec()

    def test(self):
        """
        This function is intended for CI testing. To test at least some parts of the code, the app is initialized, but never
        started. Instead, all windows are closed again.
        """
        app = self.initApp()
        self.initGUI()
        app.closeAllWindows()

    def initApp(self):
        # Init the QApplication in a robust way.
        # See https://stackoverflow.com/questions/54281439/pyside2-not-closing-correctly-with-basic-example
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)
        return app

    def initGUI(self):
        # Use one Widget as a main container.
        self.container = QWidget()
        # Init all sub-widgets.
        self.initMatplotlibFigure()
        self.initTabs()
        self.initWindow()
        # Arrange the layout inside the container.
        layout = QGridLayout(self.container)
        # Notation: layout.addWidget(widget, row, column, rowSpan, columnSpan)
        layout.addWidget(self.tabs_widget, 0, 0, 2, 1)
        layout.addWidget(self.canvas, 1, 1)
        layout.addWidget(self.toolbar, 0, 1)

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
        self.cb_color.activated.connect(self.update_color)

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
        layout.addWidget(self.lb_dataset, 0, 0, 1, 1)
        layout.addWidget(self.lb_mon, 0, 1, 1, 1)
        # one after the other
        layout.addWidget(self.cb_color, 1, 0, 1, 2)
        layout.addWidget(self.cb_xaxis, 2, 0, 1, 2)
        layout.addWidget(self.cb_yaxis, 3, 0, 1, 2)
        layout.addWidget(self.cb_hull, 4, 0, 1, 2)
        layout.addWidget(self.cb_labels, 5, 0, 1, 2)
        layout.addWidget(self.cb_minmax, 6, 0, 1, 2)
        layout.addWidget(self.label_n_loadcases, 7, 0, 1, 2)

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
        self.datasets['color'][self.lb_dataset.currentRow()] = self.colors[color]
        self.update_plot()

    def update_desc(self, *args):
        self.datasets['desc'][self.lb_dataset.currentRow()] = self.lb_dataset.currentItem().text()
        self.update_plot()

    def update_plot(self):
        if self.lb_dataset.currentItem() is not None and self.lb_mon.currentItem() is not None:
            # Get the items selected by the user.
            mon_sel = self.common_monstations[self.lb_mon.currentRow()]
            dataset_sel = [item.row() for item in self.lb_dataset.selectedIndexes()]
            # Check that monitoring stations exists in all selected datasets.
            mon_existing = [mon_sel in self.datasets['dataset'][i] for i in dataset_sel]
            dataset_sel = list(compress(dataset_sel, mon_existing))
            # Get the selected datasets, colors and a description.
            datasets = [self.datasets['dataset'][i] for i in dataset_sel]
            colors = [self.datasets['color'][i] for i in dataset_sel]
            desciptions = [self.datasets['desc'][i] for i in dataset_sel]
            # Call the plotting function.
            self.plotting.potato_plots(datasets,
                                       mon_sel,
                                       desciptions,
                                       colors,
                                       self.cb_xaxis.currentIndex(),
                                       self.cb_yaxis.currentIndex(),
                                       self.cb_xaxis.currentText(),
                                       self.cb_yaxis.currentText(),
                                       self.cb_hull.isChecked(),
                                       self.cb_labels.isChecked(),
                                       self.cb_minmax.isChecked(),
                                       )
            # Update the text box with number of plotted load cases.
            n_subcases = [len(dataset[mon_sel]['subcases']) for dataset in datasets]
            self.label_n_loadcases.setText('Selected load case: {}'.format(np.sum(n_subcases)))
        else:
            self.plotting.plot_nothing()
        self.canvas.draw()

    def merge_monstation(self):
        if len(self.lb_dataset.selectedItems()) > 1:
            # Init new dataset.
            new_dataset = {}
            for x in [item.row() for item in self.lb_dataset.selectedIndexes()]:
                print('Working on {} ...'.format(self.datasets['desc'][x]))
                for station in self.common_monstations:
                    if station not in new_dataset.keys():
                        # create (empty) entries for new monstation
                        new_dataset[station] = {'CD': self.datasets['dataset'][x][station]['CD'][()],
                                                'CP': self.datasets['dataset'][x][station]['CP'][()],
                                                'offset': self.datasets['dataset'][x][station]['offset'][()],
                                                'subcases': [],
                                                'loads': [],
                                                't': [],
                                                }
                    # Merge.
                    new_dataset[station]['loads'] += list(self.datasets['dataset'][x][station]['loads'][()])
                    new_dataset[station]['subcases'] += list(self.datasets['dataset'][x][station]['subcases'][()])
                    new_dataset[station]['t'] += list(self.datasets['dataset'][x][station]['t'][()])

            # Save into data structure.
            self.datasets['ID'].append(self.datasets['n'])
            self.datasets['dataset'].append(new_dataset)
            self.datasets['color'].append(self.colors[self.datasets['n']])
            self.datasets['desc'].append('dataset ' + str(self.datasets['n']))
            self.datasets['n'] += 1
            # Update fields.
            self.update_fields()

    def load_monstation(self):
        # open file dialog
        filename = QFileDialog.getOpenFileName(self.window, self.file_opt['title'], self.file_opt['initialdir'],
                                               self.file_opt['filters'])[0]
        if filename != '':
            if '.pickle' in filename:
                with open(filename, 'rb') as f:
                    dataset = data_handling.load_pickle(f)
            elif '.hdf5' in filename:
                dataset = data_handling.load_hdf5(filename)

            # save into data structure
            self.datasets['ID'].append(self.datasets['n'])
            self.datasets['dataset'].append(dataset)
            self.datasets['color'].append(self.colors[self.datasets['n']])
            self.datasets['desc'].append('dataset ' + str(self.datasets['n']))
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
            filename = QFileDialog.getSaveFileName(self.window, self.file_opt['title'], self.file_opt['initialdir'],
                                                   self.file_opt['filters'])[0]
            if filename != '' and '.pickle' in filename:
                with open(filename, 'wb') as f:
                    data_handling.dump_pickle(dataset_sel, f)
            if filename != '' and '.hdf5' in filename:
                data_handling.dump_hdf5(filename, dataset_sel)

    def update_fields(self):
        self.lb_dataset.clear()
        for desc in self.datasets['desc']:
            item = QListWidgetItem(desc)
            item.setFlags(item.flags() | QtCore.Qt.ItemIsEditable)
            self.lb_dataset.addItem(item)

        keys = []
        for dataset in self.datasets['dataset']:
            keys += dataset.keys()
        self.common_monstations = np.unique(keys)
        self.lb_mon.clear()
        for x in self.common_monstations:
            self.lb_mon.addItem(QListWidgetItem(x))


def command_line_interface():
    c = Compare()
    c.run()


if __name__ == "__main__":
    command_line_interface()
