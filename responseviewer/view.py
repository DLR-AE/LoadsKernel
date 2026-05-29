# -*- coding: utf-8 -*-
import os
import sys

from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QApplication, QWidget, QTabWidget, QSizePolicy, QGridLayout, QMainWindow, QListWidget,
    QListWidgetItem, QAbstractItemView, QFileDialog, QMessageBox)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure

from responseviewer import plotting
from loadskernel.io_functions import data_handling


class ResponseViewer():

    def __init__(self):
        self.responses = None
        self.colors = ['cornflowerblue', 'limegreen', 'violet', 'darkviolet', 'turquoise', 'orange', 'tomato', 'darkgrey',
                       'black']

        # define file options
        self.file_opt = {}
        self.file_opt['filters'] = "HDF5 response files (response*.hdf5);;Pickle response files \
            (response*.pickle);;all files (*.*)"
        self.file_opt['initialdir'] = os.getcwd()
        self.file_opt['title'] = 'Load Responses'

        # GUI attributes
        self.container = None
        self.tabs_widget = None
        self.canvas = None
        self.toolbar = None
        self.plotting = None
        self.window = None
        self.lb_subcase = None
        self.lb_states = None

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
        self.initStatesTab()

    def initStatesTab(self):
        tab_loads = QWidget()
        self.tabs_widget.addTab(tab_loads, 'Time Histories')
        # Elements of loads tab
        self.lb_subcase = QListWidget()
        self.lb_subcase.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.lb_subcase.itemSelectionChanged.connect(self.show_choice)

        self.lb_states = QListWidget()
        self.lb_states.setSelectionMode(QAbstractItemView.ExtendedSelection)
        for item in self.plotting.all_quantities:
            self.lb_states.addItem(QListWidgetItem(item))
        self.lb_states.setCurrentRow(4)
        self.lb_states.itemSelectionChanged.connect(self.show_choice)

        layout = QGridLayout(tab_loads)
        # Notation: layout.addWidget(widget, row, column, rowSpan, columnSpan)
        layout.addWidget(self.lb_subcase, 0, 0, 1, 2)
        layout.addWidget(self.lb_states, 1, 0, 1, 2)

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
        action = QAction('Load Responses', self.window)
        action.setShortcut('Ctrl+L')
        action.triggered.connect(self.load_response)
        fileMenu.addAction(action)

        # Add exit button
        action = QAction('Exit', self.window)
        action.setShortcut('Ctrl+Q')
        action.triggered.connect(self.window.close)
        fileMenu.addAction(action)

        self.window.setCentralWidget(self.container)
        self.window.setWindowTitle("Response Viewer")
        self.window.show()

    def show_choice(self):
        # called on change in listbox, combobox, etc
        self.update_plot()

    def update_plot(self):
        if self.lb_subcase.currentItem() is not None and self.lb_states.currentItem() is not None:
            # Get the items selected by the user.
            subcases_selected = [item.text() for item in self.lb_subcase.selectedItems()]
            quantities_selected = [item.text() for item in self.lb_states.selectedItems()]
            # Call the plotting function.
            self.plotting.timehistories(subcases_selected, quantities_selected)
        else:
            self.plotting.plot_nothing()
        self.canvas.draw()

    def load_response(self):
        # open file dialog
        filename, _ = QFileDialog.getOpenFileName(
            self.window,
            self.file_opt['title'],
            self.file_opt['initialdir'],
            self.file_opt['filters']
        )
        if filename != '':
            dataset = None
            if '.pickle' in filename:
                with open(filename, 'rb') as f:
                    dataset = data_handling.load_pickle(f)
            elif '.hdf5' in filename:
                dataset = data_handling.load_hdf5(filename)
            else:
                QMessageBox.warning(self.window, "Unsupported File", "Please select a .pickle or .hdf5 file.")

            if dataset is not None:
                # Store dataset
                self.responses = dataset
                # Update fields
                self.update_fields()
                # Update response in plotting class
                self.plotting.add_responses(self.responses)
                self.file_opt['initialdir'] = os.path.split(filename)[0]

    def update_fields(self):
        if self.responses is not None:
            self.lb_subcase.clear()
            list_of_subcases = list(self.responses)
            list_of_subcases.sort(key=int)
            self.lb_subcase.addItems(list_of_subcases)


def command_line_interface():
    r = ResponseViewer()
    r.run()


if __name__ == "__main__":
    command_line_interface()
