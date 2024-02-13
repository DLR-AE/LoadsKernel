import tables
import numpy as np


class NastranSOL101():

    def __init__(self):
        self.celldata = {}

    def load_file(self, filename):
        self.filename = filename
        self.file = tables.open_file(self.filename)
        self.read_data()
        self.file.close()

    def add_model(self, model):
        self.model = model
        self.strcgrid = model.strcgrid
        self.strcshell = model.strcshell

    def prepare_celldata(self):
        self.merge_shells()
        self.map_nastran2strcgrid()
        self.shell_thickness()

    def read_bin(self, bin_stream, components):
        # create empty data structure
        data = {}
        for component in components:
            data[component] = []
        # read binary data stream and sort into python data structure
        for row in bin_stream:
            for component in components:
                data[component].append(row[component])
        return data

    def read_data(self):
        # INPUTs
        if 'CQUAD4' in self.file.root.NASTRAN.INPUT.ELEMENT._v_children:
            bin_stream = self.file.root.NASTRAN.INPUT.ELEMENT.CQUAD4
            components = ["EID", "PID", "T"]
            self.cquad4 = self.read_bin(bin_stream, components)
            self.cquad4['n'] = self.cquad4['EID'].__len__()
        else:
            self.cquad4 = {"EID": [], "PID": [], "T": []}

        if 'CTRIA3' in self.file.root.NASTRAN.INPUT.ELEMENT._v_children:
            bin_stream = self.file.root.NASTRAN.INPUT.ELEMENT.CTRIA3
            components = ["EID", "PID", "T"]
            self.ctria3 = self.read_bin(bin_stream, components)
            self.ctria3['n'] = self.ctria3['EID'].__len__()
        else:
            self.ctria3 = {"EID": [], "PID": [], "T": []}

        if 'PSHELL' in self.file.root.NASTRAN.INPUT.PROPERTY._v_children:
            bin_stream = self.file.root.NASTRAN.INPUT.PROPERTY.PSHELL
            components = ['PID', 'MID1', 'MID2', 'MID3', 'T']
            self.pshell = self.read_bin(bin_stream, components)

    def merge_shells(self):
        self.shells = {}
        components = ["EID", "PID", "T"]
        for component in components:
            self.shells[component] = np.array(
                self.cquad4[component] + self.ctria3[component])
        self.shells['n'] = len(self.shells['EID'])

    def map_nastran2strcgrid(self):
        self.nastran2strcgrid = [np.where(self.shells['EID'] == ID)[
            0][0] for ID in self.strcshell['ID']]

    def shell_thickness(self):
        # map elements and properties from model and results
        PIDs = self.shells['PID'][self.nastran2strcgrid]
        pos_PID = [self.pshell['PID'].index(ID) for ID in PIDs]
        self.celldata['thickness'] = np.array(self.pshell['T'])[pos_PID]
