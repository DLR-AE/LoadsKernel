
import numpy as np
import tables
        
class NastranSOL101:
    def __init__(self):
        self.celldata = {}
        pass
    
    def load_file(self, filename):
        self.filename = filename 
        self.file= tables.open_file(self.filename)
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
        #self.shell_stress_faster()
        #self.set_material_properties()
        #self.shell_FI()
    
    def read_bin(self, bin_stream, components):
        # create empty data structure
        dict = {} 
        for component in components:
            dict[component] = []
        # read binary data stream and sort into python data structure
        for row in bin_stream:
            for component in components:
                dict[component].append(row[component])
        return dict
                
    def read_data(self):
        print 'Reading data...',
        # INPUTs        
        if self.file.root.NASTRAN.INPUT.ELEMENT._v_children.has_key('CQUAD4'):        
            bin_stream = self.file.root.NASTRAN.INPUT.ELEMENT.CQUAD4
            components = ["EID", "PID", "T"]
            self.cquad4 = self.read_bin(bin_stream, components)
            self.cquad4['n'] = self.cquad4['EID'].__len__()
        else:
            self.cquad4 = {"EID":[], "PID":[], "T":[]}
            
        if self.file.root.NASTRAN.INPUT.ELEMENT._v_children.has_key('CTRIA3'):  
            bin_stream = self.file.root.NASTRAN.INPUT.ELEMENT.CTRIA3
            components = ["EID", "PID", "T"]
            self.ctria3 = self.read_bin(bin_stream, components)
            self.ctria3['n'] = self.ctria3['EID'].__len__()
        else:
            self.ctria3 = {"EID":[], "PID":[], "T":[]}
        
        if self.file.root.NASTRAN.INPUT.PROPERTY._v_children.has_key('PSHELL'): 
            bin_stream = self.file.root.NASTRAN.INPUT.PROPERTY.PSHELL
            components = ['PID', 'MID1', 'MID2', 'MID3', 'T']
            self.pshell = self.read_bin(bin_stream, components)
            
            # RESULTs for Shells
            bin_stream = self.file.root.NASTRAN.RESULT.ELEMENTAL.STRESS.QUAD4
            components = ["EID", "X1", "Y1", "XY1", "X2", "Y2", "XY2", "DOMAIN_ID"]
            self.stress_quad4 = self.read_bin(bin_stream, components)
            
        
        if self.file.root.NASTRAN.INPUT.PROPERTY._v_children.has_key('PCOMP'): 
            bin_stream = self.file.root.NASTRAN.INPUT.PROPERTY.PCOMP.IDENTITY
            components = ['PID', 'NPLIES']
            self.pcomp = self.read_bin(bin_stream, components) 
                
            bin_stream = self.file.root.NASTRAN.INPUT.PROPERTY.PCOMP.PLY
            components = ['MID', 'T', 'THETA']
            plies = self.read_bin(bin_stream, components)
            self.pcomp['plies'] = plies
        
            # RESULTs for Composite
            bin_stream = self.file.root.NASTRAN.RESULT.ELEMENTAL.STRESS.QUAD4_COMP
            components = ["EID", "PLY", "X1", "Y1", "T1", "DOMAIN_ID"]
            self.stress_quad4_comp = self.read_bin(bin_stream, components)
             
            bin_stream = self.file.root.NASTRAN.RESULT.ELEMENTAL.STRESS.TRIA3_COMP
            components = ["EID", "PLY", "X1", "Y1", "T1", "DOMAIN_ID"]
            self.stress_tria3_comp = self.read_bin(bin_stream, components)
            
            bin_stream = self.file.root.NASTRAN.RESULT.ELEMENTAL.STRAIN.QUAD4_COMP
            components = ["EID", "PLY", "X1", "Y1", "T1", "DOMAIN_ID"]
            self.strain_quad4_comp = self.read_bin(bin_stream, components)
             
            bin_stream = self.file.root.NASTRAN.RESULT.ELEMENTAL.STRAIN.TRIA3_COMP
            components = ["EID", "PLY", "X1", "Y1", "T1", "DOMAIN_ID"]
            self.strain_tria3_comp = self.read_bin(bin_stream, components)
            
        bin_stream = self.file.root.NASTRAN.RESULT.DOMAINS
        components = ["ID", 'SUBCASE']
        self.domains = self.read_bin(bin_stream, components)
        self.domains['n'] = self.domains['ID'].__len__()
            
        print 'done'
    
    def merge_shells(self):
        print 'Merging shells...',
        self.shells = {}
        components = ["EID", "PID", "T"]
        for component in components:
            self.shells[component] = np.array(self.cquad4[component] + self.ctria3[component])
        self.shells['n'] = self.shells['EID'].__len__()
        
#         self.results_shells_comp = {}
#         components = ["EID", "PLY", "X1", "Y1", "T1", "DOMAIN_ID"]
#         for component in components:
#             self.results_shells_comp[component] = np.array(self.result_quad4_comp[component] + self.result_tria3_comp[component])
        print 'done'
    
    def map_nastran2strcgrid(self):
        self.map_nastran2strcgrid = [np.where(self.shells['EID']==ID)[0][0] for ID in self.strcshell['ID']]
        
    def shell_thickness(self):
        # map elements and properties from model and results
        print 'Mapping thickness...',
        PIDs = self.shells['PID'][self.map_nastran2strcgrid]
        pos_PID = [self.pshell['PID'].index(ID) for ID in PIDs]
        self.celldata['thickness'] = np.array(self.pshell['T'])[pos_PID]
        print 'done'
    
    def shell_stress_faster(self):
        print 'Mapping stresses...',
        sigma1_quad4 = np.array(self.result_quad4_comp['X1']).reshape(self.domains['n'], self.cquad4['n'],-1)
        sigma2_quad4 = np.array(self.result_quad4_comp['Y1']).reshape(self.domains['n'], self.cquad4['n'],-1)
        tau12_quad4  = np.array(self.result_quad4_comp['T1']).reshape(self.domains['n'], self.cquad4['n'],-1)
        sigma1_tria3 = np.array(self.result_tria3_comp['X1']).reshape(self.domains['n'], self.ctria3['n'],-1)
        sigma2_tria3 = np.array(self.result_tria3_comp['Y1']).reshape(self.domains['n'], self.ctria3['n'],-1)
        tau12_tria3  = np.array(self.result_tria3_comp['T1']).reshape(self.domains['n'], self.ctria3['n'],-1)
        self.celldata['sigma1'] = np.concatenate((sigma1_quad4, sigma1_tria3), axis=1)[:,self.map_nastran2strcgrid,:]
        self.celldata['sigma2'] = np.concatenate((sigma2_quad4, sigma2_tria3), axis=1)[:,self.map_nastran2strcgrid,:]
        self.celldata['tau12']  = np.concatenate((tau12_quad4, tau12_tria3), axis=1)[:,self.map_nastran2strcgrid,:]
        print 'done'
    
    def set_material_properties(self):
        #MAT8    11000001 1.55+11   8.5+9      .3   3.7+9   3.7+9   3.7+9   1510.+       
        #+            0.0     0.0     0.0 0.833+9  0.25+9 16.66+6 66.66+6   25.E6+       
        #+            0.0     0.0        
        self.Xt = 0.833e+9
        self.Xc = 0.25e+9
        self.Yt = 16.66e+6
        self.Yc = 66.66e+6
        self.S = 25.0e+6
    
    def shell_FI(self):
        print 'Calculating subcase FI...',
        
        self.celldata['TsaiWu'] = np.zeros(self.celldata['sigma1'].shape)
        self.celldata['TsaiHill']   = np.zeros(self.celldata['sigma1'].shape)
        
        F11 = (self.Xt*self.Xc)**-1
        F22 = (self.Yt*self.Yc)**-1
        Fss = (self.S**2)**-1
        F1 = 1.0/self.Xt - 1.0/self.Xc
        F2 = 1.0/self.Yt - 1.0/self.Yc
        F12 = 0.0 #-0.5*(F11*F22)**0.5
        
        for i_subcase in range(self.domains['n']):
            print str(self.domains['ID'][i_subcase]),
            for i_shell in range(self.shells['n']):
                nplies = self.celldata['sigma1'][i_subcase,i_shell].__len__()
                for i_ply in range(nplies):
                    sigma1 = self.celldata['sigma1'][i_subcase,i_shell,i_ply]
                    sigma2 = self.celldata['sigma2'][i_subcase,i_shell,i_ply]
                    tau12  = self.celldata['tau12'][i_subcase,i_shell,i_ply]
                    FI_tsaiwu = F11*sigma1**2 + 2*F12*sigma1*sigma2 + F22*sigma2**2 + Fss*tau12**2  + F1*sigma1 + F2*sigma2 
                    # distinguish tension and compression for Tsai-Hill
                    if sigma1 > 0.0: X = self.Xt
                    else:            X = self.Xc
                    if sigma2 > 0.0: Y = self.Yt
                    else:            Y = self.Yc
                    S = self.S
                    FI_tsaihill = (sigma1/X)**2 + (sigma2/Y)**2 + (tau12/S)**2 - sigma1*sigma2/X**2
                    
                    self.celldata['TsaiWu'][i_subcase,i_shell,i_ply] = FI_tsaiwu
                    self.celldata['TsaiHill'][i_subcase,i_shell,i_ply] = FI_tsaihill
        print 'done'
