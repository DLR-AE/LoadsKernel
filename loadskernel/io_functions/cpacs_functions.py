'''
Created on Apr 9, 2019

@author: voss_ar
'''

class CpacsFunctions:
    def __init__(self, tixi):
        self.tixi = tixi
        
    def write_cpacs_loadsvector(self, parent, grid, Pg, ):
        self.add_elem(parent, 'fx', Pg[grid['set'][:,0]], 'vector')
        self.add_elem(parent, 'fy', Pg[grid['set'][:,1]], 'vector')
        self.add_elem(parent, 'fz', Pg[grid['set'][:,2]], 'vector')
        self.add_elem(parent, 'mx', Pg[grid['set'][:,3]], 'vector')
        self.add_elem(parent, 'my', Pg[grid['set'][:,4]], 'vector')
        self.add_elem(parent, 'mz', Pg[grid['set'][:,5]], 'vector')
    
    def write_cpacs_grid(self, parent, grid):
        self.add_elem(parent, 'uID', grid['ID'], 'vector_int')
        self.add_elem(parent, 'x', grid['offset'][:,0], 'vector')
        self.add_elem(parent, 'y', grid['offset'][:,1], 'vector')
        self.add_elem(parent, 'z', grid['offset'][:,2], 'vector')
        
    def write_cpacs_grid_orientation(self, parent, grid, coord):
        dircos = [coord['dircos'][coord['ID'].index(x)] for x in grid['CD']]
        #self.add_elem(parent, 'dircos', dircos, 'vector')
        # Wie schreibt man MAtrizen in CPACS ???
    
    def create_path(self, parent,path):
        # adopted from cps2mn 
        # Create all elements in CPACS defined by a path string with '/' between elements.
        #
        # INPUTS
        #   parent:     [string] parent node in CPACS for the elements to be created
        #   path:       [string] path of children elements to be created
        #
        # Institute of Aeroelasticity
        # German Aerospace Center (DLR) 
        
        #tixi, tigl, modelUID = getTixiTiglModelUID()
        #Split the path at the '/' creating all the new elements names
        tmp = path.split('/')
    
        #Temporary path containing the name of the parent node
        tmp_path = parent
        
        #Loop over all elements found at 'path'
        for i in range(len(tmp)):
    
            #Create a new element under the current parent node
            self.tixi.createElement(tmp_path,tmp[i])
            
            #Expands the parent node to include the current element
            tmp_path = tmp_path + '/' + tmp[i]

    def add_elem(self, path, elem, data, data_type):
        # adopted from cps2mn 
        # Add element data to cpacs. Can be double, integer, text, vector
        #
        # INPUTS
        #   path:       [string] path of the parent element in CPACS
        #   elem:       [string] name of the element to be created in CPACS
        #   data_type   [double/integer/text/vector] type of the element to be created
        #   
        # Institute of Aeroelasticity
        # German Aerospace Center (DLR) 
    
        #tixi, tigl, modelUID = getTixiTiglModelUID()
        #---------------------------------------------------------------------#
        #Add data with TIXI
        if data_type == 'double':        
            self.tixi.addTextElement(path, elem, str(data))
            
        #if data_type == 'integer':
            #error = TIXI.tixiGetIntegerElement( tixiHandle, path, byref(data))
        if data_type == 'text':
            self.tixi.addTextElement(path, elem, data)
    
        #Add float vector
        if data_type == 'vector':
            format='%f'
            self.tixi.addFloatVector(path, elem, data, len(data),format)
            
        #Add integer vector
        if data_type == 'vector_int':
            format='%0.0f'
            self.tixi.addFloatVector(path, elem, data, len(data),format)