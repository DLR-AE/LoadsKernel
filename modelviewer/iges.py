
import sys, os
try:
    import pyiges
except:
    pass

class IgesMesh(object):
    def  __init__(self):
        self.meshes = []
        pass

    def load_file(self, filename):
        self.filename = filename 
        self.read_iges()
    
    def read_iges(self):
        if 'pyiges' in sys.modules:
            iges_object = pyiges.read(self.filename)
            vtk = iges_object.to_vtk(lines=False, bsplines=False, surfaces=True, points=False, merge=False)
            self.meshes.append({'desc': os.path.split(self.filename)[-1],
                                'vtk': vtk})
        else:
            print('Pyiges modul not found, can not read iges file.')
        
