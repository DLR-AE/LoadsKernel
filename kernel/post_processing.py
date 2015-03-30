# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 13:34:50 2015

@author: voss_ar
"""
import numpy as np

class post_processing:
    def __init__(self, jcl, model, response):
        self.jcl = jcl
        self.model = model
        self.response = response
    
    def modal_displacment_method(self):
        
        i_trimcase = 0
        response   = self.response[i_trimcase]
        trimcase   = self.jcl.trimcase[i_trimcase]
        
        i_mass     = self.model.mass['key'].index(trimcase['mass'])
        Mgg        = self.model.mass['MGG'][i_mass]
        PHIf_strc  = self.model.mass['PHIf_strc'][i_mass]
        PHIstrc_cg  = self.model.mass['PHIstrc_cg'][i_mass]
        
        
        d2Ug_dt2_r = PHIstrc_cg.dot(response['d2Ucg_dt2']) 
        d2Ug_dt2_f = PHIf_strc.T.dot(response['d2Uf_dt2'])
        Pg_iner_r = - Mgg.dot(d2Ug_dt2_r)
        Pg_iner_f = - Mgg.dot(d2Ug_dt2_f)  
        Pg_aero = response['Pg_aero']
        Ug_flex = PHIf_strc.T.dot(response['Uf'])
        Pg_flex = - self.model.Kgg.dot(Ug_flex)
        
        Pg = Pg_aero + Pg_iner_r + Pg_iner_f + Pg_flex
        
        PHIstrc_cg.T.dot(Pg)
        
        from mayavi import mlab
        x = self.model.strcgrid['offset'][:,0]
        y = self.model.strcgrid['offset'][:,1]
        z = self.model.strcgrid['offset'][:,2]
        
        mlab.figure() 
        mlab.points3d(x, y, z, scale_factor=0.1)
        #mlab.quiver3d(x, y, z, Pg_iner_r[self.model.strcgrid['set'][:,0]], Pg_iner_r[self.model.strcgrid['set'][:,1]], Pg_iner_r[self.model.strcgrid['set'][:,2]], color=(1,0,0), scale_factor=0.01)            
        #mlab.quiver3d(x, y, z, Pg_iner_f[self.model.strcgrid['set'][:,0]], Pg_iner_f[self.model.strcgrid['set'][:,1]], Pg_iner_f[self.model.strcgrid['set'][:,2]], color=(0,1,0), scale_factor=0.01)            
        mlab.quiver3d(x, y, z, Pg_flex[self.model.strcgrid['set'][:,0]], Pg_flex[self.model.strcgrid['set'][:,1]], Pg_flex[self.model.strcgrid['set'][:,2]], color=(0,1,1), scale_factor=0.01)         
        #mlab.quiver3d(x, y, z, Pg_aero[self.model.strcgrid['set'][:,0]], Pg_aero[self.model.strcgrid['set'][:,1]], Pg_aero[self.model.strcgrid['set'][:,2]], color=(0,0,1), scale_factor=0.01)            

        mlab.title('Pg', size=0.2, height=0.95)
        mlab.show()
        print 'Done.'
        
        