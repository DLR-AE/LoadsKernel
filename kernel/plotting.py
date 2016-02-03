# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 11:26:27 2015

@author: voss_ar
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
#from mayavi import mlab
from matplotlib.backends.backend_pdf import PdfPages
from scipy.spatial import ConvexHull
import csv
import build_aero


class plotting:
    def __init__(self, jcl, model, response):
        self.jcl = jcl
        self.model = model
        self.response = response
        
    def plot_pressure_distribution(self):
        for i_trimcase in range(len(self.jcl.trimcase)):
            response   = self.response[i_trimcase]
            trimcase   = self.jcl.trimcase[i_trimcase]
            print 'interactive plotting of resulting pressure distributions for trim {:s}'.format(trimcase['desc'])
            Pk = response['Pk_aero'] #response['Pk_rbm'] + response['Pk_cam']
            i_atmo = self.model.atmo['key'].index(trimcase['altitude'])
            rho = self.model.atmo['rho'][i_atmo]
            Vtas = trimcase['Ma'] * self.model.atmo['a'][i_atmo]
            F = np.linalg.norm(Pk[self.model.aerogrid['set_k'][:,:3]], axis=1)
            cp = F / (rho/2.0*Vtas**2) / self.model.aerogrid['A']
            ax = build_aero.plot_aerogrid(self.model.aerogrid, cp, 'jet', -0.5, 0.5)
            ax.set_title('Cp for {:s}'.format(trimcase['desc']))
            ax.set_xlim(0, 16)
            ax.set_ylim(-8, 8)
            plt.show()
    
    
    def plot_forces_deformation_interactive(self):
        from mayavi import mlab

        for i_trimcase in range(len(self.jcl.trimcase)):
            response   = self.response[i_trimcase]
            trimcase   = self.jcl.trimcase[i_trimcase]
            print 'interactive plotting of forces and deformations for trim {:s}'.format(trimcase['desc'])

            x = self.model.aerogrid['offset_k'][:,0]
            y = self.model.aerogrid['offset_k'][:,1]
            z = self.model.aerogrid['offset_k'][:,2]
            fx, fy, fz = response['Pk_rbm'][self.model.aerogrid['set_k'][:,0]],response['Pk_rbm'][self.model.aerogrid['set_k'][:,1]], response['Pk_rbm'][self.model.aerogrid['set_k'][:,2]]
            f_scale = 0.002 # vectors
            p_scale = 0.03 # points
            mlab.figure()
            mlab.points3d(x, y, z, scale_factor=p_scale)
            mlab.quiver3d(x, y, z, response['Pk_rbm'][self.model.aerogrid['set_k'][:,0]], response['Pk_rbm'][self.model.aerogrid['set_k'][:,1]], response['Pk_rbm'][self.model.aerogrid['set_k'][:,2]], color=(0,1,0), scale_factor=f_scale)            
            #mlab.quiver3d(x, y, z, fx*f_scale, fy*f_scale, fz*f_scale , color=(0,1,0),  mode='2ddash', opacity=0.4,  scale_mode='vector', scale_factor=1.0)
            #mlab.quiver3d(x+fx*f_scale, y+fy*f_scale, z+fz*f_scale,fx*f_scale, fy*f_scale, fz*f_scale , color=(0,1,0),  mode='cone', scale_mode='scalar', scale_factor=0.5, resolution=16)
            mlab.title('Pk_rbm', size=0.2, height=0.95)
            
            mlab.figure() 
            mlab.points3d(x, y, z, scale_factor=p_scale)
            mlab.quiver3d(x, y, z, response['Pk_cam'][self.model.aerogrid['set_k'][:,0]], response['Pk_cam'][self.model.aerogrid['set_k'][:,1]], response['Pk_cam'][self.model.aerogrid['set_k'][:,2]], color=(0,1,1), scale_factor=f_scale)            
            mlab.title('Pk_camber_twist', size=0.2, height=0.95)
            
            mlab.figure()        
            mlab.points3d(x, y, z, scale_factor=p_scale)
            mlab.quiver3d(x, y, z, response['Pk_cs'][self.model.aerogrid['set_k'][:,0]], response['Pk_cs'][self.model.aerogrid['set_k'][:,1]], response['Pk_cs'][self.model.aerogrid['set_k'][:,2]], color=(1,0,0), scale_factor=f_scale)
            mlab.title('Pk_cs', size=0.2, height=0.95)
            
            mlab.figure()   
            mlab.points3d(x, y, z, scale_factor=p_scale)
            mlab.quiver3d(x, y, z, response['Pk_f'][self.model.aerogrid['set_k'][:,0]], response['Pk_f'][self.model.aerogrid['set_k'][:,1]], response['Pk_f'][self.model.aerogrid['set_k'][:,2]], color=(1,0,1), scale_factor=f_scale)
            mlab.title('Pk_flex', size=0.2, height=0.95)
            
            mlab.figure()   
            mlab.points3d(x, y, z, scale_factor=p_scale)
            mlab.quiver3d(x, y, z, response['Pk_cfd'][self.model.aerogrid['set_k'][:,0]], response['Pk_cfd'][self.model.aerogrid['set_k'][:,1]], response['Pk_cfd'][self.model.aerogrid['set_k'][:,2]], color=(0,1,1), scale_factor=f_scale)
            mlab.title('Pk_cfd', size=0.2, height=0.95)
            
            mlab.figure()   
            mlab.points3d(x, y, z, scale_factor=p_scale)
            mlab.quiver3d(x, y, z, response['Pk_aero'][self.model.aerogrid['set_k'][:,0]], response['Pk_aero'][self.model.aerogrid['set_k'][:,1]], response['Pk_aero'][self.model.aerogrid['set_k'][:,2]], color=(0,1,0), scale_factor=f_scale)
            mlab.title('Pk_aero', size=0.2, height=0.95)
            
            x = self.model.strcgrid['offset'][:,0]
            y = self.model.strcgrid['offset'][:,1]
            z = self.model.strcgrid['offset'][:,2]
            x_r = self.model.strcgrid['offset'][:,0] + response['Ug_r'][self.model.strcgrid['set'][:,0]]
            y_r = self.model.strcgrid['offset'][:,1] + response['Ug_r'][self.model.strcgrid['set'][:,1]]
            z_r = self.model.strcgrid['offset'][:,2] + response['Ug_r'][self.model.strcgrid['set'][:,2]]
            x_f = self.model.strcgrid['offset'][:,0] + response['Ug_f'][self.model.strcgrid['set'][:,0]] * 10.0
            y_f = self.model.strcgrid['offset'][:,1] + response['Ug_f'][self.model.strcgrid['set'][:,1]] * 10.0
            z_f = self.model.strcgrid['offset'][:,2] + response['Ug_f'][self.model.strcgrid['set'][:,2]] * 10.0
            
            mlab.figure()
            mlab.points3d(x, y, z, color=(0,0,0), scale_factor=p_scale)
            mlab.points3d(x_r, y_r, z_r, color=(0,1,0), scale_factor=p_scale)
            mlab.points3d(x_f, y_f, z_f, color=(0,0,1), scale_factor=p_scale)
            mlab.title('rbm (green) and flexible deformation x10 (blue)', size=0.2, height=0.95)

            mlab.figure()   
            mlab.points3d(x, y, z, color=(0,0,0), scale_factor=p_scale/5.0)
            mlab.quiver3d(x, y, z, response['Pg'][self.model.strcgrid['set'][:,0]], response['Pg'][self.model.strcgrid['set'][:,1]], response['Pg'][self.model.strcgrid['set'][:,2]], color=(1,1,0), scale_factor=f_scale)
            mlab.title('Pg', size=0.2, height=0.95)
            
            mlab.show()
                

    def plot_monstations(self, monstations, filename_pdf):
        # Allegra
        if self.jcl.general['aircraft'] == 'ALLEGRA':
            potatos_Fz_Mx = ['ZWCUT01', 'ZWCUT04', 'ZWCUT08', 'ZWCUT12', 'ZWCUT16', 'ZWCUT20', 'ZWCUT24', 'ZHCUT01' ]
            potatos_Mx_My = ['ZWCUT01', 'ZWCUT04', 'ZWCUT08', 'ZWCUT12', 'ZWCUT16', 'ZWCUT20', 'ZWCUT24', 'ZHCUT01' ]
            potatos_Fz_My = ['ZWCUT01', 'ZWCUT04', 'ZWCUT08', 'ZWCUT12', 'ZWCUT16', 'ZWCUT20', 'ZWCUT24', 'ZHCUT01', 'ZFCUT27', 'ZFCUT28']
            cuttingforces_wing = ['ZWCUT01', 'ZWCUT04', 'ZWCUT08', 'ZWCUT12', 'ZWCUT16', 'ZWCUT20', 'ZWCUT24', ]
        # DLR-F19
        elif self.jcl.general['aircraft'] == 'DLR F-19-S':
            potatos_Fz_Mx = ['MON1', 'MON2', 'MON3', 'MON33', 'MON8', 'MON6', 'MON7', 'MON9']
            potatos_Mx_My = ['MON1', 'MON2', 'MON3', 'MON33', 'MON8', 'MON6', 'MON7', 'MON9']
            potatos_Fz_My = ['MON1', 'MON2', 'MON3', 'MON33', 'MON4', 'MON5']
            cuttingforces_wing = ['MON1', 'MON2', 'MON3', 'MON33', 'MON8']
        else:
            print 'Error: unknown aircraft: ' + str(self.jcl.general['aircraft'])
            return

        print 'start potato-plotting...'
        # get data needed for plotting from monstations
        loads = []
        offsets = []
        potato = np.unique(potatos_Fz_Mx + potatos_Mx_My + potatos_Fz_My)
        for station in potato:
            loads.append(monstations[station]['loads'])
            offsets.append(monstations[station]['offset'])
        loads = np.array(loads)
        offsets = np.array(offsets)
        
        pp = PdfPages(filename_pdf)
        self.crit_trimcases = []
        for i_station in range(len(potato)):
            
            if potato[i_station] in potatos_Fz_Mx:
                points = np.vstack((loads[i_station][:,2], loads[i_station][:,3])).T
                labels = ['Fz [N]', 'Mx [Nm]' ]
                # --- plot ---
                plt.figure()
                plt.scatter(points[:,0], points[:,1], color='cornflowerblue') # plot points
                if points.shape[0] >= 3:
                    hull = ConvexHull(points) # calculated convex hull from scattered points
                    for simplex in hull.simplices:                   # plot convex hull
                        plt.plot(points[simplex,0], points[simplex,1], color='cornflowerblue', linewidth=2.0, linestyle='--')
                    for i_case in range(hull.nsimplex):              # plot text   
                        plt.text(points[hull.vertices[i_case],0], points[hull.vertices[i_case],1], str(monstations[potato[i_station]]['subcase'][hull.vertices[i_case]]), fontsize=8)
                        self.crit_trimcases.append(monstations[potato[i_station]]['subcase'][hull.vertices[i_case]])
                else:
                    self.crit_trimcases += monstations[potato[i_station]]['subcase'][:]
                plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
                plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
                plt.title(potato[i_station])
                plt.grid('on')
                plt.xlabel(labels[0])
                plt.ylabel(labels[1])
                pp.savefig()
                plt.close()
            if potato[i_station] in potatos_Mx_My:
                points = np.vstack((loads[i_station][:,3], loads[i_station][:,4])).T
                labels = ['Mx [N]', 'My [Nm]' ]
                # --- plot ---
                plt.figure()
                plt.scatter(points[:,0], points[:,1], color='cornflowerblue') # plot points
                if points.shape[0] >= 3:
                    for simplex in hull.simplices:                   # plot convex hull
                        plt.plot(points[simplex,0], points[simplex,1], color='cornflowerblue', linewidth=2.0, linestyle='--')
                    for i_case in range(hull.nsimplex):              # plot text   
                        plt.text(points[hull.vertices[i_case],0], points[hull.vertices[i_case],1], str(monstations[potato[i_station]]['subcase'][hull.vertices[i_case]]), fontsize=8)
                        self.crit_trimcases.append(monstations[potato[i_station]]['subcase'][hull.vertices[i_case]])
                else:
                    self.crit_trimcases += monstations[potato[i_station]]['subcase'][:]
                plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
                plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
                plt.title(potato[i_station])
                plt.grid('on')
                plt.xlabel(labels[0])
                plt.ylabel(labels[1])
                pp.savefig()
                plt.close()   
                
            if potato[i_station] in potatos_Fz_My:
                points = np.vstack((loads[i_station][:,2], loads[i_station][:,4])).T
                labels = ['Fz [N]', 'My [Nm]' ]
                # --- plot ---
                plt.figure()
                plt.scatter(points[:,0], points[:,1], color='cornflowerblue') # plot points
                if points.shape[0] >= 3:
                    for simplex in hull.simplices:                   # plot convex hull
                        plt.plot(points[simplex,0], points[simplex,1], color='cornflowerblue', linewidth=2.0, linestyle='--')
                    for i_case in range(hull.nsimplex):              # plot text   
                        plt.text(points[hull.vertices[i_case],0], points[hull.vertices[i_case],1], str(monstations[potato[i_station]]['subcase'][hull.vertices[i_case]]), fontsize=8)
                        self.crit_trimcases.append(monstations[potato[i_station]]['subcase'][hull.vertices[i_case]])
                else:
                    self.crit_trimcases += monstations[potato[i_station]]['subcase'][:]
                plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
                plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
                plt.title(potato[i_station])
                plt.grid('on')
                plt.xlabel(labels[0])
                plt.ylabel(labels[1])
                pp.savefig()
                plt.close()   
        
        
        print 'start plotting cutting forces...'
        cuttingforces = ['Fx [N]', 'Fy [N]', 'Fz [N]', 'Mx [Nm]', 'My [Nm]', 'Mz [Nm]']
        
        loads = []
        offsets = []
        for station in cuttingforces_wing:
            loads.append(monstations[station]['loads'])
            offsets.append(monstations[station]['offset'])
        loads = np.array(loads)
        offsets = np.array(offsets)
        
        for i_cuttingforce in range(len(cuttingforces)):
            i_max = np.argmax(loads[:,:,i_cuttingforce], 1)
            i_min = np.argmin(loads[:,:,i_cuttingforce], 1)
            plt.figure()
            plt.plot(offsets[:,1],loads[:,:,i_cuttingforce], color='cornflowerblue', linestyle='-', marker='.')
            for i_station in range(len(cuttingforces_wing)):
                # verticalalignment or va 	[ 'center' | 'top' | 'bottom' | 'baseline' ]
                # max
                plt.scatter(offsets[i_station,1],loads[i_station,i_max[i_station],i_cuttingforce], color='r')
                plt.text(   offsets[i_station,1],loads[i_station,i_max[i_station],i_cuttingforce], str(monstations[monstations.keys()[0]]['subcase'][i_max[i_station]]), fontsize=8, verticalalignment='bottom' )
                # min
                plt.scatter(offsets[i_station,1],loads[i_station,i_min[i_station],i_cuttingforce], color='r')
                plt.text(   offsets[i_station,1],loads[i_station,i_min[i_station],i_cuttingforce], str(monstations[monstations.keys()[0]]['subcase'][i_min[i_station]]), fontsize=8, verticalalignment='top' )
            plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            plt.title('Wing')        
            plt.grid('on')
            plt.xlabel('y [m]')
            plt.ylabel(cuttingforces[i_cuttingforce]) 
            #plt.show()
            pp.savefig()
            plt.close()
            
        pp.close()
        print 'plots saved as ' + filename_pdf
        #print 'opening '+ filename_pdf
        #os.system('evince ' + filename_pdf + ' &')
        return 
        
        
    def write_critical_trimcases(self, crit_trimcases, trimcases, filename_csv):
        
        crit_trimcases_info = []
        for trimcase in trimcases:
            if trimcase['subcase'] in crit_trimcases:
                crit_trimcases_info.append(trimcase)
        print 'writing critical trimcases cases to: ' + filename_csv
        with open(filename_csv, 'wb') as fid:
            w = csv.DictWriter(fid, crit_trimcases_info[0].keys())
            w.writeheader()
            w.writerows(crit_trimcases_info)
        return
        
    def plot_time_animation(self):
        Pb_gust = []
        for i_step in range(len(self.response[0]['t'])):        
            Pb_gust.append(np.dot(self.model.Dkx1.T, self.response[0]['Pk_gust'][i_step,:]))
        Pb_gust = np.array(Pb_gust)
        plt.figure()
        plt.subplot(2,1,1)
        plt.plot(self.response[0]['t'], Pb_gust[:,2], 'b-')
        plt.xlabel('t [sec]')
        plt.ylabel('Pb_gust [N]')
        plt.grid('on')
        plt.subplot(2,1,2)
        plt.plot(self.response[0]['t'], self.response[0]['Nxyz'][:,2], 'b-')
        plt.plot(self.response[0]['t'], self.response[0]['alpha']/np.pi*180.0, 'r-')
        plt.plot(self.response[0]['t'], self.response[0]['X'][:,4]/np.pi*180.0, 'g-')
        plt.plot(self.response[0]['t'], np.arctan(self.response[0]['X'][:,8]/self.response[0]['X'][:,6])/np.pi*180.0, 'k-')
        plt.xlabel('t [sec]')
        plt.legend(['Nz', 'alpha', 'alpha/pitch', 'alpha/heave'])
        plt.grid('on')
        plt.ylabel('[-]/[deg]')
        #plt.show()
        
        if self.jcl.general['aircraft'] == 'ALLEGRA':
            lim=25.0
            length=45
        elif self.jcl.general['aircraft'] == 'DLR F-19-S':
            lim=10.0
            length=15
        else:
            print 'Error: unknown aircraft: ' + str(self.jcl.general['aircraft'])
            return
        # Set up data
        x = self.model.strcgrid['offset'][:,0] + self.response[0]['Ug'][:,self.model.strcgrid['set'][:,0]]
        y = self.model.strcgrid['offset'][:,1] + self.response[0]['Ug'][:,self.model.strcgrid['set'][:,1]]
        z = self.model.strcgrid['offset'][:,2] + self.response[0]['Ug'][:,self.model.strcgrid['set'][:,2]]
        data = np.array([x,y,z])
        t = self.response[0]['t']
        # Set up plot
        fig = plt.figure()
        ax1 = plt.subplot(1,2,1)
        line1, = ax1.plot([], [], 'r.')
        time_text = ax1.text(-lim+2,lim-2, '')
        ax1.set_xlim((-lim, lim))
        ax1.set_ylim((-lim, lim))
        ax1.grid('on')
        
        ax2 = plt.subplot(1,2,2)
        line2, = ax2.plot([], [], 'r.')
        ax2.set_ylim((-lim, lim))
        ax2.grid('on')
        #update_line(0,data,line1, line2, t, time_text)

        line_ani = animation.FuncAnimation(fig, self.update_line, fargs=(data, line1, line2, ax2, t, time_text, length), frames=len(t), interval=50, repeat=True, repeat_delay=3000)
        # Set up formatting for the movie files
        #Writer = animation.writers['ffmpeg']
        #writer = Writer(fps=20, bitrate=2000)        
        #line_ani.save('ref.mp4', writer) 
        plt.show()

    def update_line(self, num, data, line1, line2, ax2, t, time_text, length):
        line1.set_data(data[1,num,:], data[2,num,:])
        line2.set_data(data[0,num,:], data[2,num,:])
        ax2.set_xlim((-5+data[0,num,0], length+data[0,num,0]))
        time_text.set_text('Time = ' + str(t[num,0]))   