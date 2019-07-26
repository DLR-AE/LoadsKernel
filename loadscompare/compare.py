# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 17:45:02 2017

@author: voss_ar
"""


import tkinter as tk
import tkinter.ttk as ttk
import tkinter.filedialog as filedialog
import matplotlib as mpl
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.backend_bases import key_press_handler # implement the default mpl key bindings
import numpy as np
import pickle, os, copy

from loadscompare import plotting
import loadskernel.io_functions as io_functions
import loadskernel.io_functions.specific_functions

class App:
    
    def __init__(self, root):
        # --- init data structure ---
        self.root = root
        self.datasets = {   'ID':[], 
                            'dataset':[],
                            'desc': [],
                            'color':[],
                            'n': 0,
                        }
        self.common_monstations = np.array([])
        self.colors = ['cornflowerblue', 'limegreen', 'violet', 'darkviolet', 'turquoise', 'orange', 'tomato','darkgrey', 'black']
        self.var_color = tk.StringVar()
        self.var_desc  = tk.StringVar()
        
        # mapping of Fxyz and Mxyz to dof
        self.var_xaxis = tk.StringVar()
        self.var_xaxis.set('Fz [N]')

        self.var_yaxis = tk.StringVar()
        self.var_yaxis.set('Mx [Nm]')
        
        self.dof = {
            'Fx [N]':  0,
            'Fy [N]':  1,
            'Fz [N]':  2,
            'Mx [Nm]': 3,
            'My [Nm]': 4,
            'Mz [Nm]': 5,
            }
            
        # define file options
        self.file_opt = {}
        self.file_opt['filetypes']  = [('Loads Kernel files', 'monstation*.pickle'), ('all pickle files', '.pickle'), ('all files', '.*')]
        self.file_opt['initialdir'] = os.getcwd()
        self.file_opt['title']      = 'Load Monstations'
        
        # --- init GUI ---

        # init menu
        root.option_add('*tearOff', tk.FALSE)        
        menubar = tk.Menu(self.root)
        menu_file = tk.Menu(menubar)
        menu_file.add_command(label='Load Monstations', command=self.load_monstation)
        menu_file.add_command(label='Close', command=self.close_app)
        menubar.add_cascade(menu=menu_file, label='File')
        menu_action = tk.Menu(menubar)
        menu_action.add_command(label='Merge Monstations', command=self.merge_monstation)
        menu_action.add_command(label='Superpose Monstations (by subcase)', command=self.superpose_monstation_by_subcase)
        menu_action.add_command(label='Save Monstations', command=self.save_monstation)
        menubar.add_cascade(menu=menu_action, label='Action')
        root['menu'] = menubar
        
        # init frames
        # frame left side
        frame_left_top = ttk.Frame(root)
        frame_left_top.grid(row=0, column=0,)
        frame_left_bot = ttk.Frame(root)
        frame_left_bot.grid(row=1, column=0)
        # frame center
        frame_center = ttk.Frame(root)
        frame_center.grid(row=0, column=1, rowspan=2)
        # frame rigth side
        frame_right = ttk.Frame(root)
        frame_right.grid(row=0, column=2, rowspan=2, sticky=(tk.N,tk.W,tk.E,tk.S))
        # resize only the bottom and the right frame --> plot area is resized
        root.grid_columnconfigure(2, weight=1)
        root.grid_rowconfigure(1, weight=1)

        # ListBox to select several datasets 
        self.lb_dataset = tk.Listbox(frame_left_top, height=20, selectmode=tk.EXTENDED, exportselection=False)
        self.lb_dataset.grid(row=0, column=0, sticky=(tk.N,tk.W,tk.E,tk.S))
        self.lb_dataset.bind('<<ListboxSelect>>', self.show_choice)
        s_dataset = ttk.Scrollbar(frame_left_top, orient=tk.VERTICAL, command=self.lb_dataset.yview)
        s_dataset.grid(row=0, column=1, sticky=(tk.N,tk.S))
        # attach listbox to scrollbar
        self.lb_dataset.config(yscrollcommand=s_dataset.set)
        s_dataset.config(command=self.lb_dataset.yview)
        
        # change desc
        self.en_desc = ttk.Entry(frame_left_top, textvariable=self.var_desc, state='disabled', exportselection=False)
        self.en_desc.grid(row = 1, column =0, columnspan=2, sticky=(tk.W,tk.E))
        #self.en_desc.bind('<<Change>>', self.update_desc ) 
        self.var_desc.trace('w', self.update_desc)        
        
        # color selector
        self.cb_color = ttk.Combobox(frame_left_top, textvariable=self.var_color, values=self.colors, state='disabled')
        self.cb_color.grid(row = 2, column =0, columnspan=2, sticky=(tk.W,tk.E))
        self.cb_color.bind('<<ComboboxSelected>>', self.update_color )        

        # ListBox to select monstations 
        self.lb_mon = tk.Listbox(frame_left_top, height=20, selectmode=tk.SINGLE, exportselection=False)
        self.lb_mon.grid(row=0, column=2, rowspan=8, sticky=(tk.N,tk.W,tk.E,tk.S))
        self.lb_mon.bind('<<ListboxSelect>>', self.show_choice)
        s_mon = ttk.Scrollbar(frame_left_top, orient=tk.VERTICAL, command=self.lb_mon.yview)
        s_mon.grid(row=0, column=3, rowspan=8, sticky=(tk.N,tk.S))
        # attach listbox to scrollbar
        self.lb_mon.config(yscrollcommand=s_mon.set)
        s_mon.config(command=self.lb_mon.yview)

        # Eintraege hinzufuegen mit self.l.insert('end', 'Line 1')
        # abfrage der Werte mit self.l.curselection()

        # x- and y-axis
        cb_xaxis = ttk.Combobox(frame_left_top, textvariable=self.var_xaxis, values=list(self.dof), state='readonly')
        cb_xaxis.grid(row = 3, column =0, columnspan=2, sticky=(tk.W,tk.E))
        cb_xaxis.bind('<<ComboboxSelected>>', self.show_choice )
        
        cb_yaxis = ttk.Combobox(frame_left_top, textvariable=self.var_yaxis, values=list(self.dof), state='readonly')
        cb_yaxis.grid(row = 4, column =0, columnspan=2, sticky=(tk.W,tk.E))
        cb_yaxis.bind('<<ComboboxSelected>>', self.show_choice )
        
        # options / check boxes
        self.show_hull = tk.BooleanVar()
        cb_hull = ttk.Checkbutton( frame_left_top, text="show convex hull", variable=self.show_hull,  onvalue=tk.TRUE, offvalue=tk.FALSE, command=self.update_plot )        
        cb_hull.grid(row = 5, column =0, columnspan=2, sticky=(tk.W,tk.E))
        
        self.show_labels = tk.BooleanVar()
        cb_labels = ttk.Checkbutton( frame_left_top, text="show labels", variable=self.show_labels,  onvalue=tk.TRUE, offvalue=tk.FALSE, command=self.update_plot )        
        cb_labels.grid(row = 6, column =0, columnspan=2, sticky=(tk.W,tk.E))
        
        self.label_n_loadcases =tk.Label(frame_left_top, anchor='w', justify='left', padx=0, text='') 
        self.label_n_loadcases.grid(row=7, column=0, columnspan=2)
        
        # init Matplotlib Plot
        fig1 = mpl.figure.Figure()
        # hand over subplot to plotting class
        self.plotting = plotting.Plotting(fig1)
        # embed figure
        self.canvas = FigureCanvasTkAgg(fig1, master=frame_right)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        toolbar = NavigationToolbar2Tk(self.canvas, frame_right)
        toolbar.update()
        self.canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=1)     
    
    def show_choice(self, *args):
        # called on change in listbox, combobox, etc
        # discard extra variables
        self.update_plot()
        if len(self.lb_dataset.curselection()) == 1:
            self.var_color.set(self.datasets['color'][self.lb_dataset.curselection()[0]])
            self.cb_color.config(state='readonly')
            self.var_desc.set(self.datasets['desc'][self.lb_dataset.curselection()[0]])
            self.en_desc.config(state='ensabled')
        else:
            self.cb_color.config(state='disabled')
            self.en_desc.config(state='disabled')
            
    def update_color(self, *args):
        self.datasets['color'][self.lb_dataset.curselection()[0]] = self.var_color.get() 
        self.update_plot()
        
    def update_desc(self, *args):
        i = copy.deepcopy(self.lb_dataset.curselection()[0])
        self.datasets['desc'][i] = self.var_desc.get() 
        #self.update_fields()
        self.lb_dataset.delete(i)
        self.lb_dataset.insert(i,  self.datasets['desc'][i])
        self.lb_dataset.select_set(i)
        self.update_plot()
            
    def update_plot(self):
        if self.lb_dataset.curselection() != () and self.lb_mon.curselection() != ():
            # Reverse current selection of datasets for plotting. The dataset added/created last is plotted first.
            # This is useful for example after merging different datasets. The resulting dataset would obscure the view if plotted last. 
            current_selection_reversed = list(reversed(self.lb_dataset.curselection()))
            dataset_sel = [self.datasets['dataset'][i] for i in current_selection_reversed]
            color_sel   = [self.datasets['color'][i] for i in current_selection_reversed]
            desc_sel    = [self.datasets['desc'][i] for i in current_selection_reversed]
            mon_sel     = self.common_monstations[self.lb_mon.curselection()]
            n_subcases = [dataset[mon_sel]['subcase'].__len__() for dataset in dataset_sel ]
            try:
                n_subcases_dyn2stat = [dataset[mon_sel]['subcases_dyn2stat'].__len__() for dataset in dataset_sel ]
            except:
                n_subcases_dyn2stat = [0]
            
            self.plotting.potato_plots( dataset_sel, 
                                        mon_sel, 
                                        desc_sel, 
                                        color_sel, 
                                        self.dof[self.var_xaxis.get()], 
                                        self.dof[self.var_yaxis.get()], 
                                        self.var_xaxis.get(),
                                        self.var_yaxis.get(),
                                        self.show_hull.get(),
                                        self.show_labels.get(),
                                      )
            self.label_n_loadcases.config(text='Selected load case: {} \nDyn2Stat: {}'.format(np.sum(n_subcases), np.sum(n_subcases_dyn2stat)))
        else:    
            self.plotting.plot_nothing()
        self.canvas.draw()
    
    def get_loads_string(self, x, station):
        # Check for dynamic loads.
        if np.size(self.datasets['dataset'][x][station]['t'][0]) == 1:
            # Scenario 1: There are only static loads.
            print( '- {}: found static loads'.format(station))
            loads_string   = 'loads'
            subcase_string = 'subcase'
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
        if len(self.lb_dataset.curselection()) > 1:
            # Init new dataset.
            new_dataset = {}
            for x in self.lb_dataset.curselection():
                print ('Working on {} ...'.format(self.datasets['desc'][x]))
                for station in self.common_monstations:
                    if station not in new_dataset.keys():
                        # create (empty) entries for new monstation
                        new_dataset[station] = {'CD': self.datasets['dataset'][x][station]['CD'],
                                                'CP': self.datasets['dataset'][x][station]['CP'],
                                                'offset': self.datasets['dataset'][x][station]['offset'],
                                                'subcase': [],
                                                'loads':[],
                                                't':[],
                                                }
                    loads_string, subcase_string, t_string = self.get_loads_string(x, station)
                    # Merge.   
                    new_dataset[station]['loads']           += self.datasets['dataset'][x][station][loads_string]
                    new_dataset[station]['subcase']         += self.datasets['dataset'][x][station][subcase_string]
                    new_dataset[station]['t']               += self.datasets['dataset'][x][station][t_string]
            # Save into data structure.
            self.datasets['ID'].append(self.datasets['n'])  
            self.datasets['dataset'].append(new_dataset)
            self.datasets['color'].append(self.colors[self.datasets['n']])
            self.datasets['desc'].append('dataset '+ str(self.datasets['n']))
            self.datasets['n'] += 1
            # Update fields.
            self.update_fields()
    
    def superpose_monstation_by_subcase(self):
        if len(self.lb_dataset.curselection()) > 1:
            # Init new dataset.
            new_dataset = {}
            for x in self.lb_dataset.curselection():
                print( 'Working on {} ...'.format(self.datasets['desc'][x]))
                for station in self.common_monstations:
                    loads_string, subcase_string, t_string = self.get_loads_string(x, station)
                    
                    n = len(self.datasets['dataset'][x][station][t_string])
                    if station not in new_dataset.keys():
                        # create (empty) entries for new monstation
                        new_dataset[station] = {'CD': self.datasets['dataset'][x][station]['CD'],
                                                'CP': self.datasets['dataset'][x][station]['CP'],
                                                'offset': self.datasets['dataset'][x][station]['offset'],
                                                'subcase': self.datasets['dataset'][x][station]['subcase'],
                                                'loads':[np.array([0.0,0.0,0.0,0.0,0.0,0.0,])]*n,
                                                't':self.datasets['dataset'][x][station]['t'],
                                                }
                    # Superpose subcases. 
                    for i_case in range(n): 
                        subcase = self.datasets['dataset'][x][station][subcase_string][i_case]
                        if subcase in new_dataset[station]['subcase']:
                            pos = new_dataset[station]['subcase'].index(subcase)
                            new_dataset[station][loads_string][pos] = new_dataset[station][loads_string][pos] + self.datasets['dataset'][x][station][loads_string][i_case]
                        else:
                            print( '- {}: found no match for subcases {}. Superposition not possible'.format(station, subcase ))                            
                    
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
        filename = filedialog.askopenfilename(**self.file_opt)
        if filename != '':
            with open(filename, 'rb') as f:
                dataset = io_functions.specific_functions.load_pickle(f)
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
        if self.lb_dataset.curselection() != () and len(self.lb_dataset.curselection()) == 1:
            dataset_sel = self.datasets['dataset'][self.lb_dataset.curselection()[0]]
            # open file dialog
            filename = filedialog.asksaveasfilename(**self.file_opt)
            if filename != '':
                with open(filename, 'wb') as f:
                    io_functions.specific_functions.dump_pickle(dataset_sel, f)

    def update_fields(self):
        self.lb_dataset.delete(0,tk.END)
        for i in range(self.datasets['n']):
            self.lb_dataset.insert('end', self.datasets['desc'][i])

        keys = [list(dataset) for dataset in self.datasets['dataset']]
        self.common_monstations = np.unique(keys)
        self.lb_mon.delete(0,tk.END)
        for x in self.common_monstations:
            self.lb_mon.insert('end', x)
    
    def close_app(self):  
        self.root.quit()
        self.root.destroy()

class Compare():
    def __init__(self):
        pass
    
    def run(self):
        root = tk.Tk()
        root.title("Loads Compare")
        root.resizable(True, True)
                
        style = ttk.Style()
        style.theme_use('clam')
        app = App(root)
        root.mainloop()
    
    
if __name__ == "__main__":
    c = Compare()
    c.run()
    
    
    