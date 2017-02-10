# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 17:45:02 2017

@author: voss_ar
"""


import Tkinter as tk
import ttk
import tkFileDialog
import matplotlib as mpl
mpl.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.backend_bases import key_press_handler # implement the default mpl key bindings
from plotting import Plotting

import numpy as np
import cPickle, os, copy

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
        self.colors = ['cornflowerblue', 'limegreen', 'violet']
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
        #self.file_opt['defaultextension'] = '.txt'
        self.file_opt['filetypes']  = [('pickle files', '.pickle'), ('all files', '.*')]
        self.file_opt['initialdir'] = os.getcwd()
        self.file_opt['title']      = 'This is a title'
        
        # --- init GUI ---

        # init menu
        root.option_add('*tearOff', tk.FALSE)        
        menubar = tk.Menu(self.root)
        menu_file = tk.Menu(menubar)
        menu_file.add_command(label='Load Monstations', command=self.load_monstation)
        menu_file.add_command(label='Close', command=self.close_app)
        menubar.add_cascade(menu=menu_file, label='File')
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
        #frame_right.grid_columnconfigure(2, weigth=1)
        #frame_right.grid_rowconfigure(0, weight=1)
        root.grid_columnconfigure(2, weight=1)
        root.grid_rowconfigure(1, weight=1)

        # ListBox to select several datasets 
        self.lb_dataset = tk.Listbox(frame_left_top, height=5, selectmode=tk.EXTENDED, exportselection=False)
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
        self.lb_mon = tk.Listbox(frame_left_top, height=5, selectmode=tk.SINGLE, exportselection=False)
        self.lb_mon.grid(row=0, column=2, sticky=(tk.N,tk.W,tk.E,tk.S))
        self.lb_mon.bind('<<ListboxSelect>>', self.show_choice)
        s_mon = ttk.Scrollbar(frame_left_top, orient=tk.VERTICAL, command=self.lb_mon.yview)
        s_mon.grid(row=0, column=3, sticky=(tk.N,tk.S))
        # attach listbox to scrollbar
        self.lb_mon.config(yscrollcommand=s_mon.set)
        s_mon.config(command=self.lb_mon.yview)

        # Eintraege hinzufuegen mit self.l.insert('end', 'Line 1')
        # abfrage der Werte mit self.l.curselection()

        # x- and y-axis
        cb_xaxis = ttk.Combobox(frame_left_top, textvariable=self.var_xaxis, values=self.dof.keys(), state='readonly')
        cb_xaxis.grid(row = 3, column =0, columnspan=2, sticky=(tk.W,tk.E))
        cb_xaxis.bind('<<ComboboxSelected>>', self.show_choice )
        
        cb_yaxis = ttk.Combobox(frame_left_top, textvariable=self.var_yaxis, values=self.dof.keys(), state='readonly')
        cb_yaxis.grid(row = 4, column =0, columnspan=2, sticky=(tk.W,tk.E))
        cb_yaxis.bind('<<ComboboxSelected>>', self.show_choice )
        
        # options / check boxes
        self.show_hull = tk.BooleanVar()
        cb_hull = ttk.Checkbutton( frame_left_top, text="show convex hull", variable=self.show_hull,  onvalue=tk.TRUE, offvalue=tk.FALSE, command=self.update_plot )        
        cb_hull.grid(row = 5, column =0, columnspan=2, sticky=(tk.W,tk.E))
        
        self.show_labels = tk.BooleanVar()
        cb_labels = ttk.Checkbutton( frame_left_top, text="show labels", variable=self.show_labels,  onvalue=tk.TRUE, offvalue=tk.FALSE, command=self.update_plot )        
        cb_labels.grid(row = 6, column =0, columnspan=2, sticky=(tk.W,tk.E))
        
        
        # init Matplotlib Plot
        fig1 = mpl.figure.Figure()
        subplot = fig1.add_subplot(111)  
        # hand over subplot to plotting class
        self.plotting = Plotting(subplot)
        # embed figure
        self.canvas = FigureCanvasTkAgg(fig1, master=frame_right)
        self.canvas.show()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        toolbar = NavigationToolbar2TkAgg(self.canvas, frame_right)
        toolbar.update()
        self.canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=1)    
        
        # button that will update the plot
        button_update_plot  = ttk.Button(frame_center, text='>', width=2,  command=self.update_plot).grid(row=0, column=2)            
    
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
            dataset_sel = [self.datasets['dataset'][i] for i in self.lb_dataset.curselection()]
            color_sel   = [self.datasets['color'][i] for i in self.lb_dataset.curselection()]
            desc_sel    = [self.datasets['desc'][i] for i in self.lb_dataset.curselection()]
            mon_sel     = self.common_monstations[self.lb_mon.curselection()]
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
        else:    
            self.plotting.plot_nothing()
        self.canvas.draw()
            
    def load_monstation(self):
        # open file dialog
        filename = tkFileDialog.askopenfilename(**self.file_opt)
        if filename != '':
            # load pickle
            dataset = self.load_pickle(filename)
            # save into data structure
            self.datasets['ID'].append(self.datasets['n'])  
            self.datasets['dataset'].append(dataset)
            self.datasets['color'].append(self.colors[self.datasets['n']])
            self.datasets['desc'].append('dataset '+ str(self.datasets['n']))
            self.datasets['n'] += 1
            # update fields
            self.update_fields()

        
    def load_pickle(self, filename):
        with open(filename, 'r') as f:
            data = cPickle.load(f)
        return data
    
    def update_fields(self):
        self.lb_dataset.delete(0,tk.END)
        for i in range(self.datasets['n']):
            self.lb_dataset.insert('end', self.datasets['desc'][i])

        keys = [dataset.keys() for dataset in self.datasets['dataset']]
        self.common_monstations = np.unique(keys)
        self.lb_mon.delete(0,tk.END)
        for x in self.common_monstations:
            self.lb_mon.insert('end', x)
    
    def close_app(self):  
        print "Closing App."
        self.root.quit()
        self.root.destroy()


root = tk.Tk()
root.title("Loads Compare")
#root.grid_columnconfigure(2, weight=1)
#root.grid_rowconfigure(0, weight=1)
root.resizable(True, True)
        
style = ttk.Style()
style.theme_use('clam')
app = App(root)
root.mainloop()