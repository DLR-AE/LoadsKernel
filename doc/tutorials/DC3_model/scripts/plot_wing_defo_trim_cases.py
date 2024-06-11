import h5py
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import os

path_root = pathlib.Path(__file__).parent.parent.parent.resolve()
path = os.path.join(path_root, 'DC3_results', 'response_jcl_dc3_trim.hdf5')
filename = path

# open HDF5 file
responses = h5py.File(filename, 'r')

# get response of first trim case
response = responses['0']
response2 = responses['1']
response3 = responses['2']
Ug_f = response['Ug_f'][:]
Ug_f2 = response2['Ug_f'][:]
Ug_f3 = response3['Ug_f'][:]

# extract ID, offset and set matrices
filename = '/data/carn_fr/DC3_LoadsKernel/model_jcl_dc3_more_flexibility3_trim_cases.hdf5'
model = h5py.File(filename, 'r')
strcgrid = model['strcgrid']
ID = strcgrid['ID'][:]
set = strcgrid['set'][:]
offset = strcgrid['offset'][:]

# IDs of the wing's LRA
ID_wings = np.array([54090001,54090002,54090003,54090004,54090005,54090006,54090007,54090008,54090009,54090010,54090011,54090012,54090013,54090014,54090015,54090016,54090017,54090018,54090019,54090020,54090021,54090022,54090023,54090024,54090025,54090026,54090027,54090028,54090029,54090030,54090031,64090001,64090002,64090003,64090004,64090005,64090006,64090007,64090008,64090009,64090010,64090011,64090012,64090013,64090014,64090015,64090016,64090017,64090018,64090019,64090020,64090021,64090022,64090023,64090024,64090025,64090026,64090027,64090028,64090029,64090030,64090031])

y = []
dz_pos = []
ry_pos = []
for i in range(len(ID_wings)):
    x = np.where(strcgrid['ID'][:]==ID_wings[i])
    dz_pos.append(strcgrid['set'][x[0], 2])
    ry_pos.append(strcgrid['set'][x[0], 4])
    y.append(strcgrid['offset'][x[0], 1][0])

dz= []
ry = []
for i in range(len(dz_pos)):
    dz.append(Ug_f[0, dz_pos[i]])
    ry.append(Ug_f[0, ry_pos[i]])

dz2= []
ry2 = []
for i in range(len(dz_pos)):
    dz2.append(Ug_f2[0, dz_pos[i]])
    ry2.append(Ug_f2[0, ry_pos[i]])

dz3= []
ry3 = []
for i in range(len(dz_pos)):
    dz3.append(Ug_f3[0, dz_pos[i]])
    ry3.append(Ug_f3[0, ry_pos[i]])

y_sorted_indices = np.argsort(y)
y_sorted = np.sort(y)
dz_sorted = [dz[i] for i in y_sorted_indices]
dz_sorted_sym = [-dz_sorted[i] for i in range(len(dz_sorted))]
ry_sorted = [ry[i] for i in y_sorted_indices]
dz2_sorted = [dz2[i] for i in y_sorted_indices]
dz2_sorted_sym = [-dz2_sorted[i] for i in range(len(dz2_sorted))]
ry2_sorted = [ry2[i] for i in y_sorted_indices]
dz3_sorted = [dz3[i] for i in y_sorted_indices]
dz3_sorted_sym = [-dz3_sorted[i] for i in range(len(dz3_sorted))]
ry3_sorted = [ry3[i] for i in y_sorted_indices]

ry_sorted = np.degrees(ry_sorted)
ry2_sorted = np.degrees(ry2_sorted)
ry3_sorted = np.degrees(ry3_sorted)

# Plots
fig, axs = plt.subplots(2, 1, figsize=(12, 6))
axs[0].plot(y_sorted, dz_sorted_sym, marker='o', linestyle='-', color='blue', label='Horizontal Level Flight (n$_Z$=1)')
axs[0].plot(y_sorted, dz2_sorted_sym, marker='o', linestyle='-', color='red', label='Push Down Maneuver (n$_Z$=-1)')
axs[0].plot(y_sorted, dz3_sorted_sym, marker='o', linestyle='-', color='green', label='Pull Up Maneuver (n$_Z$=2.5)')
axs[0].set_xlabel('Wing span [m]', fontsize='x-large')
axs[0].set_ylabel('U$_{flex,z}$  [m]', fontsize='x-large')
axs[0].grid(True)
axs[0].legend( fontsize='x-large')

axs[1].plot(y_sorted, ry_sorted, marker='o', linestyle='-', color='blue')
axs[1].plot(y_sorted, ry2_sorted, marker='o', linestyle='-', color='red')
axs[1].plot(y_sorted, ry3_sorted, marker='o', linestyle='-', color='green')
axs[1].set_xlabel('Wing span [m]', fontsize='x-large')
axs[1].set_ylabel('U$_{flex,ry}$  [deg]', fontsize='x-large')
axs[1].grid(True)
plt.tight_layout()
plt.show()
