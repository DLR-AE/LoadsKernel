# Loads Kernel
The Loads Kernel Software allows for the calculation of quasi-steady and dynamic maneuver loads, unsteady gust loads in the time and frequency domain as well as dynamic landing loads based on a generic landing gear module.

For more information, see: https://wiki.dlr.de/display/AE/Lastenrechnung%3A+Loads+Kernel

# References

Voß, A., “An Implementation of the Vortex Lattice and the Doublet Lattice Method,” Institut für Aeroelastik, Deutsches Zentrum für Luft- und Raumfahrt, Göttingen, Germany, Technical Report DLR-IB-AE-GO-2020-137, Oktober 2020, https://elib.dlr.de/136536/.

Voß, A., “Loads Kernel User Guide, Version 1.01,” Institut für Aeroelastik, Deutsches Zentrum für Luft- und Raumfahrt, Göttingen, Germany, Technical Report DLR-IB-AE-GO-2020-136, Nov. 2021, https://elib.dlr.de/140268/.

# Continuous Integration

Master branch [![pipeline status](https://gitlab.dlr.de/loads-kernel/loads-kernel/badges/master/pipeline.svg)](https://gitlab.dlr.de/loads-kernel/loads-kernel/-/commits/master)

Development branch [![pipeline status](https://gitlab.dlr.de/loads-kernel/loads-kernel/badges/devel/pipeline.svg)](https://gitlab.dlr.de/loads-kernel/loads-kernel/-/commits/devel)

Test coverage [![coverage](https://gitlab.dlr.de/loads-kernel/loads-kernel/badges/master/coverage.svg)](https://loads-kernel.pages.gitlab.dlr.de/loads-kernel/coverage/)

# Installation & Use
## User installation 
To install everything as a python package, including dependencies:

```
pip install --user git+https://gitlab.dlr.de/loads-kernel/loads-kernel.git
```

## Examples
There are a number of typical examples, which cover different analyses and simulations. The examples are stored in an additional repository:

```
git clone https://gitlab.dlr.de/loads-kernel/loads-kernel-examples.git
```

## How can I use it?

Make sure to adjust the launch script (launch.py, located in the input folder) to your needs / for your aircraft configuration. Then, launch the python script with:

```
python launch.py
```

If ~/.local/bin is in your system PATH, you can use the following commands from the command line:

```
loads-kernel --job_name jcl_Discus2c --pre True --main True --post True --path_input /path/to/Discus2c/JCLs --path_output /path/to/Discus2c/output
```

There are two GUIs to visualize a simulation model (the Model Viewer) and to compare different sets of loads (Loads Compare), which can be started from the command line as well:

```
model-viewer
loads-compare
```

## Developer installation 
As above, but with access to the code (keep the code where it is so that you can explore and modify):

```
git clone https://gitlab.dlr.de/loads-kernel/loads-kernel.git
cd ./loads-kernel
pip install --user -e . 
```

## Feedback
Please provide your feedback via merge requests (please see [CONTRIBUTING.md](CONTRIBUTING.md)
for details) or contact contact Arne Voß, arne.voss@dlr.de.