# Loads Kernel
The Loads Kernel Software allows for the calculation of quasi-steady and dynamic maneuver loads, unsteady gust loads in the time and frequency domain as well as dynamic landing loads based on a generic landing gear module.

# References

[1] Voß, A., “Loads Kernel User Guide, Version 1.01,” Institut für Aeroelastik, Deutsches Zentrum für Luft- und Raumfahrt, Göttingen, Germany, Technical Report DLR-IB-AE-GO-2020-136, Nov. 2021, https://elib.dlr.de/140268/.

[2] Voß, A., “An Implementation of the Vortex Lattice and the Doublet Lattice Method,” Institut für Aeroelastik, Deutsches Zentrum für Luft- und Raumfahrt, Göttingen, Germany, Technical Report DLR-IB-AE-GO-2020-137, Oktober 2020, https://elib.dlr.de/136536/.

If you use this software for your scientific work, we kindly ask you to include a reference [1,2] in your publications. Thank you!

# Installation & Use
## User installation 
To install everything as a python package, including dependencies:

```
pip install --user git+https://github.com/DLR-AE/LoadsKernel.git
```

## How can I use it?

Adjust the launch script (launch.py, located in the input folder) to your needs / for your aircraft configuration. Then, launch the python script with:

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
git clone https://github.com/DLR-AE/LoadsKernel.git
cd ./loads-kernel
pip install --user -e . 
```

# License
This software is developed for scientific applications and is delivered as open source without any liability (BSD 3-Clause, please see [LICENSE](LICENSE) for details). For every new aircraft, a validation against test data and/or other simulation tools is highly recommended and in the responsibility of the user. 

If you use this software for your scientific work, we kindly ask you to include a reference [1,2] in your publications. Thank you!

# Feedback & Support
Note that this is a scientific software for users with a background in aerospace engineering and with a good understanding and experience in aeroelasticity. If you know what you are doing - go ahead and have fun! If you need specific help or assistence, we offer commerical support:
- Development of additional, proprietary features
- Consulting & Training courses
- Service & Support

We are interested in partnerships from both industry and academia, so feel free to contact us (arne.voss@dlr.de).

If you discoverd an obvious bug, please open an [issue](https://github.com/DLR-AE/LoadsKernel/issues). In case you already know how to fix it, please provide your feedback via merge requests. For details, please see the [instructions](CONTRIBUTING.md) on how to provide a contribution or contact arne.voss@dlr.de if you need any assistance with that.

# Internal Part (DLR)

## Examples
There are a number of typical examples, which cover different analyses and simulations. The examples are stored in an additional (internal) DLR GitLab repository:

```
git clone https://gitlab.dlr.de/loads-kernel/loads-kernel-examples.git
```

## Continuous Integration
Status of the (internal) DLR GitLab continuous integration pipelines:

Master branch [![pipeline status](https://gitlab.dlr.de/loads-kernel/loads-kernel/badges/master/pipeline.svg)](https://gitlab.dlr.de/loads-kernel/loads-kernel/-/commits/master)

Development branch [![pipeline status](https://gitlab.dlr.de/loads-kernel/loads-kernel/badges/devel/pipeline.svg)](https://gitlab.dlr.de/loads-kernel/loads-kernel/-/commits/devel)

Test coverage [![coverage](https://gitlab.dlr.de/loads-kernel/loads-kernel/badges/master/coverage.svg)](https://loads-kernel.pages.gitlab.dlr.de/loads-kernel/coverage/)
