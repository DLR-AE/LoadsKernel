# Note
New releases are marked in the repository using tags. Simply checkout the master branch for the lastest version or use git checkout if you require a specific release, for example 'git checkout 2022.10'.

# Release 2023.06
- Publication as Open Source under the BSD 3-Clause License 
- Update of the documentation to Version 1.04
- Extended interface to the SU2 CFD solver for time domain simulations
- Improved bdf reader

# Release 2022.10
- Administrative changes only (improved the Readme, added issue templates, created this changelog, etc.)

# Release 2022.08
- Python 3.8 compatibility
- Improved mesh deformation based on the local meshes of the mpi pcrocess
- New multiprocessing, which uses MPI instead of python's multiprocessing toolbox
- Loads from propeller aerodynmics
- Integration of system matrices (mass and stiffness) from the B2000 FE solver
