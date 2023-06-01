"""
Setup file 
Install Loads Kernel via: 
- pip install --user -e <repo_path>
In case Panel-Aero is not yet installed: 
- pip install git+https://gitlab.dlr.de/loads-kernel/panel-aero.git
"""

from setuptools import setup, find_packages

def my_setup():
    setup(name='Loads-Kernel',
          version='2023.06',
          description='The Loads Kernel Software allows for the calculation of quasi-steady and dynamic maneuver loads, unsteady gust loads in the time and frequency domain as well as dynamic landing loads based on a generic landing gear module.',
          url='https://wiki.dlr.de/display/AE/Loads+Kernel%3A+Lastenrechnung',
          author='Arne Voß',
          author_email='arne.voss@dlr.de',
          license='BSD 3-Clause License',
          packages=find_packages(),
          entry_points={'console_scripts': ['loads-kernel=loadskernel.program_flow:command_line_interface',
                                            'model-viewer=modelviewer.view:command_line_interface',
                                            'loads-compare=loadscompare.compare:command_line_interface']},
          include_package_data=True,
          package_data={'loadskernel': ['graphics/*.*'],
                        'loadscompare': ['graphics/*.*'],},
          python_requires='>=3.8',
          install_requires=[
                            'Panel-Aero @ git+https://github.com/DLR-AE/PanelAero.git',
                            'matplotlib',
                            'mayavi',
                            'traits', 
                            'traitsui', 
                            'pyface', 
                            'pyiges @ git+https://github.com/pyvista/pyiges.git',
                            'numpy',
                            'scipy',
                            'psutil',
                            'h5py',
                            'tables',
                            'mpi4py',
                            'pytest',
                            'pytest-cov',
                            'pyyaml',
                            'pandas',
                            ],
          extras_require={'FMI': ['pyfmi']},
          )

if __name__ == '__main__':
    my_setup()
