"""
Setup file, currently supports:

- installation via "pip install --user -e <repo_path>"
- installation via "python setup.py develop --user"
"""

from setuptools import setup, find_packages

def my_setup():
    setup(name='Loads-Kernel',
          version='2021.09',
          description='The Loads Kernel Software allows for the calculation of quasi-steady and dynamic maneuver loads, unsteady gust loads in the time and frequency domain as well as dynamic landing loads based on a generic landing gear module.',
          url='https://wiki.dlr.de/display/AE/Lastenrechnung%3A+Loads+Kernel',
          author='Arne Voß',
          author_email='arne.voss@dlr.de',
          license='internal use',
          packages=find_packages(),
          entry_points={'console_scripts': ['loads-kernel=loadskernel.program_flow:command_line_interface',
                                            'model-viewer=modelviewer.view:command_line_interface',
                                            'loads-compare=loadscompare.compare:command_line_interface']},
          include_package_data=True,
          package_data={'loadskernel': ['graphics/*.*'],
                        'loadscompare': ['graphics/*.*'],},
          python_requires='>=3.7',
          install_requires=[
                            'Panel-Aero @ git+https://gitlab.dlr.de/loads-kernel/panel-aero.git',
                            'matplotlib',
                            'mayavi',
                            'traits', 
                            'traitsui', 
                            'pyface', 
                            'pyiges @ git+https://github.com/pyvista/pyiges.git',
                            'numpy',
                            'scipy',
                            'psutil',
                            'pyfmi',
                            'h5py',
                            'tables',
                            'mpi4py',
                            ],
          )

if __name__ == '__main__':
    my_setup()
