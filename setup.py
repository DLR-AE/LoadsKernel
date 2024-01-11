"""
Setup file
Install Loads Kernel with core dependencies via:
- pip install -e <local_repo_path>
To use the graphical tools and other features, optional libraries definded as extras are necessary:
- pip install -e <repo_path>[extra]
Especially with mpi or the graphical libraries, pip frequently fails. In that case, try to install the packages using a
package manager such as conda.
"""

from setuptools import setup, find_packages


def my_setup():
    setup(name='Loads-Kernel',
          version='2023.08',
          description="""The Loads Kernel Software allows for the calculation of quasi-steady and dynamic maneuver loads,
          unsteady gust loads in the time and frequency domain as well as dynamic landing loads based on a generic landing
          gear module.""",
          url='https://github.com/DLR-AE/LoadsKernel',
          author='Arne Voß',
          author_email='arne.voss@dlr.de',
          license='BSD 3-Clause License',
          packages=find_packages(),
          entry_points={'console_scripts': ['loads-kernel=loadskernel.program_flow:command_line_interface',
                                            'model-viewer=modelviewer.view:command_line_interface',
                                            'loads-compare=loadscompare.compare:command_line_interface']},
          include_package_data=True,
          package_data={'loadskernel': ['graphics/*.*'],
                        'loadscompare': ['graphics/*.*'], },
          python_requires='>=3.8',
          install_requires=['Panel-Aero @ git+https://github.com/DLR-AE/PanelAero.git',
                            'matplotlib',
                            'numpy',
                            'scipy',
                            'psutil',
                            'h5py',
                            'tables',
                            'pyyaml',
                            'pandas',
                            ],
          extras_require={'extras': ['mpi4py',
                                     'pyfmi',
                                     'mayavi',
                                     'traits',
                                     'traitsui',
                                     'pyface',
                                     'pyiges @ git+https://github.com/pyvista/pyiges.git',
                                     ],
                          'test': ['pytest',
                                   'pytest-cov',
                                   'jupyter',
                                   'jupyter-book',
                                   ]},
          )


if __name__ == '__main__':
    my_setup()
