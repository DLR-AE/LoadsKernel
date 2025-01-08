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
    setup(name='LoadsKernel',
          version='2025.01',
          description="""The Loads Kernel Software allows for the calculation of quasi-steady and dynamic maneuver loads,
          unsteady gust loads in the time and frequency domain as well as dynamic landing loads based on a generic landing
          gear module.""",
          long_description=open('README.md', encoding='utf8').read(),
          long_description_content_type='text/markdown',
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
          python_requires='>=3.10',
          install_requires=['PanelAero',
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
                                     'mayavi',
                                     'traits',
                                     'traitsui',
                                     'pyface',
                                     'jupyter',
                                     'pyiges',  # only available with pip, not with conda
                                     ],
                          'difficult': ['pyfmi',  # frequent version conflicts
                                        ],
                          'test': ['pytest',
                                   'pytest-cov',
                                   'jupyter-book',
                                   'flake8',
                                   'pylint',
                                   ]},
          )


if __name__ == '__main__':
    my_setup()
