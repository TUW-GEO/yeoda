<img align="right" src="./docs/imgs/yeoda_logo.png" height="300" width="435">

# yeoda
[![Build Status](https://travis-ci.org/TUW-GEO/yeoda.svg?branch=master)](https://travis-ci.org/TUW-GEO/yeoda)
[![Build Status](https://coveralls.io/repos/TUW-GEO/yeoda/yeoda.png?branch=master)](https://coveralls.io/r/TUW-GEO/yeoda?branch=master)
[![Build Status](https://badge.fury.io/py/yeoda.svg)](https://badge.fury.io/py/yeoda)
[![Build Status](https://readthedocs.org/projects/yeoda/badge/?version=latest)](https://yeoda.readthedocs.io/en/latest/?badge=latest)


*Earth Observation (EO) data, I must read.*

## Description

*yeoda* stands for **y**our **e**arth **o**bservation **d**ata **a**ccess and provides lower and higher-level data cube 
classes to work with well-defined and structured earth observation data. These data cubes allow to filter, split and load data independently from the way the data is structured on the hard disk.  
Once the data structure is known to *yeoda*, it offers a user-friendly interface to access the data with the aforementioned operations.
Internally, the package relies on functionalities provided by [*geopathfinder*](https://github.com/TUW-GEO/geopathfinder) 
(filepath/filename and folder structure handling library), [*veranda*](https://github.com/TUW-GEO/veranda) (IO classes and higher-level data structure classes for vector and raster data)
and [*geospade*](https://github.com/TUW-GEO/geospade) (raster and vector geometry definitions and operations).
Moreover, another very important part of *yeoda* is work with pre-defined grids like the [*Equi7Grid*](https://github.com/TUW-GEO/Equi7Grid) or the [*LatLonGrid*](https://github.com/TUW-GEO/latlongrid).
These grid packages can simplify and speed up spatial operations to identify tiles/files of interest (e.g, bounding box request by a user).

## Examples
The following examples shall help to understand what the package is able to accomplish 
 and how you can use the interface *yeoda* offers to access and play around with your data.
### Setting up a data cube
In simple words, *yeoda* is a filename based data cube tool, which means that it tries to interpret the data structure via the filename.
In the future it will be also possible to create a data cube based on metadata or dataset attributes. To define a filenaming convention, 
*geopathfinder* can be used. Each (existing) filenaming convention has a ``create_[naming_convention]_filename(...)`` function to create a Python object, 
which can be handled like a dictionary to access parts of the filename. 

First, to setup a data cube, you need to prepare some input attributes:
  * A list of filepaths with the same extension. Currently GeoTIFF and NetCDF files are supported as default by *veranda*.
  * A list of dimensions you want you work with. The dimension names relate to the keys defined by filenaming convention.
  * A function to create a Python object/class instance representing a filenaming convention.
  * A grid, which is a class instance of ``pytileproj.base.TiledProjection`` being inherited to a grid package, e.g. *Equi7Grid*.
You can the initiate a data cube object with the ``EODataCube`` class:
```
dc = EODataCube(filepaths=filepaths, smart_filename_creator=smart_filename_creator,
                dimensions=['time', 'var_name', 'pol'], grid=grid)
```
### Data cube operations
*yeoda* uses a *GeoPandas* dataframe internally to store the filename and geometry information.
On top of that, data cube functions where defined to filter, split, sort, align, etc. the data. 
### Loading data
## Limitations and Outlook
## Installation
The package can be either installed via pip or if you solely want to work with *yeoda* or contribute, we recommend to 
install it as conda environment.
### pip
To install *yeoda* via pip in you own environment, use:
```
pip install yeoda
```
### conda
The packages also comes along with an own conda environment (``conda_env.yml``). 
This is especially recommended if you want to contribute to the project.
The following script will install miniconda and setup the environment on a UNIX
like system. Miniconda will be installed into ``$HOME/miniconda``.
```
wget http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p $HOME/miniconda
export PATH="$HOME/miniconda/bin:$PATH"
conda env create -f conda_env.yml
source activate yeoda
```
This script adds ``$HOME/miniconda/bin`` temporarily to the ``PATH`` to do this
permanently add ``export PATH="$HOME/miniconda/bin:$PATH"`` to your ``.bashrc``
or ``.zshrc``.

For Windows, use the following setup:
  * Download the latest [miniconda 3 installer](https://docs.conda.io/en/latest/miniconda.html) for Windows
  * Click on ``.exe`` file and complete the installation.
  * Add the folder ``condabin`` folder to your environment variable ``PATH``. 
    You can find the ``condabin`` folder usually under: ``C:\Users\username\AppData\Local\Continuum\miniconda3\condabin``
  * Finally, you can set up the conda environment via:
    ```
    conda env create -f conda_env.yml
    source activate yeoda
    ```
    
After that you should be able to run 
```
python setup.py test
```
to run the test suite.

## Contribution
We are happy if you want to contribute. Please raise an issue explaining what
is missing or if you find a bug. We will also gladly accept pull requests
against our master branch for new features or bug fixes.
If you want to contribute please follow these steps:

  * Fork the *yeoda* repository to your account
  * Clone the *yeoda* repository
  * Make a new feature branch from the *yeoda* master branch
  * Add your feature
  * Please include tests for your contributions in one of the test directories.
    We use *py.test* so a simple function called ``test_my_feature`` is enough
  * Submit a pull request to our master branch