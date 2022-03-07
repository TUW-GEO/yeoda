<img align="right" src="https://github.com/TUW-GEO/yeoda/raw/master/docs/imgs/yeoda_logo.png" height="300" width="435">

# yeoda
[![Build Status](https://travis-ci.com/TUW-GEO/yeoda.svg?branch=master)](https://travis-ci.org/TUW-GEO/yeoda)
[![Coverage Status](https://coveralls.io/repos/github/TUW-GEO/yeoda/badge.svg?branch=master)](https://coveralls.io/github/TUW-GEO/yeoda?branch=master)
[![PyPi Package](https://badge.fury.io/py/yeoda.svg)](https://badge.fury.io/py/yeoda)
[![RTD](https://readthedocs.org/projects/yeoda/badge/?version=latest)](https://yeoda.readthedocs.io/en/latest/?badge=latest)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

*Earth Observation (EO) data, I must read.*

<!-- toc -->
## Contents

  * [Description](#description)
  * [Limitations and Outlook](#limitations-and-outlook)
  * [Installation](#installation)
    * [pip](#pip)
    * [conda](#conda)
  * [Examples](#examples)
    * [Loading data from TIFF Files](#loading-data-from-tiff-files)
    * [Read raster data by pixel coordinates](#read-raster-data-by-pixel-coordinates)
    * [Read rasta data by bounding box](#read-rasta-data-by-bounding-box)
  * [Contribution](#contribution)
  * [Citation](#citation)<!-- endToc -->

## Description
*yeoda* stands for **y**our **e**arth **o**bservation **d**ata **a**ccess and provides lower and higher-level data cube 
classes to work with well-defined and structured earth observation data. These data cubes allow to filter, split and load data independently from the way the data is structured on the hard disk. Once the data structure is known to *yeoda*, it offers a user-friendly interface to access the data with the aforementioned operations.
Internally, the package relies on functionalities provided by [*geopathfinder*](https://github.com/TUW-GEO/geopathfinder) 
(filepath/filename and folder structure handling library), [*veranda*](https://github.com/TUW-GEO/veranda) (IO classes and higher-level data structure classes for vector and raster data)
and [*geospade*](https://github.com/TUW-GEO/geospade) (raster and vector geometry definitions and operations).
Moreover, another very important part of *yeoda* is that it deals with pre-defined grids like the [*Equi7Grid*](https://github.com/TUW-GEO/Equi7Grid) or the [*LatLonGrid*](https://github.com/TUW-GEO/latlongrid).
These grid packages can simplify and speed up spatial operations to identify tiles/files of interest (e.g, bounding box request by a user).

## Limitations and outlook
At the moment the functionality of *yeoda* is limited in terms of flexibility with different file types, bands and 
tiles, e.g. you can only load data from one tile and one band. This will change in the future by allowing to load data also independently from tile boundaries, bands and file types.
Most changes will take place in *veranda* and *geospade*, so the actual interface to the data given by *yeoda* should stay approximately the same.

## Installation
The package can be either installed via pip or if you solely want to work with *yeoda* or contribute, we recommend installing
it as a conda environment. If you work already with your own environment, please have look at ``conda_env.yml`` or ``setup.cfg`` for the required dependencies.

### Pip
To install *yeoda* via pip in your own environment, use:
```
pip install yeoda
```
**ATTENTION**: Packages like *gdal*, *cartopy*, or *geopandas* need more OS support and have more dependencies than other packages and can therefore not be installed solely via pip.
Thus, for a fresh setup, an existing environment with the conda dependencies listed in ``conda_env.yml`` is expected.
To create such an environment, you can run:
```
conda create -n "yeoda" -c conda-forge python=3.7 gdal=3.0.2 geopandas cartopy
```

### Conda
The packages also comes along with a pre-defined conda environment (``conda_env.yml``). 
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

## Examples
This section demonstrates basic usage of a *yeoda* datacube, specifically how to load raster data and read it using pixel 
coordinates or geometry definitions.

### Loading data from GeoTIFF files
You can create a datacube from a collection of GeoTIFF files, by passing them into the constructor of the `EODataCube` and
specifying the type of naming convention you want to use via the `filename_class` parameters. This can take any 
`SmartFilename` class, see [*geopathfinder*](https://github.com/TUW-GEO/geopathfinder) for details. The `dimensions` parameter
defines the columns you want to read into the datacube's inventory and the values for these are usually parsed from the
`SmartFilename`. The `sdim_name` defines the spatial dimension.

<!-- snippet: create_and_filter_datacube -->
<a id='snippet-create_and_filter_datacube'></a>
```py
dc = EODataCube(filepaths=filepaths, filename_class=SgrtFilename,
                dimensions=['time', 'var_name', 'pol', 'tile_name', 'orbit_direction'], sdim_name="tile_name")

dc.filter_by_dimension('VV', name='pol', inplace=True)
dc.filter_by_dimension('SIG0', name='var_name', inplace=True)
dc.filter_by_dimension('D', name='orbit_direction', inplace=True)
dc.filter_spatially_by_tilename('E042N012T6', inplace=True, use_grid=False)
```
<sup><a href='/tests/test_loading.py#L82-L90' title='Snippet source file'>snippet source</a> | <a href='#snippet-create_and_filter_datacube' title='Start of snippet'>anchor</a></sup>
<!-- endSnippet -->

### Read raster data by pixel coordinates
The datacube's `load_by_pixels` allows you to read raster data from specifying a region of interest in pixels. It will
automatically crop it the requested size and handle tile boundaries. The `dtype` parameter determines the data type the
function will return, in this case a numpy array (see [numpy](https://numpy.org/) for details).

<!-- snippet: data_cube_load_numpy_by_pixels -->
<a id='snippet-data_cube_load_numpy_by_pixels'></a>
```py
data = dc.load_by_pixels(970, 246, row_size=10, col_size=16, dtype='numpy')
```
<sup><a href='/tests/test_loading.py#L269-L271' title='Snippet source file'>snippet source</a> | <a href='#snippet-data_cube_load_numpy_by_pixels' title='Start of snippet'>anchor</a></sup>
<!-- endSnippet -->

### Read raster data by bounding box
Using the datacube's `load_by_geom` you can specify for instance a bounding box geometry and *yeoda* will load and 
return the raster data covered by it. The `dtype` parameters determines the data type that will be returned, in this
example a numpy array (see [numpy](https://numpy.org/) for details).

<!-- snippet: data_cube_load_numpy_by_bbox -->
<a id='snippet-data_cube_load_numpy_by_bbox'></a>
```py
bbox = [(4323250, 1309750), (4331250, 1314750)]
data = dc.load_by_geom(bbox, dtype='numpy')
```
<sup><a href='/tests/test_loading.py#L416-L419' title='Snippet source file'>snippet source</a> | <a href='#snippet-data_cube_load_numpy_by_bbox' title='Start of snippet'>anchor</a></sup>
<!-- endSnippet -->


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
  
## Citation

[![DOI](https://zenodo.org/badge/186986862.svg)](https://zenodo.org/badge/latestdoi/186986862)

If you use this software in a publication then please cite it using the Zenodo DOI.
Be aware that this badge links to the latest package version.

Please select your specific version at <https://doi.org/10.5281/zenodo.3540693> (link to first release) to get the DOI of that version.
You should normally always use the DOI for the specific version of your record in citations.
This is to ensure that other researchers can access the exact research artefact you used for reproducibility.

You can find additional information regarding DOI versioning at <http://help.zenodo.org/#versioning>.
