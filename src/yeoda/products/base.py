# Copyright (c) 2019, Vienna University of Technology (TU Wien), Department of
# Geodesy and Geoinformation (GEO).
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# The views and conclusions contained in the software and documentation are
# those of the authors and should not be interpreted as representing official
# policies, either expressed or implied, of the FreeBSD Project.

"""
Main code for creating a TUWGEO product data cube.
"""

# import geopathfinder modules for file and folder naming convention
from geopathfinder.naming_conventions.sgrt_naming import SgrtFilename
from geopathfinder.naming_conventions.sgrt_naming import sgrt_tree

# import TUWGEO standard grid
from equi7grid.equi7grid import Equi7Grid

# import basic data cube from yeoda
from yeoda.datacube import EODataCube


class ProductDataCube(EODataCube):
    """
    Data cube defining a TUWGEO product (preferably based on the Equi7 grid system).
    """

    def __init__(self, root_dirpath=None, var_names=None, sres=10, continent='EU', dimensions=None,
                 file_pattern=".tif$", **kwargs):

        """
        Constructor of class `ProductDataCube`.

        Parameters
        ----------
        root_dirpath : str, optional
            Root directory path to the SGRT directory tree.
        var_names : list, optional
            SGRT Variable names , e.g. ["SIG0", "SSM"].
        sres : int, optional
            Spatial sampling in grid units, e.g. 10, 500 (default is 10).
        continent : str, optional
            Continent/Subgrid of the Equi7Grid system (default is 'EU').
        dimensions : list, optional
            List of filename parts to use as dimensions. The strings have to match with the keys of the `SgrtFilename`
            fields definition.
        file_pattern : str
            Pattern to match/only select certain file names.
        **kwargs
            Arbitrary keyword arguments (e.g. containing 'inventory' or 'grid').
        """

        if root_dirpath is not None:
            dir_tree = sgrt_tree(root_dirpath, register_file_pattern=file_pattern)
        else:
            dir_tree = kwargs.get('dir_tree', None)

        filepaths = kwargs.get('filepaths', None)
        if filepaths is None and dir_tree is not None:
            filepaths = dir_tree.file_register

        inventory = kwargs.get('inventory', None)

        # ensure there is a variable dimension
        if dimensions is None:
            dimensions = ['var_name']
        elif "var_name" not in dimensions:
            dimensions.append('var_name')

        grid = kwargs.get('grid', Equi7Grid(sres).__getattr__(continent))

        # add spatial and temporal dimension
        if 'tile_name' not in dimensions:
            dimensions.append('tile_name')
        sdim_name = kwargs.get('sdim_name', 'tile_name')
        if 'time' not in dimensions:
            dimensions.append('time')
        tdim_name = kwargs.get('tdim_name', 'time')

        super().__init__(inventory=inventory, filepaths=filepaths, dimensions=dimensions,
                         smart_filename_class=SgrtFilename, grid=grid, sdim_name=sdim_name, tdim_name=tdim_name)

        # filter variable names
        if var_names is not None:
            self.filter_by_dimension(var_names, name='var_name', inplace=True)

        # file pattern has not been applied yet and file pattern is given
        if dir_tree is None and file_pattern is not None:
            self.filter_files_with_pattern(file_pattern, inplace=True)
