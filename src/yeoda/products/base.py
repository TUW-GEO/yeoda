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

import numpy as np
# import geopathfinder modules for file and folder naming convention
from geopathfinder.naming_conventions.yeoda_naming import YeodaFilename

# import TUWGEO standard grid
from equi7grid.equi7grid import Equi7Grid

# import basic data cube from yeoda
from yeoda.datacube import EODataCube


class ProductDataCube(EODataCube):
    """
    Data cube defining a TUWGEO product (preferably based on the Equi7 grid system).

    """

    def __init__(self, var_names=None, sres=10, continent='EU',
                 scale_factor=10, nodata=-9999, dtype='Int16',
                 dimensions=None, filename_class=YeodaFilename,
                 vdim_name='var_name', tdim_name='time', sdim_name='tile_name',
                 **kwargs):

        """
        Constructor of class `ProductDataCube`.

        Parameters
        ----------
        var_names : list, optional
            Variable names , e.g. ["SIG0", "SSM"].
        sres : int, optional
            Spatial sampling in grid units, e.g. 10, 500 (default is 10).
        continent : str, optional
            Continent/Subgrid of the Equi7Grid system (default is 'EU').
        scale_factor : float, optional
            Scale factor used for encoding and decoding (defaults to 10).
        nodata : float, optional
            No data value used for encoding and decoding (defaults to -9999).
        dtype : str, optional
            Data type used for encoding (defaults to 'Int16').
        dimensions : list, optional
            List of filename parts to use as dimensions. The strings have to match with the field definitions in
            `filename_class`.
        filename_class : geopathfinder.file_naming.SmartFilename, optional
            `SmartFilename` class to handle the interpretation of file names (defaults to YeodaFilename).
        vdim_name : str, optional
            Name of variable dimension (defaults to 'var_name').
        tdim_name : str, optional
            Name of temporal dimension (defaults to 'time').
        sdim_name : str, optional
            Name of spatial dimension (defaults to 'tile_name').
        **kwargs
            Arbitrary keyword arguments (e.g. containing 'inventory' or 'grid').

        """

        filepaths = kwargs.get('filepaths', None)
        inventory = kwargs.get('inventory', None)

        # ensure there is a variable dimension
        if dimensions is None:
            dimensions = [vdim_name]
        elif vdim_name not in dimensions:
            dimensions.append(vdim_name)

        # ensure there is a temporal dimension
        if tdim_name not in dimensions:
            dimensions.append(tdim_name)

        # ensure there is a spatial dimension
        if sdim_name not in dimensions:
            dimensions.append(sdim_name)

        grid = kwargs.get('grid', Equi7Grid(sres).__getattr__(continent))

        super().__init__(inventory=inventory, filepaths=filepaths, dimensions=dimensions,
                         filename_class=filename_class, grid=grid, sdim_name=sdim_name, tdim_name=tdim_name)

        # filter variable names
        if var_names is not None:
            self.filter_by_dimension(var_names, name='var_name', inplace=True)

        self._vdim_name = vdim_name
        self._scale_factor = scale_factor
        self._nodata = nodata
        self._dtype = np.dtype(dtype.lower()) if isinstance(dtype, str) else dtype

    def _assign_inventory(self, inventory, inplace=True):
        """
        Helper method for either create a new data cube or overwrite the old data cube with the given inventory.

        Parameters
        ----------
        inventory : GeoDataFrame
            Data cube inventory.
        inplace : boolean, optional
            If true, the current class instance will be altered.
            If false, a new class instance will be returned (default value is False).

        Returns
        -------
        ProductDataCube

        """

        if self.sdim_name not in list(inventory.keys()):
            sdim_name = None
        else:
            sdim_name = self.sdim_name

        if self.tdim_name not in list(inventory.keys()):
            tdim_name = None
        else:
            tdim_name = self.tdim_name

        if inplace:
            self.inventory = inventory
            self.sdim_name = sdim_name
            self.tdim_name = tdim_name
            return self
        else:
            return self.from_inventory(inventory=inventory, grid=self.grid, io_maps=self.io_map,
                                       scale_factor=self._scale_factor, nodata=self._nodata, dtype=self._dtype,
                                       dimensions=self.dimensions, filename_class=self._filename_class,
                                       vdim_name=self._vdim_name, tdim_name=tdim_name, sdim_name=sdim_name)

    def encode(self, data, **kwargs):
        """
        Joint encoding function for TUWGEO product data.

        Parameters
        ----------
        data : np.ndarray
            Input data.

        Returns
        -------
        np.ndarray
            Encoded data.

        """

        data *= self._scale_factor
        data[np.isnan(data)] = self._nodata
        data.astype(self._dtype)
        return data

    def decode(self, data, **kwargs):
        """
        Joint decoding function for TUWGEO product data.

        Parameters
        ----------
        data : np.ndarray
            Encoded input data.

        Returns
        -------
        np.ndarray
            Decoded data based on native units.

        """

        data = data.astype(float)
        data[data == self._nodata] = np.nan
        return data / self._scale_factor