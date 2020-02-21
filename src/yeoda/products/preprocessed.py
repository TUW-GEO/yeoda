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
Main code for creating a TUWGEO preprocessed data cube.
"""

# general packages
import numpy as np

# import TUWGEO product data cube
from yeoda.products.base import ProductDataCube


class PreprocessedDataCube(ProductDataCube):
    """
    Data cube defining a TUWGEO preprocessed product.
    """

    def __init__(self, root_dirpath=None, var_names=None, sres=10, continent='EU', dimensions=None,
                 file_pattern=".tif$", **kwargs):
        """
        Constructor of class `PreprocessedDataCube`.

        Parameters
        ----------
        root_dirpath : str, optional
            Root directory path to the SGRT directory tree.
        var_names : list, optional
            SGRT Variable names , e.g. ["SIG0", "GMR"].
        sres : int, optional
            Spatial sampling in grid units, e.g. 10, 500 (default is 10).
        continent : str, optional
            Continent/Subgrid of the Equi7Grid system (default is 'EU').
        dimensions : list, optional
            List of filename parts to use as dimensions. The strings have to match with the keys of the `SgrtFilename`
            fields definition.
        inventory : GeoDataFrame, optional
            Contains information about the dimensions (columns) and each filepath (rows).
        file_pattern : str
            Pattern to match/only select certain file names.
        **kwargs
            Arbitrary keyword arguments (e.g. containing 'inventory' or 'grid').
        """

        super().__init__(root_dirpath=root_dirpath, var_names=var_names, sres=sres, continent=continent,
                         dimensions=dimensions, file_pattern=file_pattern, **kwargs)

    def encode(self, data, **kwargs):
        """
        Joint encoding function for TUWGEO preprocessed data.

        Parameters
        ----------
        data : np.ndarray
            Input data.

        Returns
        -------
        np.ndarray
            Encoded data.
        """

        data *= 100
        data[np.isnan(data)] = -9999
        data.astype(np.int16)
        return data

    def decode(self, data, **kwargs):
        """
        Joint decoding function for TUWGEO preprocessed data.

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
        data[data == -9999] = None
        return data / 100.


class SIG0DataCube(PreprocessedDataCube):
    """
    Data cube defining a TUWGEO SIG0 product.
    """

    def __init__(self, root_dirpath=None, sres=10, continent='EU', file_pattern=".tif$", dimensions=None, **kwargs):
        """
        Constructor of class `SIG0DataCube`.

        Parameters
        ----------
        root_dirpath : str, optional
            Root directory path to the SGRT directory tree.
        sres : int, optional
            Spatial sampling in grid units, e.g. 10, 500 (default is 10).
        continent : str, optional
            Continent/Subgrid of the Equi7Grid system (default is 'EU').
        dimensions : list, optional
            List of filename parts to use as dimensions. The strings have to match with the keys of the `SgrtFilename`
            fields definition.
        inventory : GeoDataFrame, optional
            Contains information about the dimensions (columns) and each filepath (rows).
        file_pattern : str
            Pattern to match/only select certain file names.
        **kwargs
            Arbitrary keyword arguments (e.g. containing 'inventory' or 'grid').
        """

        super().__init__(root_dirpath=root_dirpath, var_names=["SIG0"], sres=sres, continent=continent,
                         dimensions=dimensions, file_pattern=file_pattern, **kwargs)


class GMRDataCube(PreprocessedDataCube):
    """
    Data cube defining a TUWGEO GMR product.
    """

    def __init__(self, root_dirpath=None, sres=10, continent='EU', file_pattern=".tif$", dimensions=None, **kwargs):
        """
        Constructor of class `GMRDataCube`.

        Parameters
        ----------
        root_dirpath : str, optional
            Root directory path to the SGRT directory tree.
        sres : int, optional
            Spatial sampling in grid units, e.g. 10, 500 (default is 10).
        continent : str, optional
            Continent/Subgrid of the Equi7Grid system (default is 'EU').
        dimensions : list, optional
            List of filename parts to use as dimensions. The strings have to match with the keys of the `SgrtFilename`
            fields definition.
        inventory : GeoDataFrame, optional
            Contains information about the dimensions (columns) and each filepath (rows).
        file_pattern : str
            Pattern to match/only select certain file names.
        **kwargs
            Arbitrary keyword arguments (e.g. containing 'inventory' or 'grid').
        """

        super().__init__(root_dirpath=root_dirpath, var_names=["GMR"], sres=sres, continent=continent,
                         dimensions=dimensions, file_pattern=file_pattern, **kwargs)
