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
Main code for creating a TUWGEO SCATSAR-SWI data cube.
"""

# general packages
import numpy as np

from geopathfinder.folder_naming import build_smarttree

# import TUWGEO product data cube
from yeoda.products.base import ProductDataCube


class SCATSARSWIDataCube(ProductDataCube):
    """
    Data cube defining a TUWGEO SCATSAR-SWI, with QFLAGS, and SSF product layers.
    """

    _swi_vars = ['SWI_T002', 'SWI_T005', 'SWI_T010', 'SWI_T015', 'SWI_T020', 'SWI_T040', 'SWI_T060', 'SWI_T100',
                 'QFLAG_T002', 'QFLAG_T005', 'QFLAG_T010', 'QFLAG_T015', 'QFLAG_T020', 'QFLAG_T040', 'QFLAG_T060',
                 'QFLAG_T100', 'SSF', 'SCAT_IC', 'SAR_IC']
    _swi_vars_scld = ['SWI_T002', 'SWI_T005', 'SWI_T010', 'SWI_T015', 'SWI_T020', 'SWI_T040', 'SWI_T060', 'SWI_T100',
                      'QFLAG_T002', 'QFLAG_T005', 'QFLAG_T010', 'QFLAG_T015', 'QFLAG_T020', 'QFLAG_T040',
                      'QFLAG_T060', 'QFLAG_T100']
    _swi_vars_unscld = ['SSF', 'SCAT_IC', 'SAR_IC']

    def __init__(self, root_dirpath=None, sres=500, continent='EU', dimensions=None, **kwargs):
        """
        Constructor of class `SCATSARSWIDataCube`.

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

        super().__init__(root_dirpath, ['SWI'], sres=sres, continent=continent, dimensions=dimensions,
                         **kwargs)

    def encode(self, data, band=None):
        """
        Encoding function for TUWGEO SCATSAR-SWI data.

        Parameters
        ----------
        data : np.ndarray
            Input data.
        band : str
            name of band is stored in netCDF file. one of
            'SWI_T002','SWI_T005','SWI_T010','SWI_T015','SWI_T020','SWI_T040','SWI_T060','SWI_T100',
            'QFLAG_T002' ... to ... 'QFLAG_T100' (Quality Flags),
            'SSF' (Surface State Flag),
            'SCAT_IC','SAR_IC' (input SSM data count for SCAT and SAR, i.e. how many individual
                                observations contribute to the daily SWI value update)

        Returns
        -------
        np.ndarray
            Encoded data.
        """

        if band in SCATSARSWIDataCube._swi_vars_scld:
            data *= 2
        elif band in SCATSARSWIDataCube._swi_vars_unscld:
            pass
        else:
            err_msg = 'Band {} is unknown.'.format(band)
            raise Exception(err_msg)

        data[np.isnan(data)] = 255
        data.astype(np.uint8)

        return data

    def decode(self, data, band=None):
        """
        Decoding function for TUWGEO SCATSAR-SWI data.

        Parameters
        ----------
        data : np.ndarray
            Encoded input data.
        band : str
            name of band is stored in netCDF file. one of
            'SWI_T002','SWI_T005','SWI_T010','SWI_T015','SWI_T020','SWI_T040','SWI_T060','SWI_T100',
            'QFLAG_T002' ... to ... 'QFLAG_T100' (Quality Flags),
            'SSF' (Surface State Flag),
            'SCAT_IC','SAR_IC' (input SSM data count for SCAT and SAR, i.e. how many individual
                                observations contribute to the daily SWI value update)

        Returns
        -------
        np.ndarray
            Decoded data based on native units.

        """

        if band in SCATSARSWIDataCube._swi_vars_scld:
            data = data.astype(float)
            data[data > 200] = np.nan
            data /= 2.
        elif band in SCATSARSWIDataCube._swi_vars_unscld:
            data = data.astype(int)
            data[data > 200] = 255
        else:
            err_msg = 'Band {} is unknown.'.format(band)
            raise Exception(err_msg)

        return data

    def load_by_coords(self, xs, ys, sref=None, band='1', dtype="xarray", origin="ur"):
        """
        Loads data as a 1-D array according to a given coordinate.

        Parameters
        ----------
        xs : list of floats or float
            World system coordinates in X direction.
        ys : list of floats or float
            World system coordinates in Y direction.
        sref : osr.SpatialReference, optional
            Spatial reference referring to the world system coordinates `x` and `y`.
        band : int or str, optional
            Band number or name (default is 1).
        dtype : str
            Data type of the returned array-like structure (default is 'xarray'). It can be:
                - 'xarray': loads data as an xarray.DataSet
                - 'numpy': loads data as a numpy.ndarray
                - 'dataframe': loads data as a pandas.DataFrame
        origin: str, optional
            Defines the world system origin of the pixel. It can be:
                - upper left ("ul")
                - upper right ("ur", default)
                - lower right ("lr")
                - lower left ("ll")
                - center ("c")

        Returns
        -------
        list of numpy.ndarray or list of xarray.DataSets or pandas.DataFrame
            Data as an array-like object.
        """

        decode_kwargs = {'band': band}
        return super().load_by_coords(xs, ys, sref=sref, band=band, dtype=dtype, origin=origin,
                                      decode_kwargs=decode_kwargs)

    def load_by_geom(self, geom, sref=None, band='1', apply_mask=False, dtype="xarray", origin='ur'):
        """
        Loads data according to a given geometry.

        Parameters
        ----------
        geom : OGR Geometry or Shapely Geometry or list or tuple, optional
            A geometry defining the region of interest. If it is of type list/tuple representing the extent
            (i.e. [x_min, y_min, x_max, y_max]), `sref` has to be given to transform the extent into a
            georeferenced polygon.
        sref : osr.SpatialReference, optional
            Spatial reference of the given region of interest `geom`.
        band : int or str, optional
            Band number or name (default is 1).
        apply_mask : bool, optional
            If true, a numpy mask array with a mask excluding (=1) all pixels outside `geom` (=0) will be created
            (default is True).
        dtype : str
            Data type of the returned array-like structure (default is 'xarray'). It can be:
                - 'xarray': loads data as an xarray.DataSet
                - 'numpy': loads data as a numpy.ndarray
                - 'dataframe': loads data as a pandas.DataFrame
        origin: str, optional
            Defines the world system origin of the pixel. It can be:
                - upper left ("ul")
                - upper right ("ur", default)
                - lower right ("lr")
                - lower left ("ll")
                - center ("c")

        Returns
        -------
        numpy.array or xarray.DataSet or pd.DataFrame
            Data as an array-like object.
        """

        decode_kwargs = {'band': band}
        return super().load_by_geom(geom, sref=sref, band=band, apply_mask=apply_mask, dtype=dtype, origin=origin,
                                    decode_kwargs=decode_kwargs)

    def load_by_pixels(self, rows, cols, row_size=1, col_size=1, band='1', dtype="xarray", origin="ur"):
        """
        Loads data according to given pixel numbers, i.e. the row and column numbers and optionally a certain
        pixel window (`row_size` and `col_size`).

        Parameters
        ----------
        rows : list of int or int
            Row numbers.
        cols : list of int or int
            Column numbers.
        row_size : int, optional
            Number of rows to read (counts from input argument `rows`, default is 1).
        col_size : int, optional
            Number of columns to read (counts from input argument `cols`, default is 1).
        band : int or str, optional
            Band number or name (default is 1).
        dtype : str
            Data type of the returned array-like structure (default is 'xarray'). It can be:
                - 'xarray': loads data as an xarray.DataSet
                - 'numpy': loads data as a numpy.ndarray
                - 'dataframe': loads data as a pandas.DataFrame
        origin: str, optional
            Defines the world system origin of the pixel. It can be:
                - upper left ("ul")
                - upper right ("ur", default)
                - lower right ("lr")
                - lower left ("ll")
                - center ("c")

        Returns
        -------
        list of numpy.ndarray or list of xarray.DataSets or pandas.DataFrame
            Data as an array-like object.
        """

        decode_kwargs = {'band': band}
        return super().load_by_pixels(rows, cols, row_size=row_size, col_size=col_size, band=band, dtype=dtype,
                                      origin=origin, decode_kwargs=decode_kwargs)


if __name__ == '__main__':
    pass