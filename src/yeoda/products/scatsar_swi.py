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

    def __init__(self, **kwargs):
        """
        Constructor of class `SCATSARSWIDataCube`.

        Parameters
        ----------
        **kwargs
            Keyworded arguments for `ProductDataCube`.

        """
        kwargs.update({'var_names': ["SWI"],
                       'scale_factor': 2,
                       'nodata': 255,
                       'dtype': 'UInt8'})
        super().__init__(**kwargs)

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
            data *= self._scale_factor
        elif band in SCATSARSWIDataCube._swi_vars_unscld:
            pass
        else:
            err_msg = 'Band {} is unknown.'.format(band)
            raise Exception(err_msg)

        data[np.isnan(data)] = self._nodata
        data.astype(self._dtype)

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
            data /= self._scale_factor
        elif band in SCATSARSWIDataCube._swi_vars_unscld:
            data = data.astype(int)
            data[data > 200] = self._nodata
        else:
            err_msg = 'Band {} is unknown.'.format(band)
            raise Exception(err_msg)

        return data

    def load_by_coords(self, *args, **kwargs):
        """
        Loads data as a 1-D array according to a given coordinate.

        Parameters
        ----------
        *args : tuple
            Arguments for base `load_by_coords` function.
        **kwargs : dict
            Keyword-arguments for base `load_by_coords` function.

        Returns
        -------
        list of numpy.ndarray or list of xarray.DataSets or pandas.DataFrame
            Data as an array-like object.

        """
        decode_kwargs = kwargs.pop('decode_kwargs', {})
        decode_kwargs.update({'band': kwargs.get('band', '1')})
        return super().load_by_coords(*args, decode_kwargs=decode_kwargs, **kwargs)

    def load_by_geom(self,  *args, **kwargs):
        """
        Loads data according to a given geometry.

        Parameters
        ----------
        *args : tuple
            Arguments for base `load_by_geom` function.
        **kwargs : dict
            Keyword-arguments for base `load_by_geom` function.

        Returns
        -------
        numpy.array or xarray.DataSet or pd.DataFrame
            Data as an array-like object.

        """

        decode_kwargs = kwargs.pop('decode_kwargs', {})
        decode_kwargs.update({'band': kwargs.get('band', '1')})
        return super().load_by_geom(*args, decode_kwargs=decode_kwargs, **kwargs)

    def load_by_pixels(self, *args, **kwargs):
        """
        Loads data according to given pixel numbers, i.e. the row and column numbers and optionally a certain
        pixel window (`row_size` and `col_size`).

        Parameters
        ----------
        *args : tuple
            Arguments for base `load_by_pixels` function.
        **kwargs : dict
            Keyword-arguments for base `load_by_pixels` function.

        Returns
        -------
        list of numpy.ndarray or list of xarray.DataSets or pandas.DataFrame
            Data as an array-like object.

        """

        decode_kwargs = kwargs.pop('decode_kwargs', {})
        decode_kwargs.update({'band': kwargs.get('band', '1')})
        return super().load_by_pixels(*args, decode_kwargs=decode_kwargs, **kwargs)


if __name__ == '__main__':
    pass