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
Main code for creating a TUWGEO SSM data cube.
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

    def __init__(self, root_dirpath=None, sres=500, continent='EU', dimensions=None, **kwargs):
        """
        Constructor of class `SSMDataCube`.

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

        # BBM: should that be used somehow?
        # swi_variables = list(('SWI_T002',
        #                       'SWI_T005',
        #                       'SWI_T010',
        #                       'SWI_T015',
        #                       'SWI_T020',
        #                       'SWI_T040',
        #                       'SWI_T060',
        #                       'SWI_T100',
        #                       'QFLAG_T002',
        #                       'QFLAG_T005',
        #                       'QFLAG_T010',
        #                       'QFLAG_T015',
        #                       'QFLAG_T020',
        #                       'QFLAG_T040',
        #                       'QFLAG_T060',
        #                       'QFLAG_T100',
        #                       'SSF',
        #                       'SCAT_IC',
        #                       'SAR_IC'))

        super().__init__(root_dirpath, ['SWI'], sres=sres, continent=continent, dimensions=dimensions,
                         **kwargs)

    def encode(self, data, band=None):
        """
        Encoding function for TUWGEO SSM/SSM-NOISE data.

        Parameters
        ----------
        data : np.ndarray
            Input data.

        Returns
        -------
        np.ndarray
            Encoded data.
        """

        data *= 2
        data[np.isnan(data)] = 255
        return data

    def decode(self, data, band=None):
        """
        Decoding function for TUWGEO SSM/SSM-NOISE data.

        Parameters
        ----------
        data : np.ndarray
            Encoded input data.

        Returns
        -------
        np.ndarray
            Decoded data based on native units.
        """
        # TODO: @bbm add code
        if band in ["SWI_T002"]:
            data = data.astype(float)
            data[data > 200] = np.nan
            data /= 2.
        elif band in ["QFLAG_T002"]:
            data = data.astype(float)
            data[data > 200] = np.nan
            data /= 2.
        else:
            err_msg = 'Band {} is unknown.'.format(band)
            raise Exception(err_msg)

        return data

    def load_by_coords(self, xs, ys, sref=None, band='1', dtype="xarray", origin="ur"):
        decode_kwargs = {'band': band}
        return super().load_by_coords(xs, ys, sref=sref, band=band, dtype=dtype, origin=origin,
                                      decode_kwargs=decode_kwargs)

    def load_by_geom(self, geom, sref=None, band='1', apply_mask=False, dtype="xarray", origin='ur'):
        decode_kwargs = {'band': band}
        return super().load_by_geom(geom, sref=sref, band=band, apply_mask=apply_mask, dtype=dtype, origin=origin,
                                    decode_kwargs=decode_kwargs)

    def load_by_pixels(self, rows, cols, row_size=1, col_size=1, band='1', dtype="xarray", origin="ur"):
        decode_kwargs = {'band': band}
        return super().load_by_pixels(rows, cols, row_size=row_size, col_size=col_size, band=band, dtype=dtype,
                                      origin=origin, decode_kwargs=decode_kwargs)


if __name__ == '__main__':

    root_dirpath = r'R:\Datapool_processed\SCATSAR\CGLS\C0418\202002_test'
    folder_hierarchy = ['grid', 'tile', 'var']
    dir_tree = build_smarttree(root_dirpath, folder_hierarchy, register_file_pattern='.nc')

    dc = SCATSARSWIDataCube(dir_tree=dir_tree)
    dc.filter_by_dimension('E048N012T6', name='tile_name', inplace=True)
    ts = dc.load_by_pixels(1111, 1111, band='SWI_T002')
    pass