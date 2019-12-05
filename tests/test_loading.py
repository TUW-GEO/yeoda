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
Main code for testing to load data with a data cube.
"""

# general imports
import os
import shutil
import unittest
import osr
import numpy as np
import xarray as xr
import dask.array as da

# test data imports
from tests.setup_test_data import setup_gt_test_data
from tests.setup_test_data import setup_nc_multi_test_data
from tests.setup_test_data import setup_nc_single_test_data
from tests.setup_test_data import dirpath_test

# import SGRT naming convention from geopathfinder
from geopathfinder.naming_conventions.sgrt_naming import create_sgrt_filename

# import yeoda data cube
from yeoda.datacube import EODataCube
from yeoda.products.preprocessed import SIG0DataCube


class LoadingTester(unittest.TestCase):
    """ Base class for all data loading tests. Creates and removes all necessary data """

    @classmethod
    def setUpClass(cls):
        """
        Creates GeoTIFF test, multiple single-time and single-variable NetCDF and single multi-time and
        multi-variable NetCDF data.
        """

        setup_gt_test_data()
        setup_nc_multi_test_data()
        setup_nc_single_test_data()

    @classmethod
    def tearDownClass(cls):
        """ Removes all test data. """

        shutil.rmtree(os.path.join(dirpath_test(), 'data'))

    def _create_loadable_dc(self, filepaths):
        """
        Creates a data cube and filters it so that only the temporal dimension is left for loading data
        appropriately.
        """

        dc = EODataCube(filepaths=filepaths, smart_filename_creator=create_sgrt_filename,
                        dimensions=['time', 'var_name', 'pol', 'tile_name', 'orbit_direction'])

        dc.filter_by_dimension('VV', name='pol', in_place=True)
        dc.filter_by_dimension('SIG0', name='var_name', in_place=True)
        dc.filter_by_dimension('D', name='orbit_direction', in_place=True)
        dc.filter_spatially_by_tilename('E042N012T6', dimension_name='tile_name', in_place=True, use_grid=False)

        return dc


class LoadingCoordsTester(LoadingTester):
    """ Responsible for testing the loading functionalities by specifying coordinates. """

    def setUp(self):
        """
        Retrieves test data filepaths and auxiliary data and creates temporary reference data as NumPy arrays,
        xarray arrays and Pandas data frames.
        """

        self.gt_filepaths, self.timestamps = setup_gt_test_data()
        self.nc_filepaths, _ = setup_nc_multi_test_data()
        self.nc_filepath, _ = setup_nc_single_test_data()

        self.lon = 5.
        self.lat = 44.
        sref = osr.SpatialReference()
        sref.ImportFromEPSG(4326)
        self.sref = sref
        self.row = 246
        self.col = 970
        self.x = 4323250.
        self.y = 1314750.

        self.ref_np_ar = (np.array([[[self.row + self.col]*4]]).T + np.arange(0, 4)[:, None, None]).astype(float)
        xr_ar = xr.DataArray(data=da.array(self.ref_np_ar.astype(float)).rechunk((1, 1, 1)),
                             coords={'time': self.timestamps, 'x': [self.x], 'y': [self.y]},
                             dims=['time', 'x', 'y'])
        self.ref_xr_ds = xr.Dataset(data_vars={'1': xr_ar})
        self.ref_pd_df = self.ref_xr_ds.to_dataframe()

    def test_load_gt2numpy_by_coord(self):
        """ Tests loading of a Numpy array from GeoTIFF files by geographic coordinates. """

        dc = self._create_loadable_dc(self.gt_filepaths)

        data = dc.load_by_coords(self.lon, self.lat, sref=self.sref, spatial_dim_name='tile_name', dtype='numpy')
        assert (self.ref_np_ar == data).all()

        ref_data_list = [self.ref_np_ar, self.ref_np_ar]
        data = dc.load_by_coords([self.lon, self.lon], [self.lat, self.lat], sref=self.sref,
                                 spatial_dim_name='tile_name', dtype='numpy')
        assert (ref_data_list[0] == data[0]).all() & (ref_data_list[1] == data[1]).all()

    def test_load_gt2xarray_by_coord(self):
        """ Tests loading of an xarray array from GeoTIFF files by geographic coordinates. """

        dc = self._create_loadable_dc(self.gt_filepaths)

        data = dc.load_by_coords(self.lon, self.lat, sref=self.sref, spatial_dim_name='tile_name', dtype='xarray',
                                 origin='c')
        data['1'].data = data['1'].data.astype(float)  # convert to float for comparison
        assert self.ref_xr_ds.equals(data)

        ref_data_list = [self.ref_xr_ds, self.ref_xr_ds]
        data = dc.load_by_coords([self.lon, self.lon], [self.lat, self.lat], sref=self.sref,
                                 spatial_dim_name='tile_name', dtype='xarray', origin='c')
        assert ref_data_list[0].equals(data) & ref_data_list[1].equals(data)

    def test_load_gt2dataframe_by_coord(self):
        """ Tests loading of a Pandas data frame from GeoTIFF files by geographic coordinates. """

        dc = self._create_loadable_dc(self.gt_filepaths)

        data = dc.load_by_coords(self.lon, self.lat, sref=self.sref, spatial_dim_name='tile_name', dtype='dataframe',
                                 origin='c')
        data['1'] = data['1'].astype(float)  # convert to float for comparison
        assert self.ref_pd_df.equals(data)

        ref_data_list = [self.ref_pd_df, self.ref_pd_df]
        data = dc.load_by_coords([self.lon, self.lon], [self.lat, self.lat], sref=self.sref,
                                 spatial_dim_name='tile_name', dtype='dataframe', origin='c')
        data['1'] = data['1'].astype(float)  # convert to float for comparison
        assert ref_data_list[0].equals(data) & ref_data_list[1].equals(data)

    def test_load_nc2numpy_by_coord(self):
        """ Tests loading of a Numpy array from NetCDF files by geographic coordinates. """

        dc = self._create_loadable_dc(self.nc_filepaths)

        data = dc.load_by_coords(self.lon, self.lat, sref=self.sref, spatial_dim_name='tile_name', dtype='numpy')
        assert (self.ref_np_ar == data).all()
        dc.close()

        ref_data_list = [self.ref_np_ar, self.ref_np_ar]
        data = dc.load_by_coords([self.lon, self.lon], [self.lat, self.lat], sref=self.sref,
                                 spatial_dim_name='tile_name', dtype='numpy')
        assert (ref_data_list[0] == data[0]).all() & (ref_data_list[1] == data[1]).all()
        dc.close()

    def test_load_nc2xarray_by_coord(self):
        """ Tests loading of an xarray array from NetCDF files by geographic coordinates. """

        dc = self._create_loadable_dc(self.nc_filepaths)

        data = dc.load_by_coords(self.lon, self.lat, sref=self.sref, spatial_dim_name='tile_name', dtype='xarray',
                                 origin='c')
        assert self.ref_xr_ds.equals(data)
        dc.close()

        ref_data_list = [self.ref_xr_ds, self.ref_xr_ds]
        data = dc.load_by_coords([self.lon, self.lon], [self.lat, self.lat], sref=self.sref,
                                 spatial_dim_name='tile_name', dtype='xarray', origin='c')
        assert ref_data_list[0].equals(data) & ref_data_list[1].equals(data)
        dc.close()

    def test_load_nc2dataframe_by_coord(self):
        """ Tests loading of a Pandas data frame from NetCDF files by geographic coordinates. """

        dc = self._create_loadable_dc(self.nc_filepaths)

        data = dc.load_by_coords(self.lon, self.lat, sref=self.sref, spatial_dim_name='tile_name', dtype='dataframe',
                                 origin='c')
        assert self.ref_pd_df.equals(data)
        dc.close()

        ref_data_list = [self.ref_pd_df, self.ref_pd_df]
        data = dc.load_by_coords([self.lon, self.lon], [self.lat, self.lat], sref=self.sref,
                                 spatial_dim_name='tile_name', dtype='dataframe', origin='c')
        assert ref_data_list[0].equals(data) & ref_data_list[1].equals(data)
        dc.close()

    def test_load_singlenc2xarray_by_coord(self):
        """ Tests loading of an xarray array from a multidimensional NetCDF file by geographic coordinates. """

        dc = EODataCube(filepaths=[self.nc_filepath], smart_filename_creator=create_sgrt_filename,
                        dimensions=['time'])
        data = dc.load_by_coords(self.lon, self.lat, sref=self.sref, band='SIG0', dtype='xarray', origin='c')
        assert self.ref_xr_ds.equals(data.rename({'SIG0': '1'}))
        dc.close()


class LoadingPixelsTester(LoadingTester):
    """ Responsible for testing the loading functionalities by specifying pixel numbers and and a window size. """

    def setUp(self):
        """
        Retrieves test data filepaths and auxiliary data and creates temporary reference data as NumPy arrays,
        xarray arrays and Pandas data frames.
        """

        self.gt_filepaths, self.timestamps = setup_gt_test_data()
        self.nc_filepaths, _ = setup_nc_multi_test_data()
        self.nc_filepath, _ = setup_nc_single_test_data()

        self.row = 246
        self.col = 970
        self.row_size = 10
        self.col_size = 10
        x = 4323250.
        y = 1314750.

        self.ref_np_ar = (np.array([[[self.row + self.col]*4]]).T + np.arange(0, 4)[:, None, None]).astype(float)
        xr_ar = xr.DataArray(data=self.ref_np_ar.astype(float),
                             coords={'time': self.timestamps, 'x': [x], 'y': [y]},
                             dims=['time', 'x', 'y'])
        self.ref_xr_ds = xr.Dataset(data_vars={'1': xr_ar})
        self.ref_pd_df = self.ref_xr_ds.to_dataframe()
        rows, cols = np.meshgrid(np.arange(self.row, self.row+self.row_size),
                                 np.arange(self.col, self.col+self.col_size))
        xs = np.arange(x, x + self.row_size * 500, 500.)
        ys = np.arange(y, y - self.col_size * 500, -500.)
        base_np_ar_2D = rows + cols
        base_np_ar = np.stack([base_np_ar_2D]*4, axis=0)
        self.ref_np_ar_area = (base_np_ar + np.arange(0, 4)[:, None, None]).astype(float)
        xr_ar = xr.DataArray(data=da.array(self.ref_np_ar_area.astype(float)).rechunk((1, 10, 10)),
                             coords={'time': self.timestamps, 'x': xs, 'y': ys},
                             dims=['time', 'x', 'y'])
        self.ref_xr_ds_area = xr.Dataset(data_vars={'1': xr_ar})
        self.ref_pd_df_area = self.ref_xr_ds_area.to_dataframe()

    def test_load_gt2numpy_by_pixels(self):
        """ Tests loading of a Numpy array from GeoTIFF files by pixel coordinates. """

        dc = self._create_loadable_dc(self.gt_filepaths)

        data = dc.load_by_pixels(self.row, self.col, spatial_dim_name='tile_name', dtype='numpy')
        assert (self.ref_np_ar == data).all()

        data = dc.load_by_pixels(self.row, self.col, row_size=self.row_size, col_size=self.col_size,
                                 spatial_dim_name='tile_name', dtype='numpy')
        assert (self.ref_np_ar_area == data).all()

    def test_load_gt2xarray_by_pixels(self):
        """ Tests loading of an xarray array from GeoTIFF files by pixel coordinates. """

        dc = self._create_loadable_dc(self.gt_filepaths)

        data = dc.load_by_pixels(self.row, self.col, spatial_dim_name='tile_name', dtype='xarray', origin='c')
        data['1'].data = data['1'].data.astype(float)  # convert to float for comparison
        assert self.ref_xr_ds.equals(data)

        data = dc.load_by_pixels(self.row, self.col, row_size=self.row_size, col_size=self.col_size,
                                 spatial_dim_name='tile_name', dtype='xarray', origin='c')
        data['1'].data = data['1'].data.astype(float)  # convert to float for comparison
        assert self.ref_xr_ds_area.equals(data)

    def test_load_gt2dataframe_by_pixels(self):
        """ Tests loading of a Pandas data frame from GeoTIFF files by pixel coordinates. """

        dc = self._create_loadable_dc(self.gt_filepaths)

        data = dc.load_by_pixels(self.row, self.col, spatial_dim_name='tile_name', dtype='dataframe', origin='c')
        # convert to float
        data['1'] = data['1'].astype(float)  # convert to float for comparison
        assert self.ref_pd_df.equals(data)

        data = dc.load_by_pixels(self.row, self.col, row_size=self.row_size, col_size=self.col_size,
                                 spatial_dim_name='tile_name', dtype='dataframe', origin='c')
        data['1'] = data['1'].astype(float)  # convert to float for comparison
        assert self.ref_pd_df_area.equals(data)

    def test_load_nc2numpy_by_pixels(self):
        """ Tests loading of a Numpy array from NetCDF files by pixel coordinates. """

        dc = self._create_loadable_dc(self.nc_filepaths)

        data = dc.load_by_pixels(self.row, self.col, spatial_dim_name='tile_name', dtype='numpy')
        assert (self.ref_np_ar == data).all()
        dc.close()

        data = dc.load_by_pixels(self.row, self.col, row_size=self.row_size, col_size=self.col_size,
                                 spatial_dim_name='tile_name', dtype='numpy')
        assert (self.ref_np_ar_area == data).all()
        dc.close()

    def test_load_nc2xarray_by_pixels(self):
        """ Tests loading of an xarray array from NetCDF files by pixel coordinates. """

        dc = self._create_loadable_dc(self.nc_filepaths)

        data = dc.load_by_pixels(self.row, self.col, spatial_dim_name='tile_name', dtype='xarray', origin='c')
        data['1'].data = data['1'].data.astype(float)  # convert to float for comparison
        assert self.ref_xr_ds.equals(data)
        dc.close()

        data = dc.load_by_pixels(self.row, self.col, row_size=self.row_size, col_size=self.col_size,
                                 spatial_dim_name='tile_name', dtype='xarray', origin='c')
        data['1'].data = data['1'].data.astype(float)  # convert to float for comparison
        assert self.ref_xr_ds_area.equals(data)
        dc.close()

    def test_load_nc2dataframe_by_pixels(self):
        """ Tests loading of a Pandas data frame from NetCDF files by pixel coordinates. """

        dc = self._create_loadable_dc(self.nc_filepaths)

        data = dc.load_by_pixels(self.row, self.col, spatial_dim_name='tile_name', dtype='dataframe', origin='c')
        data['1'] = data['1'].astype(float)  # convert to float for comparison
        assert self.ref_pd_df.equals(data)
        dc.close()

        data = dc.load_by_pixels(self.row, self.col, row_size=self.row_size, col_size=self.col_size,
                                 spatial_dim_name='tile_name', dtype='dataframe', origin='c')
        data['1'] = data['1'].astype(float)  # convert to float for comparison
        assert self.ref_pd_df_area.equals(data)
        dc.close()

    def test_load_singlenc2xarray_by_pixels(self):
        """ Tests loading of an xarray array from a multidimensional NetCDF file by pixel coordinates. """

        dc = EODataCube(filepaths=[self.nc_filepath], smart_filename_creator=create_sgrt_filename,
                        dimensions=['time'])
        data = dc.load_by_pixels(self.row, self.col, row_size=self.row_size, col_size=self.col_size, band='SIG0',
                                 dtype='xarray', origin='c')
        assert self.ref_xr_ds_area.equals(data.rename({'SIG0': '1'}))
        dc.close()


class LoadingGeomTester(LoadingTester):
    """ Responsible for testing the loading functionalities by specifying a region/geometry of interest. """

    def setUp(self):
        """
        Retrieves test data filepaths and auxiliary data and creates temporary reference data as NumPy arrays,
        xarray arrays and Pandas data frames.
        """

        self.gt_filepaths, self.timestamps = setup_gt_test_data()
        self.nc_filepaths, _ = setup_nc_multi_test_data()
        self.nc_filepath, _ = setup_nc_single_test_data()

        row = 246
        col = 970
        x = 4323250.
        y = 1314750.
        row_size = 10
        col_size = 10
        x_min = x
        x_max = x + (col_size-1) * 500.
        y_min = y - (row_size-1) * 500.
        y_max = y
        self.bbox = [(x_min, y_min), (x_max, y_max)]
        self.partial_outside_bbox = [(4800000.0 - 10 * 500., 1200000.0 - 20*500),
                                     (4800000.0 + 20*500, 1200000.0 + 10 * 500.)]

        rows, cols = np.meshgrid(np.arange(row, row+row_size),
                                 np.arange(col, col+col_size))
        xs = np.arange(x, x + row_size * 500, 500.)
        ys = np.arange(y, y - col_size * 500, -500.)
        base_np_ar_2D = rows + cols
        base_np_ar = np.stack([base_np_ar_2D]*4, axis=0)
        self.ref_np_ar_area = (base_np_ar + np.arange(0, 4)[:, None, None]).astype(float)
        xr_ar = xr.DataArray(data=da.array(self.ref_np_ar_area.astype(float)).rechunk((1, 10, 10)),
                             coords={'time': self.timestamps, 'x': xs, 'y': ys},
                             dims=['time', 'x', 'y'])
        self.ref_xr_ds_area = xr.Dataset(data_vars={'1': xr_ar})
        self.ref_pd_df_area = self.ref_xr_ds_area.to_dataframe()

    def test_load_gt2numpy_by_geom(self):
        """ Tests loading of a Numpy array from GeoTIFF files by a bounding box. """

        dc = self._create_loadable_dc(self.gt_filepaths)
        data = dc.load_by_geom(self.bbox, spatial_dim_name='tile_name', dtype='numpy')
        assert (self.ref_np_ar_area == data).all()

    def test_load_gt2xarray_by_geom(self):
        """ Tests loading of an xarray array from GeoTIFF files by a bounding box. """

        dc = self._create_loadable_dc(self.gt_filepaths)
        data = dc.load_by_geom(self.bbox, spatial_dim_name='tile_name', dtype='xarray', origin='c')
        data['1'].data = data['1'].data.astype(float)  # convert to float for comparison
        assert self.ref_xr_ds_area.equals(data)

    def test_load_gt2dataframe_by_geom(self):
        """ Tests loading of a Pandas data frame from GeoTIFF files by a bounding box. """

        dc = self._create_loadable_dc(self.gt_filepaths)
        data = dc.load_by_geom(self.bbox, spatial_dim_name='tile_name', dtype='dataframe', origin='c')
        data['1'] = data['1'].astype(float)  # convert to float for comparison
        assert self.ref_pd_df_area.equals(data)

    def test_load_nc2numpy_by_geom(self):
        """ Tests loading of a Numpy array from NetCDF files by a bounding box. """

        dc = self._create_loadable_dc(self.nc_filepaths)
        data = dc.load_by_geom(self.bbox, spatial_dim_name='tile_name', dtype='numpy')
        assert (self.ref_np_ar_area == data).all()
        dc.close()

    def test_load_nc2xarray_by_geom(self):
        """ Tests loading of an xarray array from NetCDF files by a bounding box. """

        dc = self._create_loadable_dc(self.nc_filepaths)
        data = dc.load_by_geom(self.bbox, spatial_dim_name='tile_name', dtype='xarray', origin='c')
        assert self.ref_xr_ds_area.equals(data)
        dc.close()

    def test_load_nc2dataframe_by_geom(self):
        """ Tests loading of a Pandas data frame from NetCDF files by a bounding box. """

        dc = self._create_loadable_dc(self.nc_filepaths)
        data = dc.load_by_geom(self.bbox, spatial_dim_name='tile_name', dtype='dataframe', origin='c')
        assert self.ref_pd_df_area.equals(data)
        dc.close()

    def test_load_singlenc2xarray_by_pixels(self):
        """ Tests loading of an xarray array from a multidimensional NetCDF file by a bounding box. """

        dc = EODataCube(filepaths=[self.nc_filepath], smart_filename_creator=create_sgrt_filename,
                        dimensions=['time'])
        data = dc.load_by_geom(self.bbox, band='SIG0', dtype='xarray', origin='c')
        assert self.ref_xr_ds_area.equals(data.rename({'SIG0': '1'}))
        dc.close()

    def test_load_geom_larger_than_tile(self):
        dc = SIG0DataCube(filepaths=self.gt_filepaths, dimensions=['time'], sres=500)
        dc.filter_spatially_by_tilename("E042N012T6", dimension_name="tile_name", in_place=True)
        data = dc.load_by_geom(self.partial_outside_bbox, spatial_dim_name='tile_name', dtype='numpy')
        assert data.shape == (16, 10, 10)

if __name__ == '__main__':
    #unittest.main()
    eodc_tester = LoadingGeomTester()
    eodc_tester.setUpClass()
    eodc_tester.setUp()
    eodc_tester.test_load_gt2xarray_by_geom()
