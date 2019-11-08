# general imports
import os
import shutil
import unittest
import osr
import numpy as np
import xarray as xr

# test data imports
from tests.setup_test_data import setup_gt_test_data
from tests.setup_test_data import setup_nc_multi_test_data
from tests.setup_test_data import setup_nc_single_test_data
from tests.setup_test_data import dirpath_test


from geopathfinder.naming_conventions.sgrt_naming import create_sgrt_filename
from yeoda.datacube import EODataCube


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
        dc = EODataCube(filepaths=filepaths, smart_filename_creator=create_sgrt_filename,
                        dimensions=['time', 'var_name', 'pol', 'tile_name', 'orbit_direction'])

        dc.filter_by_dimension('VV', name='pol', in_place=True)
        dc.filter_by_dimension('SIG0', name='var_name', in_place=True)
        dc.filter_by_dimension('D', name='orbit_direction', in_place=True)
        dc.filter_spatially_by_tilename('E042N012T6', dimension_name='tile_name', in_place=True, use_grid=False)

        return dc


class LoadingCoordsTester(LoadingTester):

    def setUp(self):
        gt_filepaths, timestamps = setup_gt_test_data()
        self.gt_filepaths = gt_filepaths
        nc_filepaths, _ = setup_nc_multi_test_data()
        self.nc_filepaths = nc_filepaths
        nc_filepath, _ = setup_nc_single_test_data()
        self.nc_filepath = nc_filepath
        self.timestamps = timestamps
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
        xr_ar = xr.DataArray(data=self.ref_np_ar.astype(float),
                             coords={'time': self.timestamps, 'x': [self.x], 'y': [self.y]},
                             dims=['time', 'x', 'y'])
        self.ref_xr_ds = xr.Dataset(data_vars={'1': xr_ar})
        self.ref_pd_df = self.ref_xr_ds.to_dataframe()

    def test_load_gt2numpy_by_coord(self):
        dc = self._create_loadable_dc(self.gt_filepaths)

        data = dc.load_by_coords(self.lon, self.lat, sref=self.sref, dimension_name='tile_name', dtype='numpy')
        assert (self.ref_np_ar == data).all()

        ref_data_list = [self.ref_np_ar, self.ref_np_ar]
        data = dc.load_by_coords([self.lon, self.lon], [self.lat, self.lat], sref=self.sref,
                                 dimension_name='tile_name', dtype='numpy')
        assert (ref_data_list[0] == data[0]).all() & (ref_data_list[1] == data[1]).all()

    def test_load_gt2xarray_by_coord(self):
        dc = self._create_loadable_dc(self.gt_filepaths)

        data = dc.load_by_coords(self.lon, self.lat, sref=self.sref, dimension_name='tile_name', dtype='xarray',
                                 origin='c')
        # convert to float
        data['1'].data = data['1'].data.astype(float)
        assert self.ref_xr_ds.equals(data)

        ref_data_list = [self.ref_xr_ds, self.ref_xr_ds]
        data = dc.load_by_coords([self.lon, self.lon], [self.lat, self.lat], sref=self.sref,
                                 dimension_name='tile_name', dtype='xarray', origin='c')
        assert ref_data_list[0].equals(data) & ref_data_list[1].equals(data)

    def test_load_gt2dataframe_by_coord(self):
        dc = self._create_loadable_dc(self.gt_filepaths)

        data = dc.load_by_coords(self.lon, self.lat, sref=self.sref, dimension_name='tile_name', dtype='dataframe',
                                 origin='c')
        # convert to float for comparison
        data['1'] = data['1'].astype(float)
        assert self.ref_pd_df.equals(data)

        ref_data_list = [self.ref_pd_df, self.ref_pd_df]
        data = dc.load_by_coords([self.lon, self.lon], [self.lat, self.lat], sref=self.sref,
                                 dimension_name='tile_name', dtype='dataframe', origin='c')
        # convert to float for comparison
        data['1'] = data['1'].astype(float)
        assert ref_data_list[0].equals(data) & ref_data_list[1].equals(data)

    def test_load_nc2numpy_by_coord(self):
        dc = self._create_loadable_dc(self.nc_filepaths)

        data = dc.load_by_coords(self.lon, self.lat, sref=self.sref, dimension_name='tile_name', dtype='numpy')
        assert (self.ref_np_ar == data).all()

        ref_data_list = [self.ref_np_ar, self.ref_np_ar]
        data = dc.load_by_coords([self.lon, self.lon], [self.lat, self.lat], sref=self.sref,
                                 dimension_name='tile_name', dtype='numpy')
        assert (ref_data_list[0] == data[0]).all() & (ref_data_list[1] == data[1]).all()

    def test_load_nc2xarray_by_coord(self):
        dc = self._create_loadable_dc(self.nc_filepaths)

        data = dc.load_by_coords(self.lon, self.lat, sref=self.sref, dimension_name='tile_name', dtype='xarray',
                                 origin='c')
        assert self.ref_xr_ds.equals(data)

        ref_data_list = [self.ref_xr_ds, self.ref_xr_ds]
        data = dc.load_by_coords([self.lon, self.lon], [self.lat, self.lat], sref=self.sref,
                                 dimension_name='tile_name', dtype='xarray', origin='c')
        assert ref_data_list[0].equals(data) & ref_data_list[1].equals(data)

    def test_load_nc2dataframe_by_coord(self):
        dc = self._create_loadable_dc(self.nc_filepaths)

        data = dc.load_by_coords(self.lon, self.lat, sref=self.sref, dimension_name='tile_name', dtype='dataframe',
                                 origin='c')
        assert self.ref_pd_df.equals(data)

        ref_data_list = [self.ref_pd_df, self.ref_pd_df]
        data = dc.load_by_coords([self.lon, self.lon], [self.lat, self.lat], sref=self.sref,
                                 dimension_name='tile_name', dtype='dataframe', origin='c')
        assert ref_data_list[0].equals(data) & ref_data_list[1].equals(data)

    def test_load_singlenc2xarray_by_coord(self):
        dc = EODataCube(filepaths=[self.nc_filepath], smart_filename_creator=create_sgrt_filename,
                        dimensions=['time'])

        data = dc.load_by_coords(self.lon, self.lat, sref=self.sref, band='SIG0', dtype='xarray', origin='c')
        assert self.ref_xr_ds.equals(data.rename({'SIG0': '1'}))


class LoadingPixelsTester(LoadingTester):

    def setUp(self):
        gt_filepaths, timestamps = setup_gt_test_data()
        self.gt_filepaths = gt_filepaths
        nc_filepaths, _ = setup_nc_multi_test_data()
        self.nc_filepaths = nc_filepaths
        nc_filepath, _ = setup_nc_single_test_data()
        self.nc_filepath = nc_filepath
        self.timestamps = timestamps
        self.row = 246
        self.col = 970
        x = 4323250.
        y = 1314750.

        self.ref_np_ar = (np.array([[[self.row + self.col]*4]]).T + np.arange(0, 4)[:, None, None]).astype(float)
        xr_ar = xr.DataArray(data=self.ref_np_ar.astype(float),
                             coords={'time': self.timestamps, 'x': [x], 'y': [y]},
                             dims=['time', 'x', 'y'])
        self.ref_xr_ds = xr.Dataset(data_vars={'1': xr_ar})
        self.ref_pd_df = self.ref_xr_ds.to_dataframe()

        self.row_size = 10
        self.col_size = 10
        rows, cols = np.meshgrid(np.arange(self.row, self.row+self.row_size),
                                 np.arange(self.col, self.col+self.col_size))
        xs = np.arange(x, x + self.row_size * 500, 500.)
        ys = np.arange(y, y - self.col_size * 500, -500.)
        base_np_ar_2D = rows + cols
        base_np_ar = np.stack([base_np_ar_2D]*4, axis=0)
        self.ref_np_ar_area = (base_np_ar + np.arange(0, 4)[:, None, None]).astype(float)
        xr_ar = xr.DataArray(data=self.ref_np_ar_area.astype(float),
                             coords={'time': self.timestamps, 'x': xs, 'y': ys},
                             dims=['time', 'x', 'y'])
        self.ref_xr_ds_area = xr.Dataset(data_vars={'1': xr_ar})
        self.ref_pd_df_area = self.ref_xr_ds_area.to_dataframe()

    def test_load_gt2numpy_by_pixels(self):
        dc = self._create_loadable_dc(self.gt_filepaths)

        data = dc.load_by_pixels(self.row, self.col, dimension_name='tile_name', dtype='numpy')
        assert (self.ref_np_ar == data).all()

        data = dc.load_by_pixels(self.row, self.col, row_size=self.row_size, col_size=self.col_size,
                                 dimension_name='tile_name', dtype='numpy')
        assert (self.ref_np_ar_area == data).all()

    def test_load_gt2xarray_by_pixels(self):
        dc = self._create_loadable_dc(self.gt_filepaths)

        data = dc.load_by_pixels(self.row, self.col, dimension_name='tile_name', dtype='xarray', origin='c')
        # convert to float
        data['1'].data = data['1'].data.astype(float)
        assert self.ref_xr_ds.equals(data)

        data = dc.load_by_pixels(self.row, self.col, row_size=self.row_size, col_size=self.col_size,
                                 dimension_name='tile_name', dtype='xarray', origin='c')
        data['1'].data = data['1'].data.astype(float)
        assert self.ref_xr_ds_area.equals(data)

    def test_load_gt2dataframe_by_pixels(self):
        dc = self._create_loadable_dc(self.gt_filepaths)

        data = dc.load_by_pixels(self.row, self.col, dimension_name='tile_name', dtype='dataframe', origin='c')
        # convert to float
        data['1'] = data['1'].astype(float)
        assert self.ref_pd_df.equals(data)

        data = dc.load_by_pixels(self.row, self.col, row_size=self.row_size, col_size=self.col_size,
                                 dimension_name='tile_name', dtype='dataframe', origin='c')
        data['1'] = data['1'].astype(float)
        assert self.ref_pd_df_area.equals(data)

    def test_load_nc2numpy_by_pixels(self):
        dc = self._create_loadable_dc(self.nc_filepaths)

        data = dc.load_by_pixels(self.row, self.col, dimension_name='tile_name', dtype='numpy')
        assert (self.ref_np_ar == data).all()

        data = dc.load_by_pixels(self.row, self.col, row_size=self.row_size, col_size=self.col_size,
                                 dimension_name='tile_name', dtype='numpy')
        assert (self.ref_np_ar_area == data).all()

    def test_load_nc2xarray_by_pixels(self):
        dc = self._create_loadable_dc(self.nc_filepaths)

        data = dc.load_by_pixels(self.row, self.col, dimension_name='tile_name', dtype='xarray', origin='c')
        # convert to float
        data['1'].data = data['1'].data.astype(float)
        assert self.ref_xr_ds.equals(data)

        data = dc.load_by_pixels(self.row, self.col, row_size=self.row_size, col_size=self.col_size,
                                 dimension_name='tile_name', dtype='xarray', origin='c')
        data['1'].data = data['1'].data.astype(float)
        assert self.ref_xr_ds_area.equals(data)

    def test_load_nc2dataframe_by_pixels(self):
        dc = self._create_loadable_dc(self.nc_filepaths)

        data = dc.load_by_pixels(self.row, self.col, dimension_name='tile_name', dtype='dataframe', origin='c')
        # convert to float
        data['1'] = data['1'].astype(float)
        assert self.ref_pd_df.equals(data)

        data = dc.load_by_pixels(self.row, self.col, row_size=self.row_size, col_size=self.col_size,
                                 dimension_name='tile_name', dtype='dataframe', origin='c')
        data['1'] = data['1'].astype(float)
        assert self.ref_pd_df_area.equals(data)

    def test_load_singlenc2xarray_by_pixels(self):
        dc = EODataCube(filepaths=[self.nc_filepath], smart_filename_creator=create_sgrt_filename,
                        dimensions=['time'])

        data = dc.load_by_pixels(self.row, self.col, row_size=self.row_size, col_size=self.col_size, band='SIG0',
                                 dtype='xarray', origin='c')
        assert self.ref_xr_ds_area.equals(data.rename({'SIG0': '1'}))


class LoadingGeomTester(LoadingTester):

    def setUp(self):
        gt_filepaths, timestamps = setup_gt_test_data()
        self.gt_filepaths = gt_filepaths
        nc_filepaths, _ = setup_nc_multi_test_data()
        self.nc_filepaths = nc_filepaths
        nc_filepath, _ = setup_nc_single_test_data()
        self.nc_filepath = nc_filepath
        self.timestamps = timestamps
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

        rows, cols = np.meshgrid(np.arange(row, row+row_size),
                                 np.arange(col, col+col_size))
        xs = np.arange(x, x + row_size * 500, 500.)
        ys = np.arange(y, y - col_size * 500, -500.)
        base_np_ar_2D = rows + cols
        base_np_ar = np.stack([base_np_ar_2D]*4, axis=0)
        self.ref_np_ar_area = (base_np_ar + np.arange(0, 4)[:, None, None]).astype(float)
        xr_ar = xr.DataArray(data=self.ref_np_ar_area.astype(float),
                             coords={'time': self.timestamps, 'x': xs, 'y': ys},
                             dims=['time', 'x', 'y'])
        self.ref_xr_ds_area = xr.Dataset(data_vars={'1': xr_ar})
        self.ref_pd_df_area = self.ref_xr_ds_area.to_dataframe()

    def test_load_gt2numpy_by_geom(self):
        dc = self._create_loadable_dc(self.gt_filepaths)
        data = dc.load_by_geom(self.bbox, dimension_name='tile_name', dtype='numpy')
        assert (self.ref_np_ar_area == data).all()

    def test_load_gt2xarray_by_geom(self):
        dc = self._create_loadable_dc(self.gt_filepaths)
        data = dc.load_by_geom(self.bbox, dimension_name='tile_name', dtype='xarray', origin='c')
        # convert to float
        data['1'].data = data['1'].data.astype(float)
        assert self.ref_xr_ds_area.equals(data)

    def test_load_gt2dataframe_by_geom(self):
        dc = self._create_loadable_dc(self.gt_filepaths)
        data = dc.load_by_geom(self.bbox, dimension_name='tile_name', dtype='dataframe', origin='c')
        # convert to float
        data['1'] = data['1'].astype(float)
        assert self.ref_pd_df_area.equals(data)

    def test_load_nc2numpy_by_geom(self):
        dc = self._create_loadable_dc(self.nc_filepaths)
        data = dc.load_by_geom(self.bbox, dimension_name='tile_name', dtype='numpy')
        assert (self.ref_np_ar_area == data).all()

    def test_load_nc2xarray_by_geom(self):
        dc = self._create_loadable_dc(self.nc_filepaths)
        data = dc.load_by_geom(self.bbox, dimension_name='tile_name', dtype='xarray', origin='c')
        assert self.ref_xr_ds_area.equals(data)

    def test_load_nc2dataframe_by_geom(self):
        dc = self._create_loadable_dc(self.nc_filepaths)
        data = dc.load_by_geom(self.bbox, dimension_name='tile_name', dtype='dataframe', origin='c')
        assert self.ref_pd_df_area.equals(data)

    def test_load_singlenc2xarray_by_pixels(self):
        dc = EODataCube(filepaths=[self.nc_filepath], smart_filename_creator=create_sgrt_filename,
                        dimensions=['time'])

        data = dc.load_by_geom(self.bbox, band='SIG0', dtype='xarray', origin='c')
        assert self.ref_xr_ds_area.equals(data.rename({'SIG0': '1'}))

if __name__ == '__main__':
    unittest.main()
