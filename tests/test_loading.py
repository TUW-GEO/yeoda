# general imports
import os
import shutil
import unittest
import osr
import netCDF4
import numpy as np
import xarray as xr
import pandas as pd

from tests.setup_test_data import setup_gt_test_data
from tests.setup_test_data import setup_nc_single_test_data
from tests.setup_test_data import setup_nc_multi_test_data
from tests.setup_test_data import dirpath_test
from tests.setup_test_data import roi_test

from geopathfinder.naming_conventions.sgrt_naming import create_sgrt_filename
from yeoda.datacube import EODataCube


class LoadingTester(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        setup_gt_test_data()
        setup_nc_multi_test_data()
        setup_nc_single_test_data()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(os.path.join(dirpath_test(), 'data'))

    def setUp(self):
        gt_filepaths, timestamps = setup_gt_test_data()
        self.gt_filepaths = gt_filepaths
        nc_filepaths, _ = setup_nc_multi_test_data()
        self.nc_filepaths = nc_filepaths
        nc_single_filepath, _ = setup_nc_single_test_data()
        self.nc_single_filepath = nc_single_filepath
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
        self.ref_pd_df = pd.DataFrame({'time': self.timestamps,
                                       'x': [self.x]*4,
                                       'y': [self.y]*4,
                                       '1': self.ref_np_ar.flatten().tolist()})
        xr_ar = xr.DataArray(data=self.ref_np_ar.astype(float),
                             coords={'time': self.timestamps, 'x': [self.x], 'y': [self.y]},
                             dims=['time', 'x', 'y'])
        self.ref_xr_ds = xr.Dataset(data_vars={'1': xr_ar})

    def test_load_gt2numpy_by_coord(self):
        dc = self.__create_loadable_dc(self.gt_filepaths)

        data = dc.load_by_coords(self.lat, self.lon, sref=self.sref, dimension_name='tile_name', dtype='numpy')
        assert (self.ref_np_ar == data).all()

        ref_data_list = [self.ref_np_ar, self.ref_np_ar]
        data = dc.load_by_coords([self.lat, self.lat], [self.lon, self.lon], sref=self.sref,
                                 dimension_name='tile_name', dtype='numpy')
        assert (ref_data_list[0] == data[0]).all() & (ref_data_list[1] == data[1]).all()

    def test_load_gt2xarray_by_coord(self):
        dc = self.__create_loadable_dc(self.gt_filepaths)

        data = dc.load_by_coords(self.lat, self.lon, sref=self.sref, dimension_name='tile_name', dtype='xarray',
                                 origin='c')
        # convert to float
        data['1'].data = data['1'].data.astype(float)
        assert self.ref_xr_ds.equals(data)

        ref_data_list = [self.ref_xr_ds, self.ref_xr_ds]
        data = dc.load_by_coords([self.lat, self.lat], [self.lon, self.lon], sref=self.sref,
                                 dimension_name='tile_name', dtype='xarray', origin='c')
        assert ref_data_list[0].equals(data) & ref_data_list[1].equals(data)

    def test_load_gt2dataframe_by_coord(self):
        dc = self.__create_loadable_dc(self.gt_filepaths)

        data = dc.load_by_coords(self.lat, self.lon, sref=self.sref, dimension_name='tile_name', dtype='dataframe',
                                 origin='c')
        # convert to float for comparison
        data['1'] = data['1'].astype(float)
        assert self.ref_pd_df.equals(data)

        ref_data_list = [self.ref_pd_df, self.ref_pd_df]
        data = dc.load_by_coords([self.lat, self.lat], [self.lon, self.lon], sref=self.sref,
                                 dimension_name='tile_name', dtype='dataframe', origin='c')
        # convert to float for comparison
        data['1'] = data['1'].astype(float)
        assert ref_data_list[0].equals(data) & ref_data_list[1].equals(data)

    def test_load_nc2numpy_by_coord(self):
        dc = self.__create_loadable_dc(self.nc_filepaths)

        data = dc.load_by_coords(self.lat, self.lon, sref=self.sref, dimension_name='tile_name', dtype='numpy')
        assert (self.ref_np_ar == data).all()

        ref_data_list = [self.ref_np_ar, self.ref_np_ar]
        data = dc.load_by_coords([self.lat, self.lat], [self.lon, self.lon], sref=self.sref,
                                 dimension_name='tile_name', dtype='numpy')
        assert (ref_data_list[0] == data[0]).all() & (ref_data_list[1] == data[1]).all()

    def test_load_nc2xarray_by_coord(self):
        dc = self.__create_loadable_dc(self.nc_filepaths)

        data = dc.load_by_coords(self.lat, self.lon, sref=self.sref, dimension_name='tile_name', dtype='xarray',
                                 origin='c')
        assert self.ref_xr_ds.equals(data)

        ref_data_list = [self.ref_xr_ds, self.ref_xr_ds]
        data = dc.load_by_coords([self.lat, self.lat], [self.lon, self.lon], sref=self.sref,
                                 dimension_name='tile_name', dtype='xarray', origin='c')
        assert ref_data_list[0].equals(data) & ref_data_list[1].equals(data)

    def test_load_nc2dataframe_by_coord(self):
        dc = self.__create_loadable_dc(self.nc_filepaths)

        data = dc.load_by_coords(self.lat, self.lon, sref=self.sref, dimension_name='tile_name', dtype='dataframe',
                                 origin='c')
        assert self.ref_pd_df.equals(data)

        ref_data_list = [self.ref_pd_df, self.ref_pd_df]
        data = dc.load_by_coords([self.lat, self.lat], [self.lon, self.lon], sref=self.sref,
                                 dimension_name='tile_name', dtype='dataframe', origin='c')
        assert ref_data_list[0].equals(data) & ref_data_list[1].equals(data)

    def test_load_by_geom(self):
        dc = EODataCube(dir_tree=self.dir_tree, smart_filename_creator=create_sgrt_filename,
                        dimensions=['time', 'var_name', 'pol', 'tile_name', 'orbit_direction'])
        bbox, sref = roi_test()
        dc.filter_by_dimension('VV', name='pol', in_place=True)
        dc.filter_by_dimension('SIG0', name='var_name', in_place=True)
        dc.filter_by_dimension('D', name='orbit_direction', in_place=True)
        dc.filter_spatially_by_geom(bbox, sref=sref, dimension_name='tile_name', in_place=True)
        data = dc.load_by_geom(bbox, sref=sref, dimension_name='tile_name', apply_mask=False)
        ref_data = np.ones(data.shape)
        ref_data *= np.array([0., 1., 2., 3.])[:, None, None]
        assert (data == ref_data).all()

    def __create_loadable_dc(self, filepaths):
        dc = EODataCube(filepaths=filepaths, smart_filename_creator=create_sgrt_filename,
                        dimensions=['time', 'var_name', 'pol', 'tile_name', 'orbit_direction'])

        dc.filter_by_dimension('VV', name='pol', in_place=True)
        dc.filter_by_dimension('SIG0', name='var_name', in_place=True)
        dc.filter_by_dimension('D', name='orbit_direction', in_place=True)
        dc.filter_spatially_by_tilename('E042N012T6', dimension_name='tile_name', in_place=True, use_grid=False)

        return dc

if __name__ == '__main__':
    load_test = LoadingTester()
    load_test.setUpClass()
    load_test.setUp()
    load_test.test_load_gt2xarray_by_coord()
    load_test.test_load_gt2dataframe_by_coord()
    load_test.test_load_gt2numpy_by_coord()
    load_test.test_load_nc2xarray_by_coord()
    load_test.test_load_nc2dataframe_by_coord()
    load_test.test_load_nc2numpy_by_coord()
    #unittest.main()
