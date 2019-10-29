# general imports
import os
import shutil
import unittest
import osr
import numpy as np

from tests.setup_test_data import setup_test_data, dirpath_test, roi_test
from geopathfinder.naming_conventions.sgrt_naming import create_sgrt_filename
from yeoda.datacube import EODataCube


class LoadingTester(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        setup_test_data()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(os.path.join(dirpath_test(), 'data'))

    def setUp(self):
        dir_tree, timestamps = setup_test_data(initialise=False)
        self.dir_tree = dir_tree
        self.timestamps = timestamps

    def test_load_by_coord(self):
        dc = EODataCube(dir_tree=self.dir_tree, smart_filename_creator=create_sgrt_filename,
                        dimensions=['time', 'var_name', 'pol', 'tile_name', 'orbit_direction'])
        x = 5.
        y = 44.
        sref = osr.SpatialReference()
        sref.ImportFromEPSG(4326)
        dc.filter_by_dimension('VV', name='pol', in_place=True)
        dc.filter_by_dimension('SIG0', name='var_name', in_place=True)
        dc.filter_by_dimension('D', name='orbit_direction', in_place=True)
        dc.filter_spatially_by_tilename('E042N012T6', dimension_name='tile_name', in_place=True, use_grid=False)
        data = dc.load_by_coord(x, y, sref=sref, dimension_name='tile_name')
        assert (data == np.array([0., 1., 2., 3.])).all()

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

if __name__ == '__main__':
    unittest.main()
