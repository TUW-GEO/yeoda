# general imports
import os
import shutil
import unittest

from tests.setup_test_data import setup_test_data, dirpath_test, roi_test


# import yeoda
from products.preprocessed import PreprocessedDataCube, SIG0DataCube, GMRDataCube


class FilteringTester(unittest.TestCase):

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

    def test_filter_pols(self):
        dc = PreprocessedDataCube(self.dir_tree.root, spres=500, dimensions=['time', 'var_name', 'pol'])
        assert len(set(dc['pol'])) == 2
        dc.filter_by_dimension("VV", name="pol", in_place=True)
        assert len(set(dc['pol'])) == 1

    def test_filter_pols_in_place(self):
        dc = PreprocessedDataCube(self.dir_tree.root, spres=500, dimensions=['time', 'var_name', 'pol'])
        dc_vv = dc.filter_by_dimension("VV", name="pol")
        dc_vh = dc.filter_by_dimension("VH", name="pol")
        assert len(set(dc_vv['pol'])) == 1
        assert list(set(dc_vv['pol']))[0] == "VV"
        assert len(set(dc_vh['pol'])) == 1
        assert list(set(dc_vh['pol']))[0] == "VH"

    def test_filter_pols_clone(self):
        dc = PreprocessedDataCube(self.dir_tree.root, spres=500, dimensions=['time', 'var_name', 'pol'])
        dc_clone = dc.clone()
        dc.filter_by_dimension("VV", name="pol", in_place=True)
        dc_clone.filter_by_dimension("VH", name="pol", in_place=True)
        assert len(set(dc['pol'])) == 1
        assert list(set(dc['pol']))[0] == "VV"
        assert len(set(dc_clone['pol'])) == 1
        assert list(set(dc_clone['pol']))[0] == "VH"

    def test_filter_time(self):
        dc = PreprocessedDataCube(self.dir_tree.root, spres=500, dimensions=['time', 'var_name', 'pol'])
        start_time = self.timestamps[0]
        end_time = self.timestamps[1]
        dc.filter_by_dimension([(start_time, end_time)], expressions=[(">=", "<=")], in_place=True)
        assert sorted(list(set(dc['time']))) == self.timestamps[:2]

    def test_filter_var_names(self):
        pre_dc = PreprocessedDataCube(self.dir_tree.root, spres=500, dimensions=['time', 'var_name', 'pol'])
        sig0_dc = SIG0DataCube(self.dir_tree.root, spres=500, dimensions=['time', 'pol'])
        gmr_dc = GMRDataCube(self.dir_tree.root, spres=500, dimensions=['time', 'pol'])
        dc_filt_sig0 = pre_dc.filter_by_dimension("SIG0", name="var_name")
        dc_filt_gmr = pre_dc.filter_by_dimension("GMR", name="var_name")
        assert sorted(list(sig0_dc['filepath'])) == sorted(list(dc_filt_sig0['filepath']))
        assert sorted(list(gmr_dc['filepath'])) == sorted(list(dc_filt_gmr['filepath']))

    def test_filter_files_with_pattern(self):
        pre_dc = PreprocessedDataCube(self.dir_tree.root, spres=500, dimensions=['time', 'var_name', 'pol'])
        sig0_dc = SIG0DataCube(self.dir_tree.root, spres=500, dimensions=['time', 'pol'])
        pre_dc.filter_files_with_pattern(".*SIG0.*", in_place=True)
        assert sorted(list(sig0_dc['filepath'])) == sorted(list(pre_dc['filepath']))

    def test_filter_spatially_by_tilename(self):
        dc = PreprocessedDataCube(self.dir_tree.root, spres=500, dimensions=['time', 'var_name', 'pol', 'tile_name'])
        assert len(set(dc['tile_name'])) == 2
        dc.filter_spatially_by_tilename("E042N012T6", dimension_name='tile_name', in_place=True)
        assert len(set(dc['tile_name'])) == 1
        assert list(set(dc['tile_name']))[0] == "E042N012T6"

    def test_filter_spatially_by_geom(self):
        dc = PreprocessedDataCube(self.dir_tree.root, spres=500, dimensions=['time', 'var_name', 'pol', 'tile_name'])
        bbox, sref = roi_test()
        assert len(set(dc['tile_name'])) == 2
        dc.filter_spatially_by_geom(bbox, sref=sref, dimension_name='tile_name', in_place=True)
        assert len(set(dc['tile_name'])) == 1
        assert list(set(dc['tile_name']))[0] == "E042N012T6"

    def test_filter_by_metadata(self):
        dc = PreprocessedDataCube(self.dir_tree.root, spres=500, dimensions=['time', 'orbit_direction'])
        assert len(set(dc['orbit_direction'])) == 2
        dc.filter_by_metadata({'direction': 'D'}, in_place=True)
        assert len(set(dc['orbit_direction'])) == 1
        assert list(set(dc['orbit_direction']))[0] == "D"

if __name__ == '__main__':
    unittest.main()
