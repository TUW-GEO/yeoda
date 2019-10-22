# general imports
import os
import shutil
import unittest
import osr

import matplotlib.pyplot as plt
from tests.setup_test_data import setup, test_dirpath
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import itertools

from geopathfinder.naming_conventions.sgrt_naming import sgrt_tree

# import yeoda
from products.ssm import SSMDataCube
from products.preprocessed import PreprocessedDataCube, SIG0DataCube, GMRDataCube

from equi7grid.equi7grid import Equi7Grid
from pyraster.geotiff import GeoTiffFile



class FilteringTester(unittest.TestCase):

    def setUp(self):
        dir_tree, timestamps = setup()
        self.dir_tree = dir_tree
        self.timestamps = timestamps

    def tearDown(self):
        shutil.rmtree(os.path.join(test_dirpath(), 'data'))

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
        end_time = self.timestamps[3]
        dc.filter_by_dimension([(start_time, end_time)], expressions=[(">=", "<=")], in_place=True)
        assert sorted(list(set(dc['time']))) == self.timestamps[:4]

    def test_split_time(self):
        dc = PreprocessedDataCube(self.dir_tree.root, spres=500, dimensions=['time', 'var_name', 'pol'])
        first_time_interval = (self.timestamps[0], self.timestamps[2])
        second_time_interval = (self.timestamps[3], self.timestamps[-1])
        expression = (">=", "<=")
        dcs = dc.split_by_dimension([first_time_interval, second_time_interval], expressions=[expression, expression])
        assert len(dcs) == 2
        assert sorted(list(set(dcs[0]['time']))) == self.timestamps[:3]
        assert sorted(list(set(dcs[1]['time']))) == self.timestamps[3:]

    def test_filter_var_names(self):
        pre_dc = PreprocessedDataCube(self.dir_tree.root, spres=500, dimensions=['time', 'var_name', 'pol'])
        sig0_dc = SIG0DataCube(self.dir_tree.root, spres=500, dimensions=['time', 'pol'])
        gmr_dc = GMRDataCube(self.dir_tree.root, spres=500, dimensions=['time', 'pol'])
        dc_filt_sig0 = pre_dc.filter_by_dimension("SIG0", name="var_name")
        dc_filt_gmr = pre_dc.filter_by_dimension("GMR", name="var_name")
        assert sorted(list(sig0_dc['filepath'])) == sorted(list(dc_filt_sig0['filepath']))
        assert sorted(list(gmr_dc['filepath'])) == sorted(list(dc_filt_gmr['filepath']))

    def test_read_ts(self):

        root_dirpath = os.path.join(self.path, 'Sentinel-1_CSAR')
        lon = 15.5352
        lat = 48.1724
        src_spref = osr.SpatialReference()
        src_spref.ImportFromEPSG(4326)

        ssm_dc = SSMDataCube(root_dirpath, spres=500, continent='EU', dimensions=['time', 'tile_name', 'var_name'])
        ssm_dc.rename_dimensions({'tile_name': 'tile'}, in_place=True)
        ssm_dc.filter_spatially_by_tilename(tilenames="E048N012T6", in_place=True)

        data = ssm_dc.load_by_coord(lon, lat, src_spref=src_spref)

        # prepare data for plotting
        ssm = data.flatten().tolist()
        timestamps = ssm_dc.inventory['time']
        # plot the data
        plt.figure()
        plt.stem(timestamps, ssm)
        plt.show()

        plt.close()
        fig, ax = None, None

        pass

    def test_filter_spatially(self):

        root_dirpath = os.path.join(self.path, 'Sentinel-1_CSAR')
        roi = Polygon([(4373136, 1995726), (4373136, 3221041), (6311254, 3221041), (6311254, 1995726)])
        st = sgrt_tree(root_dirpath, register_file_pattern=(".tif$"))
        eodc = EODataCube(dir_tree=st, smart_filename_creator=create_sgrt_filename,
                          dimensions=['time', 'tile_name', 'pol'], ignore_metadata=False)


        fig, ax = plt.subplots(1, 1)

        eodc.inventory.drop_duplicates(subset=['tile_name']).plot(ax=ax, edgecolor='red', facecolor="none")
        eodc_roi = eodc.filter_spatially(roi=roi)
        eodc_roi.inventory.drop_duplicates(subset=['tile_name']).plot(ax=ax, edgecolor='blue', facecolor="none")
        plt.show()

        plt.close()
        fig, ax = None, None

        pass

if __name__ == '__main__':
    filtering_tester = FilteringTester()
    filtering_tester.setUp()
    filtering_tester.test_filter_var_names()
    filtering_tester.tearDown()