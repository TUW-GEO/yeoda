# general imports
import os
import unittest
import osr

import matplotlib.pyplot as plt
import numpy as np

# import yeoda
from products.ssm import SSMDataCube


def cur_path():
    pth, _ = os.path.split(os.path.abspath(__file__))
    return pth


class TestSSMDataCube(unittest.TestCase):

    def setUp(self):
        self.path = os.path.join(cur_path(), 'testdata')

    def tearDown(self):
        pass

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


if __name__ == '__main__':
    dc_tester = TestSSMDataCube()
    dc_tester.setUp()
    dc_tester.test_read_ts()