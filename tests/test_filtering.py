# general imports
import os
import unittest

from shapely.wkt import load
from shapely.geometry import Polygon
import matplotlib.pyplot as plt

# import yeoda
from yeoda import eoDataCube, match_dimension

# import file and folder naming convention
from geopathfinder.sgrt_naming import create_sgrt_filename, sgrt_tree

# import grid
from Equi7Grid.equi7grid.equi7grid import Equi7Grid


def cur_path():
    pth, _ = os.path.split(os.path.abspath(__file__))
    return pth


class TestEODataCube(unittest.TestCase):

    def setUp(self):
        self.path = os.path.join(cur_path(), 'testdata')

    def tearDown(self):
        pass

    def test_filter_spatially(self):

        root_dirpath = os.path.join(self.path, 'Sentinel-1_CSAR')
        roi = Polygon([(4373136, 1995726), (4373136, 3221041), (6311254, 3221041), (6311254, 1995726)])
        st = sgrt_tree(root_dirpath, register_file_pattern=(".tif$"))
        eodc = eoDataCube(dir_tree=st, smart_filename_creator=create_sgrt_filename,
                          dimensions=['time', 'tile_name', 'pol'])


        fig, ax = plt.subplots(1, 1)

        eodc.inventory.drop_duplicates(subset=['tile_name']).plot(ax=ax, edgecolor='red', facecolor="none")
        eodc_roi = eodc.filter_spatially(roi=roi)
        eodc_roi.inventory.drop_duplicates(subset=['tile_name']).plot(ax=ax, edgecolor='blue', facecolor="none")
        plt.show()

        plt.close()
        fig, ax = None, None

        pass


if __name__ == '__main__':
    dc_tester = TestEODataCube()
    dc_tester.setUp()
    dc_tester.test_filter_spatially()