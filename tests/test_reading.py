# general imports
import os
import unittest
import osr

from osgeo import ogr
from shapely.wkt import load
from shapely.geometry import Polygon
#import matplotlib.pyplot as plt

# import yeoda
from yeoda.yeoda import EODataCube
from yeoda.yeoda import match_dimension

# import file and folder naming convention
from geopathfinder.naming_conventions.sgrt_naming import create_sgrt_filename
from geopathfinder.naming_conventions.sgrt_naming import sgrt_tree

# import grid
from Equi7Grid.equi7grid.equi7grid import Equi7Grid


def cur_path():
    pth, _ = os.path.split(os.path.abspath(__file__))
    return pth


class TestEODataCube(unittest.TestCase):

    def setUp(self):
        self.path = os.path.join(cur_path(), 'testdata')
        self.shp_filename = "roi.shp"

    def tearDown(self):
        pass

    def test_read_ts_by_coord(self):

        root_dirpath = os.path.join(self.path, 'Sentinel-1_CSAR')
        st = sgrt_tree(root_dirpath, register_file_pattern=(".tif$"))
        grid = Equi7Grid(500)
        sub_grid = grid.EU
        lon = 15.5352
        lat = 48.1724
        src_spref = osr.SpatialReference()
        src_spref.ImportFromEPSG(4326)

        eodc = EODataCube(dir_tree=st, smart_filename_creator=create_sgrt_filename,
                          dimensions=['time', 'tile_name', 'pol'], grid=sub_grid)
        eodc.rename_dimensions({'tile_name': 'tile'}, in_place=True)
        eodc.filter_spatially_by_tilename(tilenames="E048N012T6", in_place=True)

        data = eodc.load_by_coord(lon, lat, src_spref=src_spref)

        # prepare data for plotting
        ssm = data.flatten().tolist()
        timestamps = eodc.inventory['time']
        # plot the data
        plt.figure()
        plt.stem(timestamps, ssm)
        plt.show()

        plt.close()
        fig, ax = None, None

    def test_read_ts_by_poly(self):

        root_dirpath = os.path.join(self.path, 'Sentinel-1_CSAR')
        st = sgrt_tree(root_dirpath, register_file_pattern=(".tif$"))
        grid = Equi7Grid(500)
        sub_grid = grid.EU
        roi = self.__read_shp_file()

        eodc = EODataCube(dir_tree=st, smart_filename_creator=create_sgrt_filename,
                          dimensions=['time', 'tile_name', 'pol'], grid=sub_grid)
        eodc.rename_dimensions({'tile_name': 'tile'}, in_place=True)
        eodc.filter_spatially_by_tilename(tilenames="E048N012T6", in_place=True)

        data = eodc.load_by_geom(roi)

        # visualize the data
        plt.figure()
        plt.imshow(data[1, :, :])
        plt.close()
        fig, ax = None, None

        pass

    # TODO: fails with exit code -1073740791 (0xC0000409) if only feature.GetGeometryRef() is called
    def __read_shp_file(self):
        shp_filepath = os.path.join(self.path, self.shp_filename)
        ds = ogr.Open(shp_filepath)
        layer = ds.GetLayer()
        # first feature of the shapefile
        feature = layer.GetFeature(0)
        geom_wkt= feature.GetGeometryRef().ExportToWkt()

        ds = None
        layer = None
        feature = None

        geom = ogr.CreateGeometryFromWkt(geom_wkt)

        return geom

if __name__ == '__main__':
    dc_tester = TestEODataCube()
    dc_tester.setUp()
    dc_tester.test_read_ts_by_poly()