# general imports
import os
import shutil
import unittest
import osr

import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import numpy as np
import itertools

from geopathfinder.naming_conventions.sgrt_naming import sgrt_tree

# import yeoda
from products.ssm import SSMDataCube
from products.preprocessed import PreprocessedDataCube

from equi7grid.equi7grid import Equi7Grid
from pyraster.geotiff import GeoTiffFile

def cur_dirpath():
    pth, _ = os.path.split(os.path.abspath(__file__))
    return pth


class FilteringTester(unittest.TestCase):

    def setUp(self):
        root_dirpath = os.path.join(cur_dirpath(), 'data', 'Sentinel-1_CSAR')

        # create target folders
        sig0_dirpath = os.path.join(root_dirpath, 'IWGRDH', 'preprocessed', 'datasets', 'resampled', 'T0101',
                                    'EQUI7_EU500M', 'E048N012T6', 'sig0')
        gmr_dirpath = os.path.join(root_dirpath, 'IWGRDH', 'preprocessed', 'datasets', 'resampled', 'T0101',
                                    'EQUI7_EU500M', 'E048N012T6', 'gmr')
        if not os.path.exists(sig0_dirpath):
            os.makedirs(sig0_dirpath)
        if not os.path.exists(gmr_dirpath):
            os.makedirs(gmr_dirpath)

        var_names = ["SIG0", "GMR-"]
        pols = ["VV", "VH"]
        directions = ["A", "D"]
        ref_datetime = datetime.now()
        timestamps = [ref_datetime + timedelta(days=i) for i in range(1, 6)]
        filename_fmt = "D{}_000000--_{}-----_S1AIWGRDH1{}{}_146_T0101_EU500M_E048N012T6.tif"
        combs = itertools.product(var_names, pols, directions, timestamps)

        data = np.zeros((1600, 1600))
        equi7 = Equi7Grid(500)
        tile_oi = equi7.EU.tilesys.create_tile(name='E048N012T6')

        filepaths = []
        for comb in combs:
            var_name = comb[0]
            pol = comb[1]
            direction = comb[2]
            timestamp = comb[3]
            filename = filename_fmt.format(timestamp.strftime("%Y%m%d"), var_name, pol, direction)
            if var_name == "SIG0":
                dirpath = sig0_dirpath
            elif var_name == "GMR-":
                dirpath = gmr_dirpath
            else:
                raise Exception("Variable name {} unknown.".format(var_name))

            filepath = os.path.join(dirpath, filename)

            gt_file = GeoTiffFile(filepath, mode='w', count=1, geotransform=tile_oi.geotransform(),
                                       spatialref=tile_oi.get_geotags()['spatialreference'])

            data[:] = timestamps.index(timestamp)
            gt_file.write(data, band=1, nodata=[-9999])
            gt_file.close()
            filepaths.append(filepath)

        self.dir_tree = sgrt_tree(root_dirpath, register_file_pattern=".tif$")

    def tearDown(self):
        shutil.rmtree(os.path.join(cur_dirpath(), 'data'))

    def test_filter_pols(self):
        preprocessed_dc = PreprocessedDataCube(self.dir_tree.root, spres=500, dimensions=['time', 'var_name', 'pol'])
        assert len(set(preprocessed_dc.inventory['pol'])) == 2

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
    filtering_tester.test_filter_pols()
    filtering_tester.tearDown()