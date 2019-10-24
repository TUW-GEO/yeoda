# general imports
import os
import osr

from datetime import datetime
import numpy as np
import pandas as pd
import itertools

from geopathfinder.naming_conventions.sgrt_naming import sgrt_tree

from equi7grid.equi7grid import Equi7Grid
from pyraster.geotiff import GeoTiffFile


def dirpath_test():
    pth, _ = os.path.split(os.path.abspath(__file__))
    return pth


def setup_test_data(initialise=True):
    root_dirpath = os.path.join(dirpath_test(), 'data', 'Sentinel-1_CSAR')

    # create target folders
    dirpath = os.path.join(root_dirpath, 'IWGRDH', 'preprocessed', 'datasets', 'resampled', 'T0101', 'EQUI7_EU500M')

    timestamps = [datetime(2016, 1, 1), datetime(2016, 2, 1), datetime(2017, 1, 1), datetime(2017, 2, 1)]

    if initialise:
        var_names = ["SIG0", "GMR-"]
        pols = ["VV", "VH"]
        directions = ["A", "D"]
        tilenames = ["E048N012T6", "E042N012T6"]
        filename_fmt = "D{}_000000--_{}-----_S1AIWGRDH1{}{}_146_T0101_EU500M_{}.tif"
        combs = itertools.product(var_names, pols, directions, timestamps, tilenames)

        data = np.zeros((1600, 1600))
        equi7 = Equi7Grid(500)

        filepaths = []
        for comb in combs:
            var_name = comb[0]
            pol = comb[1]
            direction = comb[2]
            timestamp = comb[3]
            tilename = comb[4]
            filename = filename_fmt.format(timestamp.strftime("%Y%m%d"), var_name, pol, direction, tilename)
            if var_name == "SIG0":
                var_dirpath = os.path.join(dirpath, tilename, "sig0")
            elif var_name == "GMR-":
                var_dirpath = os.path.join(dirpath, tilename, "gmr")
            else:
                raise Exception("Variable name {} unknown.".format(var_name))

            if not os.path.exists(var_dirpath):
                os.makedirs(var_dirpath)
            filepath = os.path.join(var_dirpath, filename)

            if not os.path.exists(filepath):
                tile_oi = equi7.EU.tilesys.create_tile(name=tilename)
                tags = {'metadata': {'direction': direction}}
                gt_file = GeoTiffFile(filepath, mode='w', count=1, geotransform=tile_oi.geotransform(),
                                      spatialref=tile_oi.get_geotags()['spatialreference'], tags=tags)

                data[:] = timestamps.index(timestamp)
                gt_file.write(data, band=1, nodata=[-9999])
                gt_file.close()
                filepaths.append(filepath)

    dir_tree = sgrt_tree(root_dirpath, register_file_pattern=".tif$")
    timestamps = [pd.Timestamp(timestamp.strftime("%Y%m%d")) for timestamp in timestamps]

    return dir_tree, timestamps


def roi_test():
    bbox = [(4.36, 43.44), (6.48, 45.80)]
    sref = osr.SpatialReference()
    sref.ImportFromEPSG(4326)
    return bbox, sref
