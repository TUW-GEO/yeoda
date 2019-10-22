# general imports
import os

from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import itertools

from geopathfinder.naming_conventions.sgrt_naming import sgrt_tree

from equi7grid.equi7grid import Equi7Grid
from pyraster.geotiff import GeoTiffFile


def test_dirpath():
    pth, _ = os.path.split(os.path.abspath(__file__))
    return pth


def setup():
    root_dirpath = os.path.join(test_dirpath(), 'data', 'Sentinel-1_CSAR')

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

        if not os.path.exists(filepath):
            gt_file = GeoTiffFile(filepath, mode='w', count=1, geotransform=tile_oi.geotransform(),
                                  spatialref=tile_oi.get_geotags()['spatialreference'])

            data[:] = timestamps.index(timestamp)
            gt_file.write(data, band=1, nodata=[-9999])
            gt_file.close()
            filepaths.append(filepath)

    dir_tree = sgrt_tree(root_dirpath, register_file_pattern=".tif$")
    timestamps = [pd.Timestamp(timestamp.strftime("%Y%m%d")) for timestamp in timestamps]

    return dir_tree, timestamps
