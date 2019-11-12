# Copyright (c) 2019, Vienna University of Technology (TU Wien), Department of
# Geodesy and Geoinformation (GEO).
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# The views and conclusions contained in the software and documentation are
# those of the authors and should not be interpreted as representing official
# policies, either expressed or implied, of the FreeBSD Project.

"""
Main code for creating test data.
"""

# general packages
import os
import osr
import itertools
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime

# import veranda and Equi7Grid
from equi7grid.equi7grid import Equi7Grid
from veranda.geotiff import GeoTiffFile
from veranda.netcdf import NcFile


def dirpath_test():
    """ Defines root directory path of the test directory. """

    pth, _ = os.path.split(os.path.abspath(__file__))
    return pth


def setup_gt_test_data():
    """
    Creates test data as single-time and single-band GeoTIFF files.

    Returns
    -------
    list of str
        List of GeoTIFF test data filepaths.
    list of datetime
        List of timestamps as datetime objects.
    """

    root_dirpath = os.path.join(dirpath_test(), 'data', 'Sentinel-1_CSAR')

    # create target folders
    dirpath = os.path.join(root_dirpath, 'IWGRDH', 'preprocessed', 'datasets', 'resampled', 'T0101', 'EQUI7_EU500M')

    timestamps = [datetime(2016, 1, 1), datetime(2016, 2, 1), datetime(2017, 1, 1), datetime(2017, 2, 1)]

    var_names = ["SIG0", "GMR-"]
    pols = ["VV", "VH"]
    directions = ["A", "D"]
    tilenames = ["E048N012T6", "E042N012T6"]
    filename_fmt = "D{}_000000--_{}-----_S1AIWGRDH1{}{}_146_T0101_EU500M_{}.tif"
    combs = itertools.product(var_names, pols, directions, timestamps, tilenames)

    rows, cols = np.meshgrid(np.arange(0, 1600), np.arange(0, 1600))
    data = (rows + cols).astype(float)
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
        filepaths.append(filepath)

        if not os.path.exists(filepath):
            tile_oi = equi7.EU.tilesys.create_tile(name=tilename)
            tags = {'metadata': {'direction': direction}}
            gt_file = GeoTiffFile(filepath, mode='w', count=1, geotransform=tile_oi.geotransform(),
                                  spatialref=tile_oi.get_geotags()['spatialreference'], tags=tags)

            data_i = data + timestamps.index(timestamp)
            gt_file.write(data_i, band=1, nodata=[-9999])
            gt_file.close()

    timestamps = [pd.Timestamp(timestamp.strftime("%Y%m%d")) for timestamp in timestamps]

    return filepaths, timestamps


def setup_nc_multi_test_data():
    """
    Creates test data as single-time and single-variable NetCDF files.

    Returns
    -------
    list of str
        List of NetCDF test data filepaths.
    list of datetime
        List of timestamps as datetime objects.
    """

    root_dirpath = os.path.join(dirpath_test(), 'data', 'Sentinel-1_CSAR')

    # create target folders
    dirpath = os.path.join(root_dirpath, 'IWGRDH', 'parameters', 'datasets', 'resampled', 'T0101', 'EQUI7_EU500M',
                           'E042N012T6', 'sig0')

    timestamps = [datetime(2016, 1, 1), datetime(2016, 2, 1), datetime(2017, 1, 1), datetime(2017, 2, 1)]

    pols = ["VV", "VH"]
    directions = ["A", "D"]
    filename_fmt = "D{}_000000--_SIG0-----_S1AIWGRDH1{}{}_146_T0101_EU500M_E042N012T6.nc"
    combs = itertools.product(pols, directions, timestamps)

    rows, cols = np.meshgrid(np.arange(0, 1600), np.arange(0, 1600))
    data = (rows + cols).astype(float)
    equi7 = Equi7Grid(500)
    tile_oi = equi7.EU.tilesys.create_tile(name="E042N012T6")

    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

    filepaths = []
    for comb in combs:
        pol = comb[0]
        direction = comb[1]
        timestamp = comb[2]
        filename = filename_fmt.format(timestamp.strftime("%Y%m%d"), pol, direction)
        filepath = os.path.join(dirpath, filename)
        filepaths.append(filepath)

        if not os.path.exists(filepath):
            tags = {'direction': direction}
            nc_file = NcFile(filepath, mode='w', geotransform=tile_oi.geotransform(),
                         spatialref=tile_oi.get_geotags()['spatialreference'])
            data_i = data + timestamps.index(timestamp)
            xr_ar = xr.DataArray(data=data_i[None, :, :], coords={'time': [timestamp]},
                                 dims=['time', 'x', 'y'])
            xr_ds = xr.Dataset(data_vars={'1': xr_ar}, attrs=tags)
            nc_file.write(xr_ds)
            nc_file.close()

    timestamps = [pd.Timestamp(timestamp.strftime("%Y%m%d")) for timestamp in timestamps]

    return filepaths, timestamps


def setup_nc_single_test_data():
    """
    Creates test data as a multi-time and multi-variable NetCDF file.

    Returns
    -------
    str
        NetCDF test data filepath.
    list of datetime
        List of timestamps as datetime objects.
    """

    root_dirpath = os.path.join(dirpath_test(), 'data', 'Sentinel-1_CSAR')

    # create target folders
    dirpath = os.path.join(root_dirpath, 'IWGRDH', 'products', 'datasets', 'resampled', 'T0101', 'EQUI7_EU500M',
                           'E048N012T6', 'data')

    timestamps = [datetime(2016, 1, 1), datetime(2016, 2, 1), datetime(2017, 1, 1), datetime(2017, 2, 1)]

    var_names = ["SIG0", "GMR-"]
    directions = ["A", "D"]
    combs = itertools.product(var_names, directions, timestamps)

    rows, cols = np.meshgrid(np.arange(0, 1600), np.arange(0, 1600))
    data = rows + cols
    equi7 = Equi7Grid(500)
    tile_oi = equi7.EU.tilesys.create_tile(name="E042N012T6")

    xr_dss = []
    filepath = os.path.join(dirpath, "D20160101_20170201_PREPRO---_S1AIWGRDH1VV-_146_T0101_EU500M_E048N012T6.nc")
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

    if not os.path.exists(filepath):
        for comb in combs:
            var_name = comb[0]
            direction = comb[1]
            timestamp = comb[2]

            tags = {'direction': direction}

            data_i = data + timestamps.index(timestamp)
            xr_ar = xr.DataArray(data=data_i[None, :, :], coords={'time': [timestamp]},
                                 dims=['time', 'x', 'y'], attrs=tags)
            xr_dss.append(xr.Dataset(data_vars={var_name.strip('-'): xr_ar}))

        nc_file = NcFile(filepath, mode='w', geotransform=tile_oi.geotransform(),
                         spatialref=tile_oi.get_geotags()['spatialreference'])
        xr_ds = xr.merge(xr_dss)
        nc_file.write(xr_ds)
        nc_file.close()

    timestamps = [pd.Timestamp(timestamp.strftime("%Y%m%d")) for timestamp in timestamps]

    return filepath, timestamps


def roi_test():
    """ Creates a bounding box and a spatial reference object in LonLat. """

    bbox = [(4.36, 43.44), (6.48, 45.80)]
    sref = osr.SpatialReference()
    sref.ImportFromEPSG(4326)
    return bbox, sref
