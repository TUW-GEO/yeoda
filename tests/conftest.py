import glob
import os
import sys
import itertools
import xarray as xr
import numpy as np
from pathlib import Path
from datetime import datetime
from geospade.crs import SpatialRef
from geospade.raster import Tile
from geospade.raster import MosaicGeometry
from geopathfinder.naming_conventions.yeoda_naming import YeodaFilename
from veranda.raster.native.geotiff import GeoTiffFile
from veranda.raster.native.netcdf import NetCdf4File
import pytest


@pytest.fixture(scope="session")
def timestamps():
    return [datetime(2016, 1, 1), datetime(2016, 2, 1), datetime(2017, 1, 1), datetime(2017, 2, 1)]


@pytest.fixture(scope="session")
def sref():
    sref_wkt = 'PROJCS["Azimuthal_Equidistant",GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",' \
               'SPHEROID["WGS_1984",6378137.0,298.257223563]],PRIMEM["Greenwich",0.0],' \
               'UNIT["Degree",0.017453292519943295]],PROJECTION["Azimuthal_Equidistant"],' \
               'PARAMETER["false_easting",5837287.81977],PARAMETER["false_northing",2121415.69617],' \
               'PARAMETER["central_meridian",24.0],PARAMETER["latitude_of_origin",53.0],UNIT["Meter",1.0]]"'
    return SpatialRef(sref_wkt)


@pytest.fixture(scope="session")
def tiles(sref):
    n_rows, n_cols = 1200, 1200
    return [Tile(n_rows, n_cols, sref, geotrans=(42e5, 500, 0, 24e5, 0, -500), name="E042N018T6"),
            Tile(n_rows, n_cols, sref, geotrans=(48e5, 500, 0, 24e5, 0, -500), name="E048N018T6"),
            Tile(n_rows, n_cols, sref, geotrans=(48e5, 500, 0, 18e5, 0, -500), name="E048N012T6"),
            Tile(n_rows, n_cols, sref, geotrans=(42e5, 500, 0, 18e5, 0, -500), name="E042N012T6")]


@pytest.fixture(scope="session")
def gt_filepaths(tmp_path_factory, timestamps, tiles):
    tmp_path = str(tmp_path_factory.mktemp('gt_filepaths'))
    pols = ["VV", "VH"]
    var_names = ["VAR1", "VAR2"]
    ref_tile = tiles[0]
    n_rows, n_cols = ref_tile.shape
    tile_names = [tile.name for tile in tiles]
    combs = itertools.product(timestamps, pols, var_names, tile_names)
    rows, cols = np.meshgrid(np.arange(0, n_rows), np.arange(0, n_cols))
    data = (rows + cols).astype(float)

    filepaths = []
    for comb in combs:
        fields = {'var_name': comb[2],
                  'datetime_1': comb[0],
                  'band': comb[1],
                  'tile_name': comb[3]}
        filename = str(YeodaFilename(fields))
        filepath = os.path.join(tmp_path, filename)
        tile = tiles[tile_names.index(comb[3])]
        with GeoTiffFile(filepath, mode='w', geotrans=tile.geotrans, sref_wkt=tile.sref.wkt,
                         raster_shape=(n_rows, n_cols), dtypes='uint16', nodatavals=32000) as gt_file:
            gt_file.write({1: data + timestamps.index(comb[0]) + 1})
        filepaths.append(filepath)

    return filepaths


@pytest.fixture(scope="session")
def nc_filepaths(tmp_path_factory, timestamps, tiles):
    tmp_path = str(tmp_path_factory.mktemp('nc_filepaths'))
    pols = ["VV", "VH"]
    var_names = ["VAR1", "VAR2"]
    ref_tile = tiles[0]
    n_rows, n_cols = ref_tile.shape
    tile_names = [tile.name for tile in tiles]
    combs = itertools.product(timestamps, pols, var_names, tile_names)
    rows, cols = np.meshgrid(np.arange(0, n_rows), np.arange(0, n_cols))
    data = (rows + cols).astype(float)

    filepaths = []
    for comb in combs:
        fields = {'var_name': comb[2],
                  'datetime_1': comb[0],
                  'band': comb[1],
                  'tile_name': comb[3]}
        filename = str(YeodaFilename(fields, ext='.nc'))
        filepath = os.path.join(tmp_path, filename)
        tile = tiles[tile_names.index(comb[3])]
        dar = xr.DataArray(data[None, ...] + timestamps.index(comb[0]) + 1,
                           coords={'time': list(comb[0:1]), 'y': tile.y_coords, 'x': tile.x_coords},
                           dims=['time', 'y', 'x'])
        ds = xr.Dataset({f"{comb[2]}_{comb[1]}": dar})
        with NetCdf4File(filepath, mode='w', geotrans=tile.geotrans, sref_wkt=tile.sref.wkt,
                         attrs={'time': {'units': 'days since 2000-01-01 00:00:00'}}) as nc_file:
            nc_file.write(ds)
        filepaths.append(filepath)

    return filepaths

