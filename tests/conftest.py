import glob
import os
import sys
import itertools
import numpy as np
from pathlib import Path
from datetime import datetime
from geospade.crs import SpatialRef
from geospade.raster import Tile
from geospade.raster import MosaicGeometry
from geopathfinder.naming_conventions.yeoda_naming import YeodaFilename
from veranda.raster.native.geotiff import GeoTiffFile
import pytest


@pytest.fixture(scope="session")
def gt_timestamps():
    return [datetime(2016, 1, 1), datetime(2016, 2, 1), datetime(2017, 1, 1), datetime(2017, 2, 1)]


@pytest.fixture(scope="session")
def gt_filepaths(tmp_path_factory, gt_timestamps):
    tmp_path = str(tmp_path_factory.mktemp('gt_filepaths'))
    pols = ["VV", "VH"]
    var_names = ["VAR1", "VAR2"]
    n_rows, n_cols = 1200, 1200
    sref_wkt = 'PROJCS["Azimuthal_Equidistant",GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",' \
               'SPHEROID["WGS_1984",6378137.0,298.257223563]],PRIMEM["Greenwich",0.0],' \
               'UNIT["Degree",0.017453292519943295]],PROJECTION["Azimuthal_Equidistant"],' \
               'PARAMETER["false_easting",5837287.81977],PARAMETER["false_northing",2121415.69617],' \
               'PARAMETER["central_meridian",24.0],PARAMETER["latitude_of_origin",53.0],UNIT["Meter",1.0]]"'
    sref = SpatialRef(sref_wkt)
    tiles = [Tile(n_rows, n_cols, sref, geotrans=(42e5, 500, 0, 24e5, 0, -500), name="E042N018T6"),
             Tile(n_rows, n_cols, sref, geotrans=(48e5, 500, 0, 24e5, 0, -500), name="E048N018T6"),
             Tile(n_rows, n_cols, sref, geotrans=(48e5, 500, 0, 18e5, 0, -500), name="E048N012T6"),
             Tile(n_rows, n_cols, sref, geotrans=(42e5, 500, 0, 18e5, 0, -500), name="E042N012T6")]
    tile_names = [tile.name for tile in tiles]
    combs = itertools.product(gt_timestamps, pols, var_names, tile_names)
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
            gt_file.write({1: data})
        filepaths.append(filepath)

    return filepaths

