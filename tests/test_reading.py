import pytest
import numpy as np
from geospade.crs import SpatialRef
from yeoda.datacube import DataCubeReader
from geopathfinder.naming_conventions.yeoda_naming import YeodaFilename


@pytest.fixture
def gt_reader(gt_filepaths):
    dimensions = ['time', 'var_name', 'band', 'tile_name']
    dc_reader = DataCubeReader.from_filepaths(gt_filepaths, fn_class=YeodaFilename, dimensions=dimensions,
                                              stack_dimension='time', tile_dimension='tile_name')
    return dc_reader


@pytest.fixture
def nc_reader(nc_filepaths):
    dimensions = ['time', 'var_name', 'band', 'tile_name']
    dc_reader = DataCubeReader.from_filepaths(nc_filepaths, fn_class=YeodaFilename, dimensions=dimensions,
                                              stack_dimension='time', tile_dimension='tile_name')
    return dc_reader


@pytest.fixture
def latlon_poi():
    return 44, 5


@pytest.fixture
def rc_poi():
    return 970, 246


@pytest.fixture
def bbox_crossing_tiles():
    return [4799500, 1799500, 4800500, 1800500]


def test_xy_nc(nc_reader, latlon_poi, rc_poi):
    nc_reader.select_by_dimension(lambda v: v == 'VAR1', name='var_name', inplace=True)
    nc_reader.select_by_dimension(lambda b: b == 'VV', name='band', inplace=True)
    nc_reader.sort_by_dimension(name='time', inplace=True)
    nc_reader.select_xy(*latlon_poi, sref=SpatialRef(4326), inplace=True)
    nc_reader.read()
    assert (nc_reader.data_view['VAR1_VV'].data.flatten() == np.arange(1, 5) + sum(rc_poi)).all()


def test_xy_gt(gt_reader, latlon_poi, rc_poi):
    gt_reader.select_by_dimension(lambda v: v == 'VAR1', name='var_name', inplace=True)
    gt_reader.select_by_dimension(lambda b: b == 'VH', name='band', inplace=True)
    gt_reader.sort_by_dimension(name='time', inplace=True)
    gt_reader.select_xy(*latlon_poi, sref=SpatialRef(4326), inplace=True)
    gt_reader.read(bands=1, band_names='VAR1_VH')
    assert (gt_reader.data_view['VAR1_VH'].data.flatten() == np.arange(1, 5) + sum(rc_poi)).all()


def test_bbox_crossing_tiles_gt(gt_reader, bbox_crossing_tiles):
    gt_reader.select_by_dimension(lambda v: v == 'VAR2', name='var_name', inplace=True)
    gt_reader.select_by_dimension(lambda b: b == 'VV', name='band', inplace=True)
    gt_reader.sort_by_dimension(name='time', inplace=True)
    gt_reader.select_bbox(bbox_crossing_tiles, inplace=True)
    gt_reader.read(bands=1, band_names='VAR2_VV')
    ref_data = np.array([[2398, 1199], [1199, 0]])
    ref_data = np.repeat(ref_data[None, ...], repeats=4, axis=0)
    assert (gt_reader.data_view['VAR2_VV'].data == np.arange(1, 5)[:, None, None] + ref_data).all()


if __name__ == '__main__':
    pass
