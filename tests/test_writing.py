import os
import pytest
import numpy as np
import pandas as pd
import xarray as xr
from geospade.raster import MosaicGeometry
from geospade.tools import any_geom2ogr_geom
from yeoda.datacube import DataCubeWriter
from yeoda.datacube import DataCubeReader
from geopathfinder.naming_conventions.yeoda_naming import YeodaFilename


@pytest.fixture
def two_var_tile_ds(tiles):
    ref_tile = tiles[0]
    num_files = 10
    dims = ['time', 'y', 'x']
    coords = {'time': pd.date_range('2000-01-01', periods=num_files),
              'y': ref_tile.y_coords,
              'x': ref_tile.x_coords}
    var1 = np.random.randn(num_files, ref_tile.n_rows, ref_tile.n_cols)
    var2 = np.random.randn(num_files, ref_tile.n_rows, ref_tile.n_cols)
    return xr.Dataset({'VAR1': (dims, var1), 'VAR2': (dims, var2)}, coords=coords)


@pytest.fixture
def geom_bbox_crossing_tiles(sref):
    return any_geom2ogr_geom([4500000, 1500000, 5100000, 2100000], sref=sref)


@pytest.fixture
def two_var_mosaic_ds(tiles, geom_bbox_crossing_tiles):
    mosaic = MosaicGeometry.from_tile_list(tiles)
    mosaic_slcd = mosaic.slice_by_geom(geom_bbox_crossing_tiles)
    x_coords = np.arange(mosaic_slcd.outer_extent[0], mosaic_slcd.outer_extent[2], mosaic_slcd.x_pixel_size)
    y_coords = np.arange(mosaic_slcd.outer_extent[3], mosaic_slcd.outer_extent[1], -mosaic_slcd.y_pixel_size)
    num_files = 10
    dims = ['time', 'y', 'x']
    coords = {'time': pd.date_range('2000-01-01', periods=num_files),
              'y': y_coords,
              'x': x_coords}
    n_rows, n_cols = len(y_coords), len(x_coords)
    var1 = np.random.randn(num_files, n_rows, n_cols)
    var2 = np.random.randn(num_files, n_rows, n_cols)
    return xr.Dataset({'VAR1': (dims, var1), 'VAR2': (dims, var2)}, coords=coords)


def test_gt_multi_bands(tmp_path_factory, two_var_tile_ds, tiles):
    mosaic = MosaicGeometry.from_tile_list(tiles[0:1])
    tmp_dirpath = str(tmp_path_factory.mktemp('gt_multi_bands'))
    dc_writer = DataCubeWriter.from_data(two_var_tile_ds, tmp_dirpath, mosaic=mosaic,
                                         fn_class=YeodaFilename, ext='.tif', def_fields={'var_name': 'MVAR'},
                                         stack_dimension='time', tile_dimension='tile_name',
                                         fn_map={'time': 'datetime_1'})
    dc_writer.export()
    dc_writer.close()

    assert len(os.listdir(tmp_dirpath)) == 10
    ref_timestamp = pd.Timestamp(two_var_tile_ds['time'].data[0]).to_pydatetime()
    ref_ds = two_var_tile_ds.sel({'time': [ref_timestamp]})
    with DataCubeReader(dc_writer.file_register, mosaic,
                        stack_dimension='time', tile_dimension='tile_name') as dc_reader:
        dc_reader.select_by_dimension(lambda t: t == ref_timestamp, name='time', inplace=True)
        dc_reader.read(bands=[1, 2], band_names=['VAR1', 'VAR2'])
        assert np.equal(dc_reader.data_view['VAR1'].data, ref_ds['VAR1'].data).all()
        assert np.equal(dc_reader.data_view['VAR2'].data, ref_ds['VAR2'].data).all()


def test_gt_multi_vars(tmp_path_factory, two_var_tile_ds, tiles):
    mosaic = MosaicGeometry.from_tile_list(tiles[0:1])
    tmp_dirpath = str(tmp_path_factory.mktemp('gt_multi_vars'))
    dc_writer_var1 = DataCubeWriter.from_data(two_var_tile_ds[['VAR1']], tmp_dirpath, mosaic=mosaic,
                                              fn_class=YeodaFilename, ext='.tif', def_fields={'var_name': 'VAR1'},
                                              stack_dimension='time', tile_dimension='tile_name',
                                              fn_map={'time': 'datetime_1'})
    dc_writer_var1.export()
    dc_writer_var1.close()
    dc_writer_var2 = DataCubeWriter.from_data(two_var_tile_ds[['VAR2']], tmp_dirpath, mosaic=mosaic,
                                              fn_class=YeodaFilename, ext='.tif', def_fields={'var_name': 'VAR2'},
                                              stack_dimension='time', tile_dimension='tile_name',
                                              fn_map={'time': 'datetime_1'})
    dc_writer_var2.export()
    dc_writer_var2.close()

    dc_writer = dc_writer_var1.unite(dc_writer_var2)

    assert len(os.listdir(tmp_dirpath)) == 20
    ref_timestamp = pd.Timestamp(two_var_tile_ds['time'].data[0]).to_pydatetime()
    ref_ds = two_var_tile_ds.sel({'time': [ref_timestamp]})
    dimensions = ['time', 'tile_name', 'var_name']
    with DataCubeReader.from_filepaths(dc_writer.filepaths, fn_class=YeodaFilename, mosaic=mosaic,
                                       dimensions=dimensions, stack_dimension='time', tile_dimension='tile_name') \
            as dc_reader:
        dc_reader.select_by_dimension(lambda t: t == ref_timestamp, name='time', inplace=True)
        dc_reader_var1 = dc_reader.select_by_dimension(lambda v: v == 'VAR1', name='var_name')
        dc_reader_var1.read(bands=1, band_names='VAR1')
        dc_reader_var2 = dc_reader.select_by_dimension(lambda v: v == 'VAR2', name='var_name')
        dc_reader_var2.read(bands=1, band_names='VAR2')
        assert np.equal(dc_reader_var1.data_view['VAR1'].data, ref_ds['VAR1'].data).all()
        assert np.equal(dc_reader_var2.data_view['VAR2'].data, ref_ds['VAR2'].data).all()


def test_multi_nc(tmp_path_factory, two_var_tile_ds, tiles):
    mosaic = MosaicGeometry.from_tile_list(tiles[0:1])
    tmp_dirpath = str(tmp_path_factory.mktemp('nc_multi'))
    dc_writer = DataCubeWriter.from_data(two_var_tile_ds, tmp_dirpath, mosaic=mosaic,
                                         fn_class=YeodaFilename, ext='.nc', def_fields={'var_name': 'MVAR'},
                                         stack_dimension='time', tile_dimension='tile_name',
                                         fn_map={'time': 'datetime_1'})
    dc_writer.export()
    dc_writer.close()

    assert len(os.listdir(tmp_dirpath)) == 10
    ref_timestamp = pd.Timestamp(two_var_tile_ds['time'].data[0]).to_pydatetime()
    ref_ds = two_var_tile_ds.sel({'time': [ref_timestamp]})
    with DataCubeReader(dc_writer.file_register, mosaic,
                        stack_dimension='time', tile_dimension='tile_name') as dc_reader:
        dc_reader.select_by_dimension(lambda t: t == ref_timestamp, name='time', inplace=True)
        dc_reader.read()
        assert np.equal(dc_reader.data_view['VAR1'].data, ref_ds['VAR1'].data).all()
        assert np.equal(dc_reader.data_view['VAR2'].data, ref_ds['VAR2'].data).all()


def test_single_nc(tmp_path_factory, two_var_tile_ds, tiles):
    mosaic = MosaicGeometry.from_tile_list(tiles[0:1])
    tmp_dirpath = str(tmp_path_factory.mktemp('nc_single'))
    min_time, max_time = min(two_var_tile_ds['time'].data), max(two_var_tile_ds['time'].data)
    stack_groups = {ts: 0 for ts in two_var_tile_ds['time'].data}
    dc_writer = DataCubeWriter.from_data(two_var_tile_ds, tmp_dirpath, mosaic=mosaic,
                                         fn_class=YeodaFilename, ext='.nc', def_fields={'var_name': 'MVAR'},
                                         stack_dimension='time', tile_dimension='tile_name',
                                         fn_map={'time': 'datetime_1'},
                                         stack_groups=stack_groups,
                                         fn_groups_map={0: {'datetime_1': min_time, 'datetime_2': max_time}})
    dc_writer.export()
    dc_writer.close()

    assert len(os.listdir(tmp_dirpath)) == 1
    with DataCubeReader.from_filepaths(dc_writer.filepaths, fn_class=YeodaFilename, mosaic=mosaic,
                                       stack_dimension='group_id', tile_dimension='tile_name') as dc_reader:
        dc_reader.read()
        assert np.equal(dc_reader.data_view['VAR1'].data, two_var_tile_ds['VAR1'].data).all()
        assert np.equal(dc_reader.data_view['VAR2'].data, two_var_tile_ds['VAR2'].data).all()


def test_single_nc_tiling(tmp_path_factory, two_var_mosaic_ds, tiles, geom_bbox_crossing_tiles):
    mosaic = MosaicGeometry.from_tile_list(tiles)
    tmp_dirpath = str(tmp_path_factory.mktemp('nc_single_tiling'))
    min_time, max_time = min(two_var_mosaic_ds['time'].data), max(two_var_mosaic_ds['time'].data)
    stack_groups = {ts: 0 for ts in two_var_mosaic_ds['time'].data}
    dc_writer = DataCubeWriter.from_data(two_var_mosaic_ds, tmp_dirpath, mosaic=mosaic,
                                         fn_class=YeodaFilename, ext='.nc', def_fields={'var_name': 'MVAR'},
                                         stack_dimension='time', tile_dimension='tile_name',
                                         fn_map={'time': 'datetime_1'},
                                         stack_groups=stack_groups,
                                         fn_groups_map={0: {'datetime_1': min_time, 'datetime_2': max_time}})
    dc_writer.export(use_mosaic=True)
    dc_writer.close()

    assert len(os.listdir(tmp_dirpath)) == len(mosaic.tiles)
    with DataCubeReader.from_filepaths(dc_writer.filepaths, fn_class=YeodaFilename, mosaic=mosaic,
                                       stack_dimension='group_id', tile_dimension='tile_name') as dc_reader:
        dc_reader.select_polygon(geom_bbox_crossing_tiles, inplace=True)
        dc_reader.read()
        assert np.equal(dc_reader.data_view['VAR1'].data, two_var_mosaic_ds['VAR1'].data).all()
        assert np.equal(dc_reader.data_view['VAR2'].data, two_var_mosaic_ds['VAR2'].data).all()
