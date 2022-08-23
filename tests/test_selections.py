import pytest
from geospade.crs import SpatialRef
from yeoda.datacube import DataCubeReader
from geopathfinder.naming_conventions.yeoda_naming import YeodaFilename


@pytest.fixture
def dc_reader(gt_filepaths):
    dimensions = ['time', 'var_name', 'band', 'tile_name']
    dc_reader = DataCubeReader.from_filepaths(gt_filepaths, fn_class=YeodaFilename, dimensions=dimensions,
                                              stack_dimension='time', tile_dimension='tile_name')
    return dc_reader


@pytest.fixture
def bbox():
    return [(43.44, 4.36), (45.80, 6.48)]


def test_select_bands_inplace(dc_reader):
    assert len(set(dc_reader['band'])) == 2
    dc_reader.select_by_dimension(lambda x: x == "VV", name="band", inplace=True)
    assert len(set(dc_reader['band'])) == 1


def test_select_pols_not_inplace(dc_reader):
    dc_vv = dc_reader.select_by_dimension(lambda x: x == "VV", name="band")
    dc_vh = dc_reader.select_by_dimension(lambda x: x == "VH", name="band")
    assert len(set(dc_vv['band'])) == 1
    assert list(set(dc_vv['band']))[0] == "VV"
    assert len(set(dc_vh['band'])) == 1
    assert list(set(dc_vh['band']))[0] == "VH"


def test_select_pols_clone(dc_reader):
    dc_clone = dc_reader.clone()
    dc_reader.select_by_dimension(lambda x: x == "VV", name="band", inplace=True)
    dc_clone.select_by_dimension(lambda x: x == "VH", name="band", inplace=True)
    assert len(set(dc_reader['band'])) == 1
    assert list(set(dc_reader['band']))[0] == "VV"
    assert len(set(dc_clone['band'])) == 1
    assert list(set(dc_clone['band']))[0] == "VH"


def test_select_time(dc_reader, timestamps):
    start_time, end_time = timestamps[0], timestamps[1]
    dc_reader.select_by_dimension(lambda t: (t >= start_time) & (t <= end_time), inplace=True)
    assert sorted(list(set(dc_reader['time']))) == timestamps[:2]


def test_select_var_names_vs_select(dc_reader):
    dc_var1_sel = dc_reader.select_by_dimension(lambda x: x == "VAR1", name="var_name")
    dc_var2_sel = dc_reader.select_by_dimension(lambda x: x == "VAR2", name="var_name")
    dc_var1_ptrn = dc_reader.select_files_with_pattern(".*VAR1.*")
    dc_var2_ptrn = dc_reader.select_files_with_pattern(".*VAR2.*")
    assert sorted(list(dc_var1_sel['filepath'])) == sorted(list(dc_var1_ptrn['filepath']))
    assert sorted(list(dc_var2_sel['filepath'])) == sorted(list(dc_var2_ptrn['filepath']))


def test_select_tiles(dc_reader):
    assert len(set(dc_reader['tile_name'])) == 4
    dc_reader.select_tiles(["E042N012T6"], inplace=True)
    assert len(set(dc_reader['tile_name'])) == 1
    assert dc_reader.mosaic.tile_names == ["E042N012T6"]


def test_select_bbox(dc_reader, bbox):
    assert len(set(dc_reader['tile_name'])) == 4
    dc_reader.select_bbox(bbox, sref=SpatialRef(4326), inplace=True)
    assert len(set(dc_reader['tile_name'])) == 1
    assert [tile.parent.name for tile in dc_reader.mosaic.tiles] == ["E042N012T6"]

