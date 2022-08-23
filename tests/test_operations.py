import pytest
import numpy as np
from yeoda.datacube import DataCubeReader
from geopathfinder.file_naming import SmartFilename
from geopathfinder.naming_conventions.yeoda_naming import YeodaFilename


@pytest.fixture
def dc_reader(gt_filepaths):
    dimensions = ['time', 'var_name', 'band', 'tile_name']
    dc_reader = DataCubeReader.from_filepaths(gt_filepaths, YeodaFilename, dimensions=dimensions,
                                              stack_dimension='time', tile_dimension='tile_name')
    return dc_reader


class SmartFilenameFail(SmartFilename):
    @classmethod
    def from_filename(self, filename, **kwargs):
        raise Exception


def test_unknown_filename(gt_filepaths):
    dimensions = ['time', 'var_name', 'band', 'tile_name']
    dc_reader = DataCubeReader.from_filepaths(gt_filepaths, SmartFilenameFail, dimensions=dimensions,
                                              stack_dimension='time', tile_dimension='tile_name')

    assert len(dc_reader.dimensions) == 2
    assert len(dc_reader) == len(gt_filepaths)


def test_rename_dimension(dc_reader):
    dc_reader.rename_dimensions({'band': 'pol'}, inplace=True)
    assert 'pol' in dc_reader.dimensions
    assert 'band' not in dc_reader.dimensions
    assert len(set(dc_reader['pol'])) == 2


def test_add_dimension(dc_reader):
    dim_values = np.random.rand(len(dc_reader))
    dc_reader.add_dimension("value", dim_values, inplace=True)
    assert "value" in dc_reader.dimensions
    assert list(dc_reader['value']) == dim_values.tolist()


def test_sort_by_dimension(dc_reader):
    timestamps = list(dc_reader['time'])
    dim_values = np.random.rand(len(dc_reader))
    timestamps_sorted = np.array(timestamps)[np.argsort(dim_values)].tolist()

    dc_reader.add_dimension("value", dim_values, inplace=True)
    dc_reader.sort_by_dimension("value", inplace=True)
    assert list(dc_reader['time']) == timestamps_sorted


def test_split_time(dc_reader, timestamps):
    time_interval_1 = (timestamps[0], timestamps[1])
    time_interval_2 = (timestamps[2], timestamps[-1])
    expressions = [lambda t: (t >= time_interval_1[0]) & (t <= time_interval_1[1]),
                   lambda t: (t >= time_interval_2[0]) & (t <= time_interval_2[1])]

    dcs = dc_reader.split_by_dimension(expressions)
    assert len(dcs) == 2
    assert sorted(list(set(dcs[0]['time']))) == timestamps[:2]
    assert sorted(list(set(dcs[1]['time']))) == timestamps[2:]


def test_split_yearly(dc_reader):
    yearly_dcs = dc_reader.split_by_temporal_freq('Y')
    assert len(yearly_dcs) == 2
    dc_2016 = yearly_dcs[0]
    years = [timestamp.year for timestamp in dc_2016['time']]
    assert (np.array(years) == 2016).all()


def test_split_monthly(dc_reader):
    monthly_dcs = dc_reader.split_by_temporal_freq('M')
    assert len(monthly_dcs) == 4


def test_unite(gt_filepaths):
    n = len(gt_filepaths)
    dimensions_1 = ['time', 'band', 'tile_name']
    dc_reader_1 = DataCubeReader.from_filepaths(gt_filepaths[:int(n/2)], YeodaFilename, dimensions=dimensions_1,
                                                stack_dimension='time', tile_dimension='tile_name')
    dimensions_2 = ['time', 'var_name', 'tile_name']
    dc_reader_2 = DataCubeReader.from_filepaths(gt_filepaths[int(n/2):], YeodaFilename, dimensions=dimensions_2,
                                                stack_dimension='time', tile_dimension='tile_name')

    dc_reader_united = dc_reader_1.unite(dc_reader_2)
    assert 'band' in dc_reader_united.dimensions
    assert 'var_name' in dc_reader_united.dimensions
    assert len(dc_reader_united) == (len(dc_reader_1) + len(dc_reader_2))


def test_intersect_empty(gt_filepaths):
    dimensions_1 = ['time', 'band', 'tile_name']
    dc_reader_1 = DataCubeReader.from_filepaths(gt_filepaths[0:1], YeodaFilename, dimensions=dimensions_1,
                                                stack_dimension='time', tile_dimension='tile_name')
    dimensions_2 = ['time', 'var_name', 'tile_name']
    dc_reader_2 = DataCubeReader.from_filepaths(gt_filepaths[1:2], YeodaFilename, dimensions=dimensions_2,
                                                stack_dimension='time', tile_dimension='tile_name')
    dc_reader_intsct = dc_reader_1.intersect(dc_reader_2, on_dimension='tile_name')
    assert len(dc_reader_intsct) == 0


def test_intersect_non_empty(gt_filepaths):
    dimensions_1 = ['time', 'band', 'tile_name']
    dc_reader_1 = DataCubeReader.from_filepaths(gt_filepaths[0:1], YeodaFilename, dimensions=dimensions_1,
                                                stack_dimension='time', tile_dimension='tile_name')
    dimensions_2 = ['time', 'var_name', 'tile_name']
    dc_reader_2 = DataCubeReader.from_filepaths(gt_filepaths[1:2], YeodaFilename, dimensions=dimensions_2,
                                                stack_dimension='time', tile_dimension='tile_name')
    dc_reader_intsct = dc_reader_1.intersect(dc_reader_2, on_dimension='time')
    assert len(dc_reader_intsct) == 2


def test_intersect_dimensions(gt_filepaths):
    dimensions_1 = ['time', 'band', 'tile_name']
    dc_reader_1 = DataCubeReader.from_filepaths(gt_filepaths[0:1], YeodaFilename, dimensions=dimensions_1,
                                                stack_dimension='time', tile_dimension='tile_name')
    dimensions_2 = ['time', 'var_name', 'tile_name']
    dc_reader_2 = DataCubeReader.from_filepaths(gt_filepaths[1:2], YeodaFilename, dimensions=dimensions_2,
                                                stack_dimension='time', tile_dimension='tile_name')
    dc_reader_intsct = dc_reader_1.intersect(dc_reader_2)
    assert len(dc_reader_intsct) == 2
    assert 'var_name' not in dc_reader_intsct.dimensions
    assert 'band' not in dc_reader_intsct.dimensions
    assert 'time' in dc_reader_intsct.dimensions


def test_align_dimension_intersect_shrink(gt_filepaths):
    dimensions = ['time', 'tile_name']
    dc_reader_1 = DataCubeReader.from_filepaths(gt_filepaths, YeodaFilename, dimensions=dimensions,
                                                stack_dimension='time', tile_dimension='tile_name')
    dc_reader_2 = dc_reader_1.clone()
    timestamps = list(set(dc_reader_1['time']))
    timestamps_sel = timestamps[2:]
    dc_reader_2.select_by_dimension(lambda t: t.isin(timestamps_sel), name='time', inplace=True)

    dc_reader_aligned = dc_reader_1.align_dimension(dc_reader_2, name='time')
    dc_reader_intersected = dc_reader_1.intersect(dc_reader_2, on_dimension='time')

    assert sorted(list(dc_reader_aligned['time'])) == sorted(list(dc_reader_intersected['time']))


def test_align_dimension_shrink(gt_filepaths):
    dimensions = ['time', 'tile_name']
    dc_reader_1 = DataCubeReader.from_filepaths(gt_filepaths, YeodaFilename, dimensions=dimensions,
                                                stack_dimension='time', tile_dimension='tile_name')
    dc_reader_2 = dc_reader_1.clone()
    timestamps = sorted(list(set(dc_reader_1['time'])))
    timestamps_sel = timestamps[2:]
    dc_reader_2.select_by_dimension(lambda t: t.isin(timestamps_sel), name='time', inplace=True)

    dc_reader_1.align_dimension(dc_reader_2, name='time', inplace=True)
    assert sorted(list(set(dc_reader_1['time']))) == timestamps_sel


def test_align_dimension_grow(gt_filepaths):
    dimensions = ['time', 'tile_name']
    dc_reader_1 = DataCubeReader.from_filepaths(gt_filepaths, YeodaFilename, dimensions=dimensions,
                                                stack_dimension='time', tile_dimension='tile_name')
    dc_reader_2 = dc_reader_1.clone()
    timestamps = sorted(list(set(dc_reader_1['time'])))
    timestamps_sel = timestamps[:1]
    dc_reader_2.select_by_dimension(lambda t: t.isin(timestamps_sel), name='time', inplace=True)
    tile_names = sorted(list(set(dc_reader_1['tile_name'])))
    tile_names_sel = tile_names[:1]
    dc_reader_2.select_by_dimension(lambda t: t.isin(tile_names_sel), name='tile_name', inplace=True)

    dc_reader_2.align_dimension(dc_reader_1, name='time', inplace=True)
    dc_reader_1.select_by_dimension(lambda t: t.isin(timestamps_sel), name='time', inplace=True)
    assert len(dc_reader_1) == len(dc_reader_2)
