import pytest
import numpy as np
from geospade.crs import SpatialRef
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


def test_split_time(dc_reader, gt_timestamps):
    time_interval_1 = (gt_timestamps[0], gt_timestamps[1])
    time_interval_2 = (gt_timestamps[2], gt_timestamps[-1])
    expressions = [lambda t: (t >= time_interval_1[0]) & (t <= time_interval_1[1]),
                   lambda t: (t >= time_interval_2[0]) & (t <= time_interval_2[1])]

    dcs = dc_reader.split_by_dimension(expressions)
    assert len(dcs) == 2
    assert sorted(list(set(dcs[0]['time']))) == gt_timestamps[:2]
    assert sorted(list(set(dcs[1]['time']))) == gt_timestamps[2:]


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
    dimensions_1 = ['time', 'band', 'tile_name']
    dc_reader_1 = DataCubeReader.from_filepaths(gt_filepaths, YeodaFilename, dimensions=dimensions_1,
                                                stack_dimension='time', tile_dimension='tile_name')
    dimensions_2 = ['time', 'var_name', 'tile_name']
    dc_reader_2 = DataCubeReader.from_filepaths(gt_filepaths, YeodaFilename, dimensions=dimensions_2,
                                                stack_dimension='time', tile_dimension='tile_name')

    dc_reader_united = dc_reader_1.unite(dc_reader_2)
    assert 'pol' in dc_reader_united.dimensions
    assert 'orbit_direction' in dc_reader_united.dimensions
    assert len(dc_reader_united) == len(dc_reader_1)
    #
    # def test_intersect_empty(self):
    #     """
    #     Tests data cube intersection on the temporal dimension, i.e. if all data from a second datacube is properly
    #     intersected with the data of to the original data cube according to matching timestamps. The result should be
    #     empty due to non-overlapping timestamps.
    #     """
    #
    #     # empty data cube when an intersection is applied
    #     dc_1 = EODataCube(filepaths=self.gt_filepaths, filename_class=SgrtFilename,
    #                       dimensions=['time', 'pol'])
    #     dc_2 = EODataCube(filepaths=self.gt_filepaths, filename_class=SgrtFilename,
    #                       dimensions=['time', 'orbit_direction'])
    #     dc_1.inventory = dc_1.inventory[dc_1['time'] == self.timestamps[0]]
    #     dc_2.inventory = dc_2.inventory[dc_2['time'] == self.timestamps[1]]
    #     dc_intersected = dc_1.intersect(dc_2, on_dimension='time')
    #     assert len(dc_intersected) == 0
    #
    # def test_intersect_dimensions(self):
    #     """
    #     Tests simple data cube intersection, i.e. if all data from a second datacube is properly
    #     intersected with the data of to the original data cube.
    #     """
    #
    #     dc_1 = EODataCube(filepaths=self.gt_filepaths, filename_class=SgrtFilename,
    #                       dimensions=['time', 'pol'])
    #     dc_2 = EODataCube(filepaths=self.gt_filepaths, filename_class=SgrtFilename,
    #                       dimensions=['time', 'orbit_direction'])
    #
    #     dc_intersected = dc_1.intersect(dc_2)
    #     assert len(dc_intersected) == len(self.gt_filepaths)
    #     assert 'pol' not in dc_intersected.dimensions
    #     assert 'orbit_direction' not in dc_intersected.dimensions
    #     assert 'time' in dc_intersected.dimensions
    #
    # def test_intersect_align_dimension_shrink(self):
    #     """
    #     Tests matching of entries with two different methods, which should yield the same result: data cube
    #     intersection and data cube alignment on the temporal dimension.
    #     """
    #
    #     dc_1 = EODataCube(filepaths=self.gt_filepaths, filename_class=SgrtFilename,
    #                       dimensions=['time'])
    #     dc_2 = dc_1.clone()
    #     dc_2.inventory = dc_2.inventory[dc_2['time'] != self.timestamps[0]]
    #     dc_2.inventory = dc_2.inventory[dc_2['time'] != self.timestamps[2]]
    #
    #     dc_aligned = dc_1.align_dimension(dc_2, name='time', inplace=False)
    #     dc_intersected = dc_1.intersect(dc_2, on_dimension='time', inplace=False)
    #
    #     assert sorted(list(dc_aligned['time'])) == sorted(list(dc_intersected['time']))
    #
    # def test_align_dimension_shrink(self):
    #     """
    #     Tests alignment of a data cube with another data cube along the temporal dimension. Since the second
    #     data cube contains less data, the original data cube will also contain less data, i.e. the same timestamps as
    #     in the other data cube.
    #     """
    #
    #     dc_1 = EODataCube(filepaths=self.gt_filepaths, filename_class=SgrtFilename,
    #                       dimensions=['time'])
    #     dc_2 = dc_1.clone()
    #     dc_2.inventory = dc_2.inventory[dc_2['time'] != self.timestamps[0]]
    #     dc_2.inventory = dc_2.inventory[dc_2['time'] != self.timestamps[2]]
    #
    #     dc_1.align_dimension(dc_2, name='time', inplace=True)
    #     assert sorted(list(set(dc_1['time']))) == [self.timestamps[1], self.timestamps[3]]
    #
    # def test_align_dimension_grow(self):
    #     """
    #     Tests alignment of a data cube with another data cube along the temporal dimension. Since the second
    #     data cube contains more data, the original data cube will also contain more data, i.e. the same timestamps as
    #     in the other data cube by duplicating the entries.
    #     """
    #
    #     dc_1 = EODataCube(filepaths=self.gt_filepaths, filename_class=SgrtFilename,
    #                       dimensions=['time'])
    #     dc_2 = dc_1.clone()
    #     timestamps = list(dc_1['time'])
    #     subset_idxs = [timestamps.index(self.timestamps[0]),
    #                    timestamps.index(self.timestamps[1]),
    #                    timestamps.index(self.timestamps[2]),
    #                    timestamps.index(self.timestamps[3])]
    #     dc_1.inventory = dc_1.inventory.iloc[subset_idxs]
    #
    #     dc_1.align_dimension(dc_2, name='time', inplace=True)
    #     assert (dc_1['time'] == dc_2['time']).all()