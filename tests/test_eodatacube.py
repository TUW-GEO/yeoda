# general imports
import os
import shutil
import unittest
import numpy as np

from tests.setup_test_data import setup_test_data, dirpath_test

from geopathfinder.naming_conventions.sgrt_naming import create_sgrt_filename

# import yeoda
from yeoda.yeoda import EODataCube


class EODataCubeTester(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        setup_test_data()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(os.path.join(dirpath_test(), 'data'))

    def setUp(self):
        dir_tree, timestamps = setup_test_data(initialise=False)
        self.dir_tree = dir_tree
        self.timestamps = timestamps

    def test_rename_dimension(self):
        dc = EODataCube(dir_tree=self.dir_tree, smart_filename_creator=create_sgrt_filename,
                        dimensions=['time', 'var_name', 'pol'])
        dc.rename_dimensions({'pol': 'band'}, in_place=True)
        assert 'band' in dc.inventory.columns
        assert 'pol' not in dc.inventory.columns
        assert len(set(dc['band'])) == 2

    def test_add_dimension(self):
        dc = EODataCube(dir_tree=self.dir_tree, smart_filename_creator=create_sgrt_filename,
                        dimensions=['time', 'var_name', 'pol'])
        dim_values = np.random.rand(len(dc))
        dc.add_dimension("value", dim_values, in_place=True)
        assert "value" in dc.inventory.columns
        assert list(dc['value']) == dim_values.tolist()

    def test_sort_by_dimension(self):
        dc = EODataCube(dir_tree=self.dir_tree, smart_filename_creator=create_sgrt_filename,
                        dimensions=['time', 'var_name', 'pol'])
        timestamps = list(dc['time'])
        dim_values = np.random.rand(len(dc))
        timestamps_sorted = np.array(timestamps)[np.argsort(dim_values)].tolist()

        dc.add_dimension("value", dim_values, in_place=True)
        dc.sort_by_dimension("value", in_place=True)
        assert list(dc['time']) == timestamps_sorted

    def test_split_time(self):
        dc = EODataCube(dir_tree=self.dir_tree, smart_filename_creator=create_sgrt_filename,
                        dimensions=['time', 'var_name', 'pol'])
        first_time_interval = (self.timestamps[0], self.timestamps[1])
        second_time_interval = (self.timestamps[2], self.timestamps[-1])
        expression = (">=", "<=")
        dcs = dc.split_by_dimension([first_time_interval, second_time_interval], expressions=[expression, expression])
        assert len(dcs) == 2
        assert sorted(list(set(dcs[0]['time']))) == self.timestamps[:2]
        assert sorted(list(set(dcs[1]['time']))) == self.timestamps[2:]

    def test_split_yearly(self):
        dc = EODataCube(dir_tree=self.dir_tree, smart_filename_creator=create_sgrt_filename,
                        dimensions=['time', 'var_name', 'pol'])
        yearly_dcs = dc.split_yearly(name='time')
        assert len(yearly_dcs) == 2
        dcs_2016 = dc.split_yearly(name='time', years=2016)
        assert len(dcs_2016) == 1
        dc_2016 = dcs_2016[0]
        years = [timestamp.year for timestamp in dc_2016['time']]
        assert (np.array(years) == 2016).all()

    def test_split_monthly(self):
        dc = EODataCube(dir_tree=self.dir_tree, smart_filename_creator=create_sgrt_filename,
                        dimensions=['time', 'var_name', 'pol'])
        monthly_dcs = dc.split_monthly(name='time')
        assert len(monthly_dcs) == 4
        dcs_feb = dc.split_monthly(name='time', months=2)
        assert len(dcs_feb) == 2
        months = [timestamp.month for dc_feb in dcs_feb for timestamp in dc_feb['time']]
        assert (np.array(months) == 2).all()

    def test_unite(self):
        dc_1 = EODataCube(dir_tree=self.dir_tree, smart_filename_creator=create_sgrt_filename,
                          dimensions=['time', 'pol'])
        dc_2 = EODataCube(dir_tree=self.dir_tree, smart_filename_creator=create_sgrt_filename,
                          dimensions=['time', 'orbit_direction'])

        dc_merged = dc_1.unite(dc_2)
        assert 'pol' not in dc_merged.inventory.columns
        assert 'orbit_direction' not in dc_merged.inventory.columns

    def test_match_dimension(self):
        dc_1 = EODataCube(dir_tree=self.dir_tree, smart_filename_creator=create_sgrt_filename,
                          dimensions=['time', 'pol'])
        dc_2 = EODataCube(dir_tree=self.dir_tree, smart_filename_creator=create_sgrt_filename,
                          dimensions=['time', 'orbit_direction'])
        dc_2.inventory = dc_2.inventory[dc_2['time'] != self.timestamps[0]]
        dc_2.inventory = dc_2.inventory[dc_2['time'] != self.timestamps[2]]

        dc_1.match_dimension(dc_2, name='time', in_place=True)
        assert sorted(list(set(dc_1['time']))) == [self.timestamps[1], self.timestamps[3]]

if __name__ == '__main__':
    unittest.main()

