# general imports
import os
import shutil
import unittest

from tests.setup_test_data import setup, test_dirpath

from geopathfinder.naming_conventions.sgrt_naming import create_sgrt_filename

# import yeoda
from yeoda.yeoda import EODataCube

class EODataCubeTester(unittest.TestCase):

    def setUp(self):
        dir_tree, timestamps = setup()
        self.dir_tree = dir_tree
        self.timestamps = timestamps

    def tearDown(self):
        shutil.rmtree(os.path.join(test_dirpath(), 'data'))

    def test_rename_dimension(self):
        dc = EODataCube(dir_tree=self.dir_tree, smart_filename_creator=create_sgrt_filename,
                        dimensions=['time', 'var_name', 'pol'])
        dc.rename_dimensions({'pol': 'band'}, in_place=True)
        assert 'band' in dc.inventory.columns
        assert 'pol' not in dc.inventory.columns
        assert len(set(dc['band'])) == 2

if __name__ == '__main__':
    eodc_tester = EODataCubeTester()
    eodc_tester.setUp()
    eodc_tester.test_rename_dimension()
    eodc_tester.tearDown()