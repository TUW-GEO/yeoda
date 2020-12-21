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
Main code for testing the filtering of data cubes.
"""

# general imports
import os
import shutil
import unittest

# test data imports
from tests.setup_test_data import setup_gt_test_data
from tests.setup_test_data import dirpath_test
from tests.setup_test_data import roi_test

# yeoda product imports
from yeoda.products.preprocessed import PreprocessedDataCube
from yeoda.products.preprocessed import SIG0DataCube
from yeoda.products.preprocessed import GMRDataCube


class FilteringTester(unittest.TestCase):
    """ Responsible for testing all the filtering functionalities of a data cube. """

    @classmethod
    def setUpClass(cls):
        """ Creates GeoTIFF test data. """

        setup_gt_test_data()

    @classmethod
    def tearDownClass(cls):
        """ Removes all test data. """

        shutil.rmtree(os.path.join(dirpath_test(), 'data', 'Sentinel-1_CSAR'))

    def setUp(self):
        """ Retrieves test data filepaths and auxiliary data. """

        self.filepaths, self.timestamps = setup_gt_test_data()
        self.data_dirpath = os.path.join(dirpath_test(), 'data', 'Sentinel-1_CSAR')

    def test_filter_pols_inplace(self):
        """ Creates a `PreprocessedDataCube` and tests filtering of polarisations on the original data cube. """

        dc = PreprocessedDataCube(self.data_dirpath, sres=500, dimensions=['time', 'var_name', 'pol'])
        assert len(set(dc['pol'])) == 2
        dc.filter_by_dimension("VV", name="pol", inplace=True)
        assert len(set(dc['pol'])) == 1

    def test_filter_pols_not_inplace(self):
        """ Creates a `PreprocessedDataCube` and tests filtering of polarisations on a newly created data cube. """

        dc = PreprocessedDataCube(self.data_dirpath, sres=500, dimensions=['time', 'var_name', 'pol'])
        dc_vv = dc.filter_by_dimension("VV", name="pol")
        dc_vh = dc.filter_by_dimension("VH", name="pol")
        assert len(set(dc_vv['pol'])) == 1
        assert list(set(dc_vv['pol']))[0] == "VV"
        assert len(set(dc_vh['pol'])) == 1
        assert list(set(dc_vh['pol']))[0] == "VH"

    def test_filter_pols_clone(self):
        """ Creates a `PreprocessedDataCube` and tests filtering of polarisations on a cloned data cube. """

        dc = PreprocessedDataCube(self.data_dirpath, sres=500, dimensions=['time', 'var_name', 'pol'])
        dc_clone = dc.clone()
        dc.filter_by_dimension("VV", name="pol", inplace=True)
        dc_clone.filter_by_dimension("VH", name="pol", inplace=True)
        assert len(set(dc['pol'])) == 1
        assert list(set(dc['pol']))[0] == "VV"
        assert len(set(dc_clone['pol'])) == 1
        assert list(set(dc_clone['pol']))[0] == "VH"

    def test_filter_time(self):
        """ Creates a `PreprocessedDataCube` and tests filtering of timestamps. """

        dc = PreprocessedDataCube(self.data_dirpath, sres=500, dimensions=['time', 'var_name', 'pol'])
        start_time = self.timestamps[0]
        end_time = self.timestamps[1]
        dc.filter_by_dimension([(start_time, end_time)], expressions=[(">=", "<=")], inplace=True)
        assert sorted(list(set(dc['time']))) == self.timestamps[:2]

    def test_filter_var_names(self):
        """
        Creates a `PreprocessedDataCube`, a `SIG0DataCube` and a `GMRDataCube` and tests filtering of variable names of
        the `PreprocessedDataCube` in comparison to the other data cubes.
        """

        pre_dc = PreprocessedDataCube(self.data_dirpath, sres=500, dimensions=['time', 'var_name', 'pol'])
        sig0_dc = SIG0DataCube(self.data_dirpath, sres=500, dimensions=['time', 'pol'])
        gmr_dc = GMRDataCube(self.data_dirpath, sres=500, dimensions=['time', 'pol'])
        dc_filt_sig0 = pre_dc.filter_by_dimension("SIG0", name="var_name")
        dc_filt_gmr = pre_dc.filter_by_dimension("GMR", name="var_name")
        assert sorted(list(sig0_dc['filepath'])) == sorted(list(dc_filt_sig0['filepath']))
        assert sorted(list(gmr_dc['filepath'])) == sorted(list(dc_filt_gmr['filepath']))

    def test_filter_files_with_pattern(self):
        """
        Creates a `PreprocessedDataCube` and a `SIG0DataCube` and tests filtering of a file pattern of
        the `PreprocessedDataCube` in comparison to `SIG0DataCube`.
        """

        pre_dc = PreprocessedDataCube(self.data_dirpath, sres=500, dimensions=['time', 'var_name', 'pol'])
        sig0_dc = SIG0DataCube(self.data_dirpath, sres=500, dimensions=['time', 'pol'])
        pre_dc.filter_files_with_pattern(".*SIG0.*", inplace=True)
        assert sorted(list(sig0_dc['filepath'])) == sorted(list(pre_dc['filepath']))

    def test_filter_spatially_by_tilename(self):
        """ Creates a `PreprocessedDataCube` and tests filtering of tile names. """

        dc = PreprocessedDataCube(self.data_dirpath, sres=500, dimensions=['time', 'var_name', 'pol', 'tile_name'],
                                  sdim_name="tile_name")
        assert len(set(dc['tile_name'])) == 2
        dc.filter_spatially_by_tilename("E042N012T6", inplace=True)
        assert len(set(dc['tile_name'])) == 1
        assert list(set(dc['tile_name']))[0] == "E042N012T6"

    def test_filter_spatially_by_geom(self):
        """
        Creates a `PreprocessedDataCube` and tests filtering of the spatial/tile dimension according to a given
        geometry/region of interest.
        """

        dc = PreprocessedDataCube(self.data_dirpath, sres=500, dimensions=['time', 'var_name', 'pol', 'tile_name'],
                                  sdim_name="tile_name")
        bbox, sref = roi_test()
        assert len(set(dc['tile_name'])) == 2
        dc.filter_spatially_by_geom(bbox, sref=sref, inplace=True)
        assert len(set(dc['tile_name'])) == 1
        assert list(set(dc['tile_name']))[0] == "E042N012T6"

    def test_filter_by_metadata(self):
        """ Creates a `PreprocessedDataCube` and tests filtering by metadata. """

        dc = PreprocessedDataCube(self.data_dirpath, sres=500, dimensions=['time', 'orbit_direction'])
        assert len(set(dc['orbit_direction'])) == 2
        dc.filter_by_metadata({'direction': 'D'}, inplace=True)
        assert len(set(dc['orbit_direction'])) == 1
        assert list(set(dc['orbit_direction']))[0] == "D"


if __name__ == '__main__':
    unittest.main()
