# Copyright (c) 2020, Vienna University of Technology (TU Wien), Department of
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
Main code for testing product data cubes.
"""

# general imports
import unittest
import numpy as np
# yeoda imports
from yeoda.products.preprocessed import SIG0DataCube
from yeoda.products.preprocessed import GMRDataCube
from yeoda.products.scatsar_swi import SCATSARSWIDataCube
from yeoda.products.ssm import SSMDataCube
from yeoda.products.parameter import ParameterDataCube

# import test data
from tests.setup_test_data import setup_sig0_test_data
from tests.setup_test_data import setup_gmr_test_data
from tests.setup_test_data import setup_parameter_test_data
from tests.setup_test_data import setup_scatsarswi_test_data
from tests.setup_test_data import setup_ssm_test_data


class SIG0DataCubeTester(unittest.TestCase):
    """ Responsible for testing the `SIG0DataCube` setup. """

    def setUp(self):
        """ Retrieves test data file paths and creates an `SIG0DataCube`. """

        self.filepaths, self.timestamps = setup_sig0_test_data()

        dimensions = ["pol", "time", "grid_name", "tile_name"]
        self.dc = SIG0DataCube(filepaths=self.filepaths, sres=10, continent='EU', dimensions=dimensions)

    def test_loading_data(self):
        """ Tests proper loading and decoding of data. """
        poi_x, poi_y = 4588097.3, 1599913.1

        # filter data cubes
        dc_vv = self.dc.filter_by_dimension("VV", name="pol", inplace=False)
        dc_vh = self.dc.filter_by_dimension("VH", name="pol", inplace=False)
        # check VV data values
        result = dc_vv.load_by_coords(poi_x, poi_y, band=1, dtype="numpy")
        assert np.array_equal(float(result), -8.74)
        # check VH data values
        result = dc_vh.load_by_coords(poi_x, poi_y, band=1, dtype="numpy")
        assert np.array_equal(float(result), -14.15)
        # check no data values
        result = dc_vv.load_by_coords(poi_x, poi_y - 5., band=1, dtype="numpy")
        assert np.all(np.isnan(result))

    def test_check_timestamps(self):
        """ Tests if data and data cube time stamps match. """
        assert np.all(self.timestamps == self.dc['time'])


class GMRDataCubeTester(unittest.TestCase):
    """ Responsible for testing the `GMRDataCube` setup. """

    def setUp(self):
        """ Retrieves test data file paths and creates an `GMRDataCube`. """

        self.filepaths, self.timestamps = setup_gmr_test_data()

        dimensions = ["pol", "time", "grid_name", "tile_name"]
        self.dc = GMRDataCube(filepaths=self.filepaths, sres=10, continent='EU', dimensions=dimensions)

    def test_loading_data(self):
        """ Tests proper loading and decoding of data. """
        poi_x, poi_y = 4588097.3, 1599913.1

        # filter data cubes
        dc_vv = self.dc.filter_by_dimension("VV", name="pol", inplace=False)
        dc_vh = self.dc.filter_by_dimension("VH", name="pol", inplace=False)
        # check VV data values
        result = dc_vv.load_by_coords(poi_x, poi_y, band=1, dtype="numpy")
        assert np.array_equal(float(result), -7.93)
        # check VH data values
        result = dc_vh.load_by_coords(poi_x, poi_y, band=1, dtype="numpy")
        assert np.array_equal(float(result), -13.39)
        # check no data values
        result = dc_vv.load_by_coords(poi_x, poi_y - 5., band=1, dtype="numpy")
        assert np.all(np.isnan(result))

    def test_check_timestamps(self):
        """ Tests if data and data cube time stamps match. """
        assert np.all(self.timestamps == self.dc['time'])


class ParameterDataCubeTester(unittest.TestCase):
    """ Responsible for testing the `ParameterDataCube` setup. """

    def setUp(self):
        """ Retrieves test data file paths and creates an `ParameterDataCube`. """

        self.filepaths, self.start_times, self.end_times = setup_parameter_test_data()

        self.dc = ParameterDataCube(filepaths=self.filepaths, sres=10, continent='EU')

    def test_loading_data(self):
        """ Tests proper loading and decoding of data. """
        poi_x, poi_y = 5200996.1, 1699971.2

        # check data values
        result = self.dc.load_by_coords(poi_x, poi_y, band=1, dtype="numpy")
        assert np.array_equal(float(result), 42.)
        # check no data values
        result = self.dc.load_by_coords(poi_x, poi_y - 5., band=1, dtype="numpy")
        assert np.all(np.isnan(result))

    def test_check_timestamps(self):
        """ Tests if data and data cube time stamps match. """
        assert np.all(self.start_times == self.dc['stime'])
        assert np.all(self.end_times == self.dc['etime'])


class SCATSARSWIDataCubeTester(unittest.TestCase):
    """ Responsible for testing the `SCATSARSWIDataCube` setup. """

    def setUp(self):
        """ Retrieves test data file paths and creates an `SCATSARSWIDataCube`. """

        self.filepaths, self.timestamps = setup_scatsarswi_test_data()

        dimensions = ["time", "grid_name", "tile_name", "band"]
        self.dc = SCATSARSWIDataCube(filepaths=self.filepaths, sres=500, continent='EU', dimensions=dimensions)

    def test_loading_data(self):
        """ Tests proper loading and decoding of data. """
        poi_x, poi_y = 4251192, 3031169

        # check data values
        result = self.dc.load_by_coords(poi_x, poi_y, band='SWI_T002')
        assert np.array_equal(np.float(result['SWI_T002'].data[:]), 30.)

        result = self.dc.load_by_coords(poi_x, poi_y, band='QFLAG_T015')
        assert np.array_equal(np.float(result['QFLAG_T015'].data[:]), 99.0)

        result = self.dc.load_by_coords(poi_x, poi_y, band='SSF')
        assert np.array_equal(np.int(result['SSF'].data[:]), 1)

        result = self.dc.load_by_coords(poi_x, poi_y, band='SAR_IC')
        assert np.array_equal(np.int(result['SAR_IC'].data[:]), 0)

    def test_check_timestamps(self):
        """ Tests if data and data cube time stamps match. """
        assert np.all(self.timestamps == self.dc['time'])


class SSMDataCubeTester(unittest.TestCase):
    """ Responsible for testing the `SSMDataCube` setup. """

    def setUp(self):
        """ Retrieves test data file paths and creates an `SSMDataCube`. """

        self.filepaths, self.timestamps = setup_ssm_test_data()

        dimensions = ["time", "grid_name", "tile_name"]
        self.dc = SSMDataCube(filepaths=self.filepaths, sres=500, continent='EU', dimensions=dimensions)

    def test_loading_data(self):
        """ Tests proper loading and decoding of data. """
        poi_x, poi_y = 4867901, 1336603

        # check data values
        result = self.dc.load_by_coords(poi_x, poi_y, band=1, dtype="numpy")
        assert np.array_equal(float(result), 76)

    def test_check_timestamps(self):
        """ Tests if data and data cube time stamps match. """
        assert np.all(self.timestamps == self.dc['time'])


if __name__ == '__main__':
    unittest.main()