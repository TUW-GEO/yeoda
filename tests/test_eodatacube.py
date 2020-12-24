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
Main code for testing general functionalities of a data cube.
"""

# general imports
import os
import shutil
import unittest
import numpy as np
from osgeo import ogr

from tests.setup_test_data import setup_gt_test_data
from tests.setup_test_data import dirpath_test

from equi7grid.equi7grid import Equi7Grid
from geopathfinder.file_naming import SmartFilename
from geopathfinder.naming_conventions.sgrt_naming import SgrtFilename

# import yeoda
from yeoda.datacube import EODataCube
from yeoda.products.preprocessed import SIG0DataCube

from yeoda.errors import SpatialInconsistencyError


class EODataCubeTester(unittest.TestCase):
    """ Responsible for testing all the data cube operations and functionalities of a data cube. """

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

        self.gt_filepaths, self.timestamps = setup_gt_test_data()

    def test_unknown_filename(self):
        """ Checks data cube structure if filename translation fails. """

        # class which is not able to interpret the filename
        class SmartFilenameFail(SmartFilename):
            @classmethod
            def from_filename(self, filename, **kwargs):
                raise Exception

        dc = EODataCube(filepaths=self.gt_filepaths, smart_filename_class=SmartFilenameFail,
                        dimensions=['time', 'var_name', 'pol'])
        assert len(dc.dimensions) == 1
        assert len(dc) == len(self.gt_filepaths)

    def test_rename_dimension(self):
        """ Tests renaming a dimension of a data cube. """

        dc = EODataCube(filepaths=self.gt_filepaths, smart_filename_class=SgrtFilename,
                        dimensions=['time', 'var_name', 'pol'])
        dc.rename_dimensions({'pol': 'band'}, inplace=True)
        assert 'band' in dc.inventory.columns
        assert 'pol' not in dc.inventory.columns
        assert len(set(dc['band'])) == 2

    def test_add_dimension(self):
        """ Tests adding a dimension to a data cube. """

        dc = EODataCube(filepaths=self.gt_filepaths, smart_filename_class=SgrtFilename,
                        dimensions=['time', 'var_name', 'pol'])
        dim_values = np.random.rand(len(dc))
        dc.add_dimension("value", dim_values, inplace=True)
        assert "value" in dc.inventory.columns
        assert list(dc['value']) == dim_values.tolist()

    def test_sort_by_dimension(self):
        """ Tests sorting a dimension of the data cube. """

        dc = EODataCube(filepaths=self.gt_filepaths, smart_filename_class=SgrtFilename,
                        dimensions=['time', 'var_name', 'pol'])
        timestamps = list(dc['time'])
        dim_values = np.random.rand(len(dc))
        timestamps_sorted = np.array(timestamps)[np.argsort(dim_values)].tolist()

        dc.add_dimension("value", dim_values, inplace=True)
        dc.sort_by_dimension("value", inplace=True)
        assert list(dc['time']) == timestamps_sorted

    def test_split_time(self):
        """ Tests splitting of the data cube in time to create multiple data cubes. """

        dc = EODataCube(filepaths=self.gt_filepaths, smart_filename_class=SgrtFilename,
                        dimensions=['time', 'var_name', 'pol'])
        first_time_interval = (self.timestamps[0], self.timestamps[1])
        second_time_interval = (self.timestamps[2], self.timestamps[-1])
        expression = (">=", "<=")
        dcs = dc.split_by_dimension([first_time_interval, second_time_interval], expressions=[expression, expression])
        assert len(dcs) == 2
        assert sorted(list(set(dcs[0]['time']))) == self.timestamps[:2]
        assert sorted(list(set(dcs[1]['time']))) == self.timestamps[2:]

    def test_split_yearly(self):
        """ Test splitting of the data cube in yearly intervals to create yearly data cubes. """

        dc = EODataCube(filepaths=self.gt_filepaths, smart_filename_class=SgrtFilename,
                        dimensions=['time', 'var_name', 'pol'], tdim_name='time')
        yearly_dcs = dc.split_yearly()
        assert len(yearly_dcs) == 2
        dcs_2016 = dc.split_yearly(years=2016)
        assert len(dcs_2016) == 1
        dc_2016 = dcs_2016[0]
        years = [timestamp.year for timestamp in dc_2016['time']]
        assert (np.array(years) == 2016).all()

    def test_split_monthly(self):
        """ Test splitting of the data cube in monthly intervals to create monthly data cubes. """

        dc = EODataCube(filepaths=self.gt_filepaths, smart_filename_class=SgrtFilename,
                        dimensions=['time', 'var_name', 'pol'], tdim_name='time')
        monthly_dcs = dc.split_monthly()
        assert len(monthly_dcs) == 4
        dcs_feb = dc.split_monthly(months=2)
        assert len(dcs_feb) == 2
        months = [timestamp.month for dc_feb in dcs_feb for timestamp in dc_feb['time']]
        assert (np.array(months) == 2).all()

    def test_unite(self):
        """
        Tests data cube union, i.e. if all data from a second datacube is added to the original data cube
        (rows and columns).
        """

        n = len(self.gt_filepaths)
        dc_1 = EODataCube(filepaths=self.gt_filepaths[:int(n/2)], smart_filename_class=SgrtFilename,
                          dimensions=['time', 'pol'])
        dc_2 = EODataCube(filepaths=self.gt_filepaths[int(n/2):], smart_filename_class=SgrtFilename,
                          dimensions=['time', 'orbit_direction'])

        dc_united = dc_1.unite(dc_2)
        assert 'pol' in dc_united.dimensions
        assert 'orbit_direction' in dc_united.dimensions

    def test_intersect_empty(self):
        """
        Tests data cube intersection on the temporal dimension, i.e. if all data from a second datacube is properly
        intersected with the data of to the original data cube according to matching timestamps. The result should be
        empty due to non-overlapping timestamps.
        """

        # empty data cube when an intersection is applied
        dc_1 = EODataCube(filepaths=self.gt_filepaths, smart_filename_class=SgrtFilename,
                          dimensions=['time', 'pol'])
        dc_2 = EODataCube(filepaths=self.gt_filepaths, smart_filename_class=SgrtFilename,
                          dimensions=['time', 'orbit_direction'])
        dc_1.inventory = dc_1.inventory[dc_1['time'] == self.timestamps[0]]
        dc_2.inventory = dc_2.inventory[dc_2['time'] == self.timestamps[1]]
        dc_intersected = dc_1.intersect(dc_2, on_dimension='time')
        assert len(dc_intersected) == 0

    def test_intersect_dimensions(self):
        """
        Tests simple data cube intersection, i.e. if all data from a second datacube is properly
        intersected with the data of to the original data cube.
        """

        dc_1 = EODataCube(filepaths=self.gt_filepaths, smart_filename_class=SgrtFilename,
                          dimensions=['time', 'pol'])
        dc_2 = EODataCube(filepaths=self.gt_filepaths, smart_filename_class=SgrtFilename,
                          dimensions=['time', 'orbit_direction'])

        dc_intersected = dc_1.intersect(dc_2)
        assert len(dc_intersected) == len(self.gt_filepaths)
        assert 'pol' not in dc_intersected.dimensions
        assert 'orbit_direction' not in dc_intersected.dimensions
        assert 'time' in dc_intersected.dimensions

    def test_intersect_align_dimension_shrink(self):
        """
        Tests matching of entries with two different methods, which should yield the same result: data cube
        intersection and data cube alignment on the temporal dimension.
        """

        dc_1 = EODataCube(filepaths=self.gt_filepaths, smart_filename_class=SgrtFilename,
                          dimensions=['time'])
        dc_2 = dc_1.clone()
        dc_2.inventory = dc_2.inventory[dc_2['time'] != self.timestamps[0]]
        dc_2.inventory = dc_2.inventory[dc_2['time'] != self.timestamps[2]]

        dc_aligned = dc_1.align_dimension(dc_2, name='time', inplace=False)
        dc_intersected = dc_1.intersect(dc_2, on_dimension='time', inplace=False)

        assert sorted(list(dc_aligned['time'])) == sorted(list(dc_intersected['time']))

    def test_align_dimension_shrink(self):
        """
        Tests alignment of a data cube with another data cube along the temporal dimension. Since the second
        data cube contains less data, the original data cube will also contain less data, i.e. the same timestamps as
        in the other data cube.
        """

        dc_1 = EODataCube(filepaths=self.gt_filepaths, smart_filename_class=SgrtFilename,
                          dimensions=['time'])
        dc_2 = dc_1.clone()
        dc_2.inventory = dc_2.inventory[dc_2['time'] != self.timestamps[0]]
        dc_2.inventory = dc_2.inventory[dc_2['time'] != self.timestamps[2]]

        dc_1.align_dimension(dc_2, name='time', inplace=True)
        assert sorted(list(set(dc_1['time']))) == [self.timestamps[1], self.timestamps[3]]

    def test_align_dimension_grow(self):
        """
        Tests alignment of a data cube with another data cube along the temporal dimension. Since the second
        data cube contains more data, the original data cube will also contain more data, i.e. the same timestamps as
        in the other data cube by duplicating the entries.
        """

        dc_1 = EODataCube(filepaths=self.gt_filepaths, smart_filename_class=SgrtFilename,
                          dimensions=['time'])
        dc_2 = dc_1.clone()
        timestamps = list(dc_1['time'])
        subset_idxs = [timestamps.index(self.timestamps[0]),
                       timestamps.index(self.timestamps[1]),
                       timestamps.index(self.timestamps[2]),
                       timestamps.index(self.timestamps[3])]
        dc_1.inventory = dc_1.inventory.iloc[subset_idxs]

        dc_1.align_dimension(dc_2, name='time', inplace=True)
        assert (dc_1['time'] == dc_2['time']).all()

    def test_boundary_fail(self):
        """ Tests exception triggering when multiple tiles are present in the data cube and a boundary is requested. """

        dc = SIG0DataCube(filepaths=self.gt_filepaths, dimensions=['time'], sdim_name="tile_name")
        try:
            boundary = dc.boundary
        except SpatialInconsistencyError:
            assert True

    def test_boundary(self):
        """ Tests equality of tile and data cube boundary. """

        dc = SIG0DataCube(filepaths=self.gt_filepaths, dimensions=['time'], sres=500, sdim_name="tile_name")
        dc.filter_spatially_by_tilename("E042N012T6", inplace=True)
        boundary = dc.boundary
        equi7 = Equi7Grid(500)
        tile_oi = equi7.EU.tilesys.create_tile(name="E042N012T6")
        assert ogr.CreateGeometryFromWkt(boundary.wkt).ExportToWkt() == tile_oi.get_extent_geometry_proj().ConvexHull().ExportToWkt()


if __name__ == '__main__':
    unittest.main()