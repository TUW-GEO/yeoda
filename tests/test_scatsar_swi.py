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
Main code for testing general functionalities of a SCATSARSWIDataCube.
"""

# general imports
import unittest
import numpy as np

from tests.setup_test_data import setup_scatsarswi_single_test_data

from equi7grid.equi7grid import Equi7Grid
from geopathfinder.naming_conventions.sgrt_naming import SgrtFilename
from geopathfinder.folder_naming import build_smarttree

# import yeoda
from yeoda.products.scatsar_swi import SCATSARSWIDataCube


class SCATSARSWIDataCubeTester(unittest.TestCase):
    """ Responsible for testing all the data cube operations and functionalities of a data cube. """

    @classmethod
    def setUpClass(cls):
        """ Creates GeoTIFF test data. """

        setup_scatsarswi_single_test_data()

    @classmethod
    def tearDownClass(cls):
        """ Removes all test data. """

        # shutil.rmtree(os.path.join(dirpath_test(), 'data', 'SCATSAR_SWI'))

        pass

    def setUp(self):
        """ Retrieves test data filepaths and auxiliary data. """

        self.root_dirpath, self.ssswi_filepaths, self.timestamps = setup_scatsarswi_single_test_data()


    def test_load_by_coords(self):

        e7g = Equi7Grid(500)

        hierarchy = ['wflow', 'grid', 'tile', 'var']
        st = build_smarttree(self.root_dirpath, hierarchy, register_file_pattern='.nc')

        dimensions = ["grid_name", "tile_name", "band"]
        swi_dc = SCATSARSWIDataCube(filepaths=st.file_register,
                               sres=500, continent='EU', dimensions=dimensions,
                               smart_filename_class=SgrtFilename)

        swi_dc.filter_by_dimension('E042N030T6', name='tile_name', inplace=True)

        result = swi_dc.load_by_coords((4251192), (3031169), sref=e7g.EU.core.projection.osr_spref, band='SWI_T002')
        assert np.array_equal(np.float(result['SWI_T002'].data[:]), 25.5)

        result = swi_dc.load_by_coords((4251192), (3031169), sref=e7g.EU.core.projection.osr_spref, band='QFLAG_T015')
        assert np.array_equal(np.float(result['QFLAG_T015'].data[:]), 99.0)

        result = swi_dc.load_by_coords((4251192), (3031169), sref=e7g.EU.core.projection.osr_spref, band='SSF')
        assert np.array_equal(np.int(result['SSF'].data[:]), 1)

        result = swi_dc.load_by_coords((4251192), (3031169), sref=e7g.EU.core.projection.osr_spref, band='SAR_IC')
        assert np.array_equal(np.int(result['SAR_IC'].data[:]), 0)


if __name__ == '__main__':
    unittest.main()