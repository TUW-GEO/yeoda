import numpy as np

# import yeoda
from products.base import ProductDataCube

# import file and folder naming convention
from geopathfinder.naming_conventions.sgrt_naming import create_sgrt_filename
from geopathfinder.naming_conventions.sgrt_naming import sgrt_tree

# import grid
from Equi7Grid.equi7grid.equi7grid import Equi7Grid


class PreprocessedDataCube(ProductDataCube):
    def __init__(self, root_dirpath, var_names, spres=10, continent='EU', dimensions=None, inventory=None,
                 file_pattern=".tif$"):
        super().__init__(root_dirpath, var_names, spres=spres, continent=continent, dimensions=dimensions,
                         inventory=inventory, file_pattern=file_pattern)

    def encode(self, data):
        data *= 100
        data[np.isnan(data)] = -9999
        data.astype(np.int16)
        return data

    def decode(self, data):
        data = data.astype(float)
        data[data == -9999] = None
        return data / 100.


class SIG0DataCube(PreprocessedDataCube):
    def __init__(self, root_dirpath, spres=10, continent='EU', dimensions=None, inventory=None, file_pattern=".tif$"):
        super().__init__(root_dirpath, ["SIG0"], spres=spres, continent=continent, dimensions=dimensions,
                         inventory=inventory, file_pattern=file_pattern)


class GMRDataCube(PreprocessedDataCube):
    def __init__(self, root_dirpath, spres=10, continent='EU', dimensions=None, inventory=None, file_pattern=".tif$"):
        super().__init__(root_dirpath, ["GMR"], spres=spres, continent=continent, dimensions=dimensions,
                         inventory=inventory, file_pattern=file_pattern)