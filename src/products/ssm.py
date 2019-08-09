import numpy as np

# import yeoda
from products.base import ProductDataCube

# import file and folder naming convention
from geopathfinder.naming_conventions.sgrt_naming import create_sgrt_filename
from geopathfinder.naming_conventions.sgrt_naming import sgrt_tree

# import grid
from Equi7Grid.equi7grid.equi7grid import Equi7Grid


class SSMDataCube(ProductDataCube):
    def __init__(self, root_dirpath, spres=500, continent='EU', dimensions=None, inventory=None):
        super().__init__(root_dirpath, ["SSM", "SSM-NOISE"], spres=spres, continent=continent, dimensions=dimensions,
                         inventory=inventory)

    def encode(self, data):
        data *= 2
        data[np.isnan(data)] = 255
        return data

    def decode(self, data):
        data = data.astype(float)
        data[data > 200] = None
        return data / 2.