import numpy as np

# import yeoda
from yeoda.yeoda import EODataCube

# import file and folder naming convention
from geopathfinder.naming_conventions.sgrt_naming import create_sgrt_filename
from geopathfinder.naming_conventions.sgrt_naming import sgrt_tree

# import grid
from Equi7Grid.equi7grid.equi7grid import Equi7Grid


class SSMDataCube(EODataCube):
    def __init__(self, root_dirpath, spres=500, continent='EU', dimensions=None, inventory=None):
        dir_tree = sgrt_tree(root_dirpath, register_file_pattern=(".tif$"))
        grid = Equi7Grid(spres)
        sub_grid = grid.__getattr__(continent)
        super().__init__(dir_tree=dir_tree, grid=sub_grid, inventory=inventory, dimensions=dimensions,
                         smart_filename_creator=create_sgrt_filename)
        self.__check_inventory()

    def __check_inventory(self):
        pass

    def encode(self, data):
        data *= 2
        data[np.isnan(data)] = 255
        return data

    def decode(self, data):
        data = data.astype(float)
        data[data > 200] = None
        return data / 2.