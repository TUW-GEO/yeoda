# general imports
import numpy as np

# import yeoda
from yeoda.yeoda import EODataCube

# import file and folder naming convention
from geopathfinder.naming_conventions.sgrt_naming import create_sgrt_filename
from geopathfinder.naming_conventions.sgrt_naming import sgrt_tree

# import grid
from Equi7Grid.equi7grid.equi7grid import Equi7Grid


class ProductDataCube(EODataCube):
    def __init__(self, root_dirpath, var_names, spres=10, continent='EU', dimensions=None, inventory=None,
                 file_pattern=".tif$"):
        dir_tree = sgrt_tree(root_dirpath, register_file_pattern=(file_pattern))
        # ensure there is a variable dimension
        if dimensions is None:
            dimensions = ['var_name']
        elif "var_name" not in dimensions:
            dimensions.append('var_name')

        grid = Equi7Grid(spres)
        sub_grid = grid.__getattr__(continent)
        super().__init__(dir_tree=dir_tree, grid=sub_grid, inventory=inventory, dimensions=dimensions,
                         smart_filename_creator=create_sgrt_filename)
        self.__check_inventory()

        # filter variable names
        self.filter_by_dimension(var_names, name='var_name', in_place=True)

    def __check_inventory(self):
        pass