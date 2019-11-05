# import geopathfinder modules for file and folder naming convention
from geopathfinder.naming_conventions.sgrt_naming import create_sgrt_filename
from geopathfinder.naming_conventions.sgrt_naming import sgrt_tree

# import TUWGEO standard grid
from equi7grid.equi7grid import Equi7Grid

# import basic data cube from yeoda
from yeoda.datacube import EODataCube


class ProductDataCube(EODataCube):
    """
    Data cube defining a TUWGEO product (preferably based on the Equi7 grid system).
    """

    def __init__(self, root_dirpath=None, var_names=None, sres=10, continent='EU', dimensions=None,
                 file_pattern=".tif$", **kwargs):

        """
        Constructor of class `ProductDataCube`.

        Parameters
        ----------
        root_dirpath : str, optional
            Root directory path to the SGRT directory tree.
        var_names : list, optional
            SGRT Variable names , e.g. ["SIG0", "SSM"].
        sres : int, optional
            Spatial sampling in grid units, e.g. 10, 500 (default is 10).
        continent : str, optional
            Continent/Subgrid of the Equi7Grid system (default is 'EU').
        dimensions : list, optional
            List of filename parts to use as dimensions. The strings have to match with the keys of the `SgrtFilename`
            fields definition.
        file_pattern : str
            Pattern to match/only select certain file names.
        **kwargs
            Arbitrary keyword arguments (e.g. containing 'inventory' or 'grid').
        """

        if root_dirpath is not None:
            dir_tree = sgrt_tree(root_dirpath, register_file_pattern=(file_pattern))
        else:
            dir_tree = kwargs.get('dir_tree', None)

        inventory = kwargs.get('inventory', None)

        # ensure there is a variable dimension
        if dimensions is None:
            dimensions = ['var_name']
        elif "var_name" not in dimensions:
            dimensions.append('var_name')

        grid = kwargs.get('grid', Equi7Grid(sres).__getattr__(continent))
        if 'tile_name' not in dimensions:
            dimensions.append('tile_name')

        super().__init__(inventory=inventory, filepaths=dir_tree.file_register, dimensions=dimensions,
                         smart_filename_creator=create_sgrt_filename, grid=grid)

        # filter variable names
        if var_names is not None:
            self.filter_by_dimension(var_names, name='var_name', in_place=True)
