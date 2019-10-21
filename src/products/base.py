# import yeoda
from yeoda.yeoda import EODataCube

# import file and folder naming convention
from geopathfinder.naming_conventions.sgrt_naming import create_sgrt_filename
from geopathfinder.naming_conventions.sgrt_naming import sgrt_tree

# import grid
from Equi7Grid.equi7grid.equi7grid import Equi7Grid


class ProductDataCube(EODataCube):
    """
    Data cube defining a TUWGEO product.
    """
    def __init__(self, root_dirpath=None, var_names=None, spres=10, continent='EU', dimensions=None, inventory=None,
                 file_pattern=".tif$", **kwargs):

        """
        Constructor of class `ProductDataCube`.

        Parameters
        ----------
        root_dirpath: str, optional
            Root directory path to the SGRT directory tree.
        var_names: list, optional
            SGRT Variable names , e.g. ["SIG0", "SSM"].
        spres: int, optional
            Spatial sampling in grid units, e.g. 10, 500 (default is 10).
        continent: str, optional
            Continent/Subgrid of the Equi7Grid system (default is 'EU').
        dimensions: list, optional
            List of filename parts to use as dimensions. The strings have to match with the keys of the `SgrtFilename`
            fields definition.
        inventory: GeoDataFrame, optional
            Contains information about the dimensions (columns) and each filepath (rows).
        file_pattern: str
            Pattern to match/only select certain file names.
        """

        if root_dirpath is not None:
        #if (dir_tree is None) and (root_dirpath is not None):
            dir_tree = sgrt_tree(root_dirpath, register_file_pattern=(file_pattern))
        # ensure there is a variable dimension
        if dimensions is None:
            dimensions = ['var_name']
        elif "var_name" not in dimensions:
            dimensions.append('var_name')

        grid = Equi7Grid(spres)
        sub_grid = grid.__getattr__(continent)

        super().__init__(inventory=inventory, dimensions=dimensions,
                         smart_filename_creator=create_sgrt_filename, grid=sub_grid, **kwargs)

        # filter variable names
        self.filter_by_dimension(var_names, name='var_name', in_place=True)
