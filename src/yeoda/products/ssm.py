import numpy as np

# import yeoda
from products.base import ProductDataCube

# import file and folder naming convention
from geopathfinder.naming_conventions.sgrt_naming import create_sgrt_filename
from geopathfinder.naming_conventions.sgrt_naming import sgrt_tree

# import grid
from Equi7Grid.equi7grid.equi7grid import Equi7Grid


class SSMDataCube(ProductDataCube):
    """
    Data cube defining a TUWGEO SSM/SSM-NOISE product.
    """

    def __init__(self, root_dirpath, spres=500, continent='EU', dimensions=None, inventory=None):
        """
        Constructor of class `SSMDataCube`.

        Parameters
        ----------
        root_dirpath: str, optional
            Root directory path to the SGRT directory tree.
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
        super().__init__(root_dirpath, ["SSM", "SSM-NOISE"], spres=spres, continent=continent, dimensions=dimensions,
                         inventory=inventory)

    def encode(self, data):
        """
        Encoding function for TUWGEO SSM/SSM-NOISE data.

        Parameters
        ----------
        data: NumPy array
            Input data.

        Returns
        -------
        data: NumPy array
            Encoded data.
        """

        data *= 2
        data[np.isnan(data)] = 255
        return data

    def decode(self, data):
        """
        Decoding function for TUWGEO SSM/SSM-NOISE data.

        Parameters
        ----------
        data: NumPy array
            Encoded input data.

        Returns
        -------
        data: NumPy array
            Decoded data based on native units.
        """

        data = data.astype(float)
        data[data > 200] = None
        return data / 2.
