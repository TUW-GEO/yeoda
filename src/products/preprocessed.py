import numpy as np

# import yeoda
from products.base import ProductDataCube


class PreprocessedDataCube(ProductDataCube):
    """
    Data cube defining a TUWGEO preprocessed product.
    """
    def __init__(self, root_dirpath=None, var_names=None, spres=10, continent='EU', dimensions=None, inventory=None,
                 file_pattern=".tif$", **kwargs):
        """
        Constructor of class `PreprocessedDataCube`.

        Parameters
        ----------
        root_dirpath: str, optional
            Root directory path to the SGRT directory tree.
        var_names: list, optional
            SGRT Variable names , e.g. ["SIG0", "GMR"].
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

        super().__init__(root_dirpath=root_dirpath, var_names=var_names, spres=spres, continent=continent,
                         dimensions=dimensions, inventory=inventory, file_pattern=file_pattern, **kwargs)

    def encode(self, data):
        """
        Joint encoding function for TUWGEO preprocessed data.

        Parameters
        ----------
        data: NumPy array
            Input data.

        Returns
        -------
        data: NumPy array
            Encoded data.
        """

        data *= 100
        data[np.isnan(data)] = -9999
        data.astype(np.int16)
        return data

    def decode(self, data):
        """
        Joint decoding function for TUWGEO preprocessed data.

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
        data[data == -9999] = None
        return data / 100.


class SIG0DataCube(PreprocessedDataCube):
    """
    Data cube defining a TUWGEO SIG0 product.
    """
    def __init__(self, root_dirpath=None, spres=10, continent='EU', dimensions=None, inventory=None, file_pattern=".tif$",
                 **kwargs):
        """
        Constructor of class `SIG0DataCube`.

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

        super().__init__(root_dirpath=root_dirpath, var_names=["SIG0"], spres=spres, continent=continent,
                         dimensions=dimensions, inventory=inventory, file_pattern=file_pattern, **kwargs)


class GMRDataCube(PreprocessedDataCube):
    """
    Data cube defining a TUWGEO GMR product.
    """
    def __init__(self, root_dirpath=None, spres=10, continent='EU', dimensions=None, inventory=None, file_pattern=".tif$",
                 **kwargs):
        """
        Constructor of class `GMRDataCube`.

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

        super().__init__(root_dirpath=root_dirpath, var_names=["GMR"], spres=spres, continent=continent,
                         dimensions=dimensions, inventory=inventory, file_pattern=file_pattern, **kwargs)
