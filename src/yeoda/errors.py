# general packages
import pprint


class IOClassNotFound(Exception):
    """
    Class to handle exceptions thrown by an unavailable IO map/IO class necessary to read a specific file type.
    """

    def __init__(self, io_map, file_type):
        """
        Constructor of `IOClassNotFound`.

        Parameters
        ----------
        io_map : dictionary
            Map that represents the relation of an EO file type (e.g. GeoTIFF) with an appropriate reader.
        file_type : str
            File type, e.g. 'GeoTIFF'.
        """

        self.message = "The file type '{}' can not be related to any IO class. " \
                       "The available IO class are: \n {}".format(file_type, pprint.pformat(io_map))

    def __str__(self):
        """ String representation of this class. """

        return self.message


class DataTypeUnknown(Exception):
    """ Class to handle exceptions thrown by mismatching data types."""

    def __init__(self, dtype_in, dtype_out):
        """
        Constructor of `DataTypeUnknown`.

        Parameters
        ----------
        dtype_in : str
            Data type of the input data. It can be:
                - 'xarray.DataSet'
                - 'numpy.ndarray'
                - 'pd.DataFrame'
        dtype_out : str
            Data type of the output data (default is 'xarray'). It can be:
                - 'xarray'
                - 'numpy'
                - 'dataframe'
        """

        self.message = "Data conversion not possible for requested data type '{}' " \
                       "and actual data type '{}'.".format(dtype_out, dtype_in)

    def __str__(self):
        """ String representation of this class. """

        return self.message


class FileTypeUnknown(Exception):
    """ Class to handle exceptions thrown by unknown file types."""

    def __init__(self, file_type):
        """
        Constructor of `FileTypeUnknown`.

        Parameters
        ----------
        io_map : dictionary
            Map that represents the relation of an EO file type (e.g. GeoTIFF) with an appropriate reader.
        file_type : str
            File type, e.g. 'GeoTIFF'.
        """

        self.message = "The file type '{}' is not known.".format(file_type)

    def __str__(self):
        """ String representation of this class. """

        return self.message


class GeometryUnkown(Exception):
    """ Class to handle exceptions thrown by unknown geometry types."""

    def __init__(self, geometry):
        """
        Constructor of `GeometryUnknown`.

        Parameters
        ----------
        geometry : ogr.Geometry or shapely.geometry or list or tuple, optional
            A vector geometry.
        """

        self.message = "The given geometry type '{}' cannot be used.".format(type(geometry))

    def __str__(self):
        """ String representation of this class. """

        return self.message


class DimensionUnkown(KeyError):
    """ Class to handle exceptions thrown by unknown dimensions/columns in the data cube inventory."""

    def __init__(self, dimension_name):
        """
        Constructor of `DimensionUnknown`.

        Parameters
        ----------
        dimension_name : str
            Column/Dimension name of the data cube inventory.
        """

        self.message = "Dimension {} is unknown. Please add it to the data cube.".format(dimension_name)

    def __str__(self):
        """ String representation of this class. """

        return self.message


class TileNotAvailable(Exception):
    """ Class to handle exceptions thrown by unknown tile names of a grid."""

    def __init__(self, tilename):
        """
        Constructor of `DimensionUnknown`.

        Parameters
        ----------
        tilename : str
            Tile name corresponding to a grid and/or the inventory.
        """

        self.message = "The given tile '{}' is not available with the provided grid.".format(tilename)

    def __str__(self):
        """ String representation of this class. """

        return self.message


class LoadingDataError(IOError):
    """ Class to handle exceptions thrown when it is not possible to read data."""

    def __init__(self):
        """Constructor of `LoadingDataError`."""

        self.message = "Failed loading the data."

    def __str__(self):
        """ String representation of this class. """

        return self.message
