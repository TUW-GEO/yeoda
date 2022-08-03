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
