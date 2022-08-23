class FileTypeUnknown(Exception):
    """ Class to handle exceptions thrown by unknown file types/extensions."""

    def __init__(self, ext):
        """
        Constructor of `FileTypeUnknown`.

        Parameters
        ----------
        ext : str
            File extension, e.g. '.nc'.
        """

        self.message = "The file type/extension '{}' is not known.".format(ext)

    def __str__(self):
        """ String representation of this class. """

        return self.message


class DimensionUnkown(KeyError):
    """ Class to handle exceptions thrown by unknown dimensions/columns in the file register of the datacube."""

    def __init__(self, dimension_name):
        """
        Constructor of `DimensionUnknown`.

        Parameters
        ----------
        dimension_name : str
            Column/Dimension name of the file register of the datacube.
        """

        self.message = "Dimension {} is unknown. Please add it to the data cube.".format(dimension_name)

    def __str__(self):
        """ String representation of this class. """

        return self.message
