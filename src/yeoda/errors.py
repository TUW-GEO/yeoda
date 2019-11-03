import pprint


class IOClassNotFound(Exception):
    def __init__(self, io_map, file_type):
        self.message = "The file type '{}' can not be related to any IO class. " \
                       "The available IO class are: \n {}".format(file_type, pprint.pformat(io_map))

    def __str__(self):
        return self.message


class DataTypeUnknown(Exception):
    def __init__(self, data_type):
        self.message = "The data type '{}' is not known.".format(data_type)

    def __str__(self):
        return self.message


class FileTypeUnknown(Exception):
    def __init__(self, file_type):
        self.message = "The file type '{}' is not known.".format(file_type)

    def __str__(self):
        return self.message


class GeometryUnkown(Exception):
    def __init__(self, geometry):
        self.message = "The given geometry type '{}' cannot be used.".format(type(geometry))

    def __str__(self):
        return self.message


class TileNotAvailable(Exception):
    def __init__(self, tilename):
        self.message = "The given tile '{}' is not available with the provided grid.".format(tilename)

    def __str__(self):
        return self.message