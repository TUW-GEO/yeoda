import copy
import pyraster
import pandas as pd
from geopathfinder.file_naming import create_smart_filename

# TODO: handle file extensions correctly
# TODO: synchronise filepaths with inventory
class eoDataCube(object):
    """
    A filename based data cube.
    """
    def __init__(self, grid, filepaths, filename_def, pad="-", delim="_", dimensions=None, inventory=None):
        """
        Constructor of the eoDataCube class.
        :param grid: object,
            Grid object/class (e.g. Equi7Grid).
        :param filepaths: list
            List of filepaths or filenames
        :param filename_def: OrderedDict
            Name of fields (keys) in right order and length (values).
        :param pad: str [optional]
            Padding used in the filenames.
        :param delim: str [optional]
            Delimiter used in the filenames.
        :param dimensions: list [optional]
            List of filename parts to use as dimensions. The strings have to match with the keys in 'filename_def'.
        :param inventory: Pandas DataFrame [optional]
            contains information about the dimensions (columns) and each filename (rows)
        """
        self.grid = grid
        self.filepaths = filepaths
        self.filename_def = filename_def
        if dimensions:
            self.dimensions = dimensions
        else:
           self.dimensions = filename_def.keys()

        if not inventory:
            # initialise inventory
            inventory = dict()
            inventory['filepath'] = []
            for dimension in self.dimensions:
                inventory[dimension] = []

            # fill inventory
            for filepath in filepaths:
                inventory['filepath'].append(filepath)
                smart_filename = create_smart_filename(filepath, filename_def, pad=pad, delim=delim)
                for dimension in dimensions:
                    inventory[dimension].append(smart_filename[dimension])

            inventory = pd.DataFrame(inventory, columns=dimensions)

        self.inventory = inventory

        # TODO: add bands and timestamps from multidimensional files using pyraster

    @classmethod
    def from_inventory(cls, inventory, grid, filepaths, filename_def, pad="-", delim="_", dimensions=None):
        return cls(grid, filepaths, filename_def, pad=pad, delim=delim, dimensions=dimensions, inventory=inventory)

    def rename_dimensions(self, dimensions_map):
        """
        Renames the dimensions of the datacube.
        :param dimensions_map: dict
            A dictionary representing the relation between old and new dimension names. The keys are the old dimension names,
            the values the new dimension names (e.g., {'time_begin': 'time'}).
        :return: eoDataCube
        """
        for dimension_name in dimensions_map.keys():
            idx = self.dimensions.index(dimension_name)
            self.dimensions[idx] = dimensions_map[dimension_name]
        # TODO: rename inventory
        return self.from_inventory(inventory, self.grid, self.filepaths, self.filename_def, dimensions=self.dimensions)

    def add_dimension(self, name, values):
        """
        Adds a new dimension to the datacube.
        :param name: str
            Name of the new dimension
        :param values: list
            Values of the new dimension (e.g., cloud cover, quality flag, ...).
            They have to have the same length as all the rows in the inventory.
        :return: eoDataCube
        """

    def filter_dimension(self, values, expressions=None, name="time"):
        """
        Filters the datacube according to the given extent and returns a new datacube.

        :param values: tuple, list
            Values of interest to filter, e.g., timestamps: ('2019-01-01', '2019-02-01'), polarisations: ('VV')
        :param expressions: tuple, list [optional]
            Mathematical expressions to filter the data accordingly. If none are given, the exact values from 'values'
            are taken, otherwise the expressions are applied for each value and linked with an AND (e.g., ('>=', '<=')).
            They have to have the same length as 'values'.
        :param name:
        :return: eoDataCube
        """

    def split_dimension(self, values, expressions=None, name="time"):
        """
        Splits the datacube according to the given extents and expressions and returns a list of new datacubes.

        :param values: tuple, list
            Values of interest to filter, e.g., timestamps: ('2019-01-01', '2019-02-01'), polarisations: ('VV')
        :param expressions: tuple, list [optional]
            Mathematical expressions to filter the data accordingly. If none are given, the exact values from 'values'
            are taken, otherwise the expressions are applied for each value and linked with an AND (e.g., ('>=', '<=')).
            They have to have the same length as 'values'.
        :param name:
        :return: list of eoDataCubes
        """

    def filter_with_pattern(self, pattern):
        """
        Filter all filepaths according to the given pattern.

        :param pattern: str
            A regex (e.g., ".*S1A.*GRD.*").
        :return: eoDataCube
        """

    def filter_spatial(self, tilenames=None, bbox=None, geom=None, sref=None, name="tile"):
        """
        Spatially filters the datacube by tilenames, a bounding box and/or a geometry.
        :param tilenames: list [optional]
            Tilenames corresponding to the given grid and inventory.
        :param bbox: tuple, list [optional]
            List containing the extent of the region of interest, i.e. [x_min, y_min, x_max, y_max]
        :param geom: str, ogr.Geometry [optional]
            A geometry defining the region of interest. It can be an OGR geometry, i.e. a polygon, or a shape file.
        :param sref: osr.SpatialReference [optional]
            It is a mandatory parameter if the bounding box coordinate system differs from the grid coordinate system.
        :param name: str [optional]
            Name of the tile/spatial dimension in the filenames.
        :return:
        """
        if tilenames:
            #TODO: filter inventory according to full tilename list

        if geom:
            #TODO: handle the geometry correctly
            bbox = geom.getEnvelope()

        if bbox:
            ftile_names = self.grid.search_tiles_in_roi(bbox, osr_spref=sref)
            # TODO: filter inventory according to full tilename list

    def filter_bands(self, bands):
        """
        Filters the datacube according to the given bands.
        :return: eoDataCube
        """

    def filter_metadata(self, keys, values):
        """
        Filters the datacube according to given key-value relations in the metadata of the files.
        :return: eoDataCube
        """

    def load(self, tilenames=None, bbox=None, geom=None, sref=None, dimension_name="tile", data_variables=None):
        """
        Loads data as an xarray.
        :param tilenames: list [optional]
            Tilenames corresponding to the given grid and inventory.
        :param bbox: tuple, list [optional]
            List containing the extent of the region of interest, i.e. [x_min, y_min, x_max, y_max]
        :param geom: str, ogr.Geometry [optional]
            A geometry defining the region of interest. It can be an OGR geometry, i.e. a polygon, or a shape file.
        :param sref: osr.SpatialReference [optional]
            It is a mandatory parameter if the bounding box coordinate system differs from the grid coordinate system.
        :param dimension_name: str [optional]
            Name of the tile/spatial dimension in the filenames.
        :param data_variables: [optional]
            Name of the tile/spatial dimension in the filenames.
        :return: xarray
        """
        dc = self.filter_spatial(tilenames=tilenames, bbox=bbox, geom=geom, sref=sref, name=dimension_name)
        # TODO: use pyraster functionalities for the remaining tasks:
            # TODO: pyraster could get a functionality directly working with the inventory given herein. This would create an xarray with the dimensions specified.
        # TODO: merge the cubes using the coordinates and dimensional properties of an xarray.

    def match_dimension(self, dc_other, name):
        """
        Matches the datacubes with respect to another datacube along the specified dimension.
        :param dc_other: eoDataCube
            Other datacube to match with.
        :return: eoDataCube
        """
        dc_this, _ = match_dimension(self, dc_other, name)
        return dc_this


def match_dimension(dc_1, dc_2, name):
    """
    Matches the given datacubes along the specified dimension.
    :param dc_1: eoDataCube
        First datacube.
    :param dc_2: eoDataCube
        Second datacube.
    :return:
    """

def merge_dcs(dc_1, dc_2, name=None, values=None):
    """
    Merges two datacubes in one datacube. By doing so, duplicates are removed.
    If 'name' and 'values' are given, the datacubes are merged over a new dimension.
    :param dc_1: eoDataCube
        First datacube.
    :param dc_2: eoDataCube
        Second datacube.
    :param name: str [optional]
        Name of the new dimension
    :param values: list [optional]
        Values of the new dimension (e.g., cloud cover, quality flag, ...).
        They have to have the same length as all the rows in the inventory.
    :return:
    """


# class eoDataSystem(object):
#     def __init__(self, datacubes, labels=None):
#         if not isinstance(datacubes, (list, dict, OrderedDict)):
#             datacubes = [datacubes]
#
#         if isinstance(datacubes, list):
#             if labels and len(labels) == len(datacubes):
#                 self.datacubes = OrderedDict()
#                 for i, label in enumerate(labels):
#                     self.datacubes[label] = datacubes[i]
#             else:
#                 self.datacubes = datacubes
#
#     @classmethod
#     def from_dc_def(cls, grid, filepaths, filename_def, pad="-", delim="_", dimensions=None, inventory=None,
#                     label=None):
#         dc = eoDataCube(filepaths, filename_def, grid, pad=pad, delim=delim, dimensions=dimensions, inventory=inventory)
#         return cls(dc, labels=[label])
#
#     def apply(self):




