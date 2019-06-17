import copy
import os
import geopandas as geopd
from geopandas import GeoSeries, GeoDataFrame
import pytileproj.geometry as geometry
from osgeo import osr

from pyraster.geotiff import GeoTiffFile
from pyraster.netcdf import NcFile
from pyraster.gdalport import GdalImage
from shapely.wkt import loads

# TODO: handle file extensions correctly
# TODO: synchronise filepaths with inventory
# TODO: include exceptions
class eoDataCube(object):
    """
    A filename based data cube.
    """
    def __init__(self, filepaths=None, grid=None, dir_tree=None, create_smart_filename=None, dimensions=None,
                 inventory=None):
        """
        Constructor of the eoDataCube class.

        Parameters
        ----------
        filepaths: list of str, optional
            List of filepaths.
        grid: optional
            Grid object/class (e.g. Equi7Grid, LatLonGrid).
        dir_tree: SmartTree, optional
            Folder tree class managing folders and files.
        create_smart_filename: function, optional
            A function that allows to create a SmartFilename instance from a filepath.
        dimensions: list of str, optional
            List of filename parts to use as dimensions. The strings have to match with the keys of the SmartFilename
            fields definition.
        inventory: GeoDataFrame, optional
            Contains information about the dimensions (columns) and each filename (rows).
            If `grid` is not specified, a geometry is added to the GeoDataFrame.
        """
        # initialise simple class variables
        self.dimensions = dimensions
        self.history = []  # TODO

        # initialise/find filepaths
        self.filepaths = None
        if filepaths:
            self.filepaths = filepaths
        elif dir_tree:
            self.filepaths = dir_tree.file_register
        elif inventory:
            self.filepaths = inventory['filepath']

        # create inventory from found filepaths
        self.inventory = None
        if inventory:
            self.inventory = inventory
        else:
            self.__inventory_from_filepaths(create_smart_filename)

        self.grid = None
        if grid:
            self.grid = grid
        elif self.inventory is not None:
            geometries = [self.__geometry_from_file(filepath) for filepath in self.filepaths]
            self.add_dimension('geometry', geometries)

    def __file_type(self, filepath):
        ext = os.path.splitext(filepath)[1]
        if ext in ['.tif', '.tiff']:
            return 'GeoTIFF'
        elif ext in ['.nc']:
            return "NetCDF"
        else:
            return None

    def __geometry_from_file(self, filepath):
        file_type = self.__file_type(filepath)
        ds = None
        if file_type == "GeoTIFF":
            gt = GeoTiffFile(filepath)
            ds = gt.src
        elif file_type == "NetCDF":
            nc = NcFile(filepath)
            ds = nc.src

        if ds:
            gdal_img = GdalImage(ds, filepath)
            boundary_extent = gdal_img.get_extent()
            boundary_spref = osr.SpatialReference()
            boundary_spref.ImportFromWkt(gdal_img.projection())
            boundary_geom = geometry.extent2polygon(boundary_extent, boundary_spref)  # TODO: directly convert it to shapely geometry
            return loads(boundary_geom.ExportToWkt())
        else:
            return

    def __inventory_from_filepaths(self, create_smart_filename=None):
        inventory = dict()
        inventory['filepath'] = []

        # fill inventory
        if self.filepaths:
            untrans_filepaths = []
            if create_smart_filename:
                for filepath in self.filepaths:
                    try:
                        smart_filename = create_smart_filename(os.path.basename(filepath))
                    except:
                        untrans_filepaths.append(filepath)
                        continue

                    inventory['filepath'].append(filepath)
                    for key, value in smart_filename.fields.items():
                        if self.dimensions:
                            if key in self.dimensions:
                                if key not in inventory.keys():
                                    inventory[key] = []
                                inventory[key].append(value)
                        else:
                            if key not in inventory.keys():
                                inventory[key] = []
                            inventory[key].append(value)
            else:
                untrans_filepaths = self.filepaths
                for untrans_filepath in untrans_filepaths:
                    for col in inventory.keys():
                        if col == 'filepath':
                            inventory[col].append(untrans_filepath)
                        else:
                            inventory[col] = None

            self.dimensions = list(inventory.keys())
            self.inventory = GeoDataFrame(inventory, columns=self.dimensions)
            self.dimensions.remove('filepath')

    def rename_dimensions(self, dimensions_map):
        """
        Renames the dimensions of the datacube.
        :param dimensions_map: dict
            A dictionary representing the relation between old and new dimension names. The keys are the old dimension names,
            the values the new dimension names (e.g., {'time_begin': 'time'}).
        :return: eoDataCube
        """
        for dimension_name in dimensions_map.keys():
            if self.dimensions:
                idx = self.dimensions.index(dimension_name)
                self.dimensions[idx] = dimensions_map[dimension_name]

            if self.inventory:
                self.inventory.rename(columns=dimensions_map)


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
        if self.inventory is not None:
            self.inventory = self.inventory.assign(**{name: GeoSeries(values, index=self.inventory.index)})

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
        pass

    def filter_with_pattern(self, pattern):
        """
        Filter all filepaths according to the given pattern.

        :param pattern: str
            A regex (e.g., ".*S1A.*GRD.*").
        :return: eoDataCube
        """
        pass

    def filter_spatial(self, tilenames=None, roi=None, sref=None, name="tile"):
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
        if roi:
            if self.grid:
                pass
            elif self.inventory is not None:
                self.inventory = self.inventory[self.inventory.intersects(roi)]


    def filter_bands(self, bands):
        """
        Filters the datacube according to the given bands.
        :return: eoDataCube
        """
        pass

    def filter_metadata(self, keys, values):
        """
        Filters the datacube according to given key-value relations in the metadata of the files.
        :return: eoDataCube
        """
        pass

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
        pass
        # dc = self.filter_spatial(tilenames=tilenames, bbox=bbox, geom=geom, sref=sref, name=dimension_name)
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
        pass
        #dc_this, _ = match_dimension(self, dc_other, name)
        #return dc_this


def match_dimension(dc_1, dc_2, name):
    """
    Matches the given datacubes along the specified dimension.
    :param dc_1: eoDataCube
        First datacube.
    :param dc_2: eoDataCube
        Second datacube.
    :return:
    """
    pass

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
    pass




