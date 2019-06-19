import copy
import os
import re
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
# TODO: clone a data cube?
class eoDataCube(object):
    """
    A filename based data cube.
    """
    def __init__(self, filepaths=None, grid=None, dir_tree=None, smart_filename_creator=None, dimensions=None,
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
        self.dir_tree = dir_tree
        self.smart_filename_creator = smart_filename_creator
        self.history = []  # TODO

        # initialise/find filepaths
        self.filepaths = None
        if filepaths:
            self.filepaths = filepaths
        elif dir_tree:
            self.filepaths = dir_tree.file_register
        elif inventory:
            self.filepaths = inventory['filepath']

        # create list of dimensions
        self.dimensions = None
        if dimensions:
            self.dimensions = dimensions
        elif inventory is not None:
            self.dimensions = list(inventory.keys())
            if 'filepath' in self.dimensions:
                self.dimensions.remove('filepath')

        # create inventory from found filepaths
        self.inventory = None
        if inventory is not None:
            self.inventory = inventory
        else:
            self.__inventory_from_filepaths(smart_filename_creator)

        self.grid = None
        if grid:
            self.grid = grid
        elif self.inventory is not None:
            geometries = [self.__geometry_from_file(filepath) for filepath in self.filepaths]
            self.add_dimension('geometry', geometries)

    def __file_type(self, filepath):
        """
        Determines the file type of types understood by yeoda, which are "GeoTiff" and "NetCDF".

        Parameters
        ----------
        filepath: str
            Filepath or filename.

        Returns
        -------
        str
            File type if it is understood by yeoda or None.
        """
        ext = os.path.splitext(filepath)[1]
        if ext in ['.tif', '.tiff']:
            return 'GeoTIFF'
        elif ext in ['.nc']:
            return "NetCDF"
        else:
            return None

    def __geometry_from_file(self, filepath):
        """
        Retrieves boundary geometry from a geospatial file.

        Parameters
        ----------
        filepath: str
            Filepath or filename of a geospatial file (e.g. NetCDF or GEOTiff).

        Returns
        -------
        Shapely Geometry
            Shapely Polygon representing the boundary of the file or None if the file can not be identified.
        """
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
        """
        Creates GeoDataFrame (`inventory`) based on all filepaths.
        Each filepath/filename is translated to a SmartFilename object using a translation function
        `create_smart_filename`.

        Parameters
        ----------
        create_smart_filename: function
            Translates a filepath/filename to a SmartFilename object.

        """
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

    @classmethod
    def from_inventory(cls, inventory, grid=None, dir_tree=None):
        cls(inventory=inventory, grid=grid, dir_tree=dir_tree)

    def rename_dimensions(self, dimensions_map, in_place=False):
        """
        Renames the dimensions of the data cube.

        Parameters
        ----------
        dimensions_map: dict
            A dictionary representing the relation between old and new dimension names. The keys are the old dimension names,
            the values the new dimension names (e.g., {'time_begin': 'time'}).
        in_place: boolean, optional
            If true, the current class instance will be altered.
            If false, a new class instance will be returned (default value is False).

        Returns
        -------
        eoDataCube
            eoDataCube object with renamed dimensions/columns of the inventory.
        """
        dimensions = copy.deepcopy(self.dimensions)
        inventory = self.inventory
        for dimension_name in dimensions_map.keys():
            if dimensions:
                idx = dimensions.index(dimension_name)
                dimensions[idx] = dimensions_map[dimension_name]

            if self.inventory:
                inventory = inventory.rename(columns=dimensions_map)

        if in_place:
            self.inventory = inventory
            self.dimensions = dimensions
            return self
        else:
            return self.from_inventory(inventory=inventory, grid=self.grid, dir_tree=self.dir_tree)

    def add_dimension(self, name, values, in_place=False):
        """
        Adds a new dimension to the data cube.

        Parameters
        ----------
        name: str
            Name of the new dimension
        values: list
            Values of the new dimension (e.g., cloud cover, quality flag, ...).
            They have to have the same length as all the rows in the inventory.
        in_place: boolean, optional
            If true, the current class instance will be altered.
            If false, a new class instance will be returned (default value is False).

        Returns
        -------
        eoDataCube
            eoDataCube object with an additional dimension in the inventory.
        """
        if self.inventory is not None:
            inventory = self.inventory.assign(**{name: GeoSeries(values, index=self.inventory.index)})
            if in_place:
                self.inventory = inventory
                return self
            else:
                return self.from_inventory(inventory=inventory, grid=self.grid, dir_tree=self.dir_tree)

    def filter_by_dimension(self, values, expressions=None, name="time", split=False, in_place=False):
        """
        Filters the data cube according to the given extent and returns a new data cube.

        Parameters
        ----------
        values: tuple or list
            Values of interest to filter, e.g., timestamps: ('2019-01-01', '2019-02-01'), polarisations: ('VV')
        expressions: tuple, list, optional
            Mathematical expressions to filter the data accordingly. If none are given, the exact values from 'values'
            are taken, otherwise the expressions are applied for each value and linked with an AND (e.g., ('>=', '<=')).
            They have to have the same length as 'values'.
        name: str, optional
            Name of the dimension.
        split: boolean, optional
            If true, a list of data cubes will be returned according to the length of the input data
            (i.e. `values` and `expressions`)(default value is False).
        in_place: boolean, optional
            If true, the current class instance will be altered.
            If false, a new class instance will be returned (default value is False).

        Returns
        -------
        eoDataCube or list of eoDataCubes
            If `split` is true and multiple filters are specified, a list of eoDataCube objects will be returned.
            If not, the inventory of the eoDataCube is filtered.
        """
        pass


    def filter_files_with_pattern(self, pattern, full_path=False, in_place=False):
        """
        Filters all filepaths according to the given pattern.

        Parameters
        ----------
        pattern: str
            A regular expression (e.g., ".*S1A.*GRD.*").
        full_path: boolean, optional
            Uses the full filepaths for filtering if it is set to True.
            Otherwise the filename is used (default value is False).
        in_place: boolean, optional
            If true, the current class instance will be altered.
            If false, a new class instance will be returned (default value is False).

        Returns
        -------
        eoDataCube
            eoDataCube object with a filtered inventory according to the given pattern.
        """
        pass

    # TODO: also allow shapefiles and more complex geometries
    def filter_spatially(self, tilenames=None, roi=None, sref=None, name="tile", in_place=False):
        """
        Spatially filters the data cube by tile names, a bounding box and/or a geometry.

        Parameters
        ----------
        tilenames: list of str, optional
            Tile names corresponding to the given grid and inventory.
        roi: OGR Geometry, Shapely Geometry or list, optional
            A geometry defining the region of interest. If it is of type list representing the extent
            (i.e. [x_min, y_min, x_max, y_max]), `sref` has to be given to transform the extent into a
            georeferenced polygon.
        sref: osr.SpatialReference, optional
            Spatial reference of the given region of interest `roi`.
        name: str, optional
            Name of the tile/spatial dimension in the inventory.
        in_place: boolean, optional
            If true, the current class instance will be altered.
            If false, a new class instance will be returned (default value is False).

        Returns
        -------
        eoDataCube
            eoDataCube object with a filtered inventory according to the given region of interest `roi` or tile names
            `tilenames`.
        """
        if roi:
            if self.grid:
                pass
            elif self.inventory is not None:
                inventory = self.inventory[self.inventory.intersects(roi)]
                if in_place:
                    self.inventory = inventory
                    return self
                else:
                    return self.from_inventory(inventory=inventory, grid=self.grid, dir_tree=self.dir_tree)



    def filter_by_metadata(self, keys, values):
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

    def __deepcopy__(self, memodict={}):
        filepaths = copy.deepcopy(self.filepaths)
        grid = copy.deepcopy(self.grid)
        dir_tree = copy.deepcopy(self.dir_tree)
        smart_filename_creator = self.smart_filename_creator
        dimensions = copy.deepcopy(self.dimensions)
        inventory = copy.deepcopy(self.inventory)

        return eoDataCube(filepaths=filepaths, grid=grid, dir_tree=dir_tree,
                          smart_filename_creator=smart_filename_creator, dimensions=dimensions,
                          inventory=inventory)

    def clone(self):
        return copy.deepcopy(self)


def match_dimension(dc_1, dc_2, name):
    """
    Matches the given data cubes along the specified dimension.

    Parameters
    ----------
    dc_1: eoDataCube
        First datacube.
    dc_2: eoDataCube
        Second datacube.

    Returns
    -------
    eoDataCube
            New eoDataCube object with a merged inventory from both input data cubes.
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




