# general packages
import copy
import os
import re
import itertools
import pandas as pd
import numpy as np
from collections import OrderedDict

# geo packages
from shapely.wkt import loads
from osgeo import osr
from geopandas import GeoSeries
from geopandas import GeoDataFrame
import shapely.wkt
from shapely.geometry import Point
import pytileproj.geometry as geometry
from pyraster.geotiff import GeoTiffFile
from pyraster.netcdf import NcFile
from pyraster.gdalport import GdalImage
from pyraster.timestack import GeoTiffRasterTimeStack
from yeoda.utils import get_file_type, any_geom2ogr_geom, xy2ij, ij2xy

# error packages
from yeoda.errors import IOClassNotFound
from yeoda.errors import TileNotAvailable


class EODataCube(object):
    """
    A file(name) based data cube for EO data.
    """
    def __init__(self, filepaths=None, grid=None, dir_tree=None, smart_filename_creator=None, dimensions=None,
                 inventory=None, io_map=None, io_md_map=None, ignore_metadata=True):
        """
        Constructor of the eoDataCube class.

        Parameters
        ----------
        filepaths: list of str, optional
            List of filepaths.
        grid: pytileproj.base.TiledProjection, optional
            Tiled projection/grid object/class (e.g. `Equi7Grid`, `LatLonGrid`).
        dir_tree: SmartTree, optional
            Directory tree class managing folders and files.
        smart_filename_creator: function, optional
            A function that allows to create a `SmartFilename` instance from a filepath.
        dimensions: list of str, optional
            List of filename parts to use as dimensions. The strings have to match with the keys of the `SmartFilename`
            fields definition.
        inventory: GeoDataFrame, optional
            Contains information about the dimensions (columns) and each filepath (rows).
            If `grid` is not specified, a `Shapely` geometry object is added to the `GeoDataFrame`.
        io_map: dictionary, optional
            Map that represents the relation of an EO file type (e.g. GeoTiff) with an appropriate reader
            (e.g. `GeoTiffFile`).
        io_md_map: dictionary, optional
            Map that represents the relation of an EO file type (e.g. GeoTiff) with a metadata tag - dimension mapping
            of the files, e.g. {1: 'SSM', 2: 'SSM-NOISE'}
        """
        # initialise simple class variables
        self.dir_tree = dir_tree
        self.smart_filename_creator = smart_filename_creator

        # initialise IO classes responsible for reading and writing
        if io_map is not None:
            self.io_map = io_map
        else:
            self.io_map = {'GeoTIFF': GeoTiffFileExt, 'NetCDF': NcFileExt}

        # initialise IO classes metadata translation
        if io_md_map is not None:
            self.io_md_map = io_md_map
        else:
            self.io_md_map = {'GeoTIFF': {'band': None}, 'NetCDF': {'time': ['t', 'time'],
                                                                    'var_name': ['var_name']}}

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
            self.__inventory_from_filepaths(smart_filename_creator, ignore_metadata)

        self.grid = None
        if grid:
            self.grid = grid
        elif (self.inventory is not None) and (self.filepaths is not None) and ('geometry' not in self.inventory.keys()):
            geometries = [self.__geometry_from_file(filepath) for filepath in self.filepaths]
            self.add_dimension('geometry', geometries, in_place=True)

    @classmethod
    def from_inventory(cls, inventory, grid=None, dir_tree=None):
        """
        Creates an EODataCube instance from a given inventory.

        Parameters
        ----------
        inventory: GeoDataFrame
            Contains information about the dimensions (columns) and each filepath (rows).
        grid: pytileproj.base.TiledProjection, optional
            Tiled projection/grid object/class (e.g. `Equi7Grid`, `LatLonGrid`).
        dir_tree: SmartTree, optional
            Directory tree class managing folders and files.

        Returns
        -------
        EODataCube
            Data cube consisting of data stored in `inventory`.
        """

        return cls(inventory=inventory, grid=grid, dir_tree=dir_tree)

    def rename_dimensions(self, dimensions_map, in_place=False):
        """
        Renames the dimensions of the data cube.

        Parameters
        ----------
        dimensions_map: dict
            A dictionary representing the relation between old and new dimension names. The keys are the old dimension
            names, the values the new dimension names (e.g., {'time_begin': 'time'}).
        in_place: boolean, optional
            If true, the current class instance will be altered.
            If false, a new class instance will be returned (default value is False).

        Returns
        -------
        EODataCube
            EODataCube object with renamed dimensions/columns of the inventory.
        """

        dimensions = copy.deepcopy(self.dimensions)
        inventory = self.inventory
        for dimension_name in dimensions_map.keys():
            if dimensions:
                idx = dimensions.index(dimension_name)
                dimensions[idx] = dimensions_map[dimension_name]

        if self.inventory is not None:
            inventory = inventory.rename(columns=dimensions_map)

        if in_place:
            self.dimensions = dimensions

        return self.__assign_inventory(inventory, in_place=in_place)

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
        EODataCube
            EODataCube object with an additional dimension in the inventory.
        """

        if self.inventory is not None:
            inventory = self.inventory.assign(**{name: GeoSeries(values, index=self.inventory.index)})
            return self.__assign_inventory(inventory, in_place=in_place)

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
        EODataCube
            EODataCube object with a filtered inventory according to the given pattern.
        """

        if self.inventory:
            filepaths = self.inventory['filepath']
            pattern = re.compile(pattern)
            if not full_path:
                file_filter = lambda x: re.match(os.path.basename(x), pattern)
            else:
                file_filter = lambda x: re.match(x, pattern)
            idx_filter = [file_filter(filepath) for filepath in filepaths]
            inventory = self.inventory[idx_filter]
            return self.__assign_inventory(inventory, in_place=in_place)
        else:
            return None

    def filter_by_metadata(self, metadata, in_place=False):
        """
        Filters all filepaths according to the given metadata.

        Parameters
        ----------
        metadata: dict
            Key value relationships being expected to be in the metadata.
        in_place: boolean, optional
            If true, the current class instance will be altered.
            If false, a new class instance will be returned (default value is False).

        Returns
        -------
        EODataCube
            EODataCube object with a filtered inventory according to the given metadata.
        """

        bool_filter = []
        filepaths = self.inventory['filepath']  # use inventory to ensure the same order
        for filepath in filepaths:
            io_class = self.__io_class(get_file_type(filepath))
            ds = io_class.src  # IO class has to have a "src" class variable which is a pointer to the data set

            select = False
            if ds:
                ds_metadata = ds.get_metadata()
                select = True
                for key, value in metadata.items():
                    if key not in ds_metadata.keys():
                        select = False
                    else:
                        if ds_metadata[key] != value:
                            select = False

            bool_filter.append(select)

        inventory = self.inventory[bool_filter]
        return self.__assign_inventory(inventory, in_place=in_place)

    def sort_by_dimension(self, name, ascending=True, in_place=False):
        """
        Sorts the data cube/inventory according to the given dimension.

        Parameters
        ----------
        name: str
            Name of the dimension.
        ascending: bool, optional
            If true, sorts in ascending order, otherwise in descending order.
        in_place: boolean, optional
            If true, the current class instance will be altered.
            If false, a new class instance will be returned (default value is False).

        Returns
        -------
        EODataCube
            Sorted EODataCube object.
        """

        inventory = copy.deepcopy(self.inventory)
        inventory_sorted = inventory.sort_values(by=name, ascending=ascending)

        if in_place:
            self.inventory = inventory_sorted
            return self
        else:
            return self.from_inventory(inventory=inventory_sorted, grid=self.grid, dir_tree=self.dir_tree)

    def filter_by_dimension(self, values, expressions=None, name="time", in_place=False):
        """
        Filters the data cube according to the given extents and returns a (new) data cube.

        Parameters
        ----------
        values: list, tuple, list of tuples or list of lists
            Values of interest to filter, e.g., timestamps: ('2019-01-01', '2019-02-01'), polarisations: ('VV')
        expressions: list, tuple, list of tuples or list of lists, optional
            Mathematical expressions to filter the data accordingly. If none are given, the exact values from 'values'
            are taken, otherwise the expressions are applied for each value and linked with an AND (e.g., ('>=', '<=')).
            They have to have the same length as 'values'.
        name: str, optional
            Name of the dimension.
        in_place: boolean, optional
            If true, the current class instance will be altered.
            If false, a new class instance will be returned (default value is False).

        Returns
        -------
        EODataCube
            Filtered EODataCube object.
        """

        return self.__filter_by_dimension(values, expressions=expressions, name=name, in_place=in_place, split=False)

    def split_by_dimension(self, values, expressions=None, name="time"):
        """
        Splits the data cube according to the given extents and returns a list of data cubes.

        Parameters
        ----------
        values: list, tuple, list of tuples or list of lists
            Values of interest to filter, e.g., timestamps: ('2019-01-01', '2019-02-01'), polarisations: ('VV')
        expressions: list, tuple, list of tuples or list of lists, optional
            Mathematical expressions to filter the data accordingly. If none are given, the exact values from 'values'
            are taken, otherwise the expressions are applied for each value and linked with an AND (e.g., ('>=', '<=')).
            They have to have the same length as 'values'.
        name: str, optional
            Name of the dimension.

        Returns
        -------
        List of EODataCube objects.
        """

        return self.__filter_by_dimension(values, expressions=expressions, name=name, split=True)

    def filter_spatially_by_tilename(self, tilenames, dimension_name="tile", in_place=False, use_grid=True):
        """
        Spatially filters the data cube by tile names.

        Parameters
        ----------
        tilenames: list of str
            Tile names corresponding to a grid and/or the inventory.
        dimension_name: str, optional
            Name of the tile/spatial dimension in the inventory.
        in_place: boolean, optional
            If true, the current class instance will be altered.
            If false, a new class instance will be returned (default value is False).
        use_grid: bool
            If true, the given tilenames are compared with the filenames defined by the grid.
            If false, pre-defined grid information is ignored.

        Returns
        -------
        EODataCube
            EODataCube object with a filtered inventory according to the given tile names.
        """

        if not isinstance(tilenames, (tuple, list)):
            tilenames = [tilenames]

        if use_grid:
            if self.grid is not None:
                available_tilenames = self.grid.tilesys.list_tiles_covering_land()
                for tilename in tilenames:
                    if tilename not in available_tilenames:
                        raise TileNotAvailable(tilename)
                return self.filter_by_dimension(tilenames, name=dimension_name, in_place=in_place)
            else:
                print('No grid is provided to extract tile information.')
                return self
        else:
            return self.filter_by_dimension(tilenames, name=dimension_name, in_place=in_place)

    def filter_spatially_by_geom(self, geom, sref=None, dimension_name="tile", in_place=False):
        """
        Spatially filters the data cube by a bounding box or a geometry.

        Parameters
        ----------
        geom: OGR Geometry or Shapely Geometry or list or tuple, optional
            A geometry defining the region of interest. If it is of type list/tuple representing the extent
            (i.e. [x_min, y_min, x_max, y_max]), `sref` has to be given to transform the extent into a
            georeferenced polygon.
        sref: osr.SpatialReference, optional
            Spatial reference of the given region of interest `geom`.
        dimension_name: str, optional
            Name of the tile/spatial dimension in the inventory.
        in_place: boolean, optional
            If true, the current class instance will be altered.
            If false, a new class instance will be returned (default value is False).

        Returns
        -------
        EODataCube
            EODataCube object with a filtered inventory according to the given region of interest `geom`.
        """

        geom_roi = any_geom2ogr_geom(geom, osr_spref=sref)

        if self.grid:
            tilenames = self.grid.search_tiles_in_roi(geom_area=geom_roi)
            self.filter_spatially_by_tilename(tilenames, dimension_name=dimension_name, in_place=in_place,
                                              use_grid=False)
        elif self.inventory is not None and 'geometry' in self.inventory.keys():
            geom_roi = shapely.wkt.loads(geom_roi.ExportToWkt())
            inventory = self.inventory[self.inventory.intersects(geom_roi)]
            return self.__assign_inventory(inventory, in_place=in_place)

    def get_monthly_dcs(self, name='time', months=None):
        """
        Separates the data cube into months.

        Parameters
        ----------
        name: str, optional
            Name of the dimension.
        months: list of integers
            List of integers specifying the months to select/split, i.e. each value is allowed to be in between 1-12.

        Returns
        -------
        List of monthly EODataCube objects.
        """

        sort = False
        yearly_eodcs = self.get_yearly_dcs(name=name)
        monthly_eodcs = []
        for yearly_eodc in yearly_eodcs:
            if months:
                # initialise empty dict keeping track of the months
                timestamps_months = {}
                for month in months:
                    timestamps_months[month] = []

                for timestamp in yearly_eodc.inventory[name]:
                    if timestamp.month in months:
                        timestamps_months[timestamp.month].append(timestamp)
            else:
                sort = True
                timestamps_months = {}
                for timestamp in yearly_eodc.inventory[name]:
                    if timestamp.month not in timestamps_months.keys():
                        timestamps_months[timestamp.month] = []

                    timestamps_months[timestamp.month].append(timestamp)

                months = timestamps_months.keys()

            if sort:
                months = sorted(months)  # sort in ascending order
            values = []
            expressions = [(">=", "<=")] * len(months)
            for month in months:
                min_timestamp = min(timestamps_months[month])
                max_timestamp = max(timestamps_months[month])
                values.append((min_timestamp, max_timestamp))

            monthly_eodcs.extend(self.split_by_dimension(values, expressions, name=name))

        return monthly_eodcs

    def get_yearly_dcs(self, name='time', years=None):
        """
        Separates the data cube into years.

        Parameters
        ----------
        name: str, optional
            Name of the dimension.
        years: list of integers
            List of integers specifying the years to select/split.

        Returns
        -------
        List of yearly EODataCube objects.
        """

        sort = False
        if years:
            # initialise empty dict keeping track of the years
            timestamps_years = {}
            for year in years:
                timestamps_years[year] = []

            for timestamp in self.inventory[name]:
                if timestamp.year in years:
                    timestamps_years[timestamp.year].append(timestamp)
        else:
            sort = True
            timestamps_years = {}
            for timestamp in self.inventory[name]:
                if timestamp.year not in timestamps_years.keys():
                    timestamps_years[timestamp.year] = []

                timestamps_years[timestamp.year].append(timestamp)

            years = timestamps_years.keys()

        if sort:
            years = sorted(years)  # sort in ascending order
        values = []
        expressions = [(">=", "<=")]*len(years)
        for year in years:
            min_timestamp = min(timestamps_years[year])
            max_timestamp = max(timestamps_years[year])
            values.append((min_timestamp, max_timestamp))

        return self.split_by_dimension(values, expressions, name=name)

    def load_by_geom(self, geom, sref=None, name="tile"):
        """
        Loads data as an array.

        Parameters
        ----------
        geom: OGR Geometry or Shapely Geometry or list or tuple, optional
            A geometry defining the region of interest. If it is of type list/tuple representing the extent
            (i.e. [x_min, y_min, x_max, y_max]), `sref` has to be given to transform the extent into a
            georeferenced polygon.
        sref: osr.SpatialReference, optional
            Spatial reference of the given region of interest `geom`.
        name: str, optional
            Name of the spatial dimension (default: "tile").

        Returns
        -------
        numpy, dask or xarray array
            Data as an array-like object.
        """

        geom_roi = any_geom2ogr_geom(geom, osr_spref=sref)

        if 'band' in self.dimensions:
            band = int(list(set(self.inventory['band']))[0])
            # filter datacube to only keep the remaining bands
            self.filter_by_dimension([band], name='band', in_place=True)
        else:
            band = 1

        if self.grid:
            if sref is not None:
                tar_spref = self.grid.core.projection.osr_spref
                geom_roi = geometry.transform_geometry(geom_roi, tar_spref)

            extent = geometry.get_geom_boundaries(geom_roi)
            tilenames = list(self.inventory[name])
            if len(list(set(tilenames))) > 1:
                raise Exception('Data can be loaded only from one tile. Please filter the data cube before.')
            tilename = tilenames[0]
            i_min, j_min = self.grid.tilesys.create_tile(name=tilename).xy2ij(extent[0], extent[3])
            i_max, j_max = self.grid.tilesys.create_tile(name=tilename).xy2ij(extent[1], extent[2])
            inv_traffo_fun = lambda i, j: self.grid.tilesys.create_tile(name=tilename).ij2xy(i, j)
        else:
            filepath = self.inventory['filepath'][0]
            io_class = self.__io_class(get_file_type(filepath))
            ds = io_class.src
            gdal_img = GdalImage(ds, filepath)
            if sref is not None:
                tar_spref = osr.SpatialReference()
                tar_spref.ImportFromWkt(gdal_img.projection())
                geom_roi = geometry.transform_geometry(geom_roi, tar_spref)
            extent = geometry.get_geom_boundaries(geom_roi)
            gt = gdal_img.geotransform()
            i_min, j_max = xy2ij(extent[0], extent[3], gt)
            i_max, j_min = xy2ij(extent[1], extent[2], gt)
            inv_traffo_fun = lambda i, j: ij2xy(i, j, gt)

        file_ts = {'filenames': list(self.inventory['filepath'])}
        gt_timestack = GeoTiffRasterTimeStack(file_ts=file_ts, file_band=band)
        row_size = j_max - j_min + 1
        col_size = i_max - i_min + 1
        data = gt_timestack.read_ts(i_min, j_min, col_size=col_size, row_size=row_size, decode_func=self.decode)

        geom_roi = shapely.wkt.loads(geom_roi.ExportToWkt())
        data_mask = np.ones((row_size, col_size))
        for i in range(col_size):
            for j in range(row_size):
                x_i, y_j = inv_traffo_fun(i_min + i, j_min + j)
                point = Point(x_i, y_j)
                if point.within(geom_roi):
                    data_mask[j, i] = 0

        data_mask = data_mask.astype(bool)
        data_mask = np.broadcast_to(data_mask, data.shape)
        data = data.astype(float)
        data[data_mask] = np.nan

        return data

    def load_by_coord(self, x, y, sref=None, name="tile"):
        """
        Loads data as a 1-D array according to a given coordinate.

        Parameters
        ----------
        x: float
            World system coordinate in X direction.
        y: float
            World system coordinate in Y direction.
        sref: osr.SpatialReference, optional
            Spatial reference referring to the world system coordinates `x` and `y`.
        name: str, optional
            Name of the spatial dimension (default: "tile").

        Returns
        -------
        numpy, dask or xarray array
            Data as an 1-D array-like object.
        """

        if 'band' in self.dimensions:
            band = int(list(set(self.inventory['band']))[0])
            # filter datacube to only keep the remaining bands
            self.filter_by_dimension([band], name='band', in_place=True)
        else:
            band = 1

        if self.grid:
            if sref is not None:
                tar_spref = self.grid.core.projection.osr_spref
                x, y = geometry.uv2xy(x, y, sref, tar_spref)
            tilenames = list(self.inventory[name])
            if len(list(set(tilenames))) > 1:
                raise Exception('Data can be loaded only from one tile. Please filter the data cube before.')
            tilename = tilenames[0]
            i, j = self.grid.tilesys.create_tile(name=tilename).xy2ij(x, y)
        else:
            filepath = self.inventory['filepath'][0]
            io_class = self.__io_class(get_file_type(filepath))
            ds = io_class.src
            gdal_img = GdalImage(ds, filepath)
            if sref is not None:
                tar_spref = osr.SpatialReference()
                tar_spref.ImportFromWkt(gdal_img.projection())
                x, y = geometry.uv2xy(x, y, sref, tar_spref)
            gt = gdal_img.geotransform()
            i, j = xy2ij(x, y, gt)

        file_ts = {'filenames': list(self.inventory['filepath'])}
        gt_timestack = GeoTiffRasterTimeStack(file_ts=file_ts, file_band=band)

        return gt_timestack.read_ts(i, j, decode_func=self.decode).flatten()

    def encode(self, data):
        """
        Encodes an array.

        Parameters
        ----------
        data: numpy, dask or xarray array
            Data array.

        Returns
        -------
        data: numpy, dask or xarray array
            Encoded data.
        """

        return data

    def decode(self, data):
        """
        Decodes an encoded array to retrieve the values in native units.

        Parameters
        ----------
        data: numpy, dask or xarray array
            Encoded array.

        Returns
        -------
        data: numpy, dask or xarray array
            Decoded data (original/native values).
        """
        return data

    def merge(self, dc_other, name=None, in_place=False):
        """
        Merges this data cube with respect to another data cube along the specified dimension.

        Parameters
        ----------
        dc_other: EODataCube
            Data cube to merge with.
        name: str, optional
            Name of the dimension.
        in_place: boolean, optional
            If true, the current class instance will be altered.
            If false, a new class instance will be returned (default value is False).

        Returns
        -------
        EODataCube
            Merged data cube.
        """

        dc_merged = merge_datacubes([self, dc_other], name=name)
        return self.__assign_inventory(dc_merged.inventory, in_place=in_place)

    def match_dimension(self, dc_other, name, in_place=False):
        """
        Matches this data cube with another data cubes along the specified dimension `name`.

        Parameters
        ----------
        dc_other: EODataCube
            Data cube to match dimensions with.
        name: str
            Name of the dimension, which is used for aligning/filtering the values for all data cubes.
        in_place: boolean, optional
            If true, the current class instance will be altered.
            If false, a new class instance will be returned (default value is False).

        Returns
        -------
        EODataCube
            Data cube with common values along the given dimension with respect to another data cube.
        """

        dc_matched = match_dimension([self, dc_other], name, in_place=in_place)[0]
        return self.__assign_inventory(dc_matched.inventory, in_place=in_place)

    def clone(self):
        """
        Clones, i.e. deepcopies a data cube.

        Returns
        -------
        EODataCube
            Cloned/copied data cube.
        """

        return copy.deepcopy(self)

    def __io_class(self, file_type):
        """
        Looks up appropriate file handler/IO class for a given file type.

        Parameters
        ----------
        file_type: str
            File type, e.g. "GeoTIFF".

        Returns
        -------
        object
            File handler to read and write EO data, e.g. "GeoTiffFile".

        Raises
        ------
        IOClassNotFound
            File handler for a given file type was not found.
        """

        if file_type not in self.io_map.keys():
            raise IOClassNotFound(self.io_map, file_type)
        else:
            return self.io_map[file_type]

    def __io_md_map(self, file_type):
        """
        Looks up appropriate metadata structure of a IO class for a given file type.

        Parameters
        ----------
        file_type: str
            File type, e.g. "GeoTIFF".

        Returns
        -------
        dict
            Metadata map to rename/redefine metadata entries of an IO class.

        Raises
        ------
        IOClassNotFound
            File handler for a given file type was not found.
        """

        if file_type not in self.io_md_map.keys():
            raise IOClassNotFound(self.io_md_map, file_type)
        else:
            return self.io_md_map[file_type]

    def __geometry_from_file(self, filepath):
        """
        Retrieves boundary geometry from an EO file.

        Parameters
        ----------
        filepath: str
            Filepath or filename of a geospatial file (e.g. NetCDF or GeoTIFF).

        Returns
        -------
        shapely.geometry
            Shapely polygon representing the boundary of the file or `None` if the file can not be identified.
        """

        file_type = get_file_type(filepath)
        io_class = self.__io_class(file_type)
        ds = io_class(filepath).src

        if ds:
            gdal_img = GdalImage(ds, filepath)
            boundary_extent = gdal_img.get_extent()
            boundary_spref = osr.SpatialReference()
            boundary_spref.ImportFromWkt(gdal_img.projection())
            boundary_geom = geometry.extent2polygon(boundary_extent, boundary_spref)  # TODO: directly convert it to shapely geometry
            return loads(boundary_geom.ExportToWkt())
        else:
            return

    def __inventory_from_filepaths(self, smart_filename_creator=None, ignore_metadata=True):
        """
        Creates GeoDataFrame (`inventory`) based on all filepaths.
        Each filepath/filename is translated to a SmartFilename object using a translation function
        `smart_filename_creator`.

        Parameters
        ----------
        smart_filename_creator: function
            Translates a filepath/filename to a SmartFilename object.
        """

        inventory = OrderedDict()
        inventory['filepath'] = []

        # fill inventory
        if self.filepaths:
            for filepath in self.filepaths:
                n = len(inventory['filepath'])
                local_inventory = OrderedDict()
                local_inventory['filepath'] = [filepath]

                # get information from filename
                smart_filename = None
                try:
                    smart_filename = smart_filename_creator(os.path.basename(filepath), convert=True)
                except:
                    pass

                if smart_filename:
                    if self.dimensions:
                        for dimension in self.dimensions:
                            try:
                                local_inventory[dimension] = [smart_filename[dimension]]
                            except:
                                pass
                    else:
                        for key, value in smart_filename.fields.items():
                            local_inventory[key] = [value]

                if not ignore_metadata:  # get information from data set metadata
                    file_type = get_file_type(filepath)
                    io_class = self.__io_class(file_type)
                    io_md_map = self.__io_md_map(file_type)
                    metadata = io_class(filepath).get_dimensions_info(map=io_md_map[file_type])
                    for key, value in metadata.items():
                        local_inventory[key] = value

                    # add global inventory keys to local inventory if they are not available locally
                    for key in inventory.keys():
                        if key not in local_inventory.keys():
                            local_inventory[key] = None

                    entries = tuple(list(local_inventory.values()))
                    extended_entries = list(itertools.product(*entries))
                else:
                    extended_entries = list(local_inventory.values())
                    extended_entries = list(map(list, zip(*extended_entries)))

                # add local inventory keys to global inventory if they are not available globally
                for key in local_inventory.keys():
                    if key not in inventory.keys():
                        if n == 0:  # first time
                            inventory[key] = []
                        else:
                            inventory[key] = [None] * n

                for entry in extended_entries:
                    for i, key in enumerate(inventory.keys()):
                        inventory[key].append(entry[i])

            self.dimensions = list(inventory.keys())
            self.inventory = GeoDataFrame(inventory, columns=self.dimensions)
            self.dimensions.remove('filepath')

    def __filter_by_dimension(self, values, expressions=None, name="time", split=False, in_place=False):
        """
        Filters the data cube according to the given extent and returns a new data cube.

        Parameters
        ----------
        values: list, tuple, list of tuples or list of lists
            Values of interest to filter, e.g., timestamps: ('2019-01-01', '2019-02-01'), polarisations: ('VV')
        expressions: list, tuple, list of tuples or list of lists, optional
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
        EODataCube or list of EODataCubes
            If `split` is true and multiple filters are specified, a list of EODataCube objects will be returned.
            If not, the inventory of the EODataCube is filtered.
        """

        if not isinstance(values, list):
            values = list(values)
        n_filters = len(values)
        if expressions is None:  # equal operator is the default comparison operator
            expressions = ["=="] * n_filters
        else:
            if not isinstance(expressions, list):
                values = list(expressions)

        inventory = copy.deepcopy(self.inventory)
        filtered_inventories = []
        for i in range(n_filters):
            value = values[i]
            expression = expressions[i]
            if not isinstance(value, (tuple, list)):
                value = [value]

            if not isinstance(expression, (tuple, list)):
                expression = [expression]

            if (len(value) == 2) and (len(expression) == 2):
                filter_cmd = "inventory[(inventory[name] {} value[0]) & " \
                             "(inventory[name] {} value[1])]".format(expression[0], expression[1])
            elif (len(value) == 1) and (len(expression) == 1):
                filter_cmd = "inventory[inventory[name] {} value[0]]".format(expression[0])
            else:
                raise Exception('Length of value (={}) and length of expression (={}) does not match or is larger than 2.'.format(len(value), len(expression)))

            filtered_inventories.append(eval(filter_cmd))

        if split:
            eodcs = [self.from_inventory(filtered_inventory, grid=self.grid, dir_tree=self.dir_tree)
                     for filtered_inventory in filtered_inventories]
            return eodcs
        else:
            filtered_inventory = pd.concat(filtered_inventories, ignore_index=True)
            return self.__assign_inventory(filtered_inventory, in_place=in_place)

    def __assign_inventory(self, inventory, in_place=True):
        """
        Helper method for either create a new data cube or overwrite the old data cube with the given inventory.

        Parameters
        ----------
        inventory: GeoDataFrame
            Data cube inventory.
        in_place: boolean, optional
            If true, the current class instance will be altered.
            If false, a new class instance will be returned (default value is False).

        Returns
        -------
        EODataCube
        """

        if in_place:
            self.inventory = inventory
            return self
        else:
            return self.from_inventory(inventory=inventory, grid=self.grid, dir_tree=self.dir_tree)

    def __deepcopy__(self, memodict={}):
        """
        Deepcopy method of the EODataCube class.

        Parameters
        ----------
        memodict: dict, optional

        Returns
        -------
        EODataCube
            Deepcopy of a data cube.
        """

        filepaths = copy.deepcopy(self.filepaths)
        grid = copy.deepcopy(self.grid)
        dir_tree = copy.deepcopy(self.dir_tree)
        smart_filename_creator = self.smart_filename_creator
        dimensions = copy.deepcopy(self.dimensions)
        inventory = copy.deepcopy(self.inventory)

        return EODataCube(filepaths=filepaths, grid=grid, dir_tree=dir_tree,
                          smart_filename_creator=smart_filename_creator, dimensions=dimensions,
                          inventory=inventory)


def match_dimension(dcs, name, in_place=False):
    """
    Matches the given data cubes along the specified dimension `name`.

    Parameters
    ----------
    dcs: list of EODataCube's
       List of data cubes.
    name: str
        Name of the dimension, which is used for aligning/filtering the values for all data cubes.
    in_place: boolean, optional
            If true, the current class instance will be altered.
            If false, a new class instance will be returned (default value is False).

    Returns
    -------
    list of EODataCube objects
        List of data cubes having the same length as the input list.
        Each value matches all data cubes along the given dimension.
    """

    values = []
    for dc in dcs:
        values.extend(dc.inventory[name])

    unique_values = pd.unique(values)

    matched_dcs = []
    for dc in dcs:
        expressions = [('==')]*len(unique_values)
        matched_dcs.append(dc.filter_by_dimension(unique_values, expressions=expressions, name=name, in_place=in_place))

    return matched_dcs


def merge_datacubes(dcs, name=None):
    """
    Merges data cubes into one data cube. By doing so, duplicates are removed and only
    common dimensions are kept.

    Parameters
    ----------
    dcs: list of EODataCube objects
       List of data cubes, which should be united based on the common set of dimensions.
    name: str
        Name of the dimension, which is used for aligning/filtering the values for all data cubes.

    Returns
    -------
    EODataCube
        Data cube containing all information of the given data cubes except duplicates and
        inconsistent dimensions.
    """

    merged_inventory = dcs[0].inventory
    dcs = dcs[1:]
    for dc in dcs:
        merged_inventory = merged_inventory.merge(dc.inventory, on=name)

    dc_merged = EODataCube.from_inventory(merged_inventory, grid=dcs[0].grid, dir_tree=dcs[0].dir_tree)

    return dc_merged


# extend pyraster base readers by certain function needed for yeoda. This part will be removed when veranda is ready.
class GeoTiffFileExt(GeoTiffFile):
    """
    Extended pyraster class GeoTiffFile by a metadata interpreter function.
    """

    def __init__(self, filename, mode='r', count=None, compression='LZW',
                 blockxsize=512, blockysize=512,
                 geotransform=(0, 1, 0, 0, 0, 1), spatialref=None, tags=None,
                 overwrite=True, gdal_opt=None):
        """
        Constructor of class GeoTiffFileExt.

        GDAL wrapper for reading and writing raster files. A tiled (not stripped)
        GeoTiff file is created.

        Parameters
        ----------
        filename : str
            File name.
        mode : str, optional
            File opening mode ('r' read, 'w' write). Default: read
        count : int, required if data is 2d
            Number of bands. Default: Not defined and raster data set
            assumed to be 3d [band, ncol, nrow], but if data is written for
            each band separately count needs to be set to the number of bands.
        compression : str or int, optional
            Set the compression to use. LZW ('LZW') and DEFLATE (a number between
            0 and 9) compressions can be used (default 'LZW').
        blockxsize : int, optional
            Set the block size for x dimension (default: 512).
        blockysize : int, optional
            Set the block size for y dimension (default: 512).
        geotransform : tuple or list, optional
            Geotransform parameters (default (0, 1, 0, 0, 0, 1)).
            0: Top left x
            1: W-E pixel resolution
            2: Rotation, 0 if image is "north up"
            3: Top left y
            4: Rotation, 0 if image is "north up"
            5: N-S pixel resolution (negative value if North up)
        spatialref : str, optional
            Coordinate Reference System (CRS) in Wkt form (default: None).
        tags : dict, optional
            Meta data tags (default: None).
        overwrite : boolean, optional
            Flag if file can be overwritten if it already exists (default: True).
        gdal_opt : dict, optional
            Driver specific control parameters (default: None).
        """

        super().__init__(filename, mode=mode, count=count, compression=compression,
                 blockxsize=blockxsize, blockysize=blockysize,
                 geotransform=geotransform, spatialref=spatialref, tags=tags,
                 overwrite=overwrite, gdal_opt=gdal_opt)

    def get_dimensions_info(self, map=None):
        """
        Remaps/retrieves specific metadata attributes.

        Parameters
        ----------
        map: dict, optional
            Metadata mapping dict used for retrieving/renaming/interpreting metadata entries, e.g. {'band': {1: 'SSM'}}.

        Returns
        -------
        metadata: dict
            Renamed/retrieved metadata dictionary of the data set.
        """

        if map is None:
            map = {'band': None}

        if len(map.keys()) != 1:
            raise Exception('You are only allowed to specify one map.')

        map_key = list(map.keys())[0]
        metadata = dict()

        bands = list(range(self.src.RasterCount))
        src_metadata = self.src.GetMetadata()
        band_labels = []
        if map[map_key]:
            for band in bands:
                if band in map[map_key].keys():
                    if map[map_key][band] in src_metadata.keys():
                        band_labels.append(src_metadata[map[map_key][band]])
                    else:
                        info_msg = "For band '{}' no metadata key '{}' could be found.".format(band, map[map_key][band])
                        print(info_msg)
                        band_labels.append(map[map_key][band])
                else:
                    band_labels.append(str(band))

        metadata[map_key] = band_labels

        return metadata


class NcFileExt(NcFile):
    """
    Extended pyraster class NcFile by a metadata interpreter function.
    """

    def __init__(self, filename, mode='r', complevel=2, zlib=True,
                 geotransform=(0, 1, 0, 0, 0, 1), spatialref=None,
                 overwrite=True, nc_format="NETCDF4_CLASSIC", chunksizes=None,
                 time_units="days since 1900-01-01 00:00:00",
                 var_chunk_cache=None, auto_scale=True):

        """
        Constructor of class NcFileExt.

        Wrapper for reading and writing netCDF4 files. It will create three
        predefined dimensions (time, x, y), with time as an unlimited dimension
        and x, y are defined by the shape of the data.

        The arrays to be written should have the following dimensions: time, x, y

        Parameters
        ----------
        filename : str
            File name.
        mode : str, optional
            File opening mode. Default: 'r' = xarray.open_dataset
            Other modes:
                'r'        ... reading with xarray.open_dataset
                'r_xarray' ... reading with xarray.open_dataset
                'r_netcdf' ... reading with netCDF4.Dataset
                'w'        ... writing with netCDF4.Dataset
                'a'        ... writing with netCDF4.Dataset
        complevel : int, optional
            Compression level (default 2)
        zlib : bool, optional
            If the optional keyword zlib is True, the data will be compressed
            in the netCDF file using gzip compression (default True).
        geotransform : tuple or list, optional
            Geotransform parameters (default (0, 1, 0, 0, 0, 1)).
            0: Top left x
            1: W-E pixel resolution
            2: Rotation, 0 if image is "north up"
            3: Top left y
            4: Rotation, 0 if image is "north up"
            5: N-S pixel resolution (negative value if North up)
        spatialref : str, optional
            Coordinate Reference System (CRS) in Wkt form (default None).
        overwrite : boolean, optional
            Flag if file can be overwritten if it already exists (default True).
        nc_format : str, optional
            NetCDF format (default 'NETCDF4_CLASSIC' (because it is
            needed for netCDF4.mfdatasets))
        chunksizes : tuple, optional
            Chunksizes of dimensions. The right definition can increase read
            operations, depending on the access pattern (e.g. time series or
            images) (default None).
        time_units : str, optional
            Time unit of time stamps (default "days since 1900-01-01 00:00:00").
        var_chunk_cache : tuple, optional
            Change variable chunk cache settings. A tuple containing
            size, nelems, preemption (default None, using default cache size)
        auto_scale : bool, optional
            should the data in variables automatically be encoded?
            that means: when reading ds, "scale_factor" and "add_offset" is applied.
            ATTENTION: Also the xarray dataset may applies encoding/scaling what
                    can mess up things
        """

        super().__init__(filename, mode=mode, complevel=complevel, zlib=zlib,
                 geotransform=geotransform, spatialref=spatialref,
                 overwrite=overwrite, nc_format=nc_format, chunksizes=chunksizes,
                 time_units=time_units,
                 var_chunk_cache=var_chunk_cache, auto_scale=auto_scale)

    def get_dimensions_info(self, map=None):
        """
        Retrieves coordinate and data variable attributes.

        Parameters
        ----------
        map: dict, optional
            Metadata dict used for retrieving metadata entries, e.g. {'var_name': ['SSM', 'SSM-NOISE']}.

        Returns
        -------
        metadata: dict
            Metadata dictionary of the data set.
        """

        metadata = self.src.dimensions
        metadata['var_name'] = self.src.variables

        if map is not None:
            metadata_mapped = dict()
            for map_key in map.keys():
                for dimension_name in map[map_key]:
                    if dimension_name in metadata.keys():
                        metadata_mapped[dimension_name] = metadata[dimension_name]
                        break
            return metadata_mapped
        else:
            return metadata
