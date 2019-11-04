# general packages
import copy
import os
import re
import itertools
import pandas as pd
import numpy as np
import xarray as xr
from collections import OrderedDict

# geo packages
from shapely.wkt import loads
from osgeo import osr
from geopandas import GeoSeries
from geopandas import GeoDataFrame
import shapely.wkt
from shapely.geometry import Point
import pytileproj.geometry as geometry
from veranda.geotiff import GeoTiffFile
from veranda.netcdf import NcFile
from veranda.timestack import GeoTiffRasterTimeStack
from yeoda.utils import get_file_type, any_geom2ogr_geom, xy2ij, ij2xy

# error packages
from yeoda.errors import IOClassNotFound
from yeoda.errors import TileNotAvailable
from yeoda.errors import FileTypeUnknown


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
        elif inventory is not None:
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

        if self.inventory is not None:
            filepaths = self.inventory['filepath']
            pattern = re.compile(pattern)
            if not full_path:
                file_filter = lambda x: re.match(pattern, os.path.basename(x)) is not None
            else:
                file_filter = lambda x: re.match(pattern, x) is not None
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
            io_instance = io_class(filepath, mode='r')  # IO class has to have a "src" class variable which is a pointer to the data set
            ds = io_instance.src

            select = False
            if ds:
                ds_metadata = ds.GetMetadata()
                select = True
                for key, value in metadata.items():
                    if key not in ds_metadata.keys():
                        select = False
                    else:
                        if ds_metadata[key] != value:
                            select = False
            # close data set
            io_instance.close()
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
            ftilenames = self.grid.search_tiles_over_geometry(geom_roi)
            tilenames = [ftilename.split('_')[1] for ftilename in ftilenames]
            self.filter_spatially_by_tilename(tilenames, dimension_name=dimension_name, in_place=in_place,
                                              use_grid=False)
        elif self.inventory is not None and 'geometry' in self.inventory.keys():
            # get spatial reference of data
            geom_roi = self.align_geom(geom_roi)
            geom_roi = shapely.wkt.loads(geom_roi.ExportToWkt())
            inventory = self.inventory[self.inventory.intersects(geom_roi)]
            return self.__assign_inventory(inventory, in_place=in_place)

    def split_monthly(self, name='time', months=None):
        """
        Separates the data cube into months.

        Parameters
        ----------
        name: str, optional
            Name of the dimension.
        months: int, list of int
            List of integers specifying the months to select/split, i.e. each value is allowed to be in between 1-12.

        Returns
        -------
        List of monthly EODataCube objects.
        """

        sort = False
        yearly_eodcs = self.split_yearly(name=name)
        monthly_eodcs = []
        for yearly_eodc in yearly_eodcs:
            if months:
                if not isinstance(months, list):
                    months = [months]
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

    def split_yearly(self, name='time', years=None):
        """
        Separates the data cube into years.

        Parameters
        ----------
        name: str, optional
            Name of the dimension.
        years: int, list of int
            List of integers specifying the years to select/split.

        Returns
        -------
        List of yearly EODataCube objects.
        """

        sort = False
        if years:
            if not isinstance(years, list):
                years = [years]
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

    def load_by_geom(self, geom, sref=None, dimension_name="tile", band=1, apply_mask=True):
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
        dimension_name: str, optional
            Name of the spatial dimension (default: "tile").
        apply_mask: bool, optional
            If true, all pixels being outside of the specified geometry are masked with np.nan.

        Returns
        -------
        numpy, dask or xarray array
            Data as an array-like object.
        """

        geom_roi = any_geom2ogr_geom(geom, osr_spref=sref)

        if self.grid:
            if sref is not None:
                tar_spref = self.grid.core.projection.osr_spref
                geom_roi = geometry.transform_geometry(geom_roi, tar_spref)

            extent = geometry.get_geometry_envelope(geom_roi)
            tilenames = list(self.inventory[dimension_name])
            if len(list(set(tilenames))) > 1:
                raise Exception('Data can be loaded only from one tile. Please filter the data cube before.')
            tilename = tilenames[0]
            i_min, j_min = self.grid.tilesys.create_tile(name=tilename).xy2ij(extent[0], extent[3])
            i_max, j_max = self.grid.tilesys.create_tile(name=tilename).xy2ij(extent[2], extent[1])
            inv_traffo_fun = lambda i, j: self.grid.tilesys.create_tile(name=tilename).ij2xy(i, j)
        else:
            this_sref, this_gt = self.__get_georef()
            geom_roi = geometry.transform_geometry(geom_roi, this_sref)
            extent = geometry.get_geometry_envelope(geom_roi)
            i_min, j_min = xy2ij(extent[0], extent[3], this_gt)
            i_max, j_max = xy2ij(extent[2], extent[1], this_gt)
            inv_traffo_fun = lambda i, j: ij2xy(i, j, this_gt)

        row_size = j_max - j_min + 1
        col_size = i_max - i_min + 1
        if apply_mask:
            geom_roi = shapely.wkt.loads(geom_roi.ExportToWkt())
            data_mask = np.ones((row_size, col_size))
            for i in range(col_size):
                for j in range(row_size):
                    x_i, y_j = inv_traffo_fun(i_min + i, j_min + j)
                    point = Point(x_i, y_j)
                    if point.within(geom_roi):
                        data_mask[j, i] = 0

        file_type = get_file_type(self.inventory['filepath'][0])
        if file_type == "GeoTIFF":
            file_ts = {'filenames': list(self.inventory['filepath'])}
            gt_timestack = GeoTiffRasterTimeStack(file_ts=file_ts)
            data = self.decode(gt_timestack.read_ts(i_min, j_min, col_size=col_size, row_size=row_size))
            if apply_mask:
                data = np.ma.array(data, mask=data_mask)
        elif file_type == "NetCDF":
            xr_data = xr.Dataset()
            for i, filepath in enumerate(self.inventory['filepath']):
                nc_file = NcFile(filepath, mode='r')
                xr_data = xr_data.merge(nc_file.read()[band][0, j_min:(j_max+1), i_min:(i_max+1)]) # assumes data has only one timestamp

            data = self.decode(xr_data)
            if apply_mask:
                data.data = np.ma.array(data.data, mask=data_mask)
        else:
            raise FileTypeUnknown(file_type)

        return data

    def load_by_pixels(self, rows, cols, row_size=None, col_size=None, band=1, dimension_name="tile", dtype="xarray"):
        n = len(rows)
        if not isinstance(rows, list):
            rows = [rows]
        if not isinstance(cols, list):
            cols = [cols]

        data = []
        xs = []
        ys = []
        for i in range(n):
            row = rows[i]
            col = cols[i]

            if self.grid:
                tilenames = list(self.inventory[dimension_name])
                if len(list(set(tilenames))) > 1:
                    raise Exception('Data can be loaded only from one tile. Please filter the data cube before.')
                tilename = tilenames[0]
                inv_traffo_fun = lambda i, j: self.grid.tilesys.create_tile(name=tilename).ij2xy(i, j)
            else:
                this_sref, this_gt = self.__get_georef()
                inv_traffo_fun = lambda i, j: ij2xy(i, j, this_gt)

            file_type = get_file_type(self.inventory['filepath'][0])
            if file_type == "GeoTIFF":
                file_ts = {'filenames': list(self.inventory['filepath'])}
                gt_timestack = GeoTiffRasterTimeStack(file_ts=file_ts)
                data.append(self.decode(gt_timestack.read_ts(col, row, col_size=col_size, row_size=row_size)))
                if row_size is not None and col_size is not None:
                    rows_i, cols_i = np.meshgrid(np.linspace(row, row + row_size), np.linspace(col, col + col_size))
                    xs_i, ys_i = inv_traffo_fun(rows_i.flatten(), cols_i.flatten())
                    xs.extend(xs_i)
                    ys.extend(ys_i)
                else:
                    x_i, y_i = inv_traffo_fun(row, col)
                    xs.append(x_i)
                    ys.append(y_i)

            elif file_type == "NetCDF":
                xr_data = xr.Dataset()
                for filepath in self.inventory['filepath']:
                    nc_file = NcFile(filepath, mode='r')
                    if row_size is not None and col_size is not None:
                        xr_data = xr_data.merge(self.decode(nc_file.read()[band][0, row:(row + row_size), col:(col + col_size)].data))  # assumes data has only one timestamp
                    else:
                        xr_data = xr_data.merge(self.decode(nc_file.read()[band][0, row, col].data))
                data.append(xr_data)
            else:
                raise FileTypeUnknown(file_type)

        return self.__convert_dtype(data, dtype, xs=xs, ys=ys, band=band)

    def load_by_coords(self, xs, ys, sref=None, band=1, dimension_name="tile", dtype="xarray"):
        """
        Loads data as a 1-D array according to a given coordinate.

        Parameters
        ----------
        xs: list of floats
            World system coordinates in X direction.
        ys: list of floats
            World system coordinates in Y direction.
        sref: osr.SpatialReference, optional
            Spatial reference referring to the world system coordinates `x` and `y`.
        dimension_name: str, optional
            Name of the spatial dimension (default: "tile").

        Returns
        -------
        numpy, dask or xarray array
            List of data as an 1-D array-like object.
        """

        if not isinstance(xs, list):
            xs = [xs]
        if not isinstance(ys, list):
            ys = [ys]

        n = len(xs)
        data = []
        for i in range(n):
            x = xs[i]
            y = ys[i]
            if self.grid:
                if sref is not None:
                    tar_spref = self.grid.core.projection.osr_spref
                    x, y = geometry.uv2xy(x, y, sref, tar_spref)
                tilenames = list(self.inventory[dimension_name])
                if len(list(set(tilenames))) > 1:
                    raise Exception('Data can be loaded only from one tile. Please filter the data cube before.')
                tilename = tilenames[0]
                row, col = self.grid.tilesys.create_tile(name=tilename).xy2ij(x, y)
            else:
                this_sref, this_gt = self.__get_georef()
                x, y = geometry.uv2xy(x, y, sref, this_sref)
                row, col = xy2ij(x, y, this_gt)

            file_type = get_file_type(self.inventory['filepath'][0])
            if file_type == "GeoTIFF":
                file_ts = {'filenames': list(self.inventory['filepath'])}
                gt_timestack = GeoTiffRasterTimeStack(file_ts=file_ts)
                data.append(self.decode(gt_timestack.read_ts(col, row)))
            elif file_type == "NetCDF":
                xr_data = xr.Dataset()
                for filepath in self.inventory['filepath']:
                    nc_file = NcFile(filepath, mode='r')
                    xr_data = xr_data.merge(self.decode(nc_file.read()[band][0, row, col].data))  # assumes data has only one timestamp
                data.append(xr_data)
            else:
                raise FileTypeUnknown(file_type)

        return self.__convert_dtype(data, dtype, xs=xs, ys=ys, band=band)

    def __convert_dtype(self, data, dtype, xs=None, ys=None, temporal_dim_name='time', band=None):
        err_msg = "Data conversion not possible for requested data type '{}' and actual data type '{}'."
        err_msg = err_msg.format(dtype, type(data))
        timestamps = self[temporal_dim_name]

        if dtype == "xarray":
            if isinstance(data, list) and isinstance(data[0], np.ndarray):
                ds = []
                for i, entry in enumerate(data):
                    xr_ar = xr.DataArray(entry, coords={'t': timestamps, 'x': [xs[i]], 'y': [ys[i]]},
                                         dims=['t', 'x', 'y'])
                    ds.append(xr.Dataset(data_vars={band: xr_ar}))
                converted_data = xr.merge(ds)
            elif isinstance(data, list) and isinstance(data[0], xr.Dataset):
                converted_data = xr.merge(data)
            elif isinstance(data, np.ndarray):
                xr_ar = xr.DataArray(data, coords={'t': timestamps, 'x': xs, 'y': ys}, dims=['t', 'x', 'y'])
                converted_data = xr.Dataset(data_vars={band: xr_ar})
            else:
                raise Exception(err_msg)
        elif dtype == "numpy":
            if isinstance(data, list) and isinstance(data[0], np.ndarray):
                converted_data = data
            elif isinstance(data, list) and isinstance(data[0], xr.Dataset):
                converted_data = [entry.data for entry in data]
            elif isinstance(data, np.ndarray):
                converted_data = data
            else:
                raise Exception(err_msg)
        elif dtype == "dataframe":
            if isinstance(data, list) and isinstance(data[0], np.ndarray):
                dfs = []
                for i, entry in enumerate(data):
                    data_i = {'t': timestamps, 'x': [xs[i]] * len(timestamps), 'y': [ys[i]] * len(timestamps),
                              band: entry}
                    dfs.append(pd.DataFrame(data_i))
                converted_data = pd.merge(dfs)
            elif isinstance(data, list) and isinstance(data[0], xr.Dataset):
                converted_data = pd.merge([entry.to_dataframe() for entry in data])
            elif isinstance(data, np.ndarray):
                xr_ar = xr.DataArray(data, coords={'t': timestamps, 'x': xs, 'y': ys}, dims=['t', 'x', 'y'])
                converted_data = xr.Dataset(data_vars={band: xr_ar}).to_dataframe()
            else:
                raise Exception(err_msg)
        else:
            raise Exception(err_msg)

        return converted_data

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

    def intersect(self, dc_other, on_dimension=None, in_place=False):
        dc_intersected = intersect_datacubes([self, dc_other], on_dimension=on_dimension)
        return self.__assign_inventory(dc_intersected.inventory, in_place=in_place)

    def unite(self, dc_other, in_place=False):
        """
        Unites this data cube with respect to another data cube.

        Parameters
        ----------
        dc_other: EODataCube
            Data cube to merge with.
        drop_unalike_dims: bool, optional
            If true, unalike dimensions of all data cubes are removed (default is True).
        in_place: boolean, optional
            If true, the current class instance will be altered.
            If false, a new class instance will be returned (default is False).

        Returns
        -------
        EODataCube
            Merged data cube.
        """

        dc_united = unite_datacubes([self, dc_other])
        return self.__assign_inventory(dc_united.inventory, in_place=in_place)

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

        uni_values = list(set(self.inventory[name]))
        all_values = np.array(dc_other.inventory[name])
        idxs = np.zeros(len(all_values))
        for i in range(len(uni_values)):
            val_idxs = np.where(uni_values[i] == all_values)
            idxs[val_idxs] = i

        inventory = self.inventory.iloc[idxs].reset_index(drop=True)
        return self.__assign_inventory(inventory, in_place=in_place)

    def clone(self):
        """
        Clones, i.e. deepcopies a data cube.

        Returns
        -------
        EODataCube
            Cloned/copied data cube.
        """

        return copy.deepcopy(self)

    def align_geom(self, geom, sref=None):
        """
        Transforms a geometry into the (geo-)spatial representation of the data cube.

        Parameters
        ----------
        geom: OGR Geometry or Shapely Geometry or list or tuple, optional
            A geometry defining the region of interest. If it is of type list/tuple representing the extent
            (i.e. [x_min, y_min, x_max, y_max]), `sref` has to be given to transform the extent into a
            georeferenced polygon.
        sref: osr.SpatialReference, optional
            Spatial reference of the given region of interest `geom`.

        Returns
        -------
        geom: ogr.Geometry
            Geometry with the (geo-)spatial reference of the data cube.
        """

        geom = any_geom2ogr_geom(geom, osr_spref=sref)
        this_sref, _ = self.__get_georef()
        geom = geometry.transform_geometry(geom, this_sref)

        return geom

    def __get_georef(self):
        """
        Retrieves georeference consisting of the spatialreference and the transformation parameters from the first file
        in the data cube.

        Returns
        -------
        this_sref: osr.SpatialReference()
            Spatial reference of data cube based on the first file.
        this_gt: tuple
            Geotransformation parameters based on the first file.
        """

        filepath = self['filepath'].iloc[0]
        io_class = self.__io_class(get_file_type(filepath))
        io_instance = io_class(filepath, mode='r')
        this_sref = osr.SpatialReference()
        this_sref.ImportFromWkt(io_instance.spatialref)
        this_gt = io_instance.geotransform
        # close data set
        io_instance.close()
        return this_sref, tuple(this_gt)

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
        io_instance = io_class(filepath, mode='r')
        gt = io_instance.geotransform
        boundary_extent = (gt[0], gt[3] + io_instance.shape[0] * gt[5], gt[0] + io_instance.shape[1] * gt[1], gt[3])
        boundary_spref = osr.SpatialReference()
        boundary_spref.ImportFromWkt(io_instance.spatialref)
        bbox = [(boundary_extent[0], boundary_extent[1]), (boundary_extent[2], boundary_extent[3])]
        boundary_geom = geometry.bbox2polygon(bbox, boundary_spref)
        # close data set
        io_instance.close()

        return loads(boundary_geom.ExportToWkt())

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
                    io_instance = io_class(filepath)
                    metadata = io_instance.get_dimensions_info(map=io_md_map[file_type])
                    io_instance.close()  # close data set
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
            values = [values]
        n_filters = len(values)
        if expressions is None:  # equal operator is the default comparison operator
            expressions = ["=="] * n_filters
        else:
            if not isinstance(expressions, list):
                values = [expressions]

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
            return None
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
        grid = self.grid
        dir_tree = copy.deepcopy(self.dir_tree)
        smart_filename_creator = self.smart_filename_creator
        dimensions = copy.deepcopy(self.dimensions)
        inventory = copy.deepcopy(self.inventory)

        return EODataCube(filepaths=filepaths, grid=grid, dir_tree=dir_tree,
                          smart_filename_creator=smart_filename_creator, dimensions=dimensions,
                          inventory=inventory)

    def __getitem__(self, item):
        """
        Returns a column of the internal inventory according to the given column name/item.

        Parameters
        ----------
        item: str
            Column name of the inventory of the data cube.

        Returns
        -------
        pandas.DataSeries
            Column of the internal inventory.
        """

        if self.inventory is not None and item in self.inventory.columns:
            return self.inventory[item]
        else:
            raise KeyError('Dimension {} is unknown.'.format(item))

    def __len__(self):
        """
        Returns number of inventory/data entries.

        Returns
        -------
        int
            Number of inventory/data entries.
        """

        return len(self.inventory)


def unite_datacubes(dcs):
    """
    Unites data cubes into one data cube. This is equal to an SQL UNION operation.
    In other words:
        - all columns are put into one DataFrame
        - duplicates are removed
        - gaps are filled with NaN

    Parameters
    ----------
    dcs: list of EODataCube objects
       List of data cubes, which should be united.

    Returns
    -------
    EODataCube
        Data cube containing all information of the given data cubes.
    """

    inventories = [dc.inventory for dc in dcs]

    # this is a SQL alike UNION operation
    merged_inventory = pd.concat(inventories, ignore_index=True).drop_duplicates().reset_index(drop=True)

    dc_merged = EODataCube.from_inventory(merged_inventory, grid=dcs[0].grid, dir_tree=dcs[0].dir_tree)

    return dc_merged

def intersect_datacubes(dcs, on_dimension=None):
    """
    Intersects data cubes. This is equal to an SQL INNER JOIN operation.
    In other words:
        - all uncommon columns and rows (if `on_dimension` is given) are removed
        - duplicates are removed

    Parameters
    ----------
    dcs: list of EODataCube objects
       List of data cubes, which should be intersected.

    Returns
    -------
    EODataCube
        Data cube containing all common information of the given data cubes.
    """

    inventories = [dc.inventory for dc in dcs]

    intersected_inventory = pd.concat(inventories, ignore_index=True, join='inner')
    if on_dimension is not None:
        all_vals = []
        for inventory in inventories:
            all_vals.extend(inventory[on_dimension])
        common_vals = list(set(all_vals))
        intersected_inventory = intersected_inventory[intersected_inventory[on_dimension].isin(common_vals)]

    intersected_inventory = intersected_inventory.drop_duplicates().reset_index(drop=True)
    dc_merged = EODataCube.from_inventory(intersected_inventory, grid=dcs[0].grid, dir_tree=dcs[0].dir_tree)

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
