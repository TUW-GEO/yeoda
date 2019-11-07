# general packages
import copy
import os
import re
import pandas as pd
import numpy as np
import xarray as xr
from collections import OrderedDict
from functools import reduce

# geo packages
from osgeo import osr
import shapely.wkt
from shapely.geometry import Point
from geopandas import GeoSeries
from geopandas import GeoDataFrame
import pytileproj.geometry as geometry
from veranda.geotiff import GeoTiffFile
from veranda.netcdf import NcFile
from veranda.timestack import GeoTiffRasterTimeStack

# load yeoda's utils module
from yeoda.utils import get_file_type
from yeoda.utils import any_geom2ogr_geom
from yeoda.utils import xy2ij
from yeoda.utils import ij2xy

# load classes from yeoda's error module
from yeoda.errors import IOClassNotFound
from yeoda.errors import DataTypeUnknown
from yeoda.errors import TileNotAvailable
from yeoda.errors import FileTypeUnknown
from yeoda.errors import DimensionUnkown
from yeoda.errors import LoadingDataError


def _check_inventory(f):
    """
    Decorator for `EODataCube` functions to check if the inventory exists.

    Parameters
    ----------
    f : function
        'EODataCube' function that has a keyword argument `in_place`

    Returns
    -------
    function
        Wrapper around `f`.
    """

    def f_wrapper(self, *args, **kwargs):
        in_place = kwargs.get('in_place')
        if self.inventory is not None:
            return f(self, *args, **kwargs)
        else:
            if in_place:
                return None
            else:
                return self
    return f_wrapper


class EODataCube(object):
    """
    A file(name) based data cube for preferably gridded and well-structured EO data.
    """

    def __init__(self, filepaths=None, grid=None, smart_filename_creator=None, dimensions=None, inventory=None,
                 io_map=None):
        """
        Constructor of the `EODataCube` class.

        Parameters
        ----------
        filepaths : list of str, optional
            List of file paths.
        grid : pytileproj.base.TiledProjection, optional
            Tiled projection/grid object/class (e.g. `Equi7Grid`, `LatLonGrid`).
        smart_filename_creator : function, optional
            A function that allows to create a `SmartFilename` instance from a filepath.
        dimensions : list of str, optional
            List of filename parts to use as dimensions. The strings have to match with the keys of the `SmartFilename`
            fields definition.
        inventory : GeoDataFrame, optional
            Contains information about the dimensions (columns) and each filepath (rows).
            If `grid` is not specified, a `Shapely` geometry object is added to the `GeoDataFrame`.
        io_map : dictionary, optional
            Map that represents the relation of an EO file type (e.g. GeoTIFF) with an appropriate reader
            (e.g. `GeoTiffFile` from veranda).
        """

        # initialise simple class variables
        self.smart_filename_creator = smart_filename_creator

        # initialise IO classes responsible for reading and writing
        if io_map is not None:
            self.io_map = io_map
        else:
            self.io_map = {'GeoTIFF': GeoTiffFile, 'NetCDF': NcFile}

        # create inventory from found filepaths
        self.inventory = None
        if inventory is not None:
            self.inventory = inventory
        else:
            self.__inventory_from_filepaths(filepaths, dimensions=dimensions,
                                            smart_filename_creator=smart_filename_creator)

        self.grid = None
        if grid:
            self.grid = grid
        elif (self.inventory is not None) and ('geometry' not in self.inventory.keys()):
            geometries = [self.__geometry_from_file(filepath) for filepath in self.filepaths]
            self.add_dimension('geometry', geometries, in_place=True)

    @property
    def filepaths(self):
        """
        Returns list of file paths stored in the inventory.

        Returns
        -------
        list
            List of file paths.
        """
        if self.inventory is not None:
            return list(self.inventory['filepath'])
        else:
            return None

    @property
    def dimensions(self):
        """
        Returns the dimensions of the data cube.

        Returns
        -------
        list
            List of inventory keys/dimensions of the data cube.
        """

        if self.inventory is not None:
            dimensions = list(self.inventory.keys())
            if 'filepath' in dimensions:
                dimensions.remove('filepath')
            return dimensions
        else:
            return None

    @classmethod
    def from_inventory(cls, inventory, grid=None):
        """
        Creates an `EODataCube` instance from a given inventory.

        Parameters
        ----------
        inventory : GeoDataFrame
            Contains information about the dimensions (columns) and each filepath (rows).
        grid : pytileproj.base.TiledProjection, optional
            Tiled projection/grid object/class (e.g. `Equi7Grid`, `LatLonGrid`).

        Returns
        -------
        EODataCube
            Data cube consisting of data stored in `inventory`.
        """

        return cls(inventory=inventory, grid=grid)

    @_check_inventory
    def rename_dimensions(self, dimensions_map, in_place=False):
        """
        Renames the dimensions of the data cube.

        Parameters
        ----------
        dimensions_map : dict
            A dictionary representing the relation between old and new dimension names. The keys are the old dimension
            names, the values the new dimension names (e.g., {'time_begin': 'time'}).
        in_place : boolean, optional
            If true, the current class instance will be altered.
            If false, a new class instance will be returned (default value is False).

        Returns
        -------
        EODataCube
            `EODataCube` object with renamed dimensions/columns of the inventory.
        """

        inventory = copy.deepcopy(self.inventory)
        inventory = inventory.rename(columns=dimensions_map)
        return self.__assign_inventory(inventory, in_place=in_place)

    @_check_inventory
    def add_dimension(self, name, values, in_place=False):
        """
        Adds a new dimension to the data cube.

        Parameters
        ----------
        name : str
            Name of the new dimension
        values : list
            Values of the new dimension (e.g., cloud cover, quality flag, ...).
            They have to have the same length as all the rows in the inventory.
        in_place : boolean, optional
            If true, the current class instance will be altered.
            If false, a new class instance will be returned (default value is False).

        Returns
        -------
        EODataCube
            `EODataCube` object with an additional dimension in the inventory.
        """

        inventory = self.inventory.assign(**{name: GeoSeries(values, index=self.inventory.index)})
        return self.__assign_inventory(inventory, in_place=in_place)

    @_check_inventory
    def filter_files_with_pattern(self, pattern, full_path=False, in_place=False):
        """
        Filters all filepaths according to the given pattern.

        Parameters
        ----------
        pattern : str
            A regular expression (e.g., ".*S1A.*GRD.*").
        full_path : boolean, optional
            Uses the full file paths for filtering if it is set to True.
            Otherwise the filename is used (default value is False).
        in_place : boolean, optional
            If true, the current class instance will be altered.
            If false, a new class instance will be returned (default value is False).

        Returns
        -------
        EODataCube
            `EODataCube` object with a filtered inventory according to the given pattern.
        """

        pattern = re.compile(pattern)
        if not full_path:
            file_filter = lambda x: re.match(pattern, os.path.basename(x)) is not None
        else:
            file_filter = lambda x: re.match(pattern, x) is not None
        idx_filter = [file_filter(filepath) for filepath in self.filepaths]
        inventory = self.inventory[idx_filter]
        return self.__assign_inventory(inventory, in_place=in_place)

    @_check_inventory
    def filter_by_metadata(self, metadata, in_place=False):
        """
        Filters all file paths according to the given metadata.

        Parameters
        ----------
        metadata : dict
            Key value relationships being expected to be in the metadata.
        in_place : boolean, optional
            If true, the current class instance will be altered.
            If false, a new class instance will be returned (default value is False).

        Returns
        -------
        EODataCube
            `EODataCube` object with a filtered inventory according to the given metadata.
        """

        bool_filter = []
        for filepath in self.filepaths:
            io_class = self.__io_class(get_file_type(filepath))
            io_instance = io_class(filepath, mode='r')
            ds = io_instance.src  # IO class has to have a "src" class variable which is a pointer to the data set

            select = False
            if ds:
                ds_metadata = ds.metadata
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

    @_check_inventory
    def sort_by_dimension(self, name, ascending=True, in_place=False):
        """
        Sorts the data cube/inventory according to the given dimension.

        Parameters
        ----------
        name : str
            Name of the dimension.
        ascending : bool, optional
            If true, sorts in ascending order, otherwise in descending order.
        in_place : boolean, optional
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
            return self.from_inventory(inventory=inventory_sorted, grid=self.grid)

    def filter_by_dimension(self, values, expressions=None, name="time", in_place=False):
        """
        Filters the data cube according to the given extents and returns a (new) data cube.

        Parameters
        ----------
        values : list, tuple, list of tuples or list of lists
            Values of interest to filter, e.g., timestamps: ('2019-01-01', '2019-02-01'), polarisations: ('VV')
        expressions : list, tuple, list of tuples or list of lists, optional
            Mathematical expressions to filter the data accordingly. If none are given, the exact values from 'values'
            are taken, otherwise the expressions are applied for each value and linked with an AND (e.g., ('>=', '<=')).
            They have to have the same length as 'values'.
        name : str, optional
            Name of the dimension.
        in_place : boolean, optional
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
        values : list, tuple, list of tuples or list of lists
            Values of interest to filter, e.g., timestamps: ('2019-01-01', '2019-02-01'), polarisations: ('VV')
        expressions : list, tuple, list of tuples or list of lists, optional
            Mathematical expressions to filter the data accordingly. If none are given, the exact values from 'values'
            are taken, otherwise the expressions are applied for each value and linked with an AND (e.g., ('>=', '<=')).
            They have to have the same length as 'values'.
        name : str, optional
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
        tilenames : list of str
            Tile names corresponding to a grid and/or the inventory.
        dimension_name : str, optional
            Name of the tile/spatial dimension in the inventory.
        in_place : boolean, optional
            If true, the current class instance will be altered.
            If false, a new class instance will be returned (default value is False).
        use_grid : bool
            If true, the given tile names are compared with the file names defined by the grid.
            If false, pre-defined grid information is ignored.

        Returns
        -------
        EODataCube
            `EODataCube` object with a filtered inventory according to the given tile names.
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

    @_check_inventory
    def filter_spatially_by_geom(self, geom, sref=None, dimension_name="tile", in_place=False):
        """
        Spatially filters the data cube by a bounding box or a geometry.

        Parameters
        ----------
        geom : OGR Geometry or Shapely Geometry or list or tuple, optional
            A geometry defining the region of interest. If it is of type list/tuple representing the extent
            (i.e. [x_min, y_min, x_max, y_max]), `sref` has to be given to transform the extent into a
            georeferenced polygon.
        sref : osr.SpatialReference, optional
            Spatial reference of the given region of interest `geom`.
        dimension_name : str, optional
            Name of the tile/spatial dimension in the inventory.
        in_place : boolean, optional
            If true, the current class instance will be altered.
            If false, a new class instance will be returned (default value is False).

        Returns
        -------
        EODataCube
            `EODataCube` object with a filtered inventory according to the given region of interest `geom`.
        """

        geom_roi = any_geom2ogr_geom(geom, osr_sref=sref)

        if self.grid:
            ftilenames = self.grid.search_tiles_over_geometry(geom_roi)
            tilenames = [ftilename.split('_')[1] for ftilename in ftilenames]
            self.filter_spatially_by_tilename(tilenames, dimension_name=dimension_name, in_place=in_place,
                                              use_grid=False)
        elif 'geometry' in self.inventory.keys():
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
        name : str, optional
            Name of the dimension.
        months : int, list of int
            List of integers specifying the months to select/split, i.e. each value is allowed to be in between 1-12.

        Returns
        -------
        List of monthly `EODataCube` objects.
        """

        sort = False
        yearly_eodcs = self.split_yearly(name=name)
        monthly_eodcs = []
        for yearly_eodc in yearly_eodcs:
            if months is not None:
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
        name : str, optional
            Name of the dimension.
        years : int, list of int
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

    def load_by_geom(self, geom, sref=None, dimension_name="tile", band=1, apply_mask=True, dtype="xarray"):
        """
        Loads data according to a given geometry.

        Parameters
        ----------
        geom : OGR Geometry or Shapely Geometry or list or tuple, optional
            A geometry defining the region of interest. If it is of type list/tuple representing the extent
            (i.e. [x_min, y_min, x_max, y_max]), `sref` has to be given to transform the extent into a
            georeferenced polygon.
        sref : osr.SpatialReference, optional
            Spatial reference of the given region of interest `geom`.
        band : int or str, optional
            Band number or name (default is 1).
        dimension_name : str, optional
            Name of the spatial dimension (default: 'tile').
        apply_mask : bool, optional
            If true, a numpy mask array with a mask excluding all pixels outside `geom` will be created
            (default is True).
        dtype : str
            Data type of the returned array-like structure (default is 'xarray'). It can be:
                - 'xarray': loads data as an xarray.DataSet
                - 'numpy': loads data as a numpy.ndarray
                - 'dataframe': loads data as a pandas.DataFrame

        Returns
        -------
        numpy.array or xarray.DataSet
            Data as an array-like object.
        """

        if self.inventory is None:  # no data given
            return None

        geom_roi = any_geom2ogr_geom(geom, osr_sref=sref)

        if self.grid:
            if sref is not None:
                tar_spref = self.grid.core.projection.osr_spref
                geom_roi = geometry.transform_geometry(geom_roi, tar_spref)

            extent = geometry.get_geometry_envelope(geom_roi)
            tilenames = list(self.inventory[dimension_name])
            if len(list(set(tilenames))) > 1:
                raise Exception('Data can be loaded only from one tile. Please filter the data cube before.')
            tilename = tilenames[0]
            min_row, min_col = self.grid.tilesys.create_tile(name=tilename).xy2ij(extent[0], extent[3])
            max_row, max_col = self.grid.tilesys.create_tile(name=tilename).xy2ij(extent[2], extent[1])
            inv_traffo_fun = lambda i, j: self.grid.tilesys.create_tile(name=tilename).ij2xy(i, j)
        else:
            this_sref, this_gt = self.__get_georef()
            geom_roi = geometry.transform_geometry(geom_roi, this_sref)
            extent = geometry.get_geometry_envelope(geom_roi)
            min_row, min_col = xy2ij(extent[0], extent[3], this_gt)
            max_row, max_col = xy2ij(extent[2], extent[1], this_gt)
            inv_traffo_fun = lambda i, j: ij2xy(i, j, this_gt)

        row_size = max_col - min_col + 1
        col_size = max_row - min_row + 1
        if apply_mask:
            geom_roi = shapely.wkt.loads(geom_roi.ExportToWkt())
            data_mask = np.ones((row_size, col_size))
            for i in range(col_size):
                for j in range(row_size):
                    x_i, y_j = inv_traffo_fun(min_row + i, min_col + j)
                    point = Point(x_i, y_j)
                    if point.within(geom_roi):
                        data_mask[j, i] = 0

        file_type = get_file_type(self.filepaths[0])
        if file_type == "GeoTIFF":
            file_ts = {'filenames': list(self.filepaths)}
            gt_timestack = GeoTiffRasterTimeStack(file_ts=file_ts)
            data = self.decode(gt_timestack.read_ts(min_row, min_col, col_size=col_size, row_size=row_size))
            if data is None:
                raise LoadingDataError()
            if apply_mask:
                data = np.ma.array(data, mask=data_mask)

            xs, ys = inv_traffo_fun(np.linspace(min_row, max_row + 1), np.linspace(min_col, max_col + 1))
            return self.__convert_dtype(data, dtype=dtype, xs=xs, ys=ys, band=band)
        elif file_type == "NetCDF":
            xr_data = xr.Dataset()
            for i, filepath in enumerate(self.inventory['filepath']):
                nc_file = NcFile(filepath, mode='r')
                xr_data = xr_data.merge(nc_file.read()[band][0, min_col:(max_col+1), min_row:(max_row+1)]) # assumes data has only one timestamp

            data = self.decode(xr_data)
            if data is None:
                raise LoadingDataError()

            if apply_mask:
                data.data = np.ma.array(data.data, mask=data_mask)

            return self.__convert_dtype(data, dtype=dtype)
        else:
            raise FileTypeUnknown(file_type)

    def load_by_pixels(self, rows, cols, row_size=None, col_size=None, band=1, dimension_name="tile", dtype="xarray"):
        """
        Loads data according to given pixel numbers, i.e. the row and column numbers and optionally a certain
        pixel window (`row_size` and `col_size`).

        Parameters
        ----------
        rows : list of int
            Row numbers.
        cols : list of int
            Column numbers.
        row_size : int, optional
            Number of rows to read (counts from input argument `rows`).
        col_size : int, optional
            Number of columns to read (counts from input argument `cols`).
        band : int or str, optional
            Band number or name (default is 1).
        dimension_name : str, optional
            Name of the spatial dimension (default: 'tile').
        dtype : str
            Data type of the returned array-like structure (default is 'xarray'). It can be:
                - 'xarray': loads data as an xarray.DataSet
                - 'numpy': loads data as a numpy.ndarray
                - 'dataframe': loads data as a pandas.DataFrame

        Returns
        -------
        list of numpy.ndarray or list of xarray.DataSets or pandas.DataFrame
            Data as an array-like object.
        """

        if self.inventory is None:  # no data given
            return None

        n = len(rows)
        if not isinstance(rows, list):
            rows = [rows]
        if not isinstance(cols, list):
            cols = [cols]

        data = []
        xs = None
        ys = None
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
                if xs is None:
                    xs = []
                if ys is None:
                    ys = []
                file_ts = {'filenames': list(self.inventory['filepath'])}
                gt_timestack = GeoTiffRasterTimeStack(file_ts=file_ts)
                data_i = self.decode(gt_timestack.read_ts(col, row, col_size=col_size, row_size=row_size))
                if data_i is None:
                    raise LoadingDataError()
                data.append(data_i)
                if row_size is not None and col_size is not None:
                    xs_i, ys_i = inv_traffo_fun(np.linspace(row, row + row_size), np.linspace(col, col + col_size))
                else:
                    xs_i, ys_i = inv_traffo_fun(row, col)
                xs.append(xs_i)
                ys.append(ys_i)
            elif file_type == "NetCDF":
                xr_data = xr.Dataset()
                for filepath in self.inventory['filepath']:
                    nc_file = NcFile(filepath, mode='r')
                    # assumes data has only one timestamp
                    if row_size is not None and col_size is not None:
                        data_i = self.decode(nc_file.read()[band][0, row:(row + row_size), col:(col + col_size)].data)
                    else:
                        data_i = self.decode(nc_file.read()[band][0, row, col].data)

                    if data_i is None:
                        raise LoadingDataError()
                    xr_data = xr_data.merge(data_i)
                data.append(xr_data)
            else:
                raise FileTypeUnknown(file_type)

        return self.__convert_dtype(data, dtype, xs=np.unique(xs), ys=np.unique(ys), band=band)

    def load_by_coords(self, xs, ys, sref=None, band='1', dimension_name="tile", dtype="xarray", origin="ur"):
        """
        Loads data as a 1-D array according to a given coordinate.

        Parameters
        ----------
        xs : list of floats or float
            World system coordinates in X direction.
        ys : list of floats or float
            World system coordinates in Y direction.
        sref : osr.SpatialReference, optional
            Spatial reference referring to the world system coordinates `x` and `y`.
        band : int or str, optional
            Band number or name (default is 1).
        dimension_name : str, optional
            Name of the spatial dimension (default: 'tile').
        dtype : str
            Data type of the returned array-like structure (default is 'xarray'). It can be:
                - 'xarray': loads data as an xarray.DataSet
                - 'numpy': loads data as a numpy.ndarray
                - 'dataframe': loads data as a pandas.DataFrame

        Returns
        -------
        list of numpy.ndarray or list of xarray.DataSets or pandas.DataFrame
            Data as an array-like object.
        """

        if self.inventory is None:  # no data given
            return None

        if not isinstance(xs, list):
            xs = [xs]
        if not isinstance(ys, list):
            ys = [ys]

        n = len(xs)
        data = []
        for i in range(n):
            x = xs[i]
            y = ys[i]
            if self.grid is not None:
                if sref is not None:
                    tar_spref = self.grid.core.projection.osr_spref
                    x, y = geometry.uv2xy(x, y, sref, tar_spref)
                tilenames = list(self.inventory[dimension_name])
                if len(list(set(tilenames))) > 1:
                    raise Exception('Data can be loaded only from one tile. Please filter the data cube before.')
                tilename = tilenames[0]
                gt = self.grid.tilesys.create_tile(name=tilename).geotransform()
            else:
                this_sref, gt = self.__get_georef()
                x, y = geometry.uv2xy(x, y, sref, this_sref)

            row, col = xy2ij(x, y, gt)
            # replace old coordinates with transformed coordinates related to the users definition
            x_t, y_t = ij2xy(row, col, gt, origin=origin)
            xs[i] = x_t
            ys[i] = y_t

            file_type = get_file_type(self.filepaths[0])
            if file_type == "GeoTIFF":
                file_ts = {'filenames': self.filepaths}
                gt_timestack = GeoTiffRasterTimeStack(file_ts=file_ts)
                data_i = self.decode(gt_timestack.read_ts(col, row))
                if data_i is None:
                    raise LoadingDataError()
                data.append(data_i)
            elif file_type == "NetCDF":
                xr_dss = []
                attrs = None
                for filepath in self.filepaths:
                    nc_file = NcFile(filepath, mode='r')
                    xr_ds = nc_file.read()
                    if xr_ds is None:
                        raise LoadingDataError()
                    attrs = xr_ds.attrs
                    xr_ar = self.decode(xr_ds[band][:, row, col])
                    # redeclare coordinates as dimensions:
                    for dim in list(xr_ds.dims.keys()):
                        if dim not in xr_ar.dims:
                            axis = len(xr_ar.dims)
                            xr_ar = xr_ar.expand_dims(dim, axis=axis)
                    xr_dss.append(xr.Dataset({band: xr_ar}))
                xr_data = xr.merge(xr_dss)  # assumes data has only one timestamp
                if attrs is not None:
                    xr_data.attrs = attrs
                data.append(xr_data)
            else:
                raise FileTypeUnknown(file_type)

        return self.__convert_dtype(data, dtype, xs=xs, ys=ys, band=band)

    def __convert_dtype(self, data, dtype, xs=None, ys=None, temporal_dim_name='time', band=1):
        """
        Converts `data` into an array-like object defined by `dtype`. It supports NumPy arrays, Xarray arrays and
        Pandas data frames.

        Parameters
        ----------
        data : list of numpy.ndarray or list of xarray.DataSets or numpy.ndarray or xarray.DataArray
        dtype : str
            Data type of the returned array-like structure (default is 'xarray'). It can be:
                - 'xarray': loads data as an xarray.DataSet
                - 'numpy': loads data as a numpy.ndarray
                - 'dataframe': loads data as a pandas.DataFrame
        xs : list, optional
            List of world system coordinates in X direction.
        ys : list, optional
            List of world system coordinates in Y direction.
        temporal_dim_name : str, optional
            Name of the temporal dimension (default: 'tile').
        band : int or str, optional
            Band number or name (default is 1).

        Returns
        -------
        list of numpy.ndarray or list of xarray.DataSets or pandas.DataFrame or numpy.ndarray or xarray.DataSet
            Data as an array-like object.
        """

        timestamps = self[temporal_dim_name]

        if dtype == "xarray":
            if isinstance(data, list) and isinstance(data[0], np.ndarray):
                ds = []
                for i, entry in enumerate(data):
                    xr_ar = xr.DataArray(entry, coords={'time': timestamps, 'x': [xs[i]], 'y': [ys[i]]},
                                         dims=['time', 'x', 'y'])
                    ds.append(xr.Dataset(data_vars={band: xr_ar}))
                converted_data = xr.merge(ds)
            elif isinstance(data, list) and isinstance(data[0], xr.Dataset):
                converted_data = xr.merge(data)
                converted_data.attrs = data[0].attrs
            elif isinstance(data, np.ndarray):
                xr_ar = xr.DataArray(data, coords={'time': timestamps, 'x': xs, 'y': ys}, dims=['time', 'x', 'y'])
                converted_data = xr.Dataset(data_vars={band: xr_ar})
            else:
                raise DataTypeUnknown(type(data), dtype)
        elif dtype == "numpy":
            if isinstance(data, list) and isinstance(data[0], np.ndarray):
                if len(data) == 1:
                    converted_data = data[0]
                else:
                    converted_data = data
            elif isinstance(data, list) and isinstance(data[0], xr.Dataset):
                converted_data = [np.array(entry[band].data) for entry in data]
                if len(converted_data) == 1:
                    converted_data = converted_data[0]
            elif isinstance(data, np.ndarray):
                converted_data = data
            else:
                raise DataTypeUnknown(type(data), dtype)
        elif dtype == "dataframe":
            if isinstance(data, list) and isinstance(data[0], np.ndarray):
                dfs = []
                for i, entry in enumerate(data):
                    entry_list = entry.flatten().tolist()
                    data_i = {'time': timestamps, 'x': [xs[i]]*len(entry_list), 'y': [ys[i]]*len(entry_list),
                              band: entry_list}
                    dfs.append(pd.DataFrame(data_i))
                converted_data = reduce(lambda left, right: pd.merge(left, right), dfs)
            elif isinstance(data, list) and isinstance(data[0], xr.Dataset):
                dfs = [entry.to_dataframe() for entry in data]
                converted_data = pd.concat(dfs).drop_duplicates().reset_index()
            elif isinstance(data, np.ndarray):
                xr_ar = xr.DataArray(data, coords={'time': timestamps, 'x': xs, 'y': ys}, dims=['time', 'x', 'y'])
                converted_data = xr.Dataset(data_vars={band: xr_ar}).to_dataframe()
            else:
                raise DataTypeUnknown(type(data), dtype)
        else:
            raise DataTypeUnknown(type(data), dtype)

        return converted_data

    def encode(self, data):
        """
        Encodes an array.

        Parameters
        ----------
        data : numpy, dask or xarray array
            Data array.

        Returns
        -------
        numpy, dask or xarray array
            Encoded data.
        """

        return data

    def decode(self, data):
        """
        Decodes an encoded array to retrieve the values in native units.

        Parameters
        ----------
        data : numpy, dask or xarray array
            Encoded array.

        Returns
        -------
        numpy, dask or xarray array
            Decoded data (original/native values).
        """

        return data

    @_check_inventory
    def intersect(self, dc_other, on_dimension=None, in_place=False):
        """
        Intersects this data cube with another data cube. This is equal to an SQL INNER JOIN operation.
        In other words:
            - all uncommon columns and rows (if `on_dimension` is given) are removed
            - duplicates are removed

        Parameters
        ----------
        dc_other : EODataCube
            Data cube to intersect with.
        on_dimension : str, optional
            Dimension name to intersect on, meaning that only equal entries along this dimension will be retained.
        in_place : boolean, optional
            If true, the current class instance will be altered.
            If false, a new class instance will be returned (default is False).

        Returns
        -------
        EODataCube
            Intersected data cubes.
        """

        dc_intersected = intersect_datacubes([self, dc_other], on_dimension=on_dimension)
        return self.__assign_inventory(dc_intersected.inventory, in_place=in_place)

    @_check_inventory
    def unite(self, dc_other, in_place=False):
        """
        Unites this data cube with respect to another data cube. This is equal to an SQL UNION operation.
        In other words:
            - all columns are put into one DataFrame
            - duplicates are removed
            - gaps are filled with NaN

        Parameters
        ----------
        dc_other : EODataCube
            Data cube to unite with.
        in_place : boolean, optional
            If true, the current class instance will be altered.
            If false, a new class instance will be returned (default is False).

        Returns
        -------
        EODataCube
            United data cubes.
        """

        dc_united = unite_datacubes([self, dc_other])
        return self.__assign_inventory(dc_united.inventory, in_place=in_place)

    def align_dimension(self, dc_other, name, in_place=False):
        """
        Aligns this data cube with another data cube along the specified dimension `name`.

        Parameters
        ----------
        dc_other : EODataCube
            Data cube to align with.
        name : str
            Name of the dimension, which is used for aligning/filtering the values for all data cubes.
        in_place : boolean, optional
            If true, the current class instance will be altered.
            If false, a new class instance will be returned (default value is False).

        Returns
        -------
        EODataCube
            Data cube with common values along the given dimension with respect to another data cube.
        """

        this_dim_values = list(self.inventory[name])
        uni_values = list(set(this_dim_values))
        other_dim_values = dc_other.inventory[name]
        idxs = np.zeros(len(other_dim_values)) - 1  # set -1 as no data value

        for i in range(len(uni_values)):
            val_idxs = np.where(uni_values[i] == other_dim_values)
            idxs[val_idxs] = this_dim_values.index(uni_values[i])  # get index of value in this data cube

        idxs = idxs[idxs != -1]
        if len(idxs) > 0:
            inventory = self.inventory.iloc[idxs].reset_index(drop=True)
            return self.__assign_inventory(inventory, in_place=in_place)
        else:
            print('No common dimension values found. Original data cube is returned.')
            return self

    def clone(self):
        """
        Clones, i.e. deep-copies a data cube.

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
        geom : OGR Geometry or Shapely Geometry or list or tuple, optional
            A geometry defining the region of interest. If it is of type list/tuple representing the extent
            (i.e. [x_min, y_min, x_max, y_max]), `sref` has to be given to transform the extent into a
            georeferenced polygon.
        sref : osr.SpatialReference, optional
            Spatial reference of the given region of interest `geom`.

        Returns
        -------
        ogr.Geometry
            Geometry with the (geo-)spatial reference of the data cube.
        """

        geom = any_geom2ogr_geom(geom, osr_sref=sref)
        this_sref, _ = self.__get_georef()
        geom = geometry.transform_geometry(geom, this_sref)

        return geom

    def __get_georef(self):
        """
        Retrieves georeference consisting of the spatialreference and the transformation parameters from the first file
        in the data cube.

        Returns
        -------
        this_sref : osr.SpatialReference()
            Spatial reference of data cube based on the first file.
        this_gt : tuple
            Geotransformation parameters based on the first file.
        """

        if self.filepaths is not None:
            filepath = self.filepaths[0]
            io_class = self.__io_class(get_file_type(filepath))
            io_instance = io_class(filepath, mode='r')
            this_sref = osr.SpatialReference()
            this_sref.ImportFromWkt(io_instance.spatialref)
            this_gt = io_instance.geotransform
            # close data set
            io_instance.close()
            return this_sref, tuple(this_gt)
        else:
            return None, None

    def __io_class(self, file_type):
        """
        Looks up appropriate file handler/IO class for a given file type.

        Parameters
        ----------
        file_type : str
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

    def __geometry_from_file(self, filepath):
        """
        Retrieves boundary geometry from an EO file.

        Parameters
        ----------
        filepath : str
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

        return shapely.wkt.loads(boundary_geom.ExportToWkt())

    def __inventory_from_filepaths(self, filepaths, dimensions=None, smart_filename_creator=None):
        """
        Creates GeoDataFrame (`inventory`) based on all filepaths.
        Each filepath/filename is translated to a SmartFilename object using a translation function
        `smart_filename_creator`.

        Parameters
        ----------
        filepaths : list of str
            List of file paths.
        dimensions : list of str, optional
            List of filename parts to use as dimensions. The strings have to match with the keys of the `SmartFilename`
            fields definition.
        smart_filename_creator : function, optional
            Translates a filepath/filename to a SmartFilename object.
        """

        inventory = OrderedDict()
        inventory['filepath'] = []

        # fill inventory
        for filepath in filepaths:
            n = len(inventory['filepath'])
            local_inventory = OrderedDict()
            local_inventory['filepath'] = [filepath]
            ext = os.path.splitext(filepath)[1]

            # get information from filename
            smart_filename = None
            try:
                smart_filename = smart_filename_creator(os.path.basename(filepath), ext=ext, convert=True)
            except:
                pass

            if smart_filename:
                if dimensions is not None:
                    for dimension in dimensions:
                        try:
                            local_inventory[dimension] = [smart_filename[dimension]]
                        except:
                            pass
                else:
                    for key, value in smart_filename.fields.items():
                        local_inventory[key] = [value]

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

        self.inventory = GeoDataFrame(inventory)

    @_check_inventory
    def __filter_by_dimension(self, values, expressions=None, name="time", split=False, in_place=False):
        """
        Filters the data cube according to the given extent and returns a new data cube.

        Parameters
        ----------
        values : list, tuple, list of tuples or list of lists
            Values of interest to filter, e.g., timestamps: ('2019-01-01', '2019-02-01'), polarisations: ('VV')
        expressions : list, tuple, list of tuples or list of lists, optional
            Mathematical expressions to filter the data accordingly. If none are given, the exact values from 'values'
            are taken, otherwise the expressions are applied for each value and linked with an AND (e.g., ('>=', '<=')).
            They have to have the same length as 'values'.
        name : str, optional
            Name of the dimension.
        split : boolean, optional
            If true, a list of data cubes will be returned according to the length of the input data
            (i.e. `values` and `expressions`)(default value is False).
        in_place : boolean, optional
            If true, the current class instance will be altered.
            If false, a new class instance will be returned (default value is False).

        Returns
        -------
        EODataCube or list of EODataCubes
            If `split` is true and multiple filters are specified, a list of EODataCube objects will be returned.
            If not, the inventory of the data cube is filtered.
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
            eodcs = [self.from_inventory(filtered_inventory, grid=self.grid)
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
        inventory : GeoDataFrame
            Data cube inventory.
        in_place : boolean, optional
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
            return self.from_inventory(inventory=inventory, grid=self.grid)

    def __deepcopy__(self, memodict={}):
        """
        Deepcopy method of the EODataCube class.

        Parameters
        ----------
        memodict : dict, optional

        Returns
        -------
        EODataCube
            Deepcopy of a data cube.
        """

        filepaths = copy.deepcopy(self.filepaths)
        grid = self.grid
        smart_filename_creator = self.smart_filename_creator
        dimensions = copy.deepcopy(self.dimensions)
        inventory = copy.deepcopy(self.inventory)

        return EODataCube(filepaths=filepaths, grid=grid, smart_filename_creator=smart_filename_creator,
                          dimensions=dimensions, inventory=inventory)

    def __getitem__(self, dimension_name):
        """
        Returns a column of the internal inventory according to the given column name/item.

        Parameters
        ----------
        dimension_name : str
            Column/Dimension name of the data cube inventory.

        Returns
        -------
        pandas.DataSeries
            Column of the internal inventory.
        """

        if self.inventory is not None and dimension_name in self.inventory.columns:
            return self.inventory[dimension_name]
        else:
            raise DimensionUnkown(dimension_name)

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
    dcs : list of EODataCube objects
       List of data cubes, which should be united.

    Returns
    -------
    EODataCube
        Data cube containing all information of the given data cubes.
    """

    inventories = [dc.inventory for dc in dcs]

    # this is a SQL alike UNION operation
    merged_inventory = pd.concat(inventories, ignore_index=True).drop_duplicates().reset_index(drop=True)

    dc_merged = EODataCube.from_inventory(merged_inventory, grid=dcs[0].grid)

    return dc_merged


def intersect_datacubes(dcs, on_dimension=None):
    """
    Intersects data cubes. This is equal to an SQL INNER JOIN operation.
    In other words:
        - all uncommon columns and rows (if `on_dimension` is given) are removed
        - duplicates are removed

    Parameters
    ----------
    dcs : list of EODataCube objects
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
            all_vals.append(list(inventory[on_dimension]))
        common_vals = list(set.intersection(*map(set, all_vals)))
        intersected_inventory = intersected_inventory[intersected_inventory[on_dimension].isin(common_vals)]

    intersected_inventory = intersected_inventory.drop_duplicates().reset_index(drop=True)
    dc_merged = EODataCube.from_inventory(intersected_inventory, grid=dcs[0].grid)

    return dc_merged

