# Copyright (c) 2019, Vienna University of Technology (TU Wien), Department of
# Geodesy and Geoinformation (GEO).
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# The views and conclusions contained in the software and documentation are
# those of the authors and should not be interpreted as representing official
# policies, either expressed or implied, of the FreeBSD Project.

"""
Main code for creating data cubes.
"""

# general packages
import copy
import os
import re
import netCDF4
import pandas as pd
import numpy as np
import xarray as xr
from collections import OrderedDict

# geo packages
from osgeo import osr
from osgeo import ogr
import shapely.wkt
from shapely.geometry import Point
from geopandas import GeoSeries
from geopandas import GeoDataFrame
from geopandas.base import is_geometry_type
import pytileproj.geometry as geometry
from veranda.io.geotiff import GeoTiffFile
from veranda.io.netcdf import NcFile
from veranda.io.timestack import GeoTiffRasterTimeStack
from veranda.io.timestack import NcRasterTimeStack

# load yeoda's utils module
from yeoda.utils import get_file_type
from yeoda.utils import any_geom2ogr_geom
from yeoda.utils import xy2ij
from yeoda.utils import ij2xy
from yeoda.utils import boundary

# load classes from yeoda's error module
from yeoda.errors import IOClassNotFound
from yeoda.errors import DataTypeUnknown
from yeoda.errors import TileNotAvailable
from yeoda.errors import FileTypeUnknown
from yeoda.errors import DimensionUnkown
from yeoda.errors import LoadingDataError
from yeoda.errors import SpatialInconsistencyError

# TODO: resolve geometry/tile columns and simplify queries


def _check_inventory(f):
    """
    Decorator for `EODataCube` functions to check if the inventory exists.

    Parameters
    ----------
    f : function
        'EODataCube' function that has a keyword argument `inplace`

    Returns
    -------
    function
        Wrapper around `f`.
    """

    def f_wrapper(self, *args, **kwargs):
        inplace = kwargs.get('inplace')
        if self.inventory is not None:
            return f(self, *args, **kwargs)
        else:
            if inplace:
                return None
            else:
                return self
    return f_wrapper


# TODO: maybe use pandas isequal after function call to set this flag
def _set_status(status):
    """
    Decorator for `EODataCube` functions to set internal flag defining if a process changes the data cube structure or
    not.

    Parameters
    ----------
    status: str
        Flag defining the status of the data cube. It can be:
            - "changed": a filtering process was executed, therefore the structure of the data cube has changed.
            - "stable": the structure of the data cube is remains the same.

    Returns
    -------
    function
        Wrapper around `f`.
    """

    def decorator(f):
        def f_wrapper(self, *args, **kwargs):
            ret_val = f(self, *args, **kwargs)
            inplace = kwargs.get('inplace', None)
            if inplace:
                self.status = status
            return ret_val
        return f_wrapper
    return decorator


class EODataCube:
    """
    A file(name) based data cube for preferably gridded and well-structured EO data.
    """

    def __init__(self, filepaths=None, grid=None, smart_filename_class=None, dimensions=None, inventory=None,
                 io_map=None, sdim_name="tile", tdim_name="time"):
        """
        Constructor of the `EODataCube` class.

        Parameters
        ----------
        filepaths : list of str, optional
            List of file paths.
        grid : pytileproj.base.TiledProjection, optional
            Tiled projection/grid object/class (e.g. `Equi7Grid`, `LatLonGrid`).
        smart_filename_class : geopathfinder.file_naming.SmartFilename, optional
            `SmartFilename` class to handle the interpretation of filenames.
        dimensions : list of str, optional
            List of filename parts to use as dimensions. The strings have to match with the keys of the `SmartFilename`
            fields definition.
        inventory : GeoDataFrame, optional
            Contains information about the dimensions (columns) and each filepath (rows).
            If `grid` is not specified, a `Shapely` geometry object is added to the `GeoDataFrame`.
        io_map : dictionary, optional
            Map that represents the relation of an EO file type (e.g. GeoTIFF) with an appropriate reader
            (e.g. `GeoTiffFile` from veranda).
        sdim_name : str, optional
            Name of the spatial dimension (default is 'tile'). If no grid is given, then the spatial dimension name
            will be 'geometry'.
        tdim_name : str, optional
            Name of the temporal dimension (default is 'time').
        """

        # initialise simple class variables
        self._ds = None  # data set pointer
        self.status = None
        self.tdim_name = tdim_name

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
                                            smart_filename_class=smart_filename_class)

        self.grid = None
        if grid:
            self.grid = grid
            self.sdim_name = sdim_name
        elif (self.inventory is not None) and ('geometry' not in self.inventory.keys()):
            geometries = [self.__geometry_from_file(filepath) for filepath in self.filepaths]
            self.sdim_name = sdim_name if sdim_name in self.dimensions else "geometry"
            self.add_dimension('geometry', geometries, inplace=True)
        else:
            self.sdim_name = sdim_name

    @property
    def filepaths(self):
        """
        list : List of file paths.
        """

        if self.inventory is not None:
            return list(self.inventory['filepath'])
        else:
            return None

    @property
    def dimensions(self):
        """
        list : List of inventory keys/dimensions of the data cube.
        """

        if self.inventory is not None:
            dimensions = list(self.inventory.keys())
            if 'filepath' in dimensions:
                dimensions.remove('filepath')
            return dimensions
        else:
            return None

    @property
    def boundary(self):
        """
        shapely.geometry : Shapely polygon representing the boundary of the data cube or `None` if no files are
        contained in the data cube.
        """

        self.__check_spatial_consistency()
        if self.filepaths is not None:
            filepath = self.filepaths[0]
            return self.__geometry_from_file(filepath)
        else:
            return None

    @classmethod
    def from_inventory(cls, inventory, grid=None, **kwargs):
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

        return cls(inventory=inventory, grid=grid, **kwargs)

    @_check_inventory
    def rename_dimensions(self, dimensions_map, inplace=False):
        """
        Renames the dimensions of the data cube.

        Parameters
        ----------
        dimensions_map : dict
            A dictionary representing the relation between old and new dimension names. The keys are the old dimension
            names, the values the new dimension names (e.g., {'time_begin': 'time'}).
        inplace : boolean, optional
            If true, the current class instance will be altered.
            If false, a new class instance will be returned (default value is False).

        Returns
        -------
        EODataCube
            `EODataCube` object with renamed dimensions/columns of the inventory.
        """

        # reset spatial and and temporal dimension name
        for old_dimension in list(dimensions_map.keys()):
            if self.sdim_name == old_dimension:
                self.sdim_name = dimensions_map[old_dimension]

            if self.tdim_name == old_dimension:
                self.tdim_name = dimensions_map[old_dimension]

        inventory = copy.deepcopy(self.inventory)
        inventory = inventory.rename(columns=dimensions_map)
        return self.__assign_inventory(inventory, inplace=inplace)

    @_check_inventory
    def add_dimension(self, name, values, inplace=False):
        """
        Adds a new dimension to the data cube.

        Parameters
        ----------
        name : str
            Name of the new dimension
        values : list
            Values of the new dimension (e.g., cloud cover, quality flag, ...).
            They have to have the same length as all the rows in the inventory.
        inplace : boolean, optional
            If true, the current class instance will be altered.
            If false, a new class instance will be returned (default value is False).

        Returns
        -------
        EODataCube
            `EODataCube` object with an additional dimension in the inventory.
        """

        if is_geometry_type(values):
            ds = GeoSeries(values, index=self.inventory.index)
        else:
            ds = pd.Series(values, index=self.inventory.index)
        inventory = self.inventory.assign(**{name: ds})
        return self.__assign_inventory(inventory, inplace=inplace)

    @_set_status('changed')
    @_check_inventory
    def filter_files_with_pattern(self, pattern, full_path=False, inplace=False):
        """
        Filters all filepaths according to the given pattern.

        Parameters
        ----------
        pattern : str
            A regular expression (e.g., ".*S1A.*GRD.*").
        full_path : boolean, optional
            Uses the full file paths for filtering if it is set to True.
            Otherwise the filename is used (default value is False).
        inplace : boolean, optional
            If true, the current class instance will be altered.
            If false, a new class instance will be returned (default value is False).

        Returns
        -------
        EODataCube
            `EODataCube` object with a filtered inventory according to the given pattern.
        """

        pattern = re.compile(pattern)
        if not full_path:
            file_filter = lambda x: re.search(pattern, os.path.basename(x)) is not None
        else:
            file_filter = lambda x: re.search(pattern, x) is not None
        idx_filter = [file_filter(filepath) for filepath in self.filepaths]
        inventory = self.inventory[idx_filter]
        return self.__assign_inventory(inventory, inplace=inplace)

    @_set_status('changed')
    @_check_inventory
    def filter_by_metadata(self, metadata, inplace=False):
        """
        Filters all file paths according to the given metadata.

        Parameters
        ----------
        metadata : dict
            Key value relationships being expected to be in the metadata.
        inplace : boolean, optional
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

            select = False
            if io_instance.src:
                ds_metadata = io_instance.metadata
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
        return self.__assign_inventory(inventory, inplace=inplace)

    @_set_status('changed')
    @_check_inventory
    def sort_by_dimension(self, name, ascending=True, inplace=False):
        """
        Sorts the data cube/inventory according to the given dimension.

        Parameters
        ----------
        name : str
            Name of the dimension.
        ascending : bool, optional
            If true, sorts in ascending order, otherwise in descending order.
        inplace : boolean, optional
            If true, the current class instance will be altered.
            If false, a new class instance will be returned (default value is False).

        Returns
        -------
        EODataCube
            Sorted EODataCube object.
        """

        inventory = copy.deepcopy(self.inventory)
        inventory_sorted = inventory.sort_values(by=name, ascending=ascending)

        return self.__assign_inventory(inventory=inventory_sorted)

    @_set_status('changed')
    def filter_by_dimension(self, values, expressions=None, name="time", inplace=False):
        """
        Filters the data cube according to the given extents and returns a (new) data cube.

        Parameters
        ----------
        values : list, tuple, list of tuples or list of lists
            Values of interest to filter, e.g., timestamps: ('2019-01-01', '2019-02-01'), polarisations: ('VV')
        expressions : list, tuple, list of tuples or list of lists, optional
            Mathematical expressions to filter the data accordingly. If none are given, the exact values from 'values'
            are taken, otherwise the expressions are applied for each value and linked with an AND (e.g., ('>=', '<=')).
            They have to have the same length as 'values'. The following comparison operators are allowed:
            - '==': equal to
            - '>=': larger than or equal to
            - '<=': smaller than or equal to
            - '>':  larger than
            - '<':  smaller than
        name : str, optional
            Name of the dimension.
        inplace : boolean, optional
            If true, the current class instance will be altered.
            If false, a new class instance will be returned (default value is False).

        Returns
        -------
        EODataCube
            Filtered EODataCube object.
        """

        return self.__filter_by_dimension(values, expressions=expressions, name=name, inplace=inplace, split=False)

    @_set_status('changed')
    def filter_spatially_by_tilename(self, tilenames, inplace=False, use_grid=True):
        """
        Spatially filters the data cube by tile names.

        Parameters
        ----------
        tilenames : list of str
            Tile names corresponding to a grid and/or the inventory.
        inplace : boolean, optional
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
                return self.filter_by_dimension(tilenames, name=self.sdim_name, inplace=inplace)
            else:
                print('No grid is provided to extract tile information.')
                return self
        else:
            return self.filter_by_dimension(tilenames, name=self.sdim_name, inplace=inplace)

    @_set_status('changed')
    @_check_inventory
    def filter_spatially_by_geom(self, geom, sref=None, inplace=False):
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
        inplace : boolean, optional
            If true, the current class instance will be altered.
            If false, a new class instance will be returned (default value is False).

        Returns
        -------
        EODataCube
            `EODataCube` object with a filtered inventory according to the given region of interest `geom`.
        """

        geom_roi = any_geom2ogr_geom(geom, osr_sref=sref)

        if self.grid:
            # pytileproj expects latlon polygon
            sref_lonlat = osr.SpatialReference()
            sref_lonlat.ImportFromEPSG(4326)
            geom_roi.TransformTo(sref_lonlat)
            ftilenames = self.grid.search_tiles_over_geometry(geom_roi)
            tilenames = [ftilename.split('_')[1] for ftilename in ftilenames]
            return self.filter_spatially_by_tilename(tilenames, inplace=inplace, use_grid=False)
        elif 'geometry' in self.inventory.keys():
            # get spatial reference of data
            geom_roi = self.align_geom(geom_roi)
            geom_roi = shapely.wkt.loads(geom_roi.ExportToWkt())
            inventory = self.inventory[self.inventory.intersects(geom_roi)]
            return self.__assign_inventory(inventory, inplace=inplace)
        else:
            return self

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
            They have to have the same length as 'values'. The following comparison operators are allowed:
            - '==': equal to
            - '>=': larger than or equal to
            - '<=': smaller than or equal to
            - '>':  larger than
            - '<':  smaller than
        name : str, optional
            Name of the dimension.

        Returns
        -------
        List of EODataCube objects.
        """

        return self.__filter_by_dimension(values, expressions=expressions, name=name, split=True)

    def split_monthly(self, months=None):
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
        yearly_eodcs = self.split_yearly()
        monthly_eodcs = []
        for yearly_eodc in yearly_eodcs:
            if months is not None:
                yearly_months = months
                if not isinstance(months, list):
                    yearly_months = [yearly_months]
                # initialise empty dict keeping track of the months
                timestamps_months = {}
                for month in yearly_months:
                    timestamps_months[month] = []

                for timestamp in yearly_eodc.inventory[self.tdim_name]:
                    if timestamp.month in yearly_months:
                        timestamps_months[timestamp.month].append(timestamp)
            else:
                sort = True
                timestamps_months = {}
                for timestamp in yearly_eodc.inventory[self.tdim_name]:
                    if timestamp.month not in timestamps_months.keys():
                        timestamps_months[timestamp.month] = []

                    timestamps_months[timestamp.month].append(timestamp)

                yearly_months = timestamps_months.keys()

            if sort:
                yearly_months = sorted(yearly_months)  # sort in ascending order
            values = []
            expressions = [(">=", "<=")] * len(yearly_months)
            for month in yearly_months:
                min_timestamp = min(timestamps_months[month])
                max_timestamp = max(timestamps_months[month])
                values.append((min_timestamp, max_timestamp))

            monthly_eodcs.extend(self.split_by_dimension(values, expressions, name=self.tdim_name))

        return monthly_eodcs

    def split_yearly(self, years=None):
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

            for timestamp in self.inventory[self.tdim_name]:
                if timestamp.year in years:
                    timestamps_years[timestamp.year].append(timestamp)
        else:
            sort = True
            timestamps_years = {}
            for timestamp in self.inventory[self.tdim_name]:
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

        return self.split_by_dimension(values, expressions, name=self.tdim_name)

    @_set_status('stable')
    def load_by_geom(self, geom, sref=None, band=1, apply_mask=False, dtype="xarray", origin='ul',
                     decode_kwargs=None):
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
        apply_mask : bool, optional
            If true, a numpy mask array with a mask excluding (=1) all pixels outside `geom` (=0) will be created
            (default is True).
        dtype : str
            Data type of the returned array-like structure (default is 'xarray'). It can be:
                - 'xarray': loads data as an xarray.DataSet
                - 'numpy': loads data as a numpy.ndarray
                - 'dataframe': loads data as a pandas.DataFrame
        origin: str, optional
            Defines the world system origin of the pixel. It can be:
                - upper left ("ul", default)
                - upper right ("ur")
                - lower right ("lr")
                - lower left ("ll")
                - center ("c")
        decode_kwargs: dict, optional
            Keyword arguments for the decoder.

        Returns
        -------
        numpy.array or xarray.DataSet or pd.DataFrame
            Data as an array-like object.
        """

        if self.inventory is None:  # no data given
            return None

        decode_kwargs = {} if decode_kwargs is None else decode_kwargs

        if self.grid:
            if self.sdim_name not in self.dimensions:
                raise DimensionUnkown(self.sdim_name)
            this_sref = self.grid.core.projection.osr_spref
            tilenames = list(self.inventory[self.sdim_name])
            if len(list(set(tilenames))) > 1:
                raise Exception('Data can be loaded only from one tile. Please filter the data cube before.')
            tilename = tilenames[0]
            this_gt = self.grid.tilesys.create_tile(name=tilename).geotransform()
        else:
            this_sref, this_gt = self.__get_georef()

        if sref is not None:
            geom_roi = any_geom2ogr_geom(geom, osr_sref=sref)
        else:
            geom_roi = any_geom2ogr_geom(geom, osr_sref=this_sref)

        roi_sref = geom_roi.GetSpatialReference()
        if not this_sref.IsSame(roi_sref):
            geom_roi = geometry.transform_geometry(geom_roi, this_sref)

        # clip region of interest to tile boundary
        boundary_ogr = ogr.CreateGeometryFromWkt(self.boundary.wkt)
        geom_roi = geom_roi.Intersection(boundary_ogr)
        if geom_roi.ExportToWkt() == 'GEOMETRYCOLLECTION EMPTY':
            raise Exception('The given geometry does not intersect with the tile boundaries.')
        geom_roi.AssignSpatialReference(this_sref)

        extent = geometry.get_geometry_envelope(geom_roi)
        inv_traffo_fun = lambda i, j: ij2xy(i, j, this_gt, origin=origin)
        min_col, min_row = xy2ij(extent[0], extent[3], this_gt)
        max_col, max_row = xy2ij(extent[2], extent[1], this_gt)
        col_size = max_col - min_col
        row_size = max_row - min_row
        if apply_mask:
            geom_roi = shapely.wkt.loads(geom_roi.ExportToWkt())
            data_mask = np.ones((row_size, col_size))
            for col in range(col_size):
                for row in range(row_size):
                    x, y = inv_traffo_fun(min_col + col, min_row + row)
                    point = Point(x, y)
                    if point.within(geom_roi) or point.touches(geom_roi):
                        data_mask[row, col] = 0

        file_type = get_file_type(self.filepaths[0])
        xs = None
        ys = None
        if file_type == "GeoTIFF":
            if self._ds is None and self.status != "stable":
                file_ts = {'filenames': list(self.filepaths)}
                self._ds = GeoTiffRasterTimeStack(file_ts=file_ts, file_band=band)
            data = self._ds.read_ts(min_col, min_row, col_size=col_size, row_size=row_size)
            if data is None:
                raise LoadingDataError()
            data = self.decode(data, **decode_kwargs)
            if apply_mask:
                data = np.ma.array(data, mask=np.stack([data_mask]*data.shape[0], axis=0))

            cols_traffo = np.concatenate(([min_col] * row_size, np.arange(min_col, max_col))).astype(float)
            rows_traffo = np.concatenate((np.arange(min_row, max_row), [min_row] * col_size)).astype(float)
            x_traffo, y_traffo = inv_traffo_fun(cols_traffo, rows_traffo)
            xs = x_traffo[row_size:]
            ys = y_traffo[:row_size]
            data = self.__convert_dtype(data, dtype=dtype, xs=xs, ys=ys, band=band)
        elif file_type == "NetCDF":
            if self._ds is None and self.status != "stable":
                file_ts = pd.DataFrame({'filenames': list(self.filepaths)})
                self._ds = NcRasterTimeStack(file_ts=file_ts, stack_size='single', auto_decode=False)
            time_units = self._ds.time_units
            data_ar = self._ds.read()[str(band)][:, min_row:max_row, min_col:max_col]
            if data_ar is None:
                raise LoadingDataError()

            data_ar.data = self.decode(data_ar.data, **decode_kwargs)
            if apply_mask:
                data_ar.data = np.ma.array(data_ar.data, mask=np.stack([data_mask] * data_ar.data.shape[0], axis=0))
            data = data_ar.to_dataset()
            data = self.__convert_dtype(data, dtype=dtype, xs=xs, ys=ys, band=band, time_units=time_units)
        else:
            raise FileTypeUnknown(file_type)

        return data

    @_set_status('stable')
    def load_by_pixels(self, rows, cols, row_size=1, col_size=1, band=1, dtype="xarray", origin="ul",
                       decode_kwargs=None):
        """
        Loads data according to given pixel numbers, i.e. the row and column numbers and optionally a certain
        pixel window (`row_size` and `col_size`).

        Parameters
        ----------
        rows : list of int or int
            Row numbers.
        cols : list of int or int
            Column numbers.
        row_size : int, optional
            Number of rows to read (counts from input argument `rows`, default is 1).
        col_size : int, optional
            Number of columns to read (counts from input argument `cols`, default is 1).
        band : int or str, optional
            Band number or name (default is 1).
        dtype : str
            Data type of the returned array-like structure (default is 'xarray'). It can be:
                - 'xarray': loads data as an xarray.DataSet
                - 'numpy': loads data as a numpy.ndarray
                - 'dataframe': loads data as a pandas.DataFrame
        origin: str, optional
            Defines the world system origin of the pixel. It can be:
                - upper left ("ul", default)
                - upper right ("ur")
                - lower right ("lr")
                - lower left ("ll")
                - center ("c")
        decode_kwargs: dict, optional
            Keyword arguments for the decoder.

        Returns
        -------
        list of numpy.ndarray or list of xarray.DataSets or pandas.DataFrame
            Data as an array-like object.
        """

        if self.inventory is None:  # no data given
            return None

        decode_kwargs = {} if decode_kwargs is None else decode_kwargs

        if not isinstance(rows, list):
            rows = [rows]
        if not isinstance(cols, list):
            cols = [cols]

        if self.grid:
            if self.sdim_name not in self.dimensions:
                raise DimensionUnkown(self.sdim_name)
            tilenames = list(self.inventory[self.sdim_name])
            if len(list(set(tilenames))) > 1:
                raise Exception('Data can be loaded only from one tile. Please filter the data cube before.')
            tilename = tilenames[0]
            this_gt = self.grid.tilesys.create_tile(name=tilename).geotransform()
        else:
            _, this_gt = self.__get_georef()

        inv_traffo_fun = lambda i, j: ij2xy(i, j, this_gt, origin=origin)
        file_type = get_file_type(self.filepaths[0])

        time_units = "days since 1900-01-01 00:00:00"
        n = len(rows)
        data = []
        xs = []
        ys = []
        for i in range(n):
            row = rows[i]
            col = cols[i]

            if file_type == "GeoTIFF":
                if self._ds is None and self.status != "stable":
                    file_ts = {'filenames': list(self.filepaths)}
                    self._ds = GeoTiffRasterTimeStack(file_ts=file_ts, file_band=band)

                data_i = self._ds.read_ts(col, row, col_size=col_size, row_size=row_size)
                if data_i is None:
                    raise LoadingDataError()
                data.append(self.decode(data_i, **decode_kwargs))
                if row_size != 1 and col_size != 1:
                    max_col = col + col_size
                    max_row = row + row_size
                    cols_traffo = np.concatenate(([col] * row_size, np.arange(col, max_col))).astype(float)
                    rows_traffo = np.concatenate((np.arange(row, max_row), [row] * col_size)).astype(float)
                    x_traffo, y_traffo = inv_traffo_fun(cols_traffo, rows_traffo)
                    xs_i = x_traffo[row_size:].tolist()
                    ys_i = y_traffo[:row_size].tolist()
                else:
                    xs_i, ys_i = inv_traffo_fun(col, row)
                xs.append(xs_i)
                ys.append(ys_i)
            elif file_type == "NetCDF":
                if self._ds is None and self.status != "stable":
                    file_ts = pd.DataFrame({'filenames': list(self.filepaths)})
                    self._ds = NcRasterTimeStack(file_ts=file_ts, stack_size='single', auto_decode=False)
                time_units = self._ds.time_units
                if row_size != 1 and col_size != 1:
                    data_ar = self._ds.read()[str(band)][:, row:(row + row_size), col:(col + col_size)]
                else:
                    data_ar = self._ds.read()[str(band)][:, row:(row + 1), col:(col + 1)]  # +1 to keep the dimension

                if data_ar is None:
                    raise LoadingDataError()
                data_ar.data = self.decode(data_ar.data, **decode_kwargs)
                data.append(data_ar.to_dataset())
            else:
                raise FileTypeUnknown(file_type)

        return self.__convert_dtype(data, dtype, xs=xs, ys=ys, band=band, time_units=time_units)

    @_set_status('stable')
    def load_by_coords(self, xs, ys, sref=None, band=1, dtype="xarray", origin="ul", decode_kwargs=None):
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
        dtype : str
            Data type of the returned array-like structure (default is 'xarray'). It can be:
                - 'xarray': loads data as an xarray.DataSet
                - 'numpy': loads data as a numpy.ndarray
                - 'dataframe': loads data as a pandas.DataFrame
        origin: str, optional
            Defines the world system origin of the pixel. It can be:
                - upper left ("ul", default)
                - upper right ("ur")
                - lower right ("lr")
                - lower left ("ll")
                - center ("c")
        decode_kwargs: dict, optional
            Keyword arguments for the decoder.

        Returns
        -------
        list of numpy.ndarray or list of xarray.DataSets or pandas.DataFrame
            Data as an array-like object.
        """

        if self.inventory is None:  # no data given
            return None

        decode_kwargs = {} if decode_kwargs is None else decode_kwargs

        if not isinstance(xs, list):
            xs = [xs]
        if not isinstance(ys, list):
            ys = [ys]

        if self.grid is not None:
            if self.sdim_name not in self.dimensions:
                raise DimensionUnkown(self.sdim_name)
            this_sref = self.grid.core.projection.osr_spref
            tilenames = list(self.inventory[self.sdim_name])
            if len(list(set(tilenames))) > 1:
                raise Exception('Data can be loaded only from one tile. Please filter the data cube before.')
            tilename = tilenames[0]
            this_gt = self.grid.tilesys.create_tile(name=tilename).geotransform()
        else:
            this_sref, this_gt = self.__get_georef()

        time_units = "days since 1900-01-01 00:00:00"
        n = len(xs)
        data = []
        for i in range(n):
            x = xs[i]
            y = ys[i]

            if sref is not None:
                x, y = geometry.uv2xy(x, y, sref, this_sref)
            col, row = xy2ij(x, y, this_gt)
            # replace old coordinates with transformed coordinates related to the users definition
            x_t, y_t = ij2xy(col, row, this_gt, origin=origin)
            xs[i] = x_t
            ys[i] = y_t

            file_type = get_file_type(self.filepaths[0])
            if file_type == "GeoTIFF":
                if self._ds is None and self.status != "stable":
                    file_ts = {'filenames': self.filepaths}
                    self._ds = GeoTiffRasterTimeStack(file_ts=file_ts, file_band=band)
                data_i = self._ds.read_ts(col, row)
                if data_i is None:
                    raise LoadingDataError()

                data_i = self.decode(data_i, **decode_kwargs)
            elif file_type == "NetCDF":
                if self._ds is None and self.status != "stable":
                    file_ts = pd.DataFrame({'filenames': list(self.filepaths)})
                    self._ds = NcRasterTimeStack(file_ts=file_ts, stack_size='single', auto_decode=False)
                time_units = self._ds.time_units
                data_ar = self._ds.read()[str(band)][:, row:(row + 1), col:(col + 1)]  # +1 to keep the dimension
                if data_ar is None:
                    raise LoadingDataError()

                data_ar.data = self.decode(data_ar.data, **decode_kwargs)
                data_i = data_ar.to_dataset()
            else:
                raise FileTypeUnknown(file_type)

            data.append(data_i)

        return self.__convert_dtype(data, dtype, xs=xs, ys=ys, band=band, time_units=time_units)

    def encode(self, data, **kwargs):
        """
        Encodes an array.

        Parameters
        ----------
        data : numpy, dask or xarray array
            Data array.
        **kwargs
            Keyword arguments for encoding function.

        Returns
        -------
        numpy, dask or xarray array
            Encoded data.
        """

        return data

    def decode(self, data, **kwargs):
        """
        Decodes an encoded array to retrieve the values in native units.

        Parameters
        ----------
        data : numpy, dask or xarray array
            Encoded array.
        **kwargs
            Keyword arguments for decoding function.

        Returns
        -------
        numpy, dask or xarray array
            Decoded data (original/native values).
        """

        return data

    @_set_status('changed')
    @_check_inventory
    def intersect(self, dc_other, on_dimension=None, inplace=False):
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
        inplace : boolean, optional
            If true, the current class instance will be altered.
            If false, a new class instance will be returned (default is False).

        Returns
        -------
        EODataCube
            Intersected data cubes.
        """

        dc_intersected = intersect_datacubes([self, dc_other], on_dimension=on_dimension)
        return self.__assign_inventory(dc_intersected.inventory, inplace=inplace)

    @_set_status('changed')
    @_check_inventory
    def unite(self, dc_other, inplace=False):
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
        inplace : boolean, optional
            If true, the current class instance will be altered.
            If false, a new class instance will be returned (default is False).

        Returns
        -------
        EODataCube
            United data cubes.
        """

        dc_united = unite_datacubes([self, dc_other])
        return self.__assign_inventory(dc_united.inventory, inplace=inplace)

    @_set_status('changed')
    def align_dimension(self, dc_other, name, inplace=False):
        """
        Aligns this data cube with another data cube along the specified dimension `name`.

        Parameters
        ----------
        dc_other : EODataCube
            Data cube to align with.
        name : str
            Name of the dimension, which is used for aligning/filtering the values for all data cubes.
        inplace : boolean, optional
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
            return self.__assign_inventory(inventory, inplace=inplace)
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

    def close(self):
        """ Closes data set pointer. """

        self._ds.close()

    def __convert_dtype(self, data, dtype, xs=None, ys=None, band=1, time_units='days since 1900-01-01 00:00:00'):
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
        band : int or str, optional
            Band number or name (default is 1).
        time_units : str, optional
            Time units definition for NetCDF4's `num2date` function.
            Defaults to 'days since 1900-01-01 00:00:00'.

        Returns
        -------
        list of numpy.ndarray or list of xarray.DataSets or pandas.DataFrame or numpy.ndarray or xarray.DataSet
            Data as an array-like object.
        """

        if dtype == "xarray":
            timestamps = self[self.tdim_name]
            if isinstance(data, list) and isinstance(data[0], np.ndarray):
                ds = []
                for i, entry in enumerate(data):
                    x = xs[i]
                    y = ys[i]
                    if not isinstance(x, list):
                        x = [x]
                    if not isinstance(y, list):
                        y = [y]
                    xr_ar = xr.DataArray(entry, coords={self.tdim_name: timestamps, 'y': y, 'x': x},
                                         dims=[self.tdim_name, 'y', 'x'])
                    ds.append(xr.Dataset(data_vars={str(band): xr_ar}))
                converted_data = xr.merge(ds)
            elif isinstance(data, list) and isinstance(data[0], xr.Dataset):
                converted_data = xr.merge(data)
                converted_data.attrs = data[0].attrs
                if converted_data['time'].dtype == 'float':
                    conv_timestamps = netCDF4.num2date(converted_data['time'], time_units)
                    converted_data = converted_data.assign_coords({'time': conv_timestamps})
            elif isinstance(data, np.ndarray):
                xr_ar = xr.DataArray(data, coords={self.tdim_name: timestamps, 'y': ys, 'x': xs},
                                     dims=[self.tdim_name, 'y', 'x'])
                converted_data = xr.Dataset(data_vars={str(band): xr_ar})
            elif isinstance(data, xr.Dataset):
                converted_data = data
                if converted_data['time'].dtype == 'float':
                    conv_timestamps = netCDF4.num2date(converted_data['time'], time_units)
                    converted_data = converted_data.assign_coords({'time': conv_timestamps})
            else:
                raise DataTypeUnknown(type(data), dtype)
        elif dtype == "numpy":
            if isinstance(data, list) and isinstance(data[0], np.ndarray):
                if len(data) == 1:
                    converted_data = data[0]
                else:
                    converted_data = data
            elif isinstance(data, list) and isinstance(data[0], xr.Dataset):
                converted_data = [np.array(entry[str(band)].data) for entry in data]
                if len(converted_data) == 1:
                    converted_data = converted_data[0]
            elif isinstance(data, xr.Dataset):
                converted_data = np.array(data[str(band)].data)
            elif isinstance(data, np.ndarray):
                converted_data = data
            else:
                raise DataTypeUnknown(type(data), dtype)
        elif dtype == "dataframe":
            xr_ds = self.__convert_dtype(data, 'xarray', xs=xs, ys=ys, band=band)
            converted_data = xr_ds.to_dataframe()
        else:
            raise DataTypeUnknown(type(data), dtype)

        return converted_data

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
        boundary_geom = boundary(io_instance.geotransform, io_instance.spatialref, io_instance.shape)
        # close data set
        io_instance.close()

        return shapely.wkt.loads(boundary_geom.ExportToWkt())

    def __check_spatial_consistency(self):
        """
        Checks if there are multiple tiles/file extents present in the data cube.
        If so, a `SpatialInconsistencyError` is raised.
        """

        if self.sdim_name in self.dimensions:
            geoms = self[self.sdim_name]
            try:  # try apply unique function to DataSeries
                uni_vals = geoms.unique()
            except:
                # the type seems not to be hashable, it could contain shapely geometries.
                # try to convert them to strings and then apply a unique function
                uni_vals = geoms.apply(lambda x: x.wkt).unique()

            if len(uni_vals) > 1:
                raise SpatialInconsistencyError()

    def __inventory_from_filepaths(self, filepaths, dimensions=None, smart_filename_class=None):
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
        smart_filename_class : geopathfinder.file_naming.SmartFilename, optional
            `SmartFilename` class to handle the interpretation of filenames.
        """

        inventory = OrderedDict()
        inventory['filepath'] = []

        # fill inventory
        for filepath in filepaths:
            n = len(inventory['filepath'])
            local_inventory = OrderedDict()
            local_inventory['filepath'] = [filepath]

            # get information from filename
            smart_filename = None
            try:
                smart_filename = smart_filename_class.from_filename(os.path.basename(filepath), convert=True)
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
                    else:  # fill dimension with None
                        inventory[key] = [None] * n

            for entry in extended_entries:
                for i, key in enumerate(inventory.keys()):
                    inventory[key].append(entry[i])

        self.inventory = GeoDataFrame(inventory)

    @_check_inventory
    def __filter_by_dimension(self, values, expressions=None, name="time", split=False, inplace=False):
        """
        Filters the data cube according to the given extent and returns a new data cube.

        Parameters
        ----------
        values : list, tuple, list of tuples or list of lists
            Values of interest to filter, e.g., timestamps: ('2019-01-01', '2019-02-01'), polarisations: ('VV')
        expressions : list, tuple, list of tuples or list of lists, optional
            Mathematical expressions to filter the data accordingly. If none are given, the exact values from 'values'
            are taken, otherwise the expressions are applied for each value and linked with an AND (e.g., ('>=', '<=')).
            They have to have the same length as 'values'. The following comparison operators are allowed:
            - '==': equal to
            - '>=': larger than or equal to
            - '<=': smaller than or equal to
            - '>':  larger than
            - '<':  smaller than
        name : str, optional
            Name of the dimension.
        split : boolean, optional
            If true, a list of data cubes will be returned according to the length of the input data
            (i.e. `values` and `expressions`)(default value is False).
        inplace : boolean, optional
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
            eodcs = [self.__assign_inventory(filtered_inventory, inplace=False)
                     for filtered_inventory in filtered_inventories]
            return eodcs
        else:
            filtered_inventory = pd.concat(filtered_inventories, ignore_index=True)
            return self.__assign_inventory(filtered_inventory, inplace=inplace)

    def __assign_inventory(self, inventory, inplace=True):
        """
        Helper method for either create a new data cube or overwrite the old data cube with the given inventory.

        Parameters
        ----------
        inventory : GeoDataFrame
            Data cube inventory.
        inplace : boolean, optional
            If true, the current class instance will be altered.
            If false, a new class instance will be returned (default value is False).

        Returns
        -------
        EODataCube
        """

        if self.sdim_name not in list(inventory.keys()):
            sdim_name = None
        else:
            sdim_name = self.sdim_name

        if self.tdim_name not in list(inventory.keys()):
            tdim_name = None
        else:
            tdim_name = self.tdim_name

        if inplace:
            self.inventory = inventory
            self.sdim_name = sdim_name
            self.tdim_name = tdim_name
            return self
        else:
            return self.from_inventory(inventory=inventory, grid=self.grid,
                                       tdim_name=tdim_name, sdim_name=sdim_name)

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
        dimensions = copy.deepcopy(self.dimensions)
        inventory = copy.deepcopy(self.inventory)

        return EODataCube(filepaths=filepaths, grid=grid, dimensions=dimensions, inventory=inventory,
                          sdim_name=self.sdim_name, tdim_name=self.tdim_name)

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

    def __repr__(self):
        """
        Defines the string representation of the class.

        Returns
        -------
        str
            String representation of a data cube, i.e. its inventory.
        """

        return str(self.inventory)


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
    united_inventory = pd.concat(inventories, ignore_index=True, sort=False).drop_duplicates().reset_index(drop=True)

    sdim_name = dcs[0].sdim_name
    tdim_name = dcs[0].tdim_name
    if sdim_name not in list(united_inventory.keys()):
        sdim_name = None

    if tdim_name not in list(united_inventory.keys()):
        tdim_name = None

    dc_merged = EODataCube.from_inventory(united_inventory, grid=dcs[0].grid, sdim_name=sdim_name, tdim_name=tdim_name)

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

    sdim_name = dcs[0].sdim_name
    tdim_name = dcs[0].tdim_name
    if sdim_name not in list(intersected_inventory.keys()):
        sdim_name = None

    if tdim_name not in list(intersected_inventory.keys()):
        tdim_name = None

    dc_merged = EODataCube.from_inventory(intersected_inventory, grid=dcs[0].grid,
                                          sdim_name=sdim_name, tdim_name=tdim_name)

    return dc_merged

