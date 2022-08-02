# general packages
import copy
import glob
import os
import re
import uuid
import abc
import deprecation
import netCDF4
import pandas as pd
import numpy as np
import xarray as xr
import warnings
from typing import List, Tuple
from tempfile import mkdtemp
from datetime import datetime
from collections import OrderedDict
from multiprocessing import Pool
from collections import defaultdict

# geo packages
from osgeo import osr
from osgeo import ogr
import shapely.wkt
from shapely.geometry import Polygon
from geopandas import GeoSeries
from geopandas import GeoDataFrame
from geopandas.base import is_geometry_type

from veranda.raster.native.geotiff import GeoTiffFile
from veranda.raster.native.netcdf import NetCdf4File
from veranda.raster.mosaic.geotiff import GeoTiffReader, GeoTiffWriter
from veranda.raster.mosaic.netcdf import NetCdfReader, NetCdfWriter

from geospade.tools import rasterise_polygon
from geospade.raster import RasterGeometry
from geospade.raster import SpatialRef
from geospade.transform import xy2ij
from geospade.transform import ij2xy
from geospade.raster import MosaicGeometry

# load yeoda's utils module
from yeoda.utils import get_file_type
from yeoda.utils import any_geom2ogr_geom
from yeoda.utils import to_list
from yeoda.utils import swap_axis
from yeoda.utils import get_polygon_envelope


# load classes from yeoda's error module
from yeoda.errors import IOClassNotFound
from yeoda.errors import DataTypeUnknown
from yeoda.errors import TileNotAvailable
from yeoda.errors import FileTypeUnknown
from yeoda.errors import DimensionUnkown
from yeoda.errors import LoadingDataError
from yeoda.errors import SpatialInconsistencyError


FILE_CLASS = {'.tif': GeoTiffFile,
              '.nc': NetCdf4File}
RASTER_DATA_CLASS = {'.tif': (GeoTiffReader, GeoTiffWriter),
                     '.nc': (NetCdfReader, NetCdfWriter)}
PROC_OBJS = {}


def parse_init(filepaths, fn_class, file_class, fn_dims, md_dims, md_decoder, tmp_dirpath):
    PROC_OBJS['filepaths'] = filepaths
    PROC_OBJS['filename_class'] = fn_class
    PROC_OBJS['file_class'] = file_class
    PROC_OBJS['fn_dims'] = fn_dims
    PROC_OBJS['md_dims'] = md_dims
    PROC_OBJS['md_decoder'] = md_decoder
    PROC_OBJS['tmp_dirpath'] = tmp_dirpath


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


def parse_filepath(slice_proc):
    filepaths = PROC_OBJS['filepaths']
    fn_class = PROC_OBJS['filename_class']
    file_class = PROC_OBJS['file_class']
    fn_dims = PROC_OBJS['fn_dims']
    md_dims = PROC_OBJS['md_dims']
    md_decoder = PROC_OBJS['md_decoder']
    tmp_dirpath = PROC_OBJS['tmp_dirpath']

    filepaths_proc = filepaths[slice_proc]
    n_files = len(filepaths_proc)
    fn_dict = defaultdict(lambda: [None] * n_files)
    use_metadata = len(md_dims) > 0
    if use_metadata:
        md_decoder = {dim: md_decoder.get(dim, lambda x: x) for dim in md_dims}
    for i, filepath in enumerate(filepaths_proc):
        print(i)
        fn = fn_class.from_filename(os.path.basename(filepath), convert=True)
        for dim in fn_dims:
            fn_dict[dim][i] = fn[dim]

        if use_metadata:
            with file_class(filepath, mode='r') as file:
                for dim in md_dims:
                    fn_dict[dim][i] = md_decoder[dim](file.metadata.get(dim, None))

    df = pd.DataFrame(fn_dict)
    tmp_filename = f"{datetime.utcnow().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex}.df"
    tmp_filepath = os.path.join(tmp_dirpath, tmp_filename)
    df.to_pickle(tmp_filepath)


class DataCube(metaclass=abc.ABCMeta):
    def __init__(self, raster_data):
        self._raster_data = raster_data

    @property
    def mosaic(self) -> MosaicGeometry:
        """ Mosaic geometry of the raster mosaic files. """
        return self._raster_data.mosaic

    @property
    def n_tiles(self) -> int:
        """ Number of tiles. """
        return self._raster_data.n_tiles

    @property
    def data_geom(self) -> RasterGeometry:
        """ Raster/tile geometry of the raster mosaic files. """
        return self._raster_data.data_geom

    @property
    def file_register(self) -> pd.DataFrame:
        """ File register of the raster data object. """
        return self._raster_data.file_register

    @property
    def filepaths(self) -> List[str]:
        """ Unique list of file paths stored in the file register. """
        return self._raster_data.filepaths

    @property
    def data_view(self) -> xr.Dataset:
        """ View on internal raster data. """
        return self._raster_data.data_view

    def rename_dimensions(self, dimensions_map, inplace=False) -> "DataCube":
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
        DataCube
            `DataCube` object with renamed dimensions/columns of the inventory.
        """
        if not inplace:
            new_datacube = copy.deepcopy(self)
            return new_datacube.rename_dimensions(dimensions_map, inplace=True)

        for old_dimension in list(dimensions_map.keys()):
            if self._raster_data._file_dim == old_dimension:
                self._raster_data._file_dim = dimensions_map[old_dimension]
            if self._raster_data._tile_dim == old_dimension:
                self._raster_data._tile_dim = dimensions_map[old_dimension]

        self._raster_data._file_register.rename(columns=dimensions_map, inplace=True)
        return self

    def add_dimension(self, name, values, inplace=False) -> "DataCube":
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
        DataCube
            `DataCube` object with an additional dimension in the inventory.
        """
        if not inplace:
            new_datacube = copy.deepcopy(self)
            return new_datacube.add_dimension(name, values, inplace=True)

        ds = pd.Series(values, index=self.file_register.index)
        self._raster_data._file_register[name] = ds

        return self

    def select_files_with_pattern(self, pattern, full_path=False, inplace=False) -> "DataCube":
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
        DataCube
            `DataCube` object with a filtered inventory according to the given pattern.

        """
        if not inplace:
            new_datacube = copy.deepcopy(self)
            return new_datacube.select_files_with_pattern(pattern, full_path=full_path, inplace=True)

        pattern = re.compile(pattern)
        if not full_path:
            file_filter = lambda x: re.search(pattern, os.path.basename(x)) is not None
        else:
            file_filter = lambda x: re.search(pattern, x) is not None
        idx_filter = [file_filter(filepath) for filepath in self.filepaths]
        self._raster_data._file_register = self._raster_data._file_register[idx_filter]

        return self

    def sort_by_dimension(self, name, ascending=True, inplace=False) -> "DataCube":
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
        DataCube
            Sorted DataCube object.
        """
        if not inplace:
            new_datacube = copy.deepcopy(self)
            return new_datacube.sort_by_dimension(name, ascending=ascending, inplace=True)

        self._raster_data._file_register.sort_values(by=name, ascending=ascending, inplace=True)

        return self

    def select_by_dimension(self, expressions, name=None, inplace=False) -> "DataCube":
        """
        Filters the data cube according to the given extents and returns a (new) data cube.

        Parameters
        ----------
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
        DataCube
            Filtered DataCube object.
        """
        if not inplace:
            new_datacube = copy.deepcopy(self)
            return new_datacube.select_by_dimension(expressions, name=name, inplace=True)

        name = self._raster_data._file_dim if name is None else name
        sel_mask = np.zeros(len(self._raster_data._file_register), dtype=bool)
        for expression in to_list(expressions):
            sel_mask = sel_mask | expression(self._raster_data._file_register[name])
        self._raster_data._file_register = self._raster_data._file_register[sel_mask]

        return self

    def split_by_dimension(self, expressions, name=None) -> List["DataCube"]:
        """
        Filters the data cube according to the given extents and returns a (new) data cube.

        Parameters
        ----------
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
        datacubes : list
            Filtered DataCube objects.
        """

        datacubes = [self.select_by_dimension(expression, name=name, inplace=False)
                     for expression in expressions]

        return datacubes

    def split_by_temporal_freq(self, time_freq, name=None) -> List["DataCube"]:
        """
        Filters the data cube according to the given extents and returns a (new) data cube.

        Parameters
        ----------
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
        datacubes : list
            Filtered DataCube objects.
        """
        name = name if name is not None else self._raster_data._file_dim
        min_time, max_time = min(self.file_register[name]), max(self.file_register[name])
        time_ranges = pd.date_range(min_time, max_time, freq=time_freq)
        expressions = [lambda x: time_ranges[i] <= x <= time_ranges[i + 1]
                       for i in range(len(time_ranges) - 1)]

        return self.split_by_dimension(expressions, name=name)

    def select_tiles(self, tile_names, inplace=False) -> "DataCube":
        """
        Selects certain tiles from a datacube.

        Parameters
        ----------
        tile_names : list of str
            Tile names/IDs.
        inplace : bool, optional
            If True, the current datacube is modified.
            If False, a new datacube instance will be returned (default).

        Returns
        -------
        DataCube :
            DataCube object with a mosaic and a file register only consisting of the given tiles.

        """
        if not inplace:
            new_datacube = copy.deepcopy(self)
            return new_datacube.select_tiles(tile_names, inplace=True)

        self._raster_data.select_tiles(tile_names, inplace=True)

        return self

    def select_px_window(self, row, col, height=1, width=1, inplace=False) -> "DataCube":
        """
        Selects the pixel coordinates according to the given pixel window.

        Parameters
        ----------
        row : int
            Top-left row number of the pixel window anchor.
        col : int
            Top-left column number of the pixel window anchor.
        height : int, optional
            Number of rows/height of the pixel window. Defaults to 1.
        width : int, optional
            Number of columns/width of the pixel window. Defaults to 1.
        inplace : bool, optional
            If True, the current datacube is modified.
            If False, a new datacube instance will be returned (default).

        Returns
        -------
        Data :
            Raster data object with a data and a mosaic geometry only consisting of the intersected tile with the
            pixel window.

        Notes
        -----
        The mosaic will be only sliced if it consists of one tile to prevent ambiguities in terms of the definition
        of the pixel window.

        """
        if not inplace:
            new_datacube = copy.deepcopy(self)
            return new_datacube.select_px_window(row, col, height=height, width=width, inplace=True)

        self._raster_data.select_px_window(row, col, height=height, width=width, inplace=True)

        return self

    def select_xy(self, x, y, sref=None, inplace=False) -> "DataCube":
        """
        Selects a pixel according to the given coordinate tuple.

        Parameters
        ----------
        x : number
            Coordinate in X direction.
        y : number
            Coordinate in Y direction.
        sref : geospade.crs.SpatialRef, optional
            CRS of the given coordinate tuple. Defaults to the CRS of the mosaic.
        inplace : bool, optional
            If True, the current datacube is modified.
            If False, a new datacube instance will be returned (default).

        Returns
        -------
        DataCube :
            Raster data object with a file register and a mosaic only consisting of the intersected tile containing
            information on the location of the time series.

        """
        if not inplace:
            new_datacube = copy.deepcopy(self)
            return new_datacube.select_xy(x, y, sref=sref, inplace=True)

        self._raster_data.select_xy(x, y, sref=sref, inplace=True)

        return self

    def select_bbox(self, bbox, sref=None, inplace=False) -> "DataCube":
        """
        Selects tile and pixel coordinates according to the given bounding box.

        Parameters
        ----------
        bbox : list of 2 2-tuple
            Bounding box to select, i.e. [(x_min, y_min), (x_max, y_max)]
        sref : geospade.crs.SpatialRef, optional
            CRS of the given bounding box coordinates. Defaults to the CRS of the mosaic.
        inplace : bool, optional
            If True, the current datacube object is modified.
            If False, a new datacube instance will be returned (default).

        Returns
        -------
        DataCube :
            Raster data object with a file register and a mosaic only consisting of the intersected tiles.

        """
        if not inplace:
            new_datacube = copy.deepcopy(self)
            return new_datacube.select_bbox(bbox, sref=sref, inplace=True)

        return self.select_polygon(bbox, apply_mask=False, inplace=inplace)

    def select_polygon(self, polygon, sref=None, apply_mask=True, inplace=False) -> "DataCube":
        """
        Selects tile and pixel coordinates according to the given polygon.

        Parameters
        ----------
        polygon : ogr.Geometry
            Polygon specifying the pixels to collect.
        sref : geospade.crs.SpatialRef, optional
            CRS of the given bounding box coordinates. Defaults to the CRS of the mosaic.
        apply_mask : bool, optional
            True if pixels outside the polygon should be set to a no data value (default).
            False if every pixel withing the bounding box of the polygon should be included.
        inplace : bool, optional
            If True, the current datacube object is modified.
            If False, a new datacube instance will be returned (default).

        Returns
        -------
        DataCube :
            Raster data object with a file register and a mosaic only consisting of the intersected tiles.

        """
        if not inplace:
            new_datacube = copy.deepcopy(self)
            return new_datacube.select_polygon(polygon, sref=sref, apply_mask=apply_mask, inplace=True)

        self._raster_data.select_polygon(polygon, sref=sref, apply_mask=apply_mask, inplace=True)

        return self

    def intersect(self, other, on_dimension=None, inplace=False) -> "DataCube":
        """
        Intersects this data cube with another data cube. This is equal to an SQL INNER JOIN operation.
        In other words:
            - all uncommon columns and rows (if `on_dimension` is given) are removed
            - duplicates are removed

        Parameters
        ----------
        other : DataCube
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
        if not inplace:
            new_datacube = copy.deepcopy(self)
            return new_datacube.intersect(other, on_dimension=on_dimension, inplace=True)

        # close all open file handles before operation
        self.close()
        other.close()

        self._check_dc_compliance(other)

        file_registers = [self.file_register, other.file_register]
        intsct_fr = pd.concat(file_registers, ignore_index=True, join='inner')
        if on_dimension is not None:
            all_vals = []
            for file_register in file_registers:
                all_vals.append(list(file_register[on_dimension]))
            common_vals = list(set.intersection(*map(set, all_vals)))
            intsct_fr = intsct_fr[intsct_fr[on_dimension].isin(common_vals)]

        intsct_fr = intsct_fr.drop_duplicates().reset_index(drop=True)
        self._raster_data._file_register = intsct_fr
        self.add_dimension("file_id", [None] * len(self), inplace=True)

        return self

    def unite(self, other, inplace=False) -> "DataCube":
        """
        Unites this data cube with respect to another data cube. This is equal to an SQL UNION operation.
        In other words:
            - all columns are put into one DataFrame
            - duplicates are removed
            - gaps are filled with NaN

        Parameters
        ----------
        other : DataCube
            Data cube to unite with.
        inplace : boolean, optional
            If true, the current class instance will be altered.
            If false, a new class instance will be returned (default is False).

        Returns
        -------
        DataCube
            United datacubes.
        """
        if not inplace:
            new_datacube = copy.deepcopy(self)
            return new_datacube.unite(other, inplace=True)

        # close all open file handles before operation
        self.close()
        other.close()

        self._check_dc_compliance(other)

        file_registers = [self.file_register, other.file_register]
        # this is a SQL alike UNION operation
        united_frs = pd.concat(file_registers, ignore_index=True, sort=False).drop_duplicates().reset_index(drop=True)
        self._raster_data._file_register = united_frs
        self.add_dimension("file_id", [None] * len(self), inplace=True)

        return self

    def align_dimension(self, other, name, inplace=False) -> "DataCube":
        """
        Aligns this data cube with another data cube along the specified dimension `name`.

        Parameters
        ----------
        other : EDataCube
            Data cube to align with.
        name : str
            Name of the dimension, which is used for aligning/filtering the values for all data cubes.
        inplace : boolean, optional
            If true, the current class instance will be altered.
            If false, a new class instance will be returned (default value is False).

        Returns
        -------
        DataCube
            Datacube with common values along the given dimension with respect to another data cube.
        """
        if not inplace:
            new_datacube = copy.deepcopy(self)
            return new_datacube.align_dimension(other, name=name, inplace=True)

        self._check_dc_compliance(other)

        this_dim_values = list(self.file_register[name])
        uni_values = list(set(this_dim_values))
        other_dim_values = other.file_register[name]
        idxs = np.zeros(len(other_dim_values)) - 1  # set -1 as no data value

        for i in range(len(uni_values)):
            val_idxs = np.where(uni_values[i] == other_dim_values)
            idxs[val_idxs] = this_dim_values.index(uni_values[i])  # get index of value in this data cube

        idxs = idxs[idxs != -1]
        if len(idxs) > 0:
            # close all open file handles before operation
            self.close()
            other.close()

            self._raster_data._file_register = self._raster_data._file_register.iloc[idxs].reset_index(drop=True)
        else:
            wrn_msg = "No common dimension values found. Original datacube is returned."
            warnings.warn(wrn_msg)

        return self

    def _check_dc_compliance(self, other):

        if self._file_dim != other._file_dim:
            err_msg = f"Both datacubes must have the same file dimension ({self._file_dim} != {other._file_dim})."
            raise ValueError(err_msg)
        if self._tile_dim != other._tile_dim:
            err_msg = f"Both datacubes must have the same tile dimension ({self._tile_dim} != {other._tile_dim})."
            raise ValueError(err_msg)

    def apply_nan(self):
        """
        Converts no data values given as an attribute '_FillValue' to np.nan. Note that this replacement implicitly
        converts the data format to float.

        """
        self._raster_data.apply_nan()

    def close(self):
        """ Closes open file handles. """
        self._raster_data.close()

    def clear_ram(self):
        """ Releases memory allocated by the internal data object. """
        self._raster_data.clear_ram()

    def clone(self) -> "DataCube":
        """
        Clones, i.e. deep-copies a datacube.

        Returns
        -------
        DataCube
            Cloned/copied datacube.
        """

        return copy.deepcopy(self)

    def __getitem__(self, dimension_name) -> pd.Series:
        """
        Returns a column of the internal inventory according to the given column name/item.

        Parameters
        ----------
        dimension_name : str
            Column/Dimension name of the datacube file register.

        Returns
        -------
        pandas.DataSeries
            Column of the internal inventory.

        """

        if dimension_name in self.file_register.columns:
            return self.file_register[dimension_name]
        else:
            raise DimensionUnkown(dimension_name)

    def __len__(self):
        return len(self.file_register)

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        self.close()

    def __deepcopy__(self, memo):
        """
        Deepcopy method of the `DataCube` class.

        Parameters
        ----------
        memo : dict

        Returns
        -------
        DataCube
            Deepcopy of a datacube.

        """

        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memo))

        return result

    def __repr__(self) -> str:
        """ General string representation of a datacube instance. """
        return f"{self.__class__.__name__}({self._raster_data._file_dim}, {self.mosaic.__class__.__name__}):\n\n" \
               f"{repr(self.file_register)}"


class DataCubeReader(DataCube):
    def __init__(self, file_register, mosaic, stack_dimension='layer_id', tile_dimension='tile_id', **kwargs):
        ref_filepath = file_register['filepath'].iloc[0]
        reader_class = RASTER_DATA_CLASS[os.path.basename(ref_filepath)[-1]][0]
        reader = reader_class(file_register, mosaic, stack_dimension=stack_dimension, tile_dimension=tile_dimension,
                              **kwargs)
        super().__init__(reader)

    @classmethod
    def from_filepaths(cls, filepaths, filename_class, mosaic=None, dimensions=None, tile_dimension='tile',
                       stack_dimension='time', use_metadata=False, md_decoder=None, n_cores=1, **kwargs) -> "DataCube":
        """
        Creates an `EODataCube` instance from a list of filepaths.

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
        md_decoder = {} if md_decoder is None else md_decoder
        n_files = len(filepaths)
        step = int(n_files / n_cores)
        slices = []
        for i in range(0, n_files, step):
            slices.append(slice(i, i + step))
        slices[-1] = slice(slices[-1].start, n_files + 1)

        ref_filepath = filepaths[0]
        fn_dims = cls.__get_dims_from_fn(filename_class.from_filename(ref_filepath), dimensions=dimensions)
        md_dims = [] if not use_metadata else cls.__get_dims_from_md(ref_filepath, dimensions=dimensions)
        file_class = FILE_CLASS[os.path.splitext(ref_filepath)[-1]]
        tmp_dirpath = mkdtemp()
        with Pool(n_cores, initializer=parse_init, initargs=(filepaths, filename_class, file_class, fn_dims, md_dims,
                                                             md_decoder, tmp_dirpath)) as p:
            p.map(parse_filepath, slices)

        df_filepaths = glob.glob(os.path.join(tmp_dirpath, "*.df"))
        file_register = pd.concat([pd.read_pickle(df_filepath) for df_filepath in df_filepaths])

        return cls(file_register, mosaic, stack_dimension=stack_dimension, tile_dimension=tile_dimension, **kwargs)

    def read(self):
        self._raster_data.read()


class DataCubeWriter(DataCube):
    def __init__(self, mosaic, file_register=None, data=None, format='.nc', stack_dimension='layer_id',
                 tile_dimension='tile_id', **kwargs):
        format = format if file_register is None else os.path.basename(file_register['filepath'].iloc[0])[-1]
        writer_class = RASTER_DATA_CLASS[format][1]
        writer = writer_class(mosaic, file_register=file_register, data=data,
                              stack_dimension=stack_dimension, tile_dimension=tile_dimension, **kwargs)
        super().__init__(writer)

    @classmethod
    def from_data(cls, data, dirpath, filename_class, resampler, mosaic=None, stack_dimension='layer_id',
                  tile_dimension='tile_id', fn_map=None):
        pass


    def write(self, data, apply_tiling=False, data_variables=None, encoder=None, encoder_kwargs=None, overwrite=False,
              unlimited_dims=None, **kwargs):
        self._raster_data(data, apply_tiling=apply_tiling, data_variables=data_variables, encoder=encoder,
                          encoder_kwargs=encoder_kwargs, overwrite=overwrite, unlimited_dims=unlimited_dims, **kwargs)

    def export(self, apply_tiling=False, data_variables=None, encoder=None, encoder_kwargs=None, overwrite=False,
              unlimited_dims=None, **kwargs):
        self._raster_data(apply_tiling=apply_tiling, data_variables=data_variables, encoder=encoder,
                          encoder_kwargs=encoder_kwargs, overwrite=overwrite, unlimited_dims=unlimited_dims, **kwargs)




@deprecation.deprecated(deprecated_in="1.0.0",
                        current_version="1.0.0",
                        details="Use DataCubeReader or DataCubeWriter instead")
class EODataCube:
    """
    A file(name) based data cube for preferably gridded and well-structured EO data.

    """

    def __init__(self, file_register, mosaic=None, data=None, tile_dimension="tile", stack_dimension="time"):
        """
        Constructor of the `EODataCube` class.

        Parameters
        ----------
        filepaths : list of str, optional
            List of file paths.
        grid : pytileproj.base.TiledProjection, optional
            Tiled projection/grid object/class (e.g. `Equi7Grid`, `LatLonGrid`).
        filename_class : geopathfinder.file_naming.SmartFilename, optional
            `SmartFilename` class to handle the interpretation of file names.
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
        self._reader = None
        self._writer = None
        # initialise simple class variables
        self._ds = None  # data set pointer
        self.status = None
        self.tdim_name = tdim_name
        self._filename_class = filename_class

        # initialise IO classes responsible for reading and writing
        if io_map is not None:
            self.io_map = io_map
        else:
            self.io_map = {'GeoTIFF': GeoTiffFile, 'NetCDF': NetCdf4File}

        # create inventory from found filepaths
        self.inventory = None
        if inventory is not None:
            self.inventory = inventory
        else:
            self.__inventory_from_filepaths(filepaths, dimensions=dimensions)

        self.grid = None
        if grid:
            self.grid = grid
            self.sdim_name = sdim_name
        elif (self.inventory is not None) and ('geometry' not in self.inventory.keys()):
            geometries = [self.__raster_geom_from_file(filepath).boundary_shapely
                          for filepath in self.filepaths]
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
        ogr.geometry : OGR polygon representing the boundary/envelope of the data cube or `None` if no files are
        contained in the data cube.

        """

        self.__check_spatial_consistency()
        boundary = None
        if self.filepaths is not None:
            filepath = self.filepaths[0]
            raster_geom = self.__raster_geom_from_file(filepath)
            boundary = raster_geom.boundary_ogr
            boundary = swap_axis(boundary)  # ensure lon-lat order

        return boundary

    @property
    def raster_geometry(self):
        """
        geospade.raster.RasterGeometry :
            Raster geometry representing the geometric properties of the given file. Note
            that the raster geometry is extracted from the first file, so be sure that the
            datacube only holds files from the same tile of the grid.

        """
        self.__check_spatial_consistency()
        raster_geom = None
        if not self.inventory.empty:
            raster_geom = self.__raster_geom_from_file(self.filepaths[0])

        return raster_geom

    @property
    def coordinate_boundary(self):
        """
        ogr.geometry : OGR polygon representing the coordinate boundary of the data cube or `None` if no files are
        contained in the data cube.

        """

        self.__check_spatial_consistency()
        coord_boundary = None
        if self.filepaths is not None:
            filepath = self.filepaths[0]
            raster_geom = self.__raster_geom_from_file(filepath)
            coord_boundary = ogr.CreateGeometryFromWkt(Polygon(raster_geom.coord_corners).wkt)
            coord_boundary.AssignSpatialReference(raster_geom.sref.osr_sref)
            coord_boundary = swap_axis(coord_boundary)  # ensure lon-lat order

        return coord_boundary

    @classmethod
    def from_filepaths(cls, filepaths, filename_class, mosaic=None, dimensions=None, tile_dimension='tile',
                       stack_dimension='time', use_metadata=False, md_decoder=None, n_cores=1, **kwargs):
        """
        Creates an `EODataCube` instance from a list of filepaths.

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
        md_decoder = {} if md_decoder is None else md_decoder
        n_files = len(filepaths)
        step = int(n_files/n_cores)
        slices = []
        for i in range(0, n_files, step):
            slices.append(slice(i, i + step))
        slices[-1] = slice(slices[-1].start, n_files + 1)

        ref_filepath = filepaths[0]
        fn_dims = cls.__get_dims_from_fn(filename_class.from_filename(ref_filepath), dimensions=dimensions)
        md_dims = [] if not use_metadata else cls.__get_dims_from_md(ref_filepath, dimensions=dimensions)
        file_class = FILE_CLASS[os.path.splitext(ref_filepath)[-1]]
        tmp_dirpath = mkdtemp()
        with Pool(n_cores, initializer=parse_init, initargs=(filepaths, filename_class, file_class, fn_dims, md_dims,
                                                             md_decoder, tmp_dirpath)) as p:
           p.map(parse_filepath, slices)

        df_filepaths = glob.glob(os.path.join(tmp_dirpath, "*.df"))
        df = pd.concat([pd.read_pickle(df_filepath) for df_filepath in df_filepaths])

        return cls(df)


    #parse_filepath(filepaths, fn_map)
        # c_dtype = np.ctypeslib.as_ctypes_type('char')
        # for dimension in dimensions:
        #     data_nshm = np.ones((n_files,), dtype=fn_dtypes[dimension])
        #     c_dtype = np.ctypeslib.as_ctypes_type(data_nshm.dtype)
        #     shm_rar = RawArray(c_dtype, data_nshm.size)
        #     shm_data = np.frombuffer(shm_rar, dtype=fn_dtypes[dimension])
        #     shm_data[:] = data_nshm[:]
        #     shm_map[dimension] = shm_data



        pass
        return #cls(inventory=inventory, grid=grid, **kwargs)

    # @staticmethod
    # def __get_dims_from_fn(fn, dimensions=None):
    #     dimensions = list(fn.fields_def.keys()) if dimensions is None else dimensions
    #     return {dimension: np.array(fn[dimension]).dtype for dimension in dimensions}

    @staticmethod
    def __get_dims_from_fn(fn, dimensions=None):
        fn_dims = list(fn.fields_def.keys())
        if dimensions is not None:
            fn_dims = list(set(fn_dims).intersection(set(dimensions)))
        return fn_dims

    @staticmethod
    def __get_dims_from_md(filepath, dimensions=None):
        file_class = FILE_CLASS[os.path.splitext(filepath)[1]]
        with file_class(filepath, mode='r') as file:
            md = file.metadata
            md_dims = list(md.keys())
            if dimensions is not None:
                md_dims = list(set(dimensions).intersection(md_dims))

            return md_dims

    def __parse_filepath(self, file_idx):
        print(file_idx)
        shm_map = PROC_OBJS['shm_map']
        filepaths = PROC_OBJS['filepaths']
        fn_class = PROC_OBJS['filename_class']
        filepath = filepaths[file_idx]
        file_class = self.__io_class(get_file_type(filepath))
        fn = fn_class.from_filename(os.path.basename(filepath), convert=True)
        fn_dims = list(fn.fields_def.keys())
        dimensions = list(shm_map.keys())
        for dimension in dimensions:
            shm_rar = shm_map[dimension]
            if dimension in fn_dims:
                value = fn[dimension]
            else:
                with file_class(filepath, mode='r') as file:
                    value = file.metadata[dimension]
            shm_map[dimension][file_idx] = value

    @classmethod
    def from_disk(cls):
        pass

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
        return self._assign_inventory(inventory, inplace=inplace)

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
        return self._assign_inventory(inventory, inplace=inplace)

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
        return self._assign_inventory(inventory, inplace=inplace)

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
        return self._assign_inventory(inventory, inplace=inplace)

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

        return self._assign_inventory(inventory=inventory_sorted)

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

        tilenames = to_list(tilenames)

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
            geom_roi = swap_axis(geom_roi)
            ftilenames = self.grid.search_tiles_over_geometry(geom_roi)
            tilenames = [ftilename.split('_')[1] for ftilename in ftilenames]
            return self.filter_spatially_by_tilename(tilenames, inplace=inplace, use_grid=False)
        elif 'geometry' in self.inventory.keys():
            # get spatial reference of data
            geom_roi = self.align_geom(geom_roi)
            geom_roi = shapely.wkt.loads(geom_roi.ExportToWkt())
            inventory = self.inventory[self.inventory.intersects(geom_roi)]
            return self._assign_inventory(inventory, inplace=inplace)
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
                yearly_months = to_list(months)
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
            years = to_list(years)
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
    def load_by_geom(self, geom, sref=None, band=1, apply_mask=True, dtype="xarray", origin='ul',
                     decode_kwargs=None):
        """
        Loads data according to a given geometry.

        Parameters
        ----------
        geom : OGR Geometry or Shapely Geometry or list or tuple
            A geometry defining the region of interest. If it is of type list/tuple representing the extent
            (i.e. [x_min, y_min, x_max, y_max]), `sref` has to be given to transform the extent into a
            georeferenced polygon.
        sref : osr.SpatialReference, optional
            Spatial reference of the given region of interest `geom`.
        band : int or str, optional
            Band number or name (default is 1).
        apply_mask : bool, optional
            If true, a numpy mask array with a mask excluding (=1) all pixels outside `geom` (=0) will be created
            (default is False).
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

        if self.inventory is None or self.inventory.empty:  # no data given
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
            geom_roi = swap_axis(geom_roi)

        # clip region of interest to tile boundary
        geom_roi = geom_roi.Intersection(self.coordinate_boundary)
        if geom_roi.IsEmpty():
            raise Exception('The given geometry does not intersect with the tile boundaries.')
        geom_roi.AssignSpatialReference(this_sref)

        # remove third dimension from geometry
        geom_roi.FlattenTo2D()

        # retrieve extent of polygon with respect to the pixel sampling of the grid
        x_pixel_size = abs(this_gt[1])
        y_pixel_size = abs(this_gt[5])
        extent = get_polygon_envelope(shapely.wkt.loads(geom_roi.ExportToWkt()),
                                      x_pixel_size,
                                      y_pixel_size)
        inv_traffo_fun = lambda i, j: ij2xy(i, j, this_gt, origin=origin)
        min_col, min_row = [int(coord) for coord in xy2ij(extent[0], extent[3], this_gt)]
        max_col, max_row = [int(coord) for coord in xy2ij(extent[2], extent[1], this_gt)]
        max_col, max_row = max_col + 1, max_row + 1 # plus one to still include the maximum indexes

        if apply_mask:
            # pixel size extraction assumes non-rotated data
            data_mask = np.invert(rasterise_polygon(shapely.wkt.loads(geom_roi.ExportToWkt()),
                                                    x_pixel_size,
                                                    y_pixel_size).astype(bool))

        file_type = get_file_type(self.filepaths[0])
        xs = None
        ys = None
        if file_type == "GeoTIFF":
            if self._ds is None and self.status != "stable":
                file_ts = {'filenames': list(self.filepaths)}
                self._ds = GeoTiffRasterTimeStack(file_ts=file_ts, file_band=band)
            col_size = max_col - min_col
            row_size = max_row - min_row
            data = self._ds.read_ts(min_col, min_row, col_size=col_size, row_size=row_size)
            if data is None:
                raise LoadingDataError()
            data = self.decode(data, **decode_kwargs)
            if len(data.shape) == 2:  # ensure that the data is always forwarded as a 3D array
                data = data[None, :, :]
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

        if self.inventory is None or self.inventory.empty:  # no data given
            return None

        decode_kwargs = {} if decode_kwargs is None else decode_kwargs

        rows = to_list(rows)
        cols = to_list(cols)

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
                data_i = self.decode(data_i, **decode_kwargs)
                if len(data_i.shape) == 2:  # ensure that the data is always forwarded as a 3D array
                    data_i = data_i[None, :, :]
                data.append(data_i)
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

        if self.inventory is None or self.inventory.empty:  # no data given
            return None

        decode_kwargs = {} if decode_kwargs is None else decode_kwargs

        xs = to_list(xs)
        ys = to_list(ys)

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
            col, row = [int(coord) for coord in xy2ij(x, y, this_gt)]
            # check if coordinate is within datacube
            raster_geom = self.__raster_geom_from_file(self.filepaths[0])
            if (col < 0) or (row < 0)  or (col >= raster_geom.n_cols) or (row >= raster_geom.n_rows):
                raise Exception('The given coordinate does not intersect with the tile boundaries.')
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
                if len(data_i.shape) == 2:  # ensure that the data is always forwarded as a 3D array
                    data_i = data_i[None, :, :]
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
        return self._assign_inventory(dc_intersected.inventory, inplace=inplace)

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
        return self._assign_inventory(dc_united.inventory, inplace=inplace)

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
            return self._assign_inventory(inventory, inplace=inplace)
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
        geom = swap_axis(geom)

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
                    x = to_list(x)
                    y = to_list(y)
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
            dimensions = ["time", "y", "x"]
            xr_ds = self.__convert_dtype(data, 'xarray', xs=xs, ys=ys, band=band)
            converted_data = xr_ds.to_dataframe().reset_index().sort_values(dimensions).set_index(dimensions)
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

    def __raster_geom_from_file(self, filepath):
        """
        Retrieves a raster geometry from an EO file.

        Parameters
        ----------
        filepath : str
            Filepath or filename of a geospatial file (e.g. NetCDF or GeoTIFF).

        Returns
        -------
        geospade.raster.RasterGeometry
            Raster geometry representing the geometric properties of the given file.

        """

        file_type = get_file_type(filepath)
        io_class = self.__io_class(file_type)
        io_instance = io_class(filepath, mode='r')
        sref = SpatialRef(io_instance.spatialref)
        raster_geom = RasterGeometry(io_instance.shape[0], io_instance.shape[1], sref, io_instance.geotransform)
        # close data set
        io_instance.close()

        return raster_geom

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

    def __inventory_from_filepaths(self, filepaths, dimensions=None):
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
                smart_filename = self._filename_class.from_filename(os.path.basename(filepath), convert=True)
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

        values = to_list(values)
        n_filters = len(values)
        if expressions is None:  # equal operator is the default comparison operator
            expressions = ["=="] * n_filters
        else:
            expressions = to_list(expressions)

        inventory = copy.deepcopy(self.inventory)
        filtered_inventories = []
        for i in range(n_filters):
            value = to_list(values[i])
            expression = to_list(expressions[i])

            if (len(value) == 2) and (len(expression) == 2):
                filter_cmd = "inventory[(inventory[name] {} value[0]) & " \
                             "(inventory[name] {} value[1])]".format(expression[0], expression[1])
            elif (len(value) == 1) and (len(expression) == 1):
                filter_cmd = "inventory[inventory[name] {} value[0]]".format(expression[0])
            else:
                raise Exception('Length of value (={}) and length of expression (={}) does not match or is larger than 2.'.format(len(value), len(expression)))

            filtered_inventories.append(eval(filter_cmd))

        if split:
            eodcs = [self._assign_inventory(filtered_inventory, inplace=False)
                     for filtered_inventory in filtered_inventories]
            return eodcs
        else:
            filtered_inventory = pd.concat(filtered_inventories, ignore_index=True)
            return self._assign_inventory(filtered_inventory, inplace=inplace)

    def _assign_inventory(self, inventory, inplace=True):
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
                                       dimensions=self.dimensions, filename_class=self._filename_class,
                                       io_map=self.io_map, tdim_name=tdim_name, sdim_name=sdim_name)

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

