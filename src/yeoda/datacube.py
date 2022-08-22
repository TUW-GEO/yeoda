import copy
import glob
import os
import re
import uuid
import abc
import warnings
import pandas as pd
import numpy as np
import xarray as xr
from osgeo import ogr
from typing import List, Tuple
from tempfile import mkdtemp
from datetime import datetime
from multiprocessing import Pool
from collections import defaultdict

from geopathfinder.file_naming import SmartFilename

from veranda.raster.native.geotiff import GeoTiffFile
from veranda.raster.native.netcdf import NetCdf4File
from veranda.raster.mosaic.geotiff import GeoTiffReader, GeoTiffWriter
from veranda.raster.mosaic.netcdf import NetCdfReader, NetCdfWriter

from geospade.raster import RasterGeometry
from geospade.raster import MosaicGeometry
from geospade.raster import Tile
from geospade.crs import SpatialRef
from geospade.raster import find_congruent_tile_id_from_tiles

from yeoda.utils import to_list
from yeoda.utils import create_fn_class
from yeoda.errors import DimensionUnkown
from yeoda.errors import FileTypeUnknown


FILE_CLASS = {'.tif': GeoTiffFile,
              '.nc': NetCdf4File}
RASTER_DATA_CLASS = {'.tif': (GeoTiffReader, GeoTiffWriter),
                     '.nc': (NetCdfReader, NetCdfWriter)}
PROC_OBJS = {}


def parse_init(filepaths, fn_class, fields_def, fn_kwargs, file_class, fc_kwargs, fn_dims, md_dims, md_decoder,
               tmp_dirpath):
    """ Helper method for setting the entries of global variable `PROC_OBJS` to be available during multiprocessing. """
    PROC_OBJS['filepaths'] = filepaths
    PROC_OBJS['fn_class'] = fn_class
    PROC_OBJS['fields_def'] = fields_def
    PROC_OBJS['fn_kwargs'] = fn_kwargs
    PROC_OBJS['file_class'] = file_class
    PROC_OBJS['file_class_kwargs'] = fc_kwargs
    PROC_OBJS['fn_dims'] = fn_dims
    PROC_OBJS['md_dims'] = md_dims
    PROC_OBJS['md_decoder'] = md_decoder
    PROC_OBJS['tmp_dirpath'] = tmp_dirpath


def parse_filepaths(slice_proc):
    """
    Parses a portion of file paths, i.e. retrieves decoded attributes from the file name itself or the metadata, and
    writes the output as a data frame to disk (for joining it with the output all workers afterwards).

    Parameters
    ----------
    slice_proc : slice
        Index range corresponding to the filepaths to parse.

    """
    filepaths = PROC_OBJS['filepaths']
    fn_class = PROC_OBJS['fn_class']
    fields_def = PROC_OBJS['fields_def']
    fn_kwargs = PROC_OBJS['fn_kwargs']
    file_class = PROC_OBJS['file_class']
    file_class_kwargs = PROC_OBJS['file_class_kwargs']
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
        fn_dict['filepath'][i] = filepath
        try:
            fn_args = [os.path.basename(filepath)]
            if fields_def is not None:
                fn_args.append(fields_def)
            fn = fn_class.from_filename(*fn_args, convert=True, **fn_kwargs)
            for dim in fn_dims:
                fn_dict[dim][i] = fn[dim]
        except:
            pass

        if use_metadata:
            try:
                with file_class(filepath, mode='r', **file_class_kwargs) as file:
                    for dim in md_dims:
                        fn_dict[dim][i] = md_decoder[dim](file.metadata.get(dim, None))
            except:
                pass

    df = pd.DataFrame(fn_dict)
    tmp_filename = f"{datetime.utcnow().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex}.df"
    tmp_filepath = os.path.join(tmp_dirpath, tmp_filename)
    df.to_pickle(tmp_filepath)


class DataCube(metaclass=abc.ABCMeta):
    """ Basic datacube class defining all selection and datacube operations. """
    def __init__(self, raster_data):
        """
        Constructor of `DataCube`.

        Parameters
        ----------
        raster_data : veranda.raster.mosaic.base.RasterData
            Raster data object (reader or writer) storing a file register, data and a mosaic.

        """
        self._raster_data = raster_data

    @property
    def dimensions(self) -> list:
        """ Dimensions of the datacube. i.e. the columns of the file register without the 'filepath' entry. """
        fr_cols = list(self.file_register.columns)
        fr_cols.remove('filepath')
        return fr_cols

    @property
    def mosaic(self) -> MosaicGeometry:
        """ Mosaic geometry representing the spatial properties of the datacube. """
        return self._raster_data.mosaic

    @property
    def n_tiles(self) -> int:
        """ Number of tiles. """
        return self._raster_data.n_tiles

    @property
    def data_geom(self) -> RasterGeometry:
        """ Raster/tile geometry of the internal data. """
        return self._raster_data.data_geom

    @property
    def file_register(self) -> pd.DataFrame:
        """ File register of the datacube. """
        return self._raster_data.file_register

    @property
    def filepaths(self) -> List[str]:
        """
        Unique list of file paths stored in the file register. Note that this property does not preserve the order of
        the file paths in the file register.

        """
        return self._raster_data.filepaths

    @property
    def data_view(self) -> xr.Dataset:
        """ View on internal raster data. """
        return self._raster_data.data_view

    @property
    def is_empty(self) -> bool:
        """ Checks if the datacube is empty, i.e. does not contain any files. """
        return len(self) == 0

    def rename_dimensions(self, dimensions_map, inplace=False) -> "DataCube":
        """
        Renames the dimensions of the datacube.

        Parameters
        ----------
        dimensions_map : dict
            A dictionary representing the relation between old and new dimension names. The keys are the old dimension
            names, the values the new dimension names (e.g., {'time_begin': 'time'}).
        inplace : boolean, optional
            If true, the current class instance will be altered.
            If false (default), a new class instance will be returned.

        Returns
        -------
        DataCube
            `DataCube` object with renamed dimensions/columns of the file register.

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
        Adds a new dimension to the datacube.

        Parameters
        ----------
        name : str
            Name of the new dimension.
        values : list
            Values along the new dimension (e.g., cloud cover, quality flag, ...).
            They have to have the same length as all the rows in the file register.
        inplace : boolean, optional
            If true, the current class instance will be altered.
            If false (default), a new class instance will be returned.

        Returns
        -------
        DataCube
            `DataCube` object with an additional dimension in the file register.

        """
        if not inplace:
            new_datacube = copy.deepcopy(self)
            return new_datacube.add_dimension(name, values, inplace=True)

        ds = pd.Series(values, index=self.file_register.index)
        self._raster_data._file_register[name] = ds

        return self

    def select_files_with_pattern(self, pattern, full_path=False, inplace=False) -> "DataCube":
        """
        Filters all file paths according to the given pattern.

        Parameters
        ----------
        pattern : str
            A regular expression (e.g., ".*S1A.*GRD.*").
        full_path : boolean, optional
            Uses the full file paths for filtering if it is set to `True`.
            Otherwise, the file name is used (default value is `False`).
        inplace : boolean, optional
            If true, the current class instance will be altered.
            If false (default), a new class instance will be returned.

        Returns
        -------
        DataCube
            `DataCube` object with a filtered file register according to the given pattern.

        """
        if not inplace:
            new_datacube = copy.deepcopy(self)
            return new_datacube.select_files_with_pattern(pattern, full_path=full_path, inplace=True)

        pattern = re.compile(pattern)
        if not full_path:
            file_filter = lambda x: re.search(pattern, os.path.basename(x)) is not None
        else:
            file_filter = lambda x: re.search(pattern, x) is not None
        idx_filter = [file_filter(filepath) for filepath in self['filepath']]
        self._raster_data._file_register = self._raster_data._file_register[idx_filter]

        return self

    def sort_by_dimension(self, name, ascending=True, inplace=False) -> "DataCube":
        """
        Sorts the datacube/file register according to the given dimension.

        Parameters
        ----------
        name : str
            Name of the dimension.
        ascending : bool, optional
            If true (default), sorts in ascending order, otherwise in descending order.
        inplace : boolean, optional
            If true, the current class instance will be altered.
            If false (default), a new class instance will be returned.

        Returns
        -------
        DataCube
            Sorted datacube.

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
        expressions : callable
            A list of functions expecting one input argument, which will be replace with the respective column of the
            file register later on, and returning a boolean value for each entry in the file register (used for the
            decision if it will be selected or not). Two examples are given below:
                - `datacube.select_by_dimension(lambda s: s == "X", name='dim', inplace=True)`
                - `datacube.select_by_dimension(lambda t: (t >= start_time) & (t <= end_time), name='time',
                inplace=True)`
        name : str, optional
            Name of the dimension. Defaults to the name of the stack dimension.
        inplace : boolean, optional
            If true, the current class instance will be altered.
            If false (default), a new class instance will be returned.

        Returns
        -------
        DataCube
            Subset of the original datacube.

        Notes
        -----
        The results of the expressions are concatenated via an OR operation.

        """

        if not inplace:
            new_datacube = copy.deepcopy(self)
            return new_datacube.select_by_dimension(expressions, name=name, inplace=True)

        name = self._raster_data._file_dim if name is None else name
        sel_mask = np.zeros(len(self._raster_data._file_register), dtype=bool)
        for expression in to_list(expressions):
            sel_mask = sel_mask | expression(self._raster_data._file_register[name])
        self._raster_data._file_register = self._raster_data._file_register[sel_mask]

        if name == self._raster_data._tile_dim:
            self.select_tiles(list(set(self[name])), inplace=True)

        return self

    def split_by_dimension(self, expressions, name=None) -> List["DataCube"]:
        """
        Creates subsets/a new datacube from the original datacube for each expression.

        Parameters
        ----------
        expressions : callable
            A list of functions expecting one input argument, which will be replace with the respective column of the
            file register later on, and returning a boolean value for each entry in the file register (used for the
            decision if it will be selected or not). Two examples are given below:
                - `datacube.select_by_dimension(lambda s: s == "X", name='dim', inplace=True)`
                - `datacube.select_by_dimension(lambda t: (t >= start_time) & (t <= end_time), name='time',
                inplace=True)`
        name : str, optional
            Name of the dimension. Defaults to the name of the stack dimension.

        Returns
        -------
        datacubes : list
            A list of datacubes corresponding to each expression.

        """

        datacubes = [self.select_by_dimension(expression, name=name, inplace=False)
                     for expression in expressions]

        return datacubes

    def split_by_temporal_freq(self, time_freq, name=None) -> List["DataCube"]:
        """
        Temporally splits the original datacube according to a given frequency string.

        Parameters
        ----------
        time_freq : str
            Pandas DateOffset frequency string (see
            https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects)
        name : str, optional
            Name of the dimension. Defaults to the name of the stack dimension.

        Returns
        -------
        datacubes : list
            A list of datacubes corresponding to the given temporal frequency intervals.

        Notes
        -----
        Empty datacubes are discarded.

        """
        name = name if name is not None else self._raster_data._file_dim
        min_time, max_time = min(self.file_register[name]), max(self.file_register[name])
        time_ranges = pd.date_range(min_time, max_time, freq=time_freq).union([min_time, max_time])
        expressions = [lambda x: (x >= time_ranges[0]) & (x <= time_ranges[1])]
        expressions += [lambda x, i=i: (x > time_ranges[i]) & (x <= time_ranges[i + 1])
                        for i in range(1, len(time_ranges) - 1)]

        return [dc for dc in self.split_by_dimension(expressions, name=name) if not dc.is_empty]

    def select_tiles(self, tile_names, inplace=False) -> "DataCube":
        """
        Selects the given tiles from the datacube.

        Parameters
        ----------
        tile_names : list of str
            Tile names/IDs.
        inplace : boolean, optional
            If true, the current class instance will be altered.
            If false (default), a new class instance will be returned.

        Returns
        -------
        DataCube :
            Datacube with a mosaic and a file register only consisting of the given tiles.

        """
        if not inplace:
            new_datacube = copy.deepcopy(self)
            return new_datacube.select_tiles(tile_names, inplace=True)

        self._raster_data.select_tiles(tile_names, inplace=True)

        return self

    def select_px_window(self, row, col, height=1, width=1, inplace=False) -> "DataCube":
        """
        Selects a rectangular region corresponding to the given pixel window from the datacube.

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
        inplace : boolean, optional
            If true, the current class instance will be altered.
            If false (default), a new class instance will be returned.

        Returns
        -------
        DataCube :
            Datacube with a data and a mosaic geometry only consisting of the intersected tile with the
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
        Selects a pixel from the datacube according to the given coordinate tuple.

        Parameters
        ----------
        x : number
            Coordinate in X direction.
        y : number
            Coordinate in Y direction.
        sref : geospade.crs.SpatialRef, optional
            CRS of the given coordinate tuple. Defaults to the CRS of the mosaic.
        inplace : boolean, optional
            If true, the current class instance will be altered.
            If false (default), a new class instance will be returned.

        Returns
        -------
        DataCube :
            Datacube object with a file register and a mosaic only consisting of the intersected tile containing
            information on the location of the single-pixel time series.

        """
        if not inplace:
            new_datacube = copy.deepcopy(self)
            return new_datacube.select_xy(x, y, sref=sref, inplace=True)

        self._raster_data.select_xy(x, y, sref=sref, inplace=True)

        return self

    def select_bbox(self, bbox, sref=None, inplace=False) -> "DataCube":
        """
        Selects a rectangular (if it is provided in native units) region from the datacube, according to the given
        bounding box.

        Parameters
        ----------
        bbox : list of 2 2-tuple
            Bounding box to select, i.e. [(x_min, y_min), (x_max, y_max)].
        sref : geospade.crs.SpatialRef, optional
            CRS of the given bounding box coordinates. Defaults to the CRS of the mosaic.
        inplace : boolean, optional
            If true, the current class instance will be altered.
            If false (default), a new class instance will be returned.

        Returns
        -------
        DataCube :
            Datacube object with a file register and a mosaic only consisting of the intersected tiles.

        """
        if not inplace:
            new_datacube = copy.deepcopy(self)
            return new_datacube.select_bbox(bbox, sref=sref, inplace=True)

        return self.select_polygon(bbox, sref=sref, apply_mask=False, inplace=True)

    def select_polygon(self, polygon, sref=None, apply_mask=True, inplace=False) -> "DataCube":
        """
        Selects a region delineated by the given polygon from the datacube.

        Parameters
        ----------
        polygon : ogr.Geometry
            Polygon specifying the pixels to collect.
        sref : geospade.crs.SpatialRef, optional
            CRS of the given bounding box coordinates. Defaults to the CRS of the mosaic.
        apply_mask : bool, optional
            True if pixels outside the polygon should be set to a no data value (default).
            False if every pixel withing the bounding box of the polygon should be included.
        inplace : boolean, optional
            If true, the current class instance will be altered.
            If false (default), a new class instance will be returned.

        Returns
        -------
        DataCube :
            Datacube object with a file register and a mosaic only consisting of the intersected tiles.

        """
        if not inplace:
            new_datacube = copy.deepcopy(self)
            return new_datacube.select_polygon(polygon, sref=sref, apply_mask=apply_mask, inplace=True)

        self._raster_data.select_polygon(polygon, sref=sref, apply_mask=apply_mask, inplace=True)

        return self

    def intersect(self, other, on_dimension=None, inplace=False) -> "DataCube":
        """
        Intersects this datacube with another datacube. This is equal to an SQL INNER JOIN operation.
        In other words:
            - all uncommon columns and rows (if `on_dimension` is given) are removed
            - duplicates are removed

        Parameters
        ----------
        other : DataCube
            Datacube to intersect with.
        on_dimension : str, optional
            Dimension name to intersect on, meaning that only equal entries along this dimension will be retained.
        inplace : boolean, optional
            If true, the current class instance will be altered.
            If false (default), a new class instance will be returned.

        Returns
        -------
        DataCube
            Intersected datacube.

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
        Unites this datacube with respect to another datacube. This is equal to an SQL UNION operation.
        In other words:
            - all columns are put into one DataFrame
            - duplicates are removed
            - gaps are filled with NaN

        Parameters
        ----------
        other : DataCube
            Datacube to unite with.
        inplace : boolean, optional
            If true, the current class instance will be altered.
            If false (default), a new class instance will be returned.

        Returns
        -------
        DataCube
            United datacube.

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
        Aligns this datacube with another datacube along the specified dimension `name`.

        Parameters
        ----------
        other : DataCube
            Datacube to align with.
        name : str
            Name of the dimension, which is used for aligning/filtering the values for all datacubes.
        inplace : boolean, optional
            If true, the current class instance will be altered.
            If false (default), a new class instance will be returned.

        Returns
        -------
        DataCube
            Datacube with common values along the given dimension with respect to another datacube.

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
        """
        Checks if another datacube is compliant with this datacube, i.e., the stack and tile dimension have the same
        name.

        Parameters
        ----------
        other : DataCube
            Other datacube to check with.

        """
        if self._raster_data._file_dim != other._raster_data._file_dim:
            err_msg = f"Both datacubes must have the same file dimension " \
                      f"({self._raster_data._file_dim} != {other._raster_data._file_dim})."
            raise ValueError(err_msg)
        if self._raster_data._tile_dim != other._raster_data._tile_dim:
            err_msg = f"Both datacubes must have the same tile dimension " \
                      f"({self._raster_data._tile_dim} != {other._raster_data._tile_dim})."
            raise ValueError(err_msg)

    def apply_nan(self):
        """
        Converts no data values of the internal data to np.nan. Note that this replacement implicitly converts the
        data format to float.

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

    @staticmethod
    def _get_file_class(filepath, file_class=None):
        """
        Collects class for opening a single geospatial file.

        Parameters
        ----------
        filepath : str
            Full file path.
        file_class : class, optional
            External file class.

        Returns
        -------
        file_class : class
            Geospatial file class.

        """
        ext = os.path.splitext(filepath)[-1]
        file_class = FILE_CLASS[ext] if file_class is None else file_class
        if file_class is None:
            raise FileTypeUnknown(ext)
        return file_class

    def __getitem__(self, dimension_name) -> pd.Series:
        """
        Returns a column of the internal file register according to the given column/dimension name.

        Parameters
        ----------
        dimension_name : str
            Column/dimension name of the datacube's file register.

        Returns
        -------
        pandas.DataSeries
            Column of the internal file register.

        """

        if dimension_name in self.file_register.columns:
            return self.file_register[dimension_name]
        else:
            raise DimensionUnkown(dimension_name)

    def __len__(self) -> int:
        """ Length of the datacube/file register. """
        return len(self.file_register)

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        """ Closes open file handles. """
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
        return f"{self.__class__.__name__} -> {self._raster_data.__class__.__name__}({self._raster_data._file_dim}, " \
               f"{self.mosaic.__class__.__name__}):\n\n" \
               f"{repr(self.file_register)}"


class DataCubeReader(DataCube):
    """ Datacube reader class inheriting from `DataCube`. """
    def __init__(self, file_register, mosaic, stack_dimension='layer_id', tile_dimension='tile_id',
                 file_class=None, file_class_kwargs=None, **kwargs):
        """
        Constructor of `DataCubeReader`.

        Parameters
        ----------
        file_register : pd.Dataframe
            Data frame managing a stack/list of files containing the following columns:
                - 'filepath' : str
                    Full file path to a geospatial file.
                - `stack_dimension` : object
                    Specifies an ID to which layer a file belongs to, e.g. a layer counter or a timestamp. Must
                    correspond to `stack_dimension`.
                - `tile_dimension` : str
                    Tile name or ID to which tile a file belongs to.
        mosaic : geospade.raster.MosaicGeometry
            Mosaic representing the spatial allocation of the given files. The tiles of the mosaic have to match the
            ID's/names of the `tile_dimension` column.
        stack_dimension : str, optional
            Dimension/column name of the dimension, where to stack the files along (first axis), e.g. time, bands etc.
            Defaults to 'layer_id', i.e. the layer ID's are used as the main coordinates to stack the files.
        tile_dimension : str, optional
            Dimension/column name of the dimension containing tile ID's in correspondence with the tiles in `mosaic`.
            Defaults to 'tile_id'.
        file_class : class, optional
            Class used to open a reference file for retrieving basic information. Defaults to none, meaning that the
            datacube uses the default classes assigned to each file extension/data format.
        file_class_kwargs : dict, optional
            Keyword arguments for `file_class`.
        kwargs : dict
            Keywords passed to a `RasterDataReader` class.

        """
        ref_filepath = file_register['filepath'].iloc[0]
        ext = os.path.splitext(ref_filepath)[-1]
        reader_class = RASTER_DATA_CLASS[ext][0]
        file_class = DataCube._get_file_class(ref_filepath, file_class=file_class)
        reader = reader_class(file_register, mosaic, stack_dimension=stack_dimension, tile_dimension=tile_dimension,
                              file_class=file_class, file_class_kwargs=file_class_kwargs, **kwargs)
        super().__init__(reader)

    @classmethod
    def from_filepaths(cls, filepaths, fn_class=SmartFilename, fields_def=None, fn_kwargs=None,
                       mosaic=None, tile_class=Tile, sref=None, file_class=None,
                       file_class_kwargs=None, dimensions=None,
                       tile_dimension='tile', stack_dimension='time', use_metadata=False, md_decoder=None, n_cores=1,
                       **kwargs) -> "DataCubeReader":
        """
        Creates a `DataCubeReader` instance from a list of file paths.

        Parameters
        ----------
        filepaths : list of str
            List of file paths to ingest into the datacube.
        fn_class : SmartFilename, optional
            Filename class used to interpret the file name. Defaults to `SmartFilename`.
        fields_def : dict, optional
            Dictionary defining the elements of a specific file name. For further details take a look at
            geopathfinder's `SmartFilename` class. This argument can be used if `fn_class` is None.
        fn_kwargs : dict, optional
            Keyword arguments for `fn_class`.
        mosaic : geospade.raster.MosaicGeometry
            Mosaic representing the spatial allocation of the given files. The `tile_dimension` part of the file name
            must match the tile IDs/names of the mosaic. By default a mosaic is automatically retrieved from the
            spatial extent of the files.
        tile_class : geospade.raster.Tile, optional
            Tile class used for creating a default mosaic, if `mosaic` is not provided. Defaults to `Tile`.
        sref : geospade.crs.SpatialRef, optional
            CRS of the given files. Defaults to the CRS of the mosaic.
        file_class : class, optional
            Class used to open a reference file for retrieving basic information. Defaults to none, meaning that the
            datacube uses the default classes assigned to each file extension/data format.
        file_class_kwargs : dict, optional
            Keyword arguments for `file_class`.
        dimensions : list, optional
            Desired dimensions of the datacube in compliance with the chosen file naming convention.
        tile_dimension : str, optional
            Dimension/column name of the dimension containing tile ID's in correspondence with the tiles in `mosaic`.
            Defaults to 'tile_id'.
        stack_dimension : str, optional
            Dimension/column name of the dimension, where to stack the files along (first axis), e.g. time, bands etc.
            Defaults to 'layer_id', i.e. the layer ID's are used as the main coordinates to stack the files.
        use_metadata :  bool, optional
            True if dimensions should be retrieved from the metadata of the files (defaults to `False`).
        md_decoder : dict, optional
            Dictionary mapping dimension names/attribute names with decoding functions.
        n_cores : int, optional
            Number of cores used to interpret files in parallel (defaults to 1).
        kwargs : dict
            Keywords passed to the `DataCubeReader` constructor.

        Returns
        -------
        DataCubeReader
            Datacube reader instance.

        """
        file_register = cls._get_file_register_from_files(filepaths, fn_class, fields_def=fields_def,
                                                          fn_kwargs=fn_kwargs, dimensions=dimensions,
                                                          n_cores=n_cores, use_metadata=use_metadata,
                                                          md_decoder=md_decoder, file_class=file_class,
                                                          file_class_kwargs=file_class_kwargs)

        if tile_dimension in file_register.columns and mosaic is None:
            tiles = cls._get_tiles_from_file_register(file_register, tile_class=tile_class,
                                                      tile_dimension=tile_dimension, sref=sref, file_class=file_class,
                                                      file_class_kwargs=file_class_kwargs)
        elif tile_dimension not in file_register.columns:
            tiles, tile_ids = cls._get_tiles_and_ids_from_files(filepaths, tile_class=tile_class, mosaic=mosaic,
                                                                sref=sref,
                                                                file_class=file_class,
                                                                file_class_kwargs=file_class_kwargs)
            file_register[tile_dimension] = tile_ids

        if mosaic is None:
            mosaic = MosaicGeometry.from_tile_list(tiles)

        if stack_dimension not in file_register.columns:
            stack_ids = cls._get_stack_ids_from_file_register(file_register, tile_dimension=tile_dimension)
            file_register[stack_dimension] = stack_ids

        return cls(file_register, mosaic, stack_dimension=stack_dimension, tile_dimension=tile_dimension,
                   file_class=file_class, file_class_kwargs=file_class_kwargs, **kwargs)

    @staticmethod
    def _get_file_register_from_files(filepaths, fn_class, fields_def=None, fn_kwargs=None,
                                      dimensions=None, n_cores=1,
                                      use_metadata=False, md_decoder=None, file_class=None,
                                      file_class_kwargs=None):
        """

        Parameters
        ----------
        filepaths : list of str
            List of file paths to ingest into the datacube.
        fn_class : SmartFilename
            Filename class used to interpret the file name. Defaults to `SmartFilename`.
        fields_def : dict, optional
            Dictionary defining the elements of a specific file name. For further details take a look at
            geopathfinder's `SmartFilename` class. This argument can be used if `fn_class` is None.
        fn_kwargs : dict, optional
            Keyword arguments for `fn_class`.
        dimensions : list, optional
            Desired dimensions of the datacube in compliance with the chosen file naming convention.
        n_cores : int, optional
            Number of cores used to interpret files in parallel (defaults to 1).
        use_metadata :  bool, optional
            True if dimensions should be retrieved from the metadata of the files (defaults to `False`).
        md_decoder : dict, optional
            Dictionary mapping dimension names/attribute names with decoding functions.
        file_class : class, optional
            Class used to open a reference file for retrieving basic information. Defaults to none, meaning that the
            datacube uses the default classes assigned to each file extension/data format.
        file_class_kwargs : dict, optional
            Keyword arguments for `file_class`.

        Returns
        -------
        pd.DataFrame
            Data frame representing the file register.

        """
        fn_kwargs = fn_kwargs or dict()
        md_decoder = md_decoder or dict()
        file_class_kwargs = {} if file_class_kwargs is None else file_class_kwargs
        n_files = len(filepaths)
        slices = DataCubeReader._get_file_chunks(n_files, n_cores)
        ref_filepath = filepaths[0]
        try:
            fn_args = [os.path.basename(ref_filepath)]
            if fields_def is not None:
                fn_args.append(fields_def)
            fn = fn_class.from_filename(*fn_args, convert=True, **fn_kwargs)
            fn_dims = DataCubeReader._get_dims_from_fn(fn, dimensions=dimensions, fields_def=fields_def)
        except:
            fn_dims = []
        md_dims = [] if not use_metadata else DataCubeReader._get_dims_from_md(ref_filepath, dimensions=dimensions,
                                                                               file_class=file_class,
                                                                               file_class_kwargs=file_class_kwargs)
        file_class = DataCube._get_file_class(ref_filepath, file_class=file_class)
        tmp_dirpath = mkdtemp()
        with Pool(n_cores, initializer=parse_init, initargs=(filepaths, fn_class, fields_def, fn_kwargs,
                                                             file_class, file_class_kwargs,
                                                             fn_dims, md_dims, md_decoder, tmp_dirpath)) as p:
            p.map(parse_filepaths, slices)

        df_filepaths = glob.glob(os.path.join(tmp_dirpath, "*.df"))
        return pd.concat([pd.read_pickle(df_filepath) for df_filepath in df_filepaths])

    @staticmethod
    def _get_file_chunks(n_files, n_cores) -> List[slice]:
        """
        Creates a list of file index slices for parsing files in parallel.

        Parameters
        ----------
        n_files : int
            Number of files.
        n_cores : int
            Number of cores used to interpret files in parallel.

        Returns
        -------
        list of slice
            List of file index slices.

        """
        step = int(n_files / n_cores)
        slices = []
        for i in range(0, n_files, step):
            slices.append(slice(i, i + step))
        slices[-1] = slice(slices[-1].start, n_files + 1)

        return slices

    @staticmethod
    def _get_dims_from_fn(fn, dimensions=None, fields_def=None) -> List[str]:
        """
        Collects available file name dimensions and inner joins them with existing ones.

        Parameters
        ----------
        fn : SmartFilename
            Filename instance.
        dimensions : list, optional
            Desired dimensions of the datacube in compliance with the chosen file naming convention.
        fields_def : dict, optional
            Dictionary defining the elements of a specific file name. For further details take a look at
            geopathfinder's `SmartFilename` class. This argument can be used if `fn_class` is None.

        Returns
        -------
        fn_dims : list of str
            List of available or desired file dimensions.

        """
        fields_def = fields_def or dict()
        fn_dims = list(fn.fields_def.keys()) if hasattr(fn, 'fields_def') else list(fields_def.keys())
        if dimensions is not None:
            fn_dims = list(set(fn_dims).intersection(set(dimensions)))
            for dimension in dimensions:
                if getattr(fn, dimension, False):
                    fn_dims.append(dimension)
        return fn_dims

    @staticmethod
    def _get_dims_from_md(filepath, dimensions=None, file_class=None, file_class_kwargs=None):
        """
        Collects available metadata dimensions and inner joins them with existing ones.

        Parameters
        ----------
        filepath : str
            Reference filepath.
        dimensions : list, optional
            Desired dimensions of the datacube in compliance with the available metadata attributes.
        file_class : class, optional
            Class used to open a reference file for retrieving basic information. Defaults to none, meaning that the
            datacube uses the default classes assigned to each file extension/data format.
        file_class_kwargs : dict, optional
            Keyword arguments for `file_class`.

        Returns
        -------
        md_dims : list of str
            List of available or desired metadata dimensions.

        """
        file_class_kwargs = file_class_kwargs or dict()
        file_class = DataCube._get_file_class(filepath, file_class=file_class)
        md_dims = []
        try:
            with file_class(filepath, mode='r', **file_class_kwargs) as file:
                md = file.metadata
                md_dims = list(md.keys())
                if dimensions is not None:
                    md_dims = list(set(dimensions).intersection(md_dims))
        except:
            pass

        return md_dims

    @staticmethod
    def _get_tiles_from_file_register(file_register, tile_class=Tile, tile_dimension='tile_id', sref=None,
                                      file_class=None, file_class_kwargs=None) -> List[Tile]:
        """
        Retrieve all tiles using the file register and geospatial information in the files.

        Parameters
        ----------
        file_register : pd.DataFrame
            Preliminary file register.
        tile_class : geospade.raster.Tile, optional
            Tile class used for creating a default mosaic, if `mosaic` is not provided. Defaults to `Tile`.
        tile_dimension : str, optional
            Dimension/column name of the dimension containing tile ID's in correspondence with the tiles in `mosaic`.
            Defaults to 'tile_id'.
        sref : geospade.crs.SpatialRef, optional
            CRS of the given files. Defaults to the CRS of the mosaic.
        file_class : class, optional
            Class used to open a reference file for retrieving basic information. Defaults to none, meaning that the
            datacube uses the default classes assigned to each file extension/data format.
        file_class_kwargs : dict, optional
            Keyword arguments for `file_class`.

        Returns
        -------
        tiles : list of geospade.raster.Tile
            Available tiles.

        """
        tiles = []
        for tile_id, tile_group in file_register.groupby(by=tile_dimension):
            ref_filepath = tile_group['filepath'].iloc[0]
            tile = DataCubeReader._get_tile_from_file(ref_filepath, tile_class=tile_class, tile_id=tile_id, sref=sref,
                                                      file_class=file_class, file_class_kwargs=file_class_kwargs)
            tiles.append(tile)
        return tiles

    @staticmethod
    def _get_tiles_and_ids_from_files(filepaths, tile_class=Tile, mosaic=None, sref=None, file_class=None,
                                      file_class_kwargs=None) -> Tuple[list, list]:
        """
        Collects all tiles and their IDs for a given list of files.

        Parameters
        ----------
        filepaths : list of str
            List of file paths to retrieve tiles and their IDs from.
        tile_class : geospade.raster.Tile, optional
            Tile class used for creating a default mosaic, if `mosaic` is not provided. Defaults to `Tile`.
        mosaic : geospade.raster.MosaicGeometry
            Mosaic representing the spatial allocation of the given files. The `tile_dimension` part of the file name
            must match the tile IDs/names of the mosaic.
        sref : geospade.crs.SpatialRef, optional
            CRS of the given files. Defaults to the CRS of the mosaic.
        file_class : class, optional
            Class used to open a reference file for retrieving basic information. Defaults to none, meaning that the
            datacube uses the default classes assigned to each file extension/data format.
        file_class_kwargs : dict, optional
            Keyword arguments for `file_class`.

        Returns
        -------
        tiles : list of geospade.raster.Tile
            Available tiles.
        tile_ids : list of str
            Tile IDs linking `filepaths` with `tiles`.

        """
        sref = mosaic.sref if mosaic is not None else sref
        tile_ids = []
        tiles = []
        tile_id = 0
        for filepath in filepaths:
            tile = DataCubeReader._get_tile_from_file(filepath, tile_class=tile_class, tile_id=str(tile_id), sref=sref,
                                                      file_class=file_class, file_class_kwargs=file_class_kwargs)
            if mosaic is None:
                curr_tile_id = find_congruent_tile_id_from_tiles(tile, tiles)
            else:
                curr_tile_id = find_congruent_tile_id_from_tiles(tile, mosaic.all_tiles)
            if curr_tile_id is None:
                tiles.append(tile)
                curr_tile_id = str(tile_id)
                tile_id += 1
            tile_ids.append(curr_tile_id)

        return tiles, tile_ids

    @staticmethod
    def _get_tile_from_file(filepath, tile_class=Tile, tile_id='0', sref=None, file_class=None,
                            file_class_kwargs=None):
        """
        Creates a tile from the geospatial information stored in the file.

        Parameters
        ----------
        filepath : str
            File path.
        tile_class : geospade.raster.Tile, optional
            Tile class used for creating a default mosaic, if `mosaic` is not provided. Defaults to `Tile`.
        tile_id : str, optional
            Name/ID of the tile (defaults to '0').
        sref : geospade.crs.SpatialRef, optional
            CRS of the given files. Defaults to the CRS of the mosaic.
        file_class : class, optional
            Class used to open a reference file for retrieving basic information. Defaults to none, meaning that the
            datacube uses the default classes assigned to each file extension/data format.
        file_class_kwargs : dict, optional
            Keyword arguments for `file_class`.

        Returns
        -------
        geospade.raster.Tile
            Tile instance.

        """
        file_class_kwargs = {} if file_class_kwargs is None else file_class_kwargs
        file_class = DataCube._get_file_class(filepath, file_class=file_class)
        with file_class(filepath, 'r', **file_class_kwargs) as f:
            sref_wkt = f.sref_wkt
            geotrans = f.geotrans
            n_rows, n_cols = f.raster_shape
        sref = sref if sref_wkt is None else SpatialRef(sref_wkt)
        return tile_class(n_rows, n_cols, sref=sref, geotrans=geotrans, name=tile_id)

    @staticmethod
    def _get_stack_ids_from_file_register(file_register, tile_dimension='tile_id'):
        """
        Retrieves/creates stack IDs (counted separately for each tile) from the given file register.

        Parameters
        ----------
        file_register : pd.DataFrame
            Preliminary file register.
        tile_dimension : str, optional
            Dimension/column name of the dimension containing tile ID's in correspondence with the tiles in `mosaic`.
            Defaults to 'tile_id'.

        Returns
        -------
        stack_ids : np.ndarray
            Stack IDs.

        """
        n_files = len(file_register)
        tile_ids = file_register[tile_dimension]
        tile_ids_uni = list(set(tile_ids))
        stack_ids = np.zeros(n_files, dtype=int)
        for tile_id in tile_ids_uni:
            tile_idx = tile_ids == tile_id
            n_same_tiles = np.sum(tile_idx)
            stack_ids[tile_idx] = np.arange(n_same_tiles)

        return stack_ids

    def read(self, *args, **kwargs):
        """
        Reads data from disk.

        Parameters
        ----------
        args : tuple
            Positional arguments for the `RasterDataReader().read()` function.
        kwargs : dict
            Keyword arguments for the `RasterDataReader().read()` function.

        Notes
        -----
        Details about the available arguments can be retrieved from the respective `read()` functions in veranda.

        """
        self._raster_data.read(*args, **kwargs)


class DataCubeWriter(DataCube):
    """ Datacube writer class inheriting from `DataCube`. """
    def __init__(self, mosaic, file_register=None, data=None, ext='.nc', stack_dimension='layer_id',
                 tile_dimension='tile_id', **kwargs):
        """
        Constructor of `DataCubeWriter`.

        Parameters
        ----------
        mosaic : geospade.raster.MosaicGeometry
            Mosaic representing the spatial allocation of the given files. The tiles of the mosaic have to match the
            ID's/names of the `tile_dimension` column.
        file_register : pd.Dataframe, optional
            Data frame managing a stack/list of files containing the following columns:
                - 'filepath' : str
                    Full file path to a geospatial file.
                - `stack_dimension` : object
                    Specifies an ID to which layer a file belongs to, e.g. a layer counter or a timestamp. Must
                    correspond to `stack_dimension`.
                - `tile_dimension` : str
                    Tile name or ID to which tile a file belongs to.
        data : xr.Dataset, optional
            Raster data stored in memory. It must match the spatial sampling and CRS of the mosaic, but not its spatial
            extent or tiling. Moreover, the dimension of the mosaic along the first dimension (stack dimension), must
            match the entries/filepaths in `file_register`.
        ext : str, optional
            File extension/format. Defaults to ".nc".
        stack_dimension : str, optional
            Dimension/column name of the dimension, where to stack the files along (first axis), e.g. time, bands etc.
            Defaults to 'layer_id', i.e. the layer ID's are used as the main coordinates to stack the files.
        tile_dimension : str, optional
            Dimension/column name of the dimension containing tile ID's in correspondence with the tiles in `mosaic`.
            Defaults to 'tile_id'.
        kwargs : dict
            Keywords passed to a `RasterDataWriter` class.

        """
        ext = ext if file_register is None else os.path.splitext(file_register['filepath'].iloc[0])[-1]
        writer_class = RASTER_DATA_CLASS[ext][1]
        writer = writer_class(mosaic, file_register=file_register, data=data,
                              stack_dimension=stack_dimension, tile_dimension=tile_dimension, **kwargs)
        super().__init__(writer)

    @classmethod
    def from_data(cls, data, dirpath, fn_class=SmartFilename, fn_map=None, def_fields=None,
                  stack_groups=None, fn_groups_map=None,
                  ext='.nc', mosaic=None, stack_dimension='layer_id', tile_dimension='tile_id',
                  **kwargs) -> "DataCubeWriter":
        """
        Creates a `DataCubeWriter` instance from an xarray dataset.

        Parameters
        ----------
        data : xr.Dataset, optional
            Raster data stored in memory to derive the mosaic and file register from.
        dirpath : str
            Full directory path where the files are located/should be written to.
        fn_class : SmartFilename, optional
            Filename class used to create a file name from the coordinates in `data`. Defaults to `SmartFilename`.
        fn_map : dict, optional
            Dictionary mapping dimension/coordinate names of `data` with dimension names of the file naming convention.
        def_fields : dict, optional
            Dictionary containing default attributes/values used when creating all file names.
        stack_groups : dict, optional
            Defines the relation between the stack coordinates and a group ID, i.e. in what portions along the stack
            dimension the data should be written. The keys are the coordinates and the value a group ID.
        fn_groups_map : dict, optional
            If `stack_groups` is set, then you can assign new filename attributes to each group ID by using
            this argument. It's format should be a dictionary mapping group IDs (keys) with filename fields (values).
        ext : str, optional
            File extension/format. Defaults to ".nc".
        mosaic : geospade.raster.MosaicGeometry
            Mosaic representing the spatial allocation of the given files. The tiles of the mosaic have to match the
            ID's/names of the `tile_dimension` column.
        stack_dimension : str, optional
            Dimension/column name of the dimension, where to stack the files along (first axis), e.g. time, bands etc.
            Defaults to 'layer_id', i.e. the layer ID's are used as the main coordinates to stack the files.
        tile_dimension : str, optional
            Dimension/column name of the dimension containing tile ID's in correspondence with the tiles in `mosaic`.
            Defaults to 'tile_id'.
        kwargs : dict
            Keywords passed to the `DataCubeWriter` class.

        Returns
        -------
        DataCubeWriter
            Datacube writer instance.

        """
        writer_class = RASTER_DATA_CLASS[ext][1]
        if mosaic is None:
            mosaic = writer_class._mosaic_from_data(data)

        tile_ids = mosaic.tile_names
        stack_ids = data[stack_dimension].data
        filepaths, stack_ids, tile_ids = cls._get_filepaths_from_tile_stack_ids(tile_ids, stack_ids,
                                                                                fn_class, dirpath,
                                                                                ext=ext,
                                                                                tile_dimension=tile_dimension,
                                                                                stack_dimension=stack_dimension,
                                                                                fn_map=fn_map,
                                                                                def_fields=def_fields,
                                                                                stack_groups=stack_groups,
                                                                                fn_groups_map=fn_groups_map)
        fr_dict = {'filepath': filepaths,
                   stack_dimension: stack_ids,
                   tile_dimension: tile_ids}
        file_register = pd.DataFrame(fr_dict)

        return cls(mosaic, file_register=file_register, data=data,  ext=ext,
                   stack_dimension=stack_dimension, tile_dimension=tile_dimension,
                   **kwargs)

    @staticmethod
    def _get_filepaths_from_tile_stack_ids(tile_ids, stack_ids, fn_class, dirpath, ext='.nc',
                                           tile_dimension='tile_id', stack_dimension='layer_id',
                                           fn_map=None, def_fields=None,
                                           stack_groups=None, fn_groups_map=None) -> Tuple[list, list, list]:
        """
        Creates file paths from a list of tile and stack IDs by mapping them to the given file naming convention.

        Parameters
        ----------
        tile_ids : list of str
            List of tile IDs/names.
        stack_ids : list
            List of stack IDs.
        fn_class : SmartFilename, optional
            Filename class used to create a file name from the coordinates in `data`.
        dirpath : str
            Full directory path where the files are located/should be written to.
        ext : str, optional
            File extension/format. Defaults to ".nc".
        tile_dimension : str, optional
            Dimension/column name of the dimension containing tile ID's in correspondence with the tiles in `mosaic`.
            Defaults to 'tile_id'.n
        stack_dimension : str, optional
            Dimension/column name of the dimension, where to stack the files along (first axis), e.g. time, bands etc.
            Defaults to 'layer_id', i.e. the layer ID's are used as the main coordinates to stack the files.
        fn_map : dict, optional
            Dictionary mapping dimension/coordinate names of `data` with dimension names of the file naming convention.
        def_fields : dict, optional
            Dictionary containing default attributes/values used when creating all file names.
        stack_groups : dict, optional
            Defines the relation between the stack coordinates and a group ID, i.e. in what portions along the stack
            dimension the data should be written. The keys are the coordinates and the value a group ID.
        fn_groups_map : dict, optional
            If `stack_groups` is set, then you can assign new filename attributes to each group ID by using
            this argument. It's format should be a dictionary mapping group IDs (keys) with filename fields (values).

        Returns
        -------
        filepaths : list of str
            List of file paths.
        tile_ids_aligned : list of str
            List of tile IDs/names corresponding to each entry in `filepaths`.
        stack_ids_aligned : list
            List of stack IDs/names corresponding to each entry in `filepaths`.

        """
        fn_map = fn_map or dict()
        fn_groups_map = fn_groups_map or dict()
        def_fields = def_fields or dict()
        if fn_class is None:
            fields_def = dict([
                (stack_dimension, {}),
                (tile_dimension, {})])
            fn_class = create_fn_class(fields_def)

        stack_ids_aligned = []
        tile_ids_aligned = []
        filepaths = []
        for tile_id in tile_ids:
            for stack_id in stack_ids:
                fields = dict()
                fields[fn_map.get(tile_dimension, tile_dimension)] = tile_id
                fields.update(def_fields)
                if stack_groups is not None:
                    group_id = stack_groups[stack_id]
                    fields.update({fn_map.get(stack_dimension, stack_dimension): group_id})
                    fields.update(fn_groups_map.get(group_id))
                else:
                    fields.update({fn_map.get(stack_dimension, stack_dimension): stack_id})

                filename = str(fn_class(fields, ext=ext, convert=True))
                filepaths.append(os.path.join(dirpath, filename))
                stack_ids_aligned.append(stack_id)
                tile_ids_aligned.append(tile_id)

        return filepaths, stack_ids_aligned, tile_ids_aligned

    def write(self, data, use_mosaic=False, data_variables=None, encoder=None, encoder_kwargs=None, overwrite=False,
              **kwargs):
        """
        Writes a certain chunk of data to disk.

        Parameters
        ----------
        data : xr.Dataset
            Data chunk to be written to disk or being appended to existing data.
        use_mosaic : bool, optional
            True if data should be written according to the mosaic.
            False if data composes a new tile and should not be tiled (default).
        data_variables : list of str, optional
            Data variables to write. Defaults to None, i.e. all data variables are written.
        encoder : callable, optional
            Function allowing to encode data before writing it to disk.
        encoder_kwargs : dict, optional
            Keyword arguments for the encoder.
        overwrite : bool, optional
            True if data should be overwritten, False if not (default).
        kwargs : dict
            Keywords passed to the `RasterDataWriter().write()` method.

        """
        self._raster_data.write(data, use_mosaic=use_mosaic, data_variables=data_variables, encoder=encoder,
                                encoder_kwargs=encoder_kwargs, overwrite=overwrite, **kwargs)

    def export(self, use_mosaic=False, data_variables=None, encoder=None, encoder_kwargs=None, overwrite=False,
               **kwargs):
        """
        Writes all internally stored data to disk.

        Parameters
        ----------
        use_mosaic : bool, optional
            True if data should be written according to the mosaic.
            False if data composes a new tile and should not be tiled (default).
        data_variables : list of str, optional
            Data variables to write. Defaults to None, i.e. all data variables are written.
        encoder : callable, optional
            Function allowing to encode data before writing it to disk.
        encoder_kwargs : dict, optional
            Keyword arguments for the encoder.
        overwrite : bool, optional
            True if data should be overwritten, False if not (default).

        """
        self._raster_data.export(use_mosaic=use_mosaic, data_variables=data_variables, encoder=encoder,
                                 encoder_kwargs=encoder_kwargs, overwrite=overwrite, **kwargs)

