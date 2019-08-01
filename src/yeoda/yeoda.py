import copy
import os
import re
import itertools
import ogr

import pandas as pd
from collections import OrderedDict
from geopandas import GeoSeries
from geopandas import GeoDataFrame
from shapely.wkt import loads
from osgeo import osr
import shapely.wkt

# TODO: don't use this module
import pytileproj.geometry as geometry
import pytileproj.base
from pyraster.geotiff import GeoTiffFile
from pyraster.netcdf import NcFile
from pyraster.gdalport import GdalImage

from errors import IOClassNotFound
from errors import DataTypeUnknown
from errors import GeometryUnkown
from errors import TileNotAvailable

class EODataCube(object):
    """
    A filename based data cube.
    """
    def __init__(self, filepaths=None, grid=None, dir_tree=None, smart_filename_creator=None, dimensions=None,
                 inventory=None, io_map=None, io_md_map=None, ignore_metadata=True):
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

        # initialise IO classes responsible for reading and writing
        if io_map is not None:
            self.io_map = io_map
        else:
            self.io_map = {'GeoTIFF': GeoTiffFile, 'NetCDF': NcFile}

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
        elif (self.inventory is not None) and ('geometry' not in self.inventory.keys()):
            geometries = [self.__geometry_from_file(filepath) for filepath in self.filepaths]
            self.add_dimension('geometry', geometries, in_place=True)

    @classmethod
    def from_inventory(cls, inventory, grid=None, dir_tree=None):
        return cls(inventory=inventory, grid=grid, dir_tree=dir_tree)

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
        eoDataCube
            eoDataCube object with an additional dimension in the inventory.
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
        eoDataCube
            eoDataCube object with a filtered inventory according to the given pattern.
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
        eoDataCube
            eoDataCube object with a filtered inventory according to the given metadata.
        """
        bool_filter = []
        filepaths = self.inventory['filepath']  # use inventory to ensure the same order
        for filepath in filepaths:
            file_type = self.__file_type(filepath)
            io_class = self.__io_class(file_type)
            ds = io_class.src

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
        eoDataCube
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
        list of eoDataCube's
        """
        return self.__filter_by_dimension(values, expressions=expressions, name=name, split=True)

    def filter_spatially_by_tilename(self, tilenames, dimension_name="tile", in_place=False, check_grid=True):
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
        if self.grid:
            if check_grid:
                available_tilenames = self.grid.list_tiles_covering_land()  # TODO definitely has to be replaced
                for tilename in tilenames:
                    if tilename not in available_tilenames:
                        raise TileNotAvailable(tilename)
            return self.filter_by_dimension(tilenames, name=dimension_name, in_place=in_place)
        else:
            print('No grid is provided to extract tile information.')
            return self

    # TODO: also allow shapefiles and more complex geometries
    def filter_spatially_by_geom(self, geom, osr_spref=None, dimension_name="tile", in_place=False):
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
        geom_roi = None
        if isinstance(geom,(tuple, list)) and (not isinstance(geom[0],(tuple, list))) and (len(geom) == 4) and osr_spref:
            geom_roi = geometry.extent2polygon(geom, osr_spref)
        elif isinstance(geom,(tuple, list)) and isinstance(geom[0],(tuple, list)) and osr_spref:
            edge = ogr.Geometry(ogr.wkbLinearRing)
            for point in geom:
                if len(point) == 2:
                    edge.AddPoint(float(point[0]), float(point[1]))
            edge.CloseRings()
            geom_roi = ogr.Geometry(ogr.wkbPolygon)
            geom_roi.AddGeometry(edge)
            geom_roi.AssignSpatialReference(osr_spref)
        elif isinstance(geom, ogr.Geometry):
            geom_roi = geom
        else:
            raise GeometryUnkown(geom)

        if self.grid:
            tilenames = self.grid.search_tiles_in_roi(geom_area=geom_roi)
            self.filter_spatially_by_tilename(tilenames, dimension_name=dimension_name, in_place=in_place,
                                              check_grid=False)
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
        List of eoDataCube's
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
        List of eoDataCube's
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

    def load_by_geom(self, geom, ogr_spref=None):
        """
        Loads data as an array.

        Parameters
        ----------
        data_variables: [optional]
            Name of the tile/spatial dimension in the filenames.

        Returns
        -------
        object
            array-like object, e.g. numpy, dask or xarray array
        """
        pass

    def load_by_coord(self, x, y, osr_spref):
        pass

    def encode(self, data, **kwargs):
        return data

    def decode(self, data, **kwargs):
        return data

    def merge(self, dc_other, name=None):
        """
        Merges the data cubes with respect to another data cube along the specified dimension.

        Parameters
        ----------
        dc_other: eoDataCube
            Data cube to merge with.
        name: str, optional
            Name of the dimension.
        in_place: boolean, optional
            If true, the current class instance will be altered.
            If false, a new class instance will be returned (default value is False).

        Returns
        -------
        eoDataCube
        """
        return merge_datacubes([self, dc_other], name=name)

    def match_dimension(self, dc_other, name, in_place=False):
        """
        Matches the given data cubes along the specified dimension 'name'.

        Parameters
        ----------
        dc_other: eoDataCube
            Data cube to match dimensions with.
        name: str
            Name of the dimension, which is used for aligning/filtering the values for all data cubes.
        in_place: boolean, optional
            If true, the current class instance will be altered.
            If false, a new class instance will be returned (default value is False).

        Returns
        -------
        eoDataCube
        """
        return match_dimension([self, dc_other], name, in_place=in_place)[0]

    def clone(self):
        """
        Clone, i.e. deepcopy a data cube.

        Returns
        -------
        eoDataCube
            Cloned/copied data cube.

        """
        return copy.deepcopy(self)

    def trim_xarray(self, data):
        # remove xarray dimension values
        dc_dimensions = set(self.dimensions)
        xarray_dimensions = set(data.dims)
        common_dimensions = list(dc_dimensions.intersection(xarray_dimensions))
        for common_dimension in common_dimensions:
            values = list(self.inventory[common_dimension])
            data = data.drop(values, dim=common_dimension)

        # remove xarray data variables
        xarray_data_variables = set(data.variables.keys())
        common_dimensions = list(dc_dimensions.intersection(xarray_data_variables))
        data = data[common_dimensions]

        return data

    def trim_np_array(self, data):
        pass

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

    def __io_class(self, file_type):
        if file_type not in self.io_map.keys():
            raise IOClassNotFound(self.io_map, file_type)
        else:
            return self.io_map[file_type]

    def __io_md_map(self, file_type):
        if file_type not in self.io_md_map.keys():
            raise IOClassNotFound(self.io_md_map, file_type)
        else:
            return self.io_md_map[file_type]

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

    def __inventory_from_filepaths(self, create_smart_filename=None, ignore_metadata=True):
        """
        Creates GeoDataFrame (`inventory`) based on all filepaths.
        Each filepath/filename is translated to a SmartFilename object using a translation function
        `create_smart_filename`.

        Parameters
        ----------
        create_smart_filename: function
            Translates a filepath/filename to a SmartFilename object.

        """
        inventory = OrderedDict()
        inventory['filepath'] = []

        # fill inventory
        if self.filepaths:
            for filepath in self.filepaths:
                local_inventory = OrderedDict()
                local_inventory['filepath'] = [filepath]

                # get information from filename
                smart_filename = None
                try:
                    smart_filename = create_smart_filename(os.path.basename(filepath), convert=True)
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
                    file_type = self.__file_type(filepath)
                    io_class = self.__io_class(file_type)
                    io_md_map = self.__io_md_map(file_type)
                    metadata = io_class(filepath).get_dimensions_info(map=io_md_map)
                    for key, value in metadata.items():
                        local_inventory[key] = value

                    # add global inventory keys to local inventory if they are not available locally
                    for key in inventory.keys():
                        if key not in local_inventory.keys():
                            local_inventory[key] = None

                    # add local inventory keys to global inventory if they are not available globally
                    n = len(inventory['filepath'])
                    for key in local_inventory.keys():
                        if key not in inventory.keys():
                            if n == 0:  # first time
                                inventory[key] = []
                            else:
                                inventory[key] = [None] * n

                    entries = tuple(list(local_inventory.values()))
                    extended_entries = list(itertools.product(*entries))
                else:
                    extended_entries = list(local_inventory.values())

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
        eoDataCube or list of eoDataCubes
            If `split` is true and multiple filters are specified, a list of eoDataCube objects will be returned.
            If not, the inventory of the eoDataCube is filtered.
        """
        n_filters = len(values)
        if expressions is None:  # equal operator is the default comparison operator
            expressions = ["=="] * n_filters

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
                filter_cmd = "inventory[(inventory[name] expression[0] value[0]) & (inventory[name] expression[1] value[1])]"
            elif (len(value) == 1) and (len(expression) == 1):
                filter_cmd = "inventory[inventory[name] expression[0] value[0]]"
            else:
                raise Exception('Length of value (={}) and length of expression (={}) does not match or is larger than 2.'.format(len(value), len(expression)))
            inventory = eval(filter_cmd)

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
        eoDataCube
        """
        if in_place:
            self.inventory = inventory
            return self
        else:
            return self.from_inventory(inventory=inventory, grid=self.grid, dir_tree=self.dir_tree)

    def __deepcopy__(self, memodict={}):
        """
        Deepcopy method of the eoDataCube class.

        Parameters
        ----------
        memodict: dict, optional

        Returns
        -------
        eoDataCube
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
    Matches the given data cubes along the specified dimension 'name'.

    Parameters
    ----------
    dcs: list of eoDataCube's
       List of data cubes.
    name: str
        Name of the dimension, which is used for aligning/filtering the values for all data cubes.


    Returns
    -------
    list of eoDataCube objects
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
    Merges data cubes in one data cube. By doing so, duplicates are removed and only
    common dimensions are kept.

    Parameters
    ----------
    dcs: list of eoDataCube objects
       List of data cubes, which should be united based on the common set of dimensions.
    name: str
        Name of the dimension, which is used for aligning/filtering the values for all data cubes.

    Returns
    -------
    eoDataCube
        Data cube containing all information of the given data cubes except duplicates and
        inconsistent dimensions.
    """
    merged_inventory = dcs[0].inventory
    dcs = dcs[1:]
    for dc in dcs:
        merged_inventory = merged_inventory.merge(dc.inventory, on=name)

    return dcs[0].__assign_inventory(merged_inventory, in_place=False)




