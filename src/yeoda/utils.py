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
Utilities and helping functions for the other modules of yeoda.
"""

# general packages
import os
import copy
import numpy as np

# geo packages
from osgeo import ogr
from osgeo.gdal import __version__ as GDAL_VERSION
GDAL_3_ENABLED = GDAL_VERSION[0] == '3'
import pytileproj.geometry as geometry
import shapely.geometry
from geospade import DECIMALS
from geospade.crs import SpatialRef

# load classes from yeoda's error module
from yeoda.errors import GeometryUnkown


def get_file_type(filepath):
    """
    Determines the file type of types understood by yeoda, which are 'GeoTIFF' and 'NetCDF'.

    Parameters
    ----------
    filepath : str
        File path or filename.

    Returns
    -------
    str
        File type if it is understood by yeoda or `None`.
    """

    ext = os.path.splitext(filepath)[1]
    if ext in ['.tif', '.tiff']:
        return "GeoTIFF"
    elif ext in ['.nc']:
        return "NetCDF"
    else:
        return None


def any_geom2ogr_geom(geom, osr_sref):
    """
    Transforms an extent represented in different ways or a Shapely geometry object into an OGR geometry object.

    Parameters
    ----------
    geom : ogr.Geometry or shapely.geometry or list or tuple, optional
        A vector geometry. If it is of type list/tuple representing the extent (i.e. [x_min, y_min, x_max, y_max]),
        `osr_spref` has to be given to transform the extent into a georeferenced polygon.
    osr_sref : osr.SpatialReference, optional
        Spatial reference of the given geometry `geom`.

    Returns
    -------
    ogr.Geometry
        Vector geometry as an OGR Geometry object.
    """

    if isinstance(geom, (tuple, list)) and (not isinstance(geom[0], (tuple, list))) and \
            (len(geom) == 4) and osr_sref:
        geom_ogr = geometry.bbox2polygon(geom, osr_sref)
        geom_ogr = swap_axis(geom_ogr)  # ensure lon lat order
    elif isinstance(geom, (tuple, list)) and (isinstance(geom[0], (tuple, list))) and \
            (len(geom) == 2) and osr_sref:
        edge = ogr.Geometry(ogr.wkbLinearRing)
        geom = [geom[0], (geom[0][0], geom[1][1]), geom[1], (geom[1][0], geom[0][1])]
        for point in geom:
            if len(point) == 2:
                edge.AddPoint(float(point[0]), float(point[1]))
        edge.CloseRings()
        geom_ogr = ogr.Geometry(ogr.wkbPolygon)
        geom_ogr.AddGeometry(edge)
        geom_ogr.AssignSpatialReference(osr_sref)
        geom_ogr = force_axis_mapping(geom_ogr)
    elif isinstance(geom, (tuple, list)) and isinstance(geom[0], (tuple, list)) and osr_sref:
        edge = ogr.Geometry(ogr.wkbLinearRing)
        for point in geom:
            if len(point) == 2:
                edge.AddPoint(float(point[0]), float(point[1]))
        edge.CloseRings()
        geom_ogr = ogr.Geometry(ogr.wkbPolygon)
        geom_ogr.AddGeometry(edge)
        geom_ogr.AssignSpatialReference(osr_sref)
        geom_ogr = force_axis_mapping(geom_ogr)
    elif isinstance(geom, shapely.geometry.Polygon):
        geom_ogr = ogr.CreateGeometryFromWkt(geom.wkt)
        geom_ogr.AssignSpatialReference(osr_sref)
        geom_ogr = swap_axis(geom_ogr)  # ensure lon lat order
    elif isinstance(geom, ogr.Geometry):
        geom_sref = geom.GetSpatialReference()
        if geom_sref is None:
            geom.AssignSpatialReference(osr_sref)
        geom_ogr = geom
        geom_ogr = swap_axis(geom_ogr)  # ensure lon lat order
    else:
        raise GeometryUnkown(geom)

    return geom_ogr


def get_polygon_envelope(polygon, x_pixel_size, y_pixel_size):
    """
    Retrieves the envelope of the given polygon geometry in correspondence with the chosen pixel size,
    i.e. the envelope coordinates are rounded to the upper-left corner.

    Parameters
    ----------
    polygon : shapely.geometry.Polygon
        Polygon.
    x_pixel_size : number
        Absolute pixel size in X direction.
    y_pixel_size : number
        Absolute pixel size in Y direction.

    Returns
    -------
    min_x, min_y, max_x, max_y : number, number, number, number
        Envelope of the given polygon geometry in correspondence with the chosen pixel size

    """
    # retrieve polygon points
    poly_pts = list(polygon.exterior.coords)
    # split tuple points into x and y coordinates and convert them to numpy arrays
    xs, ys = [np.array(coords) for coords in zip(*poly_pts)]
    # compute bounding box
    min_x, min_y, max_x, max_y = min(xs), min(ys), max(xs), max(ys)
    # round boundary coordinates to upper-left corner
    min_x = int(round(min_x / x_pixel_size, DECIMALS)) * x_pixel_size
    min_y = int(np.ceil(round(min_y / y_pixel_size, DECIMALS))) * y_pixel_size
    max_x = int(round(max_x / x_pixel_size, DECIMALS)) * x_pixel_size
    max_y = int(np.ceil(round(max_y / y_pixel_size, DECIMALS))) * y_pixel_size

    return min_x, min_y, max_x, max_y


def to_list(value):
    """
    Takes a value and wraps it into a list if it is not already one. The result is returned.
    If None is passed, None is returned.

    Parameters
    ----------
    value : object
        value to convert

    Returns
    -------
    list or None
        A list that wraps the value.

    """
    ret_val = copy.deepcopy(value)
    whitelist = (list, tuple)
    if ret_val is not None:
        ret_val = list(ret_val) if isinstance(ret_val, whitelist) else [value]
    return ret_val


def force_axis_mapping(ogr_geom):
    """
    Forces the given geometry to follow lon-lat order ("AxisMappingStrategy" = 0), if its spatial reference system is
    geographic. I.e. for GDAL 3, the axis order will be swapped.

    Parameters
    ----------
    ogr_geom : ogr.geometry
        OGR geometry.

    Returns
    -------
    ogr.geometry :
        OGR geometry with coordinates in lon-lat order (if applicable).

    """

    osr_sref = ogr_geom.GetSpatialReference()
    sref = SpatialRef.from_osr(osr_sref)
    if sref.epsg == 4326:
        if GDAL_3_ENABLED:
            osr_sref.SetAxisMappingStrategy(0)
            ogr_geom.AssignSpatialReference(osr_sref)

    return ogr_geom


def swap_axis(ogr_geom):
    """
    Swaps axis to lon-lat order ("AxisMappingStrategy" = 0), if its spatial reference system is
    geographic. I.e. for GDAL 3, the axis order will be swapped.

    Parameters
    ----------
    ogr_geom : ogr.geometry
        OGR geometry.

    Returns
    -------
    ogr.geometry :
        OGR geometry with coordinates in lon-lat order (if applicable).

    """

    osr_sref = ogr_geom.GetSpatialReference()
    sref = SpatialRef.from_osr(osr_sref)
    if (sref.epsg == 4326) and GDAL_3_ENABLED and (osr_sref.GetAxisMappingStrategy() == 1):
        ogr_geom.SwapXY()
        osr_sref.SetAxisMappingStrategy(0)
        ogr_geom.AssignSpatialReference(osr_sref)

    return ogr_geom


if __name__ == '__main__':
    pass
