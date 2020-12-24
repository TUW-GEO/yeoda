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

# geo packages
from osgeo import ogr
from osgeo import osr
import pytileproj.geometry as geometry
import shapely.geometry

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
    elif isinstance(geom, (tuple, list)) and isinstance(geom[0], (tuple, list)) and osr_sref:
        edge = ogr.Geometry(ogr.wkbLinearRing)
        for point in geom:
            if len(point) == 2:
                edge.AddPoint(float(point[0]), float(point[1]))
        edge.CloseRings()
        geom_ogr = ogr.Geometry(ogr.wkbPolygon)
        geom_ogr.AddGeometry(edge)
        geom_ogr.AssignSpatialReference(osr_sref)
    elif isinstance(geom, shapely.geometry.Polygon):
        geom_ogr = ogr.CreateGeometryFromWkt(geom.wkt)
        geom_ogr.AssignSpatialReference(osr_sref)
    elif isinstance(geom, ogr.Geometry):
        geom_sref = geom.GetSpatialReference()
        if geom_sref is None:
            geom.AssignSpatialReference(osr_sref)
        geom_ogr = geom
    else:
        raise GeometryUnkown(geom)

    return geom_ogr


def xy2ij(x, y, gt):
    """
    Transforms global/world system coordinates to pixel coordinates/indexes.

    Parameters
    ----------
    x : float
        World system coordinate in X direction.
    y : float
        World system coordinate in Y direction.
    gt : tuple
        Geo-transformation parameters/dictionary.

    Returns
    -------
    i : int
        Column number in pixels.
    j : int
        Row number in pixels.
    """

    i = int(-1.0 * (gt[2] * gt[3] - gt[0] * gt[5] + gt[5] * x - gt[2] * y) /
            (gt[2] * gt[4] - gt[1] * gt[5]))
    j = int(-1.0 * (-1 * gt[1] * gt[3] + gt[0] * gt[4] - gt[4] * x + gt[1] * y) /
            (gt[2] * gt[4] - gt[1] * gt[5]))
    return i, j


def ij2xy(i, j, gt, origin="ul"):
    """
    Transforms global/world system coordinates to pixel coordinates/indexes.

    Parameters
    ----------
    i : int
        Column number in pixels.
    j : int
        Row number in pixels.
    gt : dict
        Geo-transformation parameters/dictionary.
    origin: str, optional
        Defines the world system origin of the pixel. It can be:
            - upper left ("ul")
            - upper right ("ur", default)
            - lower right ("lr")
            - lower left ("ll")
            - center ("c")

    Returns
    -------
    x : float
        World system coordinate in X direction.
    y : float
        World system coordinate in Y direction.
    """

    px_shift_map = {"ul": (0, 0),
                    "ur": (1, 0),
                    "lr": (1, 1),
                    "ll": (0, 1),
                    "c": (.5, .5)}

    if origin in px_shift_map.keys():
        px_shift = px_shift_map[origin]
    else:
        user_wrng = "Pixel origin '{}' unknown. Upper left origin 'ul' will be taken instead".format(origin)
        raise Warning(user_wrng)
        px_shift = (0, 0)

    i += px_shift[0]
    j += px_shift[1]
    x = gt[0] + i * gt[1] + j * gt[2]
    y = gt[3] + i * gt[4] + j * gt[5]

    return x, y


def boundary(gt, sref, shape):
    """
    Creates raster boundary polygon from geotransformation and shape parameters.

    Parameters
    ----------
    gt: tuple
        Geotransformation parameters.
    sref: osr.SpatialReference
        Spatial reference of the boundary polygon.
    shape: tuple
        Defines the size of the boundary polygon/raster (rows, columns).

    Returns
    -------
    ogr.Geometry
        Boundary polygon with the given spatial reference system assigned.
    """

    boundary_extent = (gt[0], gt[3] + shape[0] * gt[5], gt[0] + shape[1] * gt[1], gt[3])
    boundary_spref = osr.SpatialReference()
    boundary_spref.ImportFromWkt(sref)
    bbox = [(boundary_extent[0], boundary_extent[1]), (boundary_extent[2], boundary_extent[3])]
    boundary_geom = geometry.bbox2polygon(bbox, boundary_spref)

    return boundary_geom

if __name__ == '__main__':
    pass
