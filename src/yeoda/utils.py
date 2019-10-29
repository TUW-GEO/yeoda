import os
import ogr
import re
from array import array
import math
#import numba
import copy

import pytileproj.geometry as geometry
import shapely.geometry

from yeoda.errors import GeometryUnkown


def get_file_type(filepath):
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


def any_geom2ogr_geom(geom, osr_spref):
    """
    Transforms an extent represented in different ways or a Shapely geometry object into an OGR geometry object.

    Parameters
    ----------
    geom: ogr.Geometry or shapely.geometry or list or tuple, optional
        A vector geometry. If it is of type list/tuple representing the extent (i.e. [x_min, y_min, x_max, y_max]),
        `osr_spref` has to be given to transform the extent into a georeferenced polygon.
    osr_spref: osr.SpatialReference, optional
        Spatial reference of the given geometry `geom`.

    Returns
    -------
    ogr.Geometry
        Vector geometry as an OGR Geometry object.
    """

    if isinstance(geom, (tuple, list)) and (not isinstance(geom[0], (tuple, list))) and \
            (len(geom) == 4) and osr_spref:
        geom_ogr = geometry.bbox2polygon(geom, osr_spref)
    elif isinstance(geom, (tuple, list)) and (isinstance(geom[0], (tuple, list))) and \
            (len(geom) == 2) and osr_spref:
        edge = ogr.Geometry(ogr.wkbLinearRing)
        geom = [geom[0], (geom[0][0], geom[1][1]), geom[1], (geom[1][0], geom[0][1])]
        for point in geom:
            if len(point) == 2:
                edge.AddPoint(float(point[0]), float(point[1]))
        edge.CloseRings()
        geom_ogr = ogr.Geometry(ogr.wkbPolygon)
        geom_ogr.AddGeometry(edge)
        geom_ogr.AssignSpatialReference(osr_spref)
    elif isinstance(geom, (tuple, list)) and isinstance(geom[0], (tuple, list)) and osr_spref:
        edge = ogr.Geometry(ogr.wkbLinearRing)
        for point in geom:
            if len(point) == 2:
                edge.AddPoint(float(point[0]), float(point[1]))
        edge.CloseRings()
        geom_ogr = ogr.Geometry(ogr.wkbPolygon)
        geom_ogr.AddGeometry(edge)
        geom_ogr.AssignSpatialReference(osr_spref)
    elif isinstance(geom, shapely.geometry.Polygon):
        geom_ogr = ogr.CreateGeometryFromWkt(geom.wkt)
        geom_ogr.AssignSpatialReference(osr_spref)
    elif isinstance(geom, ogr.Geometry):
        geom_ogr = geom
    else:
        raise GeometryUnkown(geom)

    return geom_ogr


def xy2ij(x, y, gt):
    """
    Transforms global/world system coordinates to pixel coordinates/indexes.

    Parameters
    ----------
    x: float
        World system coordinate in X direction.
    y: float
        World system coordinate in Y direction.
    gt: tuple
        Geo-transformation parameters/dictionary.

    Returns
    -------
    i: int
        Row number in pixels.
    j: int
        Column number in pixels.
    """

    i = int(round(-1.0 * (gt[2] * gt[3] - gt[0] * gt[5] + gt[5] * x - gt[2] * y) /
                  (gt[2] * gt[4] - gt[1] * gt[5])))
    j = int(round(-1.0 * (-1 * gt[1] * gt[3] + gt[0] * gt[4] - gt[4] * x + gt[1] * y) /
                  (gt[2] * gt[4] - gt[1] * gt[5])))
    return i, j


def ij2xy(i, j, gt):
    """
    Transforms global/world system coordinates to pixel coordinates/indexes.

    Parameters
    ----------
    i: int
        Row number in pixels.
    j: int
        Column number in pixels.
    gt: dict
        Geo-transformation parameters/dictionary.

    Returns
    -------
    x: float
        World system coordinate in X direction.
    y: float
        World system coordinate in Y direction.
    """

    x = gt[0] + i * gt[1] + j * gt[2]
    y = gt[3] + i * gt[4] + j * gt[5]
    return x, y

def rasterise_polygon(points, sres=1., outer=True):
    """
    Edge-flag algorithm.

    Parameters
    ----------
    points

    Returns
    -------
    """

    xs, ys = list(zip(*points))
    x_min = min(xs)
    x_max = max(xs)
    y_min = min(ys)
    y_max = max(ys)

    x_min_l = math.floor(x_min/sres)*sres
    x_min_r = x_min_l + sres
    x_max_l = math.floor(x_max/sres)*sres
    x_max_r = x_max_l + sres
    y_min_b = math.floor(y_min/sres)*sres
    y_min_t = y_min_b + sres
    y_max_b = math.floor(y_max/sres)*sres
    y_max_t = y_max_b + sres
    half_sres = sres/2.
    if outer:
        if abs(x_min - x_min_r) < half_sres:
            x_min = x_min_r
        else:
            x_min = x_min_l

        if abs(x_max - x_max_l) < half_sres:
            x_max = x_max_l
        else:
            x_max = x_max_r

        if abs(y_min - y_min_t) < half_sres:
            y_min = y_min_t
        else:
            y_min = y_min_b

        if abs(y_max - y_max_b) < half_sres:
            y_max = y_max_b
        else:
            y_max = y_max_t
    else:
        if abs(x_min - x_min_l) < half_sres:
            x_min = x_min_l
        else:
            x_min = x_min_r

        if abs(x_max - x_max_r) < half_sres:
            x_max = x_max_r
        else:
            x_max = x_max_l

        if abs(y_min - y_min_b) < half_sres:
            y_min = y_min_b
        else:
            y_min = y_min_t

        if abs(y_max - y_max_t) < half_sres:
            y_max = y_max_t
        else:
            y_max = y_max_b

    n_rows = int((y_max - y_min)/sres) + 1
    n_cols = int((x_max - x_min)/sres) + 1
    raster = [[0]*n_cols]*n_rows
    for idx in range(1, len(xs)):
        x_1 = xs[idx - 1]
        x_2 = xs[idx]
        y_1 = ys[idx - 1]
        y_2 = ys[idx]
        x_diff = x_2 - x_1
        y_diff = y_2 - y_1
        if y_diff == 0.:
            continue
        if x_diff == 0.:
            k = None
        else:
            k = y_diff/x_diff
        q = get_quadrant(x_diff, y_diff)
        y = copy.deepcopy(y_1)
        while (y <= y_2):
            if k is not None:
                x = (y - y_1)/k + x_1
            else:
                x = x_1
            if math.floor(x/sres)*sres != x:
                x_l = math.floor(x)
                x_r = x_l + sres
                if outer:
                    if q in [1, 2]:
                        x = x_l
                    elif q in [3, 4]:
                        x = x_r
                else:
                    if q in [1, 2]:
                        x = x_r
                    elif q in [3, 4]:
                        x = x_l
            i = int((y_max - y) / sres)
            j = int((x - x_min) / sres)
            raster[i][j] = 1
            y = y + sres

    for i in range(n_rows):
        is_inner = False
        for j in range(n_cols):
            if raster[i][j]:
                is_inner = ~is_inner
            if is_inner:
                raster[i][j] = 1
            else:
                raster[i][j] = 0


def get_quadrant(x_diff, y_diff):
    if x_diff > 0 and y_diff > 0:
        return 1
    elif x_diff < 0 and y_diff > 0:
        return 2
    elif x_diff < 0 and y_diff < 0:
        return 3
    elif x_diff > 0 and y_diff < 0:
        return 4
    else:
        return None

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    points = [(19, 12), (20, 13), (21, 13), (21, 12), (20, 12), (19, 12)]
    rasterise_polygon(points, sres=0.1)