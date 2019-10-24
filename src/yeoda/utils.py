import os
import ogr
import re

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

