import numpy as np
import xarray as xr
import pandas as pd
from geospade.crs import SpatialRef
from geospade.raster import Tile
from geospade.raster import MosaicGeometry
from yeoda.datacube import DataCubeWriter
from geopathfinder.naming_conventions.yeoda_naming import YeodaFilename


def get_sref():
    sref_wkt = 'PROJCS["Azimuthal_Equidistant",GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",' \
               'SPHEROID["WGS_1984",6378137.0,298.257223563]],PRIMEM["Greenwich",0.0],' \
               'UNIT["Degree",0.017453292519943295]],PROJECTION["Azimuthal_Equidistant"],' \
               'PARAMETER["false_easting",5837287.81977],PARAMETER["false_northing",2121415.69617],' \
               'PARAMETER["central_meridian",24.0],PARAMETER["latitude_of_origin",53.0],UNIT["Meter",1.0]]"'
    return SpatialRef(sref_wkt)


def get_mosaic():
    n_rows, n_cols = 25, 25
    tiles = [Tile(n_rows, n_cols, get_sref(), geotrans=(42e5, 24e3, 0, 24e5, 0, -24e3), name="E042N018T6"),
             Tile(n_rows, n_cols, get_sref(), geotrans=(48e5, 24e3, 0, 24e5, 0, -24e3), name="E048N018T6"),
             Tile(n_rows, n_cols, get_sref(), geotrans=(48e5, 24e3, 0, 18e5, 0, -24e3), name="E048N012T6"),
             Tile(n_rows, n_cols, get_sref(), geotrans=(42e5, 24e3, 0, 18e5, 0, -24e3), name="E042N012T6")]
    return MosaicGeometry.from_tile_list(tiles)


def get_rnd_harmonic_data(noise_scale=0.1, offset=0):
    coeffs_1 = np.array([[1, 1], [0.75, 0.5], [0.5, 0.25]])
    coeffs_2 = np.array([[1, 1], [0.5, 0.5], [0.1, 0.25]])
    data_1 = np.zeros((365,)) + offset
    data_2 = np.zeros((365,)) + offset
    tdoys = np.arange(365)
    for i in range(coeffs_1.shape[0]):
        tv = (2. * np.pi) / 365. * tdoys
        data_1 += coeffs_1[i, 0] * np.cos(i * tv) + coeffs_1[i, 1] * np.sin(i * tv)
        data_2 += coeffs_2[i, 0] * np.cos(i * tv) + coeffs_2[i, 1] * np.sin(i * tv)

    data_1 = np.random.randn(365, 50, 50) * noise_scale + data_1[:, None, None]
    data_2 = np.random.randn(365, 50, 50) * noise_scale + data_2[:, None, None]

    return data_1, data_2


def get_ds(noise_scale=0.1, offset=0):
    var_1, var_2 = get_rnd_harmonic_data(noise_scale=noise_scale, offset=offset)
    mosaic = get_mosaic()
    x_coords = np.arange(mosaic.outer_extent[0], mosaic.outer_extent[2], mosaic.x_pixel_size)
    y_coords = np.arange(mosaic.outer_extent[3], mosaic.outer_extent[1], -mosaic.y_pixel_size)
    dims = ['time', 'y', 'x']
    coords = {'time': pd.date_range('2000-01-01', periods=365),
              'y': y_coords,
              'x': x_coords}
    return xr.Dataset({'VAR1': (dims, var_1), 'VAR2': (dims, var_2)}, coords=coords)


def main():
    dst_dirpath = r"D:\data\code\yeoda\2022_08__docs\general_usage"

    ds_1 = get_ds(noise_scale=0.1, offset=0)
    def_fields = {'var_name': 'DVAR', 'grid_name': 'EU024KM', 'data_version': 'V1', 'sensor_field': 'X1'}
    with DataCubeWriter.from_data(ds_1, dst_dirpath, mosaic=get_mosaic(), filename_class=YeodaFilename, ext='.nc',
                                  def_fields=def_fields, stack_dimension='time', tile_dimension='tile_name',
                                  fn_map={'time': 'datetime_1'}) as dc_writer:
        dc_writer.export(use_mosaic=True)

    ds_2 = get_ds(noise_scale=0.4, offset=5)
    def_fields = {'var_name': 'DVAR', 'grid_name': 'EU024KM', 'data_version': 'V5', 'sensor_field': 'X2'}
    with DataCubeWriter.from_data(ds_2, dst_dirpath, mosaic=get_mosaic(), filename_class=YeodaFilename, ext='.nc',
                                  def_fields=def_fields, stack_dimension='time', tile_dimension='tile_name',
                                  fn_map={'time': 'datetime_1'}) as dc_writer:
        dc_writer.export(use_mosaic=True)


if __name__ == '__main__':
    main()