## geopathfinder
- add create_sgrt_filename like functionality to SmartFilename
- should the functionality of getting the desired filenames be addded(e.g., using SmartPaths)?

## pyraster
- use an efficient way of querying and accessing the data: also an xarray for metadata? 
#### NetCDF:
- xarray with minimum dimensions/coordinates x,y,time and user defined data variables and coordinates (polarisations, sgrt_vars, bands).
- xarray.data is a Dask array.
- "views" on dimensions/coordinates necessary
- neighbouring "tiled" NetCDFs could be merged and cut internally.
#### GeoTIFF: 
- GeoTIFFs could be lazily read and stacked with PIL and Dask. Merging and cutting should be done over array indices.
- read GeoTIFFs with GDAL and do the same as before.
- apply chunking of the data
- "tiff2netcdfstack" to convert these stacks to NetCDF files.

## pytileproj
- add resampling functionality (pyresample package)
- restructure it that tiles can be easily be used as a single instance

## miscellaneous
- work on RS math packages (slope computations, soil moisture, ...)
- work on external projection package (e.g., pi)