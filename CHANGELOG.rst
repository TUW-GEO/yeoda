=========
Changelog
=========

Version 0.3.0
=============

- added more input argument checks to the data loading functions
- added geospade as a new dependency to allow for a faster polygon masking
- replaced some core functions by new tools from geospade
- upgrade to GDAL 3 and Python 3.7
- upgrade of Equi7Grid version
- updated interface to product data cubes

Version 0.2.2
=============

- bug fix: 2D arrays/datacubes are now always converted to 3D

Version 0.2.1
=============

- bug fix: decoding after load_by_pixels()

Version 0.2.0
=============

- fixed major bugs, like not-working product data cube classes and wrong rounding of pixel coordinates
- added data cube for SCATSAR-SWI data
- added data cube for Parameter data, e.g. TMENPLIA
- switched to newest version of geopathfinder and veranda

Version 0.1.4
=============

- changed 'in_place' to 'inplace' to have same syntax as Pandas
- new filename parsing syntax when using geopathfinder
- major bug fix: column and row access was partly switched

Version 0.1.3
=============

- Enhanced NetCDF stack handling by reading it as a stack
- Keep data set pointer
- Added clipping and limit reading to tile/file boundaries
- Removed 'geometry' as default dimension
- Removed bugs

Version 0.1.2
=============

- Minor changes in styling, package requirements and conda environments

Version 0.1.1
=============

- Minor changes in setup.cfg

Version 0.1.0
=============

- First release
