from yeoda import eoDataCube, match_dimension
from collections import OrderedDict

# put this somewhere else
fields_def = OrderedDict([
                     ('pflag', {'len': 1, 'delim': False}),
                     ('dtime_1', {'len': 8, 'delim': False}),
                     ('dtime_2', {'len': 8, 'delim': True}),
                     ('var_name', {'len': 9, 'delim': True}),
                     ('mission_id', {'len': 2, 'delim': True}),
                     ('spacecraft_id', {'len': 1, 'delim': False}),
                     ('mode_id', {'len': 2, 'delim': False}),
                     ('product_type', {'len': 3, 'delim': False}),
                     ('res_class', {'len': 1, 'delim': False}),
                     ('level', {'len': 1, 'delim': False}),
                     ('pol', {'len': 2, 'delim': False}),
                     ('orbit_direction', {'len': 1, 'delim': False}),
                     ('relative_orbit', {'len': 3, 'delim': True}),
                     ('workflow_id', {'len': 5, 'delim': True}),
                     ('grid_name', {'len': 6, 'delim': True}),
                     ('tile_name', {'len': 10, 'delim': True})
                    ])

# initialise cube
start_time = "2019-01-01"
end_time = "2019-03_01"
tilename = "EU10M_E048N018T1"
dimensions = ("dtime_1", "tile_name", "pol")
filepaths = None # pre-filter filepaths with SmartTree (all files in folder sig0)
grid = None # initialise the grid
eodc = eoDataCube(grid, filepaths, fields_def, dimensions=dimensions)
eodc = eodc.rename_dimensions({"dtime_1": "time", "time_name": "tile"})
eodc = eodc.filter_spatial(tilenames=[tilename]).filter_dimension((start_time, end_time), expression=(">=", "<"))

# B01 workflow, Mean over time
data = eodc.load(data_variables=("pol"))
# data should be an xarray:
# coordinates: x,y,time; data_variables: VV, VH

# B01 workflow, Mean over time per orbit
orbit_num = [create_smart_filename(filepath).relative_orbit for filepath in eodc.filepaths]
eodc = eodc.add_dimension("orbit_num", orbit_num)
# loop over orbit numbers and do the computation for each orbit number

# cross-ratio
eodc_vv = eodc.filter_dimension(("VV"), name="pol")
eodc_vh = eodc.filter_dimension(("VH"), name="pol")
eodc_vv, eodc_vh = match_dimension(eodc_vv, eodc_vh, "time")
# loop over timestamps