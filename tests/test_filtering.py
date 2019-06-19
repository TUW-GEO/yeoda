# general imports
from shapely.wkt import load
from shapely.geometry import Polygon
import matplotlib.pyplot as plt

# import yeoda
from src.yeoda.yeoda import eoDataCube, match_dimension

# import file and folder naming convention
from geopathfinder.sgrt_naming import create_sgrt_filename, sgrt_tree

# import grid
from Equi7Grid.equi7grid.equi7grid import Equi7Grid

#root_dirpath = r"R:\Projects\SHRED\07_data\Sentinel-1_CSAR"
root_dirpath = r"D:\work\data\yeoda\Sentinel-1_CSAR"
roi = Polygon([(4373136, 1995726), (4373136, 3221041), (6311254, 3221041), (6311254, 1995726)])
st = sgrt_tree(root_dirpath, register_file_pattern=(".tif$"))
eodc = eoDataCube(dir_tree=st, smart_filename_creator=create_sgrt_filename,
                  dimensions=['time', 'tile_name', 'pol'])


fig, ax = plt.subplots(1, 1)
eodc.inventory.drop_duplicates(subset=['tile_name']).plot(ax=ax, edgecolor='red', facecolor="none")
eodc_roi = eodc.filter_spatially(roi=roi)
eodc_roi.inventory.drop_duplicates(subset=['tile_name']).plot(ax=ax, edgecolor='blue', facecolor="none")
plt.show()