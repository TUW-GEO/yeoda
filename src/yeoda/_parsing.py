import os
from numba import jit, prange


#@jit
def parse_filepath_fast(filepaths, fn_map, filename_class):
    n_files = len(filepaths)
    for file_idx in range(n_files):
        print(file_idx)
        filepath = filepaths[file_idx]
        fn = filename_class.from_filename(os.path.basename(filepath), convert=True)
        dimensions = list(fn_map.keys())
        for dimension in dimensions:
            fn_map[dimension][file_idx] = fn[dimension]

    return fn_map


