"""
Utilities and helping functions for the other modules of yeoda.
"""

# general packages
import copy
import pandas as pd
import numpy as np
from geopathfinder.file_naming import SmartFilename


def create_fn_class(fields_def, pad='-', delimiter='_'):

    class DefaultFilename(SmartFilename):
        def __init__(self, fields, ext=".tif", convert=False, compact=False):
            super(DefaultFilename, self).__init__(fields, fields_def, ext=ext, pad=pad, delimiter=delimiter,
                                                  convert=convert, compact=compact)

    return DefaultFilename


# TODO: create a grouper class out of this function
def temporal_grouper(ts, freq):
    n_ts = len(ts)
    min_time, max_time = min(ts), max(ts)
    grouped_ts = pd.date_range(min_time, max_time, freq=freq).union([min_time, max_time])
    group_idxs = np.zeros(n_ts)
    for i in range(len(grouped_ts) - 1):
        group_idxs[grouped_ts[i] <= ts <= grouped_ts[i + 1]] = i
    return group_idxs


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


if __name__ == '__main__':
    pass
