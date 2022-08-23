import copy
from geopathfinder.file_naming import SmartFilename


def create_fn_class(fields_def, pad='-', delimiter='_'):
    """
    Creates a `DefaultFilename` class inheriting from `SmartFilename`.

    Parameters
    ----------
    fields_def : OrderedDict
            Name of fields (keys) in right order and length (values). It must contain:
                - "len": int
                    Length of filename part (must be given).
                    "0" to allow any length.
                - "start": int, optional
                    Start index of filename part (default is 0).
                - "delim": str, optional
                    Delimiter between this and the following filename part (default is the one from the parent class).
                - "pad": str,
                    Padding for filename part (default is the one from the parent class).
    ext : str, optional
        File name extension (default: None).
    pad : str, optional
        Padding symbol (default: '-').

    Returns
    -------
    DefaultFilename : class
        Default filename class.

    """
    class DefaultFilename(SmartFilename):
        def __init__(self, fields, ext=".tif", convert=False, compact=False):
            super(DefaultFilename, self).__init__(fields, fields_def, ext=ext, pad=pad, delimiter=delimiter,
                                                  convert=convert, compact=compact)
    return DefaultFilename


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
