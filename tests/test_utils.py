"""
Some small & concise tests for utility functions.
"""

from src.yeoda.utils import ensure_is_list


def test_ensure_list():
    """ Tests utility function that returns a list if type of variable is not list or tuple. """

    assert ensure_is_list(1) == [1]
    assert ensure_is_list([1, 2, 3]) == [1, 2, 3]
    assert ensure_is_list((1, 2, 3)) == [1, 2, 3]
    assert ensure_is_list((1, 2, 3), allow_tuples=True) == (1, 2, 3)
    assert ensure_is_list(None) is None
