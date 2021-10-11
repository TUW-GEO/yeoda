"""
Some small & concise tests for utility functions.
"""

from src.yeoda.utils import to_list


def test_ensure_list():
    """ Tests utility function that always returns a list except it is None. """

    assert to_list(1) == [1]
    assert to_list([1, 2, 3]) == [1, 2, 3]
    assert to_list((1, 2, 3)) == [1, 2, 3]
    assert to_list(None) is None
