"""Legacy geometry binding retains owned results and maps checked native failures."""
import gc
import math
import sys

import pytest
from pydatamunge import datamunge as dm


def points(values):
    result = dm.Point2DVector()
    for x, y in values:
        point = dm.Point2D()
        point.x, point.y = x, y
        result.push_back(point)
    return result


def test_closest_pair_owns_result_and_preserves_first_tie():
    data = points([(0, 0), (1, 0), (0, 1)])
    result = dm.closest_pair(data)
    del data
    gc.collect()
    assert (result.a.x, result.a.y, result.b.x, result.b.y) == (0, 0, 1, 0)
    assert result.distance == 1


@pytest.mark.parametrize("scale", [1e-200, 1e200])
def test_widened_native_distance_selects_the_right_pair(scale):
    result = dm.closest_pair(points([(0, 0), (3 * scale, 4 * scale), (0, scale)]))
    assert result.distance == scale
    assert result.b.x == 0 and result.b.y == scale


@pytest.mark.parametrize("data", [[], [(0, 0)], [(0, 0), (math.nan, 0)],
                                  [(0, 0), (0, math.inf)]])
def test_invalid_geometry_raises_value_error(data):
    with pytest.raises(ValueError):
        dm.closest_pair(points(data))


def test_unrepresentable_legacy_distance_raises_overflow_error():
    with pytest.raises(OverflowError):
        dm.closest_pair(points([(-sys.float_info.max, 0), (sys.float_info.max, 0)]))


def test_generated_help_contains_the_updated_contract():
    assert "widened" in dm.closest_pair.__doc__
    assert "nonfinite" in dm.closest_pair.__doc__
