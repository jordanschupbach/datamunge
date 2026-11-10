"""Native patch branch selection, bit preservation and live continuous paths."""
import gc
import math
import struct
import pytest
from pydatamunge import nn


def test_exact_selected_values_signed_zeros_and_owning_result():
    x = nn.from_vector_double([-0., .2, .3, .4, -0., .6, .7, .8], [2, 2, 2])
    prediction = nn.full([2, 2, 2], .9, nn.DType.Float64)
    hidden = nn.from_vector_double([0, 1, 1, 0], [2, 2])
    visible = nn.VisiblePatchValues(x, hidden, -0.)
    values = tuple(visible.values())
    assert all(math.copysign(1, values[i]) == -1 for i in [0, 2, 4])
    out = nn.CompositePatchValues(x, prediction, hidden)
    assert math.copysign(1, out.values()[0]) == -1
    assert tuple(out.values())[1:] == (.2, .9, .9, .9, .9, .7, .8)
    out.fill_(17)
    assert tuple(x.values())[1:] == (.2, .3, .4, -0., .6, .7, .8)
    retained = nn.CompositePatchValues(x, prediction, hidden)
    del x, prediction, hidden
    gc.collect()
    assert tuple(retained.shape()) == (2, 2, 2)
    assert tuple(retained.values())[1] == .2


def test_gradients_select_only_the_intended_branches():
    t = nn.DType.Float64
    x = nn.full([2, 2, 3], .2, t).requires_grad_()
    prediction = nn.full([2, 2, 3], .7, t).requires_grad_()
    hidden = nn.from_vector_double([0, 1, 1, 0], [2, 2])
    derivatives = nn.grad(nn.sum(nn.CompositePatchValues(x, prediction, hidden)), [x, prediction])
    assert tuple(derivatives[0].values()) == (1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1)
    assert tuple(derivatives[1].values()) == (0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0)
    assert nn.equal_bits(nn.grad(nn.sum(nn.VisiblePatchValues(x, hidden, 9)), [x])[0], derivatives[0])


def test_generic_strides_all_visible_all_hidden_and_float32_fill_rounding():
    t = nn.DType.Float64
    x = nn.transpose(nn.full([4, 3, 2], -3, t), [2, 1, 0])
    prediction = nn.full([2, 3, 4], 7, t)
    hidden = nn.transpose(nn.ones([3, 2], t))
    assert not x.is_contiguous() and not hidden.is_contiguous()
    assert nn.equal_bits(nn.CompositePatchValues(x, prediction, hidden), prediction)
    assert nn.equal_bits(nn.CompositePatchValues(x, prediction, nn.zeros([2, 3], t)), x)
    assert tuple(nn.VisiblePatchValues(x, hidden, -7).values()) == (-7,)*24
    fill = .123456789123
    expected = struct.unpack('f', struct.pack('f', fill))[0]
    assert nn.VisiblePatchValues(nn.zeros([1, 1, 1]), nn.ones([1, 1]), fill).item() == expected
    assert nn.VisiblePatchValues(nn.ones([1, 1, 1]), nn.ones([1, 1]), 1e-100).item() == 0


def test_invalid_masks_shapes_and_unselected_values_are_rejected():
    x = nn.zeros([2, 3, 4])
    for hidden in [nn.zeros([1, 3]), nn.full([2, 3], .5), nn.zeros([2, 3], nn.DType.Float64),
                   nn.zeros([2, 3], nn.DType.Bool), nn.ones([2, 3]).requires_grad_()]:
        with pytest.raises(RuntimeError):
            nn.VisiblePatchValues(x, hidden)
        with pytest.raises(RuntimeError):
            nn.CompositePatchValues(x, x, hidden)
    for fill in [float('inf'), float('nan'), 1e100]:
        with pytest.raises(RuntimeError):
            nn.VisiblePatchValues(x, nn.zeros([2, 3]), fill)
    with pytest.raises(RuntimeError):
        nn.CompositePatchValues(x, nn.full([2, 3, 4], float('nan')), nn.zeros([2, 3]))
    with pytest.raises(RuntimeError):
        nn.VisiblePatchValues(nn.zeros([2, 3]), nn.zeros([2, 3]))
