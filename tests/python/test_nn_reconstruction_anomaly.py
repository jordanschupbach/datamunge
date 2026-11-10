"""Native detached residual diagnostics, nominal-only calibration and tied rankings."""
import gc
import math
import struct
import pytest
from pydatamunge import nn


def test_scores_flatten_different_shapes_stop_paths_and_own_values():
    x = nn.from_vector([0, 1, 2, 3], [2, 2]).requires_grad_()
    y = nn.zeros([2, 1, 2], nn.DType.Float64).requires_grad_()
    residual = nn.ReconstructionScores(x, y)
    assert residual.dtype() == nn.DType.Float64 and tuple(residual.values()) == (.5, 6.5)
    ink = nn.InkScores(x, .5)
    assert tuple(ink.values()) == (0, 2)
    assert not residual.requires_grad() and not ink.requires_grad()
    assert x.requires_grad() and y.requires_grad()
    with nn.no_grad():
        x.fill_(9)
    del x, y
    gc.collect()
    assert tuple(residual.values()) == (.5, 6.5) and tuple(ink.values()) == (0, 2)
    cancellation = nn.from_vector_double([1e16, 1, -1e16], [1, 3])
    assert nn.InkScores(cancellation, .25).item() == .25


def test_nearest_rank_has_no_interpolation_fuzz_or_query_refit():
    scores = nn.select(nn.arange(0, 40).reshape([20, 2]), 1, 0).requires_grad_()
    assert nn.NominalThreshold(scores) == 36
    assert nn.NominalThreshold(scores, .5) == 18
    assert nn.NominalThreshold(scores, math.nextafter(.5, 1)) == 20
    assert nn.NominalThreshold(scores, 1) == 38
    assert nn.NominalThreshold(scores, math.ulp(0.0)) == 0
    assert scores.requires_grad()
    threshold = nn.NominalThreshold(nn.arange(0, 20))
    assert threshold == 18
    result = nn.RankAnomalies(nn.from_vector([18, 19], [2]), nn.from_vector([18, 1000], [2]), threshold)
    assert result.normal_fpr == result.anomaly_tpr == .5


def test_rankings_group_ties_strict_flags_mixed_dtypes_and_owning_record():
    normal = nn.from_vector_double([1, 2], [2]).requires_grad_()
    anomaly = nn.from_vector([2, 3], [2])
    result = nn.RankAnomalies(normal, anomaly, 2)
    assert result.auroc == .875 and result.average_precision == pytest.approx(5/6, abs=1e-15)
    assert result.normal_fpr == 0 and result.anomaly_tpr == .5
    lower = nn.RankAnomalies(normal, anomaly, math.nextafter(2, -math.inf))
    assert lower.normal_fpr == .5 and lower.anomaly_tpr == 1
    tied = nn.RankAnomalies(nn.ones([3]), nn.ones([2], nn.DType.Float64), 1)
    assert (tied.auroc, tied.average_precision, tied.normal_fpr, tied.anomaly_tpr) == (.5, .4, 0, 0)
    assert normal.requires_grad()
    del normal, anomaly
    gc.collect()
    assert result.auroc == .875


def test_stress_transform_pixels_strides_stopped_flags_and_owned_results():
    def f32(x):
        return struct.unpack('<f', struct.pack('<f', x))[0]
    data = [f32((i % 61-30)/30) for i in range(1568)]
    original = nn.transpose(nn.from_vector(data, [2, 28, 28, 1]), [0, 2, 1, 3]).requires_grad_()
    before = tuple(original.contiguous().values())
    outputs = []
    for kind in range(3):
        output = nn.AnomalyCorruption(original, kind)
        assert output.is_contiguous() and not output.requires_grad()
        assert tuple(output.shape()) == (2, 28, 28, 1)
        expected = []
        for b in range(2):
            for y in range(28):
                for x in range(28):
                    value = before[b*784+y*28+x]
                    if kind == 0 and y < 8 and x < 8:
                        value = 1
                    elif kind == 1:
                        value = f32(value*f32(.35))
                    elif kind == 2:
                        value = before[b*784+y*28+x-6] if x >= 6 else 0
                    expected.append(value)
        assert tuple(output.values()) == tuple(expected)
        outputs.append(output)
    assert original.requires_grad() and tuple(original.contiguous().values()) == before
    with nn.no_grad():
        original.fill_(9)
    del original
    gc.collect()
    assert outputs[2].values()[10] != 9


def test_invalid_geometry_scores_thresholds_quantiles_and_overflow_fail():
    for bad in [nn.zeros([0]), nn.ones([2, 2]), nn.ones([2], nn.DType.Int32), nn.full([2], float('nan'))]:
        with pytest.raises(RuntimeError):
            nn.NominalThreshold(bad)
        with pytest.raises(RuntimeError):
            nn.RankAnomalies(bad, nn.ones([2]), 0)
    for q in [0, -1, 1.00001, math.inf, math.nan]:
        with pytest.raises(RuntimeError):
            nn.NominalThreshold(nn.ones([2]), q)
    with pytest.raises(RuntimeError):
        nn.RankAnomalies(nn.ones([2]), nn.ones([2]), math.inf)
    with pytest.raises(RuntimeError):
        nn.ReconstructionScores(nn.ones([2, 3]), nn.ones([3, 2]))
    with pytest.raises(RuntimeError):
        nn.InkScores(nn.ones([2]), math.nan)
    for bad in [nn.zeros([0, 28, 28, 1]), nn.ones([1, 28, 28, 1], nn.DType.Float64), nn.full([1, 28, 28, 1], math.inf)]:
        with pytest.raises(RuntimeError):
            nn.AnomalyCorruption(bad, 0)
    with pytest.raises(RuntimeError):
        nn.AnomalyCorruption(nn.ones([1, 28, 28, 1]), 3)
    huge = nn.full([1, 2], float.fromhex('0x1.fffffffffffffp+1023'), nn.DType.Float64)
    with pytest.raises(RuntimeError):
        nn.ReconstructionScores(huge, nn.zeros([1, 2], nn.DType.Float64))
