"""Native three-family observations, analytic reference laws and quality diagnostics."""
import gc
import math
import pytest
from pydatamunge import nn as n


def test_all_three_explicit_transforms_and_component_labels():
    u=n.from_vector_double([0,0,0,2,1,-10],[2,3]).requires_grad_()
    banana=n.FlowStudyPoints(u,0)
    assert list(banana.points.values())==[0,-.5,2,2.5]
    assert list(banana.components.values())==[-1,-1] and banana.points.dtype()==n.DType.Float32
    ring=n.FlowStudyPoints(u,1,n.DType.Float64)
    assert list(ring.components.values())==[4,0]
    assert list(ring.points.values())==pytest.approx([-2,0,2.3,.15],abs=3e-16)
    checker=n.FlowStudyPoints(u,2,n.DType.Float64)
    assert list(checker.points.values())==pytest.approx([-1.5,.5,-1.0227501319481793,-1.1586552539314572])
    assert not banana.points.requires_grad() and not checker.points.requires_grad()


def test_erfc_tail_and_unclipped_boundary_points():
    u=n.from_vector_double([-10,0,.5,10,10,10],[2,3])
    out=n.FlowStudyPoints(u,2,n.DType.Float64)
    values=list(out.points.values())
    assert values[0]==pytest.approx(7.619853024160593e-24,rel=3e-14,abs=0)
    assert values[2:]==[2,2] and list(out.components.values())==[5,7]
    assert n.FlowStudyLogProbability(n.slice_axis(out.points,0,1,2),2).item()==-math.inf


def test_checker_half_open_truth_banana_reference_and_stable_mixture():
    x=n.from_vector_double([-2,-2,-1,-2,0,0,2,0,-2,2],[5,2]).requires_grad_()
    truth=n.FlowStudyLogProbability(x,2)
    assert list(truth.values())==[-math.log(8),-math.inf,-math.log(8),-math.inf,-math.inf]
    assert not truth.requires_grad() and truth.dtype()==n.DType.Float64
    banana=n.from_vector_double([0,-.5,2,2.5],[2,2])
    assert n.max(n.abs(n.sub(n.FlowStudyLogProbability(banana,0),n.BananaLogProbability(banana)))).item()<1e-15
    assert math.isfinite(n.FlowStudyLogProbability(n.from_vector_double([100,100],[1,2]),1).item())


def test_sample_diagnostics_inclusive_threshold_and_record_lifetime():
    samples=n.from_vector_double([2,0,-2,0,2,.45,2,math.nextafter(.45,1),0,0],[5,2])
    out=n.FlowStudySampleDiagnostics(samples);counts=out.mode_counts
    assert list(counts.values())==[2,0,0,0,1,0,0,0]
    assert out.modes_covered==2 and out.fraction_near_modes==.6
    assert out.checkerboard_support_fraction==.4
    assert out.banana_curvature_moment==pytest.approx(.54)
    points=n.FlowStudyPoints(n.zeros([1,3]),0).points
    del out,samples;gc.collect()
    assert list(counts.values())==[2,0,0,0,1,0,0,0] and list(points.values())==[0,-.5]


def test_strided_inputs_keep_values_and_do_not_mutate():
    for dtype in [n.DType.Float32,n.DType.Float64]:
        u=n.cast(n.from_vector_double([0,0,0,2,1,-1],[2,3]),dtype);before=u.clone()
        strided=n.transpose(n.transpose(u,[1,0]).contiguous(),[1,0])
        for kind in range(3):
            a=n.FlowStudyPoints(u,kind,dtype);b=n.FlowStudyPoints(strided,kind,dtype)
            assert n.equal_bits(a.points,b.points) and n.equal_bits(a.components,b.components)
        assert n.equal_bits(u,before)


def test_invalid_inputs_and_unsafe_arithmetic_reject():
    for bad in [n.zeros([3]),n.zeros([0,3]),n.zeros([1,2]),n.full([1,3],math.nan)]:
        with pytest.raises(RuntimeError):n.FlowStudyPoints(bad,0)
    for kind in [-1,3]:
        with pytest.raises(RuntimeError):n.FlowStudyPoints(n.zeros([1,3]),kind)
        with pytest.raises(RuntimeError):n.FlowStudyLogProbability(n.zeros([1,2]),kind)
    for bad in [n.zeros([2]),n.zeros([0,2]),n.zeros([1,3]),n.full([1,2],math.inf)]:
        with pytest.raises(RuntimeError):n.FlowStudySampleDiagnostics(bad)
        with pytest.raises(RuntimeError):n.FlowStudyLogProbability(bad,2)
    with pytest.raises(RuntimeError):n.FlowStudyPoints(n.zeros([1,3]),0,n.DType.Int32)
    with pytest.raises(RuntimeError):n.FlowStudyPoints(n.full([1,3],1e200,n.DType.Float64),0,n.DType.Float64)
    with pytest.raises(RuntimeError):n.FlowStudySampleDiagnostics(n.full([1,2],1e200,n.DType.Float64))
    with pytest.raises(RuntimeError):n.FlowStudyLogProbability(n.full([1,2],1e200,n.DType.Float64),1)
