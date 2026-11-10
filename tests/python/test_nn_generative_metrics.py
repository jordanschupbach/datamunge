"""Native low-rank feature moments and detached probability/split score bindings."""
import gc
import math
import pytest
from pydatamunge import nn as n


def test_unequal_gaussian_pools_match_closed_form_and_own_all_fields():
    a=n.from_vector_double([0,2],[2,1]).requires_grad_()
    b=n.from_vector([1,3,5],[3,1]).requires_grad_()
    out=n.GaussianFrechet(a,b)
    assert out.fid.item()==pytest.approx(4+(math.sqrt(2)-2)**2,abs=1e-13)
    assert out.fid.item()==pytest.approx(n.GaussianFrechet(b,a).fid.item(),abs=1e-13)
    fields=[out.fid,out.raw_fid,out.mean_term,out.trace_a,out.trace_b,out.cross_trace,out.cross_singular_values]
    del out,a,b
    gc.collect()
    assert all(not x.requires_grad() and x.dtype()==n.DType.Float64 for x in fields)
    assert list(fields[-1].shape())==[2]
    assert fields[2].item()==pytest.approx(4,abs=1e-14)


def test_probability_controls_and_ordered_unequal_splits_use_population_spread():
    p=n.from_vector_double([1,0,1,0,0,1,0,1,1,0],[5,2]).requires_grad_()
    assert n.InceptionScore(n.full([5,4],.25)).item()==pytest.approx(1,abs=1e-14)
    assert n.InceptionScore(n.from_vector_double([1,0,0,1],[2,2])).item()==pytest.approx(2,abs=1e-14)
    out=n.InceptionScoreSplits(p,3)
    score,mean,spread=out.scores,out.mean,out.population_std
    del out,p
    gc.collect()
    assert list(score.values())==pytest.approx([1,2,2],abs=1e-14)
    assert mean.item()==pytest.approx(5/3,abs=1e-14)
    assert spread.item()==pytest.approx(math.sqrt(2/9),abs=1e-14)
    assert not score.requires_grad()


def test_invalid_feature_geometry_probability_rows_and_split_counts_reject():
    with pytest.raises(RuntimeError):n.GaussianFrechet(n.zeros([1,2]),n.zeros([2,2]))
    with pytest.raises(RuntimeError):n.GaussianFrechet(n.zeros([2,2]),n.zeros([2,3]))
    with pytest.raises(RuntimeError):n.InceptionScore(n.from_vector([.5,.8],[1,2]))
    with pytest.raises(RuntimeError):n.InceptionScore(n.from_vector([-.1,1.1],[1,2]))
    with pytest.raises(RuntimeError):n.InceptionScoreSplits(n.full([2,3],1/3),3)
