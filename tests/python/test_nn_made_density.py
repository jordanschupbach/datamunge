"""Native degree masks, two MADE orders, owning records and stable joint mixtures."""
import gc
import math
import pytest
from pydatamunge import nn as n


def test_degree_inventory_and_strict_paths_survive_parent_collection():
    for reverse in [False,True]:
        o=n.MadeMasks(reverse,n.DType.Float64)
        fields=[o.input_degrees,o.hidden1_degrees,o.hidden2_degrees,o.input_mask,o.hidden_mask,o.output_mask]
        del o;gc.collect()
        assert list(fields[0].values())==list(range(64,0,-1) if reverse else range(1,65))
        assert fields[0].dtype()==n.DType.Int32
        assert sum(list(x.values()).count(0) for x in fields[3:])==1144
        paths=n.matmul(n.matmul(fields[3],fields[4]),fields[5])
        degree=list(fields[0].values());values=list(paths.values())
        assert all(values[i*64+j]==0 for i in range(64) for j in range(64) if degree[i]>=degree[j])
    a=n.from_vector_int64([0,1,2],[3]);b=n.from_vector_int64([0,2],[2])
    assert list(n.AutoregressiveDegreeMask(a,b,True).values())==[0,1,0,1,0,0]
    with pytest.raises(RuntimeError):n.AutoregressiveDegreeMask(n.ones([3]),b)


def test_made_census_order_causality_immutable_weights_and_record_lifetime():
    for reverse in [False,True]:
        p=n.ParameterStore(8803,n.DType.Float64)
        x=n.full([1,8,8,1],.25,n.DType.Float64).requires_grad_()
        o=n.MadeDensity(p,x,reverse)
        assert [p.count(),p.size()]==[2400,6]
        before=list(p.flatten());n.MadeDensity(p,x,not reverse)
        assert before==list(p.flatten())
        hidden1,hidden2,logits=o.hidden1,o.hidden2,o.logits
        del o,p;gc.collect()
        site=47 if reverse else 16
        g=n.grad(n.sum(n.slice_axis(n.reshape(logits,[64]),0,site,site+1)),[x])[0]
        assert all(v==0 for v in (list(g.values())[:site+1] if reverse else list(g.values())[site:]))
        assert list(hidden1.shape())==[1,16] and list(hidden2.shape())==[1,16]
    fresh=n.ParameterStore(1)
    with pytest.raises(RuntimeError):n.MadeDensity(fresh,n.zeros([1,4,16,1]))
    with pytest.raises(RuntimeError):n.MaskedDensityDense(fresh,'bad',n.zeros([2,3]),n.full([3,4],.5))
    assert fresh.size()==0


def test_masked_dense_stops_mask_and_keeps_owning_weight_input_graph():
    p=n.ParameterStore(1,n.DType.Float64)
    x=n.ones([2,3],n.DType.Float64).requires_grad_()
    mask=n.ones([3,4],n.DType.Float64).requires_grad_()
    y=n.MaskedDensityDense(p,'toy',x,mask)
    graph=n.GradOptions();graph.create_graph=True;graph.allow_unused=True
    g=n.grad(n.sum(n.square(y)),[x,mask],graph)
    assert not g[1].defined()
    del p,y;gc.collect()
    assert n.max(n.abs(n.grad(n.sum(g[0]),[x])[0])).item()>0


def test_equal_joint_mixture_normalizes_and_has_responsibility_gradients():
    p=n.from_vector_double([.4,.1,.1,.4,.1,.4,.4,.1],[4,2])
    nll=n.neg(n.log(p)).requires_grad_()
    o=n.EqualDensityMixture(nll)
    score,mean,weight=o.per_image_nll,o.mean_nll,o.responsibilities
    del o;gc.collect()
    assert score.dtype()==n.DType.Float64 and list(score.values())==pytest.approx([math.log(4)]*4,abs=1e-14)
    assert n.sum(n.exp(n.neg(score))).item()==pytest.approx(1,abs=1e-14)
    assert mean.item()<n.mean(nll).item()
    graph=n.GradOptions();graph.create_graph=True
    g=n.grad(n.sum(score),[nll],graph)[0]
    assert list(g.values())==pytest.approx(list(weight.values()),abs=1e-14)
    h=n.grad(n.sum(n.slice_axis(g,1,0,1)),[nll])[0]
    assert list(h.values())==pytest.approx([-.16,.16]*4,abs=1e-14)
    extreme=n.EqualDensityMixture(n.from_vector_double([1000,1002,-1002,-1000],[2,2]))
    assert extreme.mean_nll.item()==pytest.approx(-1-math.log1p(math.exp(-2))+math.log(2),abs=1e-13)
    with pytest.raises(RuntimeError):n.EqualDensityMixture(n.zeros([2]))
    with pytest.raises(RuntimeError):n.EqualDensityMixture(n.zeros([2,2],n.DType.Int64))
