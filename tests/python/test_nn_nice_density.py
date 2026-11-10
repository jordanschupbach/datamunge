"""Native NICE density graphs, inverse maps and explicit banana inputs."""
import gc
import math
import pytest
from pydatamunge import nn as n


def test_explicit_banana_coordinates_and_live_analytic_density():
    base=n.from_vector_double([0,0,2,1,-1,3],[3,2]).requires_grad_()
    points=n.BananaPoints(base)
    assert list(points.values())==[0,-.5,2,2.5,-1,3]
    assert points.dtype()==n.DType.Float32 and not points.requires_grad()
    logp=n.BananaLogProbability(base)
    assert logp.dtype()==n.DType.Float64
    assert abs(list(logp.values())[0]+math.log(2*math.pi)+.125)<1e-14
    assert abs(list(n.grad(n.sum(logp),[base])[0].values())[1]+.5)<1e-14
    assert list(n.BananaPoints(base,-.5,n.DType.Float64).values())==[0,.5,2,-.5,-1,3]


def test_global_density_and_inverse_own_graphs_after_parent_and_store_collection():
    p=n.ParameterStore(9301,n.DType.Float64)
    x=n.from_vector_double([.4,-.7,1.2,.3,-1,2],[3,2]).requires_grad_()
    path=n.NiceDensity(p,x)
    assert p.count()==198 and p.size()==17
    before=list(p.flatten());rec=n.NiceInverse(p,path.latent)
    assert n.max(n.abs(n.sub(rec.value,x))).item()<1e-14
    assert n.add(rec.logdet,path.logdet).item()==0
    density=n.NormalFlowDensity(path.latent,path.logdet,True)
    stage,loss=path.coupling2,density.loss
    del path,density,rec;gc.collect()
    graph=n.GradOptions();graph.retain_graph=True;graph.create_graph=True
    g=n.grad(loss,[x],graph)[0]
    assert list(g.shape())==[3,2]
    assert n.grad(n.sum(g),[x],graph)[0].defined()
    assert list(p.flatten())==before
    del p;gc.collect()
    assert n.grad(n.sum(stage),[x])[0].defined()


def test_both_coupling_orders_and_gaussian_precision_option():
    p=n.ParameterStore(9301)
    x=n.from_vector([.2,.7,-1,2],[2,2])
    for block in range(4):
        y=n.NiceCoupling(p,x,block);c=block%2
        assert n.equal_bits(n.slice_axis(y,1,c,c+1),n.slice_axis(x,1,c,c+1))
        rec=n.NiceCoupling(p,y,block,True)
        assert n.max(n.abs(n.sub(rec,x))).item()<3e-7
    z=n.zeros([1,2],n.DType.Float64)
    full=n.StandardNormalLogProbability(z).item()
    rounded=n.StandardNormalLogProbability(z,True).item()
    assert abs(full+math.log(2*math.pi))<1e-14 and rounded!=full
    assert abs(rounded+1.8378770351409912)<1e-15


def test_invalid_shapes_values_and_scales_reject_without_creating_initial_leaves():
    p=n.ParameterStore(9301)
    for bad in [n.zeros([2]),n.zeros([1,3]),n.zeros([0,2]),n.full([1,2],float('nan'))]:
        with pytest.raises(RuntimeError):n.NiceDensity(p,bad)
        with pytest.raises(RuntimeError):n.NiceInverse(p,bad)
        assert p.size()==0
    with pytest.raises(RuntimeError):n.NiceCoupling(p,n.zeros([1,2]),4)
    with pytest.raises(RuntimeError):n.BananaPoints(n.zeros([1,2]),float('inf'))
    with pytest.raises(RuntimeError):n.NormalFlowDensity(n.zeros([2,2]),n.zeros([1]))
    n.NiceDensity(p,n.zeros([1,2]))
    for scale in [1000,-1000,float('nan')]:
        p.set('log_scale',n.full([2],scale))
        with pytest.raises(RuntimeError):n.NiceDensity(p,n.zeros([1,2]))
        with pytest.raises(RuntimeError):n.NiceInverse(p,n.zeros([1,2]))
