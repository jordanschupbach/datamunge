"""Native bounded affine flow inverses, live fields and per-point determinants."""
import gc
import math
import pytest
from pydatamunge import nn as n


def test_couplings_keep_raw_conditioning_coordinate_and_invert_their_own_logdets():
    p=n.ParameterStore(9403,n.DType.Float64)
    x=n.from_vector_double([.4,-.7,1.2,.3,-1,2],[3,2])
    for block in range(4):
        out=n.RealNvpCoupling(p,x,block);c=block%2
        assert n.equal_bits(n.slice_axis(out.value,1,c,c+1),n.slice_axis(x,1,c,c+1))
        assert list(out.scale.shape())==[3,1] and list(out.logdet.shape())==[3]
        assert n.max(n.abs(out.scale)).item()<=.8
        assert n.equal_bits(n.reshape(out.scale,[3]),out.logdet)
        rec=n.RealNvpCoupling(p,out.value,block,True)
        assert n.max(n.abs(n.sub(rec.value,x))).item()<1e-14
        assert n.max(n.abs(n.add(rec.logdet,out.logdet))).item()<1e-14
    assert p.count()==264 and p.size()==16


def test_full_affine_fields_retain_higher_graphs_after_parent_and_store_collection():
    p=n.ParameterStore(9403,n.DType.Float64)
    x=n.from_vector_double([.4,-.7,1.2,.3,-1,2],[3,2]).requires_grad_()
    path=n.RealNvpDensity(p,x)
    assert p.count()==266 and p.size()==17 and list(path.logdet.shape())==[3]
    assert n.sub(n.max(path.logdet),n.min(path.logdet)).item()>1e-5
    before=list(p.flatten());rec=n.RealNvpInverse(p,path.latent)
    assert n.max(n.abs(n.sub(rec.value,x))).item()<1e-14
    assert n.max(n.abs(n.add(rec.logdet,path.logdet))).item()<1e-14
    density=n.NormalFlowDensity(path.latent,path.logdet,True)
    stage,scale,shift,ld,loss=path.coupling2,path.coupling_scale1,path.coupling_shift3,path.coupling_logdet0,density.loss
    del path,density,rec;gc.collect()
    graph=n.GradOptions();graph.retain_graph=True;graph.create_graph=True
    first=n.grad(loss,[x],graph)[0]
    assert n.grad(n.sum(first),[x],graph)[0].defined()
    assert list(p.flatten())==before
    del p;gc.collect()
    for field in [stage,scale,shift,ld]:
        assert n.grad(n.sum(field),[x],graph)[0].defined()


def test_diagonal_scale_supports_three_coordinates_scalar_or_per_row_logdet_and_gradients():
    p=n.ParameterStore(1,n.DType.Float64)
    p.set('log_scale',n.from_vector_double([math.log(2),math.log(.5),.2],[3]))
    x=n.ones([2,3],n.DType.Float64).requires_grad_()
    incoming=n.full([2],.3,n.DType.Float64).requires_grad_()
    out=n.FlowDiagonalScale(p,x,incoming);before=list(p.flatten())
    assert list(out.logdet.shape())==[2] and abs(list(out.logdet.values())[0]-.5)<1e-14
    assert abs(list(out.value.values())[0]-2)<1e-14
    rec=n.FlowDiagonalScale(p,out.value,out.logdet,True)
    assert n.max(n.abs(n.sub(rec.value,x))).item()<1e-14
    assert n.max(n.abs(n.sub(rec.logdet,incoming))).item()<1e-14
    scalar=n.FlowDiagonalScale(p,x,n.full([],.3,n.DType.Float64))
    assert len(list(scalar.logdet.shape()))==0
    derivative=n.grad(n.sum(out.logdet),[incoming])[0]
    assert list(derivative.values())==[1,1] and list(p.flatten())==before


def test_invalid_inputs_determinants_and_scales_reject_without_initial_leaves():
    p=n.ParameterStore(9403)
    for bad in [n.zeros([2]),n.zeros([1,3]),n.zeros([0,2]),n.full([1,2],float('nan'))]:
        with pytest.raises(RuntimeError):n.RealNvpDensity(p,bad)
        with pytest.raises(RuntimeError):n.RealNvpInverse(p,bad)
        assert p.size()==0
    with pytest.raises(RuntimeError):n.RealNvpCoupling(p,n.zeros([1,2]),4)
    for ld in [n.zeros([1]),n.zeros([2,1]),n.full([],float('nan'))]:
        with pytest.raises(RuntimeError):n.FlowDiagonalScale(p,n.zeros([2,2]),ld)
        assert p.size()==0
    n.RealNvpDensity(p,n.zeros([1,2]))
    for scale in [1000,-1000,float('nan')]:
        p.set('log_scale',n.full([2],scale))
        with pytest.raises(RuntimeError):n.RealNvpDensity(p,n.zeros([1,2]))
        with pytest.raises(RuntimeError):n.RealNvpInverse(p,n.zeros([1,2]))
