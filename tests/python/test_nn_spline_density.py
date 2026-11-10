"""Native rational-quadratic bins, analytic inverse and owning compact flow fields."""
import gc
import pytest
from pydatamunge import nn as n


def test_generic_bins_bounds_and_owning_head_metadata_higher_paths():
    x=n.from_vector_double([-4,-.7,.4,4],[4,1])
    heads=n.full([4,5],.15,n.DType.Float64).requires_grad_()
    out=n.RationalQuadraticSpline(x,heads,False,2.5,.03,.002)
    assert list(out.widths.shape())==[4,2] and list(out.derivatives.shape())==[4,3]
    rec=n.RationalQuadraticSpline(out.value,heads,True,2.5,.03,.002)
    assert n.max(n.abs(n.sub(rec.value,x))).item()<1e-14
    assert list(out.value.values())[0]==-4 and list(out.value.values())[-1]==4
    assert n.min(out.widths).item()>0 and n.min(out.derivatives).item()>0
    width=n.slice_axis(out.widths,1,0,1);del out,rec;gc.collect()
    graph=n.GradOptions();graph.retain_graph=True;graph.create_graph=True
    first=n.grad(n.sum(width),[heads],graph)[0]
    assert n.grad(n.sum(n.square(first)),[heads],graph)[0].defined()


def test_source_rounded_offset_and_actual_spline_input_derivatives():
    x=n.from_vector_double([-5,-.7,.3,5],[4,1]).requires_grad_()
    heads=n.from_vector_double([0,0,0,0,.2,-.3,.1,.4,-.2,.3,.6]*4,[4,11]).requires_grad_()
    out=n.RationalQuadraticSpline(x,heads);graph=n.GradOptions();graph.retain_graph=True;graph.create_graph=True
    j=n.grad(n.sum(out.value),[x],graph)[0]
    assert n.min(j).item()>0
    assert n.max(n.abs(n.sub(n.reshape(n.log(j),[4]),out.logdet))).item()<1e-14
    assert n.grad(n.sum(j),[heads],graph)[0].defined()
    zero=n.zeros([2,11],n.DType.Float64);points=n.full([2,1],.25,n.DType.Float64)
    precise=n.RationalQuadraticSpline(points,zero);source=n.RationalQuadraticSpline(points,zero,False,4,.01,.001,True)
    assert abs(list(precise.derivatives.values())[1]-list(source.derivatives.values())[1])>1e-10


def test_compact_spline_field_inverse_and_higher_graphs_survive_collection():
    p=n.ParameterStore(9803,n.DType.Float64)
    x=n.from_vector_double([.4,-.7,1.2,.3,-1,2],[3,2]).requires_grad_()
    path=n.SplineDensity(p,x);assert p.count()==878 and p.size()==17
    assert list(path.coupling_widths1.shape())==[3,4] and list(path.coupling_derivatives2.shape())==[3,5]
    rec=n.SplineInverse(p,path.latent);before=list(p.flatten())
    assert n.max(n.abs(n.sub(rec.value,x))).item()<1e-13
    assert n.max(n.abs(n.add(rec.logdet,path.logdet))).item()<1e-13
    loss=n.NormalFlowDensity(path.latent,path.logdet).loss
    width=n.slice_axis(path.coupling_widths3,1,0,1);height=n.slice_axis(path.coupling_heights1,1,0,1)
    derivative=n.slice_axis(path.coupling_derivatives2,1,1,2);stage=path.coupling0
    del path,rec;gc.collect();graph=n.GradOptions();graph.retain_graph=True;graph.create_graph=True
    first=n.grad(loss,[x],graph)[0];assert n.grad(n.sum(first),[x],graph)[0].defined()
    assert list(p.flatten())==before;del p;gc.collect()
    for field in [width,height,derivative,stage]:assert n.grad(n.sum(field),[x],graph)[0].defined()


def test_explicit_host_adam_updates_source_spline_leaves():
    p=n.ParameterStore(9803);x=n.from_vector([.4,-.7,1.2,.3,-1,2],[3,2]);n.SplineDensity(p,x)
    config=n.OptimizerConfig();config.kind=n.OptimizerKind.Adam;optimizer=n.Optimizer(p,.003,config)
    before=list(p.flatten());path=n.SplineDensity(p,x)
    gradients=n.grad(n.NormalFlowDensity(path.latent,path.logdet).loss,p.parameters());optimizer.step(gradients)
    assert optimizer.iterations()==1 and list(p.flatten())!=before and p.count()==878
    assert n.SplineInverse(p,n.zeros([2,2])).value.defined()


def test_invalid_spline_geometry_options_and_global_scale_reject():
    x=n.zeros([2,1]);heads=n.zeros([2,11])
    for bad in [n.zeros([2]),n.zeros([2,2]),n.zeros([0,1]),n.full([2,1],float('nan'))]:
        with pytest.raises(RuntimeError):n.RationalQuadraticSpline(bad,heads)
    with pytest.raises(RuntimeError):n.RationalQuadraticSpline(x,n.zeros([2,4]))
    with pytest.raises(RuntimeError):n.RationalQuadraticSpline(x,heads,False,0)
    with pytest.raises(RuntimeError):n.RationalQuadraticSpline(x,heads,False,4,.25)
    with pytest.raises(RuntimeError):n.RationalQuadraticSpline(x,heads,False,4,.01,1)
    p=n.ParameterStore(9803)
    with pytest.raises(RuntimeError):n.SplineDensity(p,n.zeros([1,3]))
    with pytest.raises(RuntimeError):n.SplineCoupling(p,n.zeros([1,2]),4)
    assert p.size()==0
    n.SplineDensity(p,n.zeros([1,2]));p.set('log_scale',n.full([2],1000))
    with pytest.raises(RuntimeError):n.SplineDensity(p,n.zeros([1,2]))
    with pytest.raises(RuntimeError):n.SplineInverse(p,n.zeros([1,2]))
