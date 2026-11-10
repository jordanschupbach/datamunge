"""Native elementary inverses, source derivative conventions and owning host fields."""
import gc
import math
import pytest
from pydatamunge import nn as n


def keep():
    g=n.GradOptions();g.retain_graph=True;g.create_graph=True;return g


@pytest.mark.parametrize('radial',[False,True])
def test_density_census_inverse_owning_fields_and_higher_graphs(radial):
    p=n.ParameterStore(9903,n.DType.Float64);x=n.from_vector_double([.4,-.7,1.2,.3,-1,2],[3,2]).requires_grad_()
    path=n.PlanarRadialDensity(p,x,radial);assert p.count()==(18 if radial else 22) and p.size()==(13 if radial else 21)
    assert list(path.flow_factor1.shape())==[3,1] and list(path.flow_coordinate2.shape())==[3,1]
    inv=n.PlanarRadialInverse(p,path.latent,radial)
    assert n.max(n.abs(n.sub(inv.value,x))).item()<2e-13
    assert n.max(n.abs(n.add(inv.logdet,path.logdet))).item()<2e-13
    factor=path.flow_factor1;coordinate=path.flow_coordinate2;stage=path.flow3;inverse=inv.value
    del path,inv,p;gc.collect()
    for field in [factor,coordinate,stage,inverse]:
        first=n.grad(n.sum(field),[x],keep())[0]
        assert n.grad(n.sum(n.square(first)),[x],keep())[0].defined()


@pytest.mark.parametrize('radial',[False,True])
def test_host_owned_adam_and_inverse_parameter_gradients(radial):
    p=n.ParameterStore(9903);x=n.from_vector([.4,-.7,1.2,.3,-1,2],[3,2]);n.PlanarRadialDensity(p,x,radial)
    config=n.OptimizerConfig();config.kind=n.OptimizerKind.Adam;optimizer=n.Optimizer(p,.003,config)
    before=list(p.flatten());path=n.PlanarRadialDensity(p,x,radial)
    gradients=n.grad(n.NormalFlowDensity(path.latent,path.logdet).loss,p.parameters());optimizer.step(gradients)
    assert optimizer.iterations()==1 and list(p.flatten())!=before
    inverse=n.PlanarRadialInverse(p,x,radial)
    objective=n.add(n.mean(n.sum(n.square(inverse.value),[1]),[0]),n.mul(n.mean(inverse.logdet,[0]),.3))
    partials=n.grad(objective,p.parameters());assert len(partials)==p.size() and all(g.defined() for g in partials)


def test_pure_planar_actual_jacobian_and_stopped_inverse_higher_rule():
    scalar=lambda v:n.full([],v,n.DType.Float64)
    x=n.from_vector_double([.3,-.2],[1,2]);bias=scalar(.1).requires_grad_();raw=scalar(-1)
    inv=n.PlanarFlow(x,scalar(0),scalar(0),raw,scalar(0),bias,True)
    first=n.grad(n.sum(n.slice_axis(inv.value,1,0,1)),[bias],keep())[0]
    second=n.grad(first,[bias],keep())[0];p=math.log1p(math.exp(-1))-.9;t=inv.coordinate.item();a=math.tanh(t);sech=1-a*a
    low=.4-(abs(p)+1);high=.4+(abs(p)+1)
    for _ in range(32):
        mid=.5*(low+high)
        if (mid+p*math.tanh(mid))-.4<0:low=mid
        else:high=mid
    root=.5*(low+high);den=1+p*(1-math.tanh(root)**2)
    assert abs(first.item()+p*sech/den)<2e-13
    assert abs(second.item()-2*p*a*sech/(den*den))<2e-12
    points=n.from_vector_double([.4,-.7,1.2,.3],[2,2]).requires_grad_()
    out=n.PlanarFlow(points,scalar(.6),scalar(.3),scalar(-6),scalar(.7),scalar(.1))
    j0=list(n.grad(n.sum(n.slice_axis(out.value,1,0,1)),[points],keep())[0].values())
    j1=list(n.grad(n.sum(n.slice_axis(out.value,1,1,2)),[points],keep())[0].values())
    for r,ld in enumerate(out.logdet.values()):assert abs(math.log(j0[2*r]*j1[2*r+1]-j0[2*r+1]*j1[2*r])-ld)<2e-12


def test_pure_radial_center_contraction_and_stable_inverse_branches():
    scalar=lambda v:n.full([],v,n.DType.Float64)
    center=n.from_vector_double([.3,-.2],[2]);x=n.from_vector_double([.3,-.2,.31,-.19,10,-10],[3,2]).requires_grad_()
    alpha=scalar(.4);beta=scalar(math.log(math.expm1(.12)))
    out=n.RadialFlow(x,center,alpha,beta);rec=n.RadialFlow(out.value,center,alpha,beta,True)
    assert n.max(n.abs(n.sub(rec.value,x))).item()<2e-13
    row=list(n.grad(n.sum(n.slice_axis(out.value,1,0,1)),[x],keep())[0].values())
    assert abs(row[0]-list(out.factor.values())[0])<1e-15 and row[1]==0
    radius=list(n.grad(n.sum(out.coordinate),[x],keep())[0].values());assert radius[0]==radius[1]==0


def test_invalid_input_parameter_geometry_and_unrepresentable_math():
    p=n.ParameterStore(9903)
    for bad in [n.zeros([2]),n.zeros([1,3]),n.zeros([0,2]),n.full([1,2],float('nan'))]:
        with pytest.raises(RuntimeError):n.PlanarRadialDensity(p,bad)
    with pytest.raises(RuntimeError):n.ElementaryFlow(p,n.zeros([1,2]),4)
    assert p.size()==0
    x=n.zeros([1,2]);zero=n.full([],0)
    with pytest.raises(RuntimeError):n.PlanarFlow(x,n.zeros([1]),zero,zero,zero,zero)
    with pytest.raises(RuntimeError):n.PlanarFlow(x,zero,n.full([],1000),zero,zero,zero)
    with pytest.raises(RuntimeError):n.RadialFlow(x,n.zeros([2]),zero,n.full([],-1000))
    n.PlanarRadialDensity(p,x);p.set('log_scale',n.full([2],1000))
    with pytest.raises(RuntimeError):n.PlanarRadialInverse(p,x)
