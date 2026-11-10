"""Unpaired native domains, real cycle paths and jointly captured generator writes."""
import gc
import pytest
from pydatamunge import nn as n


def inputs():
    t=n.DType.Float64
    p=[n.ParameterStore(seed,t) for seed in [6203,6211,6217,6221]]
    return p,n.full([2,28,28,1],.2,t),n.full([3,28,28,1],.8,t)


def test_native_rotation_inverse_orientation_and_input_derivative():
    x=n.from_vector_double([0,.25,.5,.75],[1,2,2,1]).requires_grad_()
    y=n.RotateInvertGrayscale(x)
    assert list(y.values())==[.5,1,.25,.75]
    assert n.equal_bits(n.RotateInvertGrayscale(y,True),x)
    assert n.equal_bits(n.grad(n.sum(y),[x])[0],n.full(x.shape(),-1,x.dtype()))
    for bad in [n.ones([2,3]),n.ones([1,2,2,3]),n.full([1,2,2,1],float('nan'))]:
        with pytest.raises(RuntimeError):n.RotateInvertGrayscale(bad)


def test_native_disjoint_per_class_domains_preserve_metadata_and_original_b():
    x=n.full([20,28,28,1],.3,n.DType.Float64).requires_grad_()
    labels=n.from_vector_int64([c for c in range(10) for _ in range(2)],[20]);ids=n.arange(1000,1020,1,n.DType.Int64)
    out=n.SplitCycleDomains(x,labels,ids)
    assert list(out.ids_a.values())==list(range(1000,1020,2))
    assert list(out.ids_b.values())==list(range(1001,1020,2))
    assert n.equal_bits(out.b,n.RotateInvertGrayscale(out.original_b))
    assert n.equal_bits(n.grad(n.add(n.sum(out.a),n.sum(out.original_b)),[x])[0],n.ones(x.shape(),x.dtype()))
    with pytest.raises(RuntimeError):n.SplitCycleDomains(x,n.zeros([20],n.DType.Int64),ids)
    with pytest.raises(RuntimeError):n.SplitCycleDomains(x,labels,n.zeros([20],n.DType.Int64))
    with pytest.raises(RuntimeError):n.SplitCycleDomains(n.full(x.shape(),1.01,x.dtype()),labels,ids)


def test_unequal_unpaired_rows_census_source_only_and_immutable_fields():
    p,a,b=inputs();out=n.CycleGan(*p,a,b)
    assert [q.count() for q in p]==[1390,1390,661,661]
    assert sum(q.size() for q in p)==32
    assert list(out.fake_a.shape())==list(b.shape()) and list(out.fake_b.shape())==list(a.shape())
    assert list(out.cycle_a.shape())==list(a.shape()) and list(out.cycle_b.shape())==list(b.shape())
    before=[list(q.flatten()) for q in p]
    assert n.equal_bits(n.Pix2PixGenerate(p[0],a).image,out.fake_b)
    assert n.equal_bits(n.CycleGan(*p,a,n.zeros(b.shape(),b.dtype())).fake_b,out.fake_b)
    assert n.equal_bits(n.Pix2PixGenerate(p[1],a).image,out.identity_a)
    assert [list(q.flatten()) for q in p]==before


def test_full_g_half_critics_and_explicit_cycle_identity_weights():
    p,a,b=inputs();n.CycleGan(*p,a,b)
    for q in p[2:]:
        for name in q.names():q.set(name,n.zeros(q.at(name).shape(),q.dtype()))
    out=n.CycleGan(*p,a,b)
    assert out.d_a_loss.item()==.5 and out.d_b_loss.item()==.5
    assert out.adversarial_a.item()==1 and out.adversarial_b.item()==1
    assert out.g_loss.item()==pytest.approx(2+10*(out.cycle_loss_a.item()+out.cycle_loss_b.item())+5*(out.identity_loss_a.item()+out.identity_loss_b.item()))
    assert n.CycleGan(*p,a,b,0,0).g_loss.item()==2


def test_both_g_captures_precede_either_write_and_bad_recompute_differs():
    p,a,b=inputs();n.CycleGan(*p,a,b);cfg=n.OptimizerConfig();cfg.kind=n.OptimizerKind.Adam;cfg.beta1=.5
    optim=[n.Optimizer(q,.0002,cfg) for q in p]
    for k in [2,3]:
        out=n.CycleGan(*p,a,b);loss=out.d_a_loss if k==2 else out.d_b_loss;before=[list(q.flatten()) for q in p]
        optim[k].step(n.grad(loss,p[k].parameters()))
        assert all(list(p[i].flatten())==before[i] for i in range(4) if i!=k)
    joint=n.CycleGan(*p,a,b);keep=n.GradOptions();keep.retain_graph=True
    first=n.grad(joint.g_loss,p[0].parameters(),keep);second=n.grad(joint.g_loss,p[1].parameters(),keep)
    before_ba=list(p[1].flatten());critics=[list(q.flatten()) for q in p[2:]];optim[0].step(first)
    assert list(p[1].flatten())==before_ba
    wrong=n.grad(n.CycleGan(*p,a,b).g_loss,p[1].parameters())
    assert max(n.max(n.abs(n.sub(x,y))).item() for x,y in zip(wrong,second))>1e-10
    after_ab=list(p[0].flatten());optim[1].step(second)
    assert list(p[0].flatten())==after_ab and [list(q.flatten()) for q in p[2:]]==critics
    assert [o.iterations() for o in optim]==[1]*4


def test_owned_cycle_fields_have_higher_paths_and_both_generator_partials():
    p,a,b=inputs();a.requires_grad_();out=n.CycleGan(*p,a,b);keep=n.GradOptions();keep.retain_graph=True
    for q in p[:2]:assert sum(n.sum(n.abs(g)).item() for g in n.grad(out.cycle_loss_a,q.parameters(),keep))>0
    cycle=out.cycle_a;loss=out.g_loss;del out,p,b;gc.collect();graph=n.GradOptions();graph.create_graph=True
    first=n.grad(loss,[a],graph)[0];second=n.grad(n.sum(n.square(first)),[a])[0]
    assert list(cycle.shape())==list(a.shape()) and n.max(n.abs(second)).item()>0


def test_invalid_aliases_inputs_weights_fail_before_leaves():
    p,a,b=inputs()
    for i in range(4):
        for j in range(i):
            alias=list(p);alias[i]=alias[j]
            with pytest.raises(RuntimeError):n.CycleGan(*alias,a,b)
    for weights in [(-1,5),(10,float('inf')),(float('nan'),5)]:
        with pytest.raises(RuntimeError):n.CycleGan(*p,a,b,*weights)
    with pytest.raises(RuntimeError):n.CycleGan(*p,a,n.ones([3,27,28,1],b.dtype()))
    assert sum(q.size() for q in p)==0
