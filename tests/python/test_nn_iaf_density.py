"""Native sequential affine IAF density and parallel observation mapping."""
import gc
import pytest
from pydatamunge import nn as n


def test_sequential_density_parallel_inverse_and_flat_higher_graphs_survive_collection():
    p=n.ParameterStore(9703,n.DType.Float64)
    x=n.from_vector_double([.4,-.7,1.2,.3,-1,2],[3,2]).requires_grad_()
    path=n.IafDensity(p,x);assert p.count()==466 and p.size()==17
    assert list(path.logdet.shape())==[3] and list(path.coupling_scale2.shape())==[3,2]
    assert n.equal_bits(path.unscaled,path.coupling0)
    assert n.equal_bits(path.conditioner_input2,path.coupling2)
    assert list(p.names())[0]=="autoregressive3/hidden/W"
    before=list(p.flatten());rec=n.IafInverse(p,path.latent)
    assert n.max(n.abs(n.sub(rec.value,x))).item()<1e-14
    assert n.max(n.abs(n.add(rec.logdet,path.logdet))).item()<1e-14
    density=n.NormalFlowDensity(path.latent,path.logdet,True)
    cond=n.MafConditioner(p,x,1)
    input2,scale,shift,mean,loss=path.conditioner_input2,path.coupling_scale3,path.coupling_shift1,cond.mean,density.loss
    del path,cond,rec,density;gc.collect()
    graph=n.GradOptions();graph.retain_graph=True;graph.create_graph=True
    first=n.grad(loss,[x],graph)[0];assert n.grad(n.sum(first),[x],graph)[0].defined()
    assert list(p.flatten())==before
    del p;gc.collect()
    for field in [input2,scale,shift,mean]:assert n.grad(n.sum(field),[x],graph)[0].defined()


def test_host_adam_leaves_all_structurally_masked_coordinates_unchanged():
    p=n.ParameterStore(9703);x=n.from_vector([.4,-.7,1.2,.3,-1,2],[3,2]);n.IafDensity(p,x)
    config=n.OptimizerConfig();config.kind=n.OptimizerKind.Adam;optimizer=n.Optimizer(p,.003,config)
    names=list(p.names());before={name:list(p.at(name).values()) for name in names}
    path=n.IafDensity(p,x);loss=n.NormalFlowDensity(path.latent,path.logdet).loss
    gradients=n.grad(loss,p.parameters());optimizer.step(gradients);checked=0
    for block in range(4):
        masks=n.MafMasks(block)
        for suffix,mask in [('hidden/W',masks.input_mask),('heads/W',masks.output_mask)]:
            name=f'autoregressive{block}/{suffix}';g=list(gradients[names.index(name)].values());after=list(p.at(name).values())
            for j,allowed in enumerate(mask.values()):
                if allowed==0:
                    assert g[j]==0 and after[j]==before[name][j];checked+=1
    assert checked==160 and optimizer.iterations()==1


def test_invalid_options_inputs_and_global_scales_reject():
    p=n.ParameterStore(9703)
    for block in [-1,4]:
        with pytest.raises(RuntimeError):n.MafMasks(block)
        with pytest.raises(RuntimeError):n.MafConditioner(p,n.zeros([2,2]),block)
    with pytest.raises(RuntimeError):n.MafMasks(0,n.DType.Int32)
    for bad in [n.zeros([2]),n.zeros([1,3]),n.zeros([0,2]),n.full([1,2],float('nan'))]:
        with pytest.raises(RuntimeError):n.IafDensity(p,bad)
        with pytest.raises(RuntimeError):n.IafInverse(p,bad)
        assert p.size()==0
    n.IafDensity(p,n.zeros([1,2]));p.set('log_scale',n.full([2],1000))
    with pytest.raises(RuntimeError):n.IafDensity(p,n.zeros([1,2]))
    with pytest.raises(RuntimeError):n.IafInverse(p,n.zeros([1,2]))
