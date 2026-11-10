"""Owning native WaveNet snapshots, copied gate vectors and explicit warmup."""
import gc
import pytest
from pydatamunge import nn as n


def setup_stack(classes=5,levels=2,cycles=1):
    c=n.WaveStackConfig();c.residual=2;c.skip=3;c.levels=levels;c.cycles=cycles;c.classes=classes
    p=n.ParameterStore(541)
    n.WaveStack(p,n.one_hot(n.zeros([1,2],n.DType.Int32),classes),c)
    return p,c


def test_snapshot_full_head_ring_wrap_and_store_collection():
    p,c=setup_stack(levels=3,cycles=2);cache=n.WaveStackCache(p,c);codes=[]
    for step in range(35):
        code=(step*3+1)%5;codes.append(code);head=cache.next(code)
        tail=codes[-n.WaveStackField(c):]
        full=n.WaveStack(p,n.one_hot(n.from_vector_int64(tail,[1,len(tail)]),5),c).logits
        last=n.reshape(n.select(full,1,len(tail)-1),[5])
        assert n.max(n.abs(n.sub(head,last))).item()<2e-6
        assert cache.consumed()==step+1 and not head.requires_grad()
    fork=cache.clone();del p,c;gc.collect()
    old=list(head.values())
    for code in [1,4,0,3,2]:assert n.equal_bits(cache.next(code),fork.next(code))
    assert list(head.values())==old


def test_clone_reset_and_invalid_step_preserve_independent_streams():
    p,c=setup_stack();cache=n.WaveStackCache(p,c)
    first=cache.next(1);cache.next(4);other=cache.clone();assert other.consumed()==2
    for code in [-1,5]:
        with pytest.raises(RuntimeError):cache.next(code)
    assert cache.consumed()==2
    assert n.equal_bits(cache.next(2),other.next(2))
    cache.reset();assert cache.consumed()==0 and other.consumed()==3
    assert n.equal_bits(first,cache.next(1))
    old=list(cache.next(2).values());p.set('output/b',n.full([5],100))
    other.reset();other.next(1);assert list(other.next(2).values())==old


def test_bad_snapshots_reject_without_creating_leaves():
    c=n.WaveStackConfig();p=n.ParameterStore(541)
    with pytest.raises(RuntimeError):n.WaveStackCache(p,c)
    assert p.size()==0
    with pytest.raises(RuntimeError):n.WaveStackCache(n.ParameterStore(541,n.DType.Float64),c)
    p,c=setup_stack();p.set('output/b',n.full([5],float('nan')))
    with pytest.raises(RuntimeError):n.WaveStackCache(p,c)
    p,c=setup_stack();c.classes=6
    with pytest.raises(RuntimeError):n.WaveStackCache(p,c)


def test_canonical_gate_vector_owns_tensor_graphs_and_warmup_is_explicit():
    p,c=setup_stack();encoded=n.one_hot(n.zeros([1,8],n.DType.Int32),5).requires_grad_()
    out=n.WaveStack(p,encoded,c);gates,logits=out.gates,out.logits
    del out,p,c;gc.collect()
    assert len(gates)==2
    gate=gates[0];del gates;gc.collect()
    assert list(gate.shape())==[1,8,2]
    graph=n.GradOptions();graph.retain_graph=True
    assert n.grad(n.sum(gate),[encoded],graph)[0].defined()
    loss=n.WaveNll(logits,n.zeros([1,8],n.DType.Int32),5,3)
    g=n.grad(loss,[logits])[0];assert all(v==0 for v in list(g.values())[:15])
    assert any(v!=0 for v in list(g.values())[15:])
