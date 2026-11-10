"""Native three-domain selection, target conditioning, shared reconstruction and explicit writes."""
import gc
import pytest
from pydatamunge import nn as n


def inputs():
    t=n.DType.Float64
    return n.ParameterStore(6301,t),n.ParameterStore(6311,t),n.full([2,28,28,1],.3,t),n.zeros([2],n.DType.Int64),n.ones([2],n.DType.Int64)


def test_native_domain_orientation_signed_pixels_and_input_paths():
    x=n.from_vector_double([0,.25,.5,.75],[1,2,2,1]).requires_grad_()
    rotate=n.StarDomainImages(x,n.full([1],2,n.DType.Int64))
    assert list(rotate.values())==[.5,0,.75,.25]
    assert n.equal_bits(n.grad(n.sum(rotate),[x])[0],n.ones(x.shape(),x.dtype()))
    inverted=n.StarDomainImages(n.full(x.shape(),2,x.dtype()),n.ones([1],n.DType.Int64));assert n.max(inverted).item()==-1
    with pytest.raises(RuntimeError):n.StarDomainImages(x,n.full([1],3,n.DType.Int64))


def test_disjoint_observed_occurrences_keep_unused_ids_and_original_gathers():
    x=n.full([40,28,28,1],.25,n.DType.Float64).requires_grad_();labels=n.from_vector_int64([c for c in range(10) for _ in range(4)],[40]);ids=n.arange(100,140,1,n.DType.Int64)
    out=n.SplitStarDomains(x,labels,ids,1)
    assert list(out.domain.values())==[0,1,2]*10
    assert list(out.unused.values())==list(range(103,140,4))
    assert out.source.dtype()==out.unused.dtype()==n.DType.Int64
    assert n.equal_bits(out.image,n.StarDomainImages(out.original,out.domain))
    dx=n.grad(n.sum(out.original),[x])[0];assert n.sum(dx).item()==30*784 and n.sum(n.take_rows(dx,[3,7,11])).item()==0
    for args in [(x,labels,n.zeros([40],n.DType.Int64),1),(x,n.zeros([40],n.DType.Int64),ids,1),(x,labels,ids,2)]:
        with pytest.raises(RuntimeError):n.SplitStarDomains(*args)


def test_local_uniform_categorical_rejection_words_and_record_ownership():
    a=n.UniformCategorical(5,0,1073741825);raw=a.raw;labels=a.labels;del a;gc.collect()
    assert list(raw.values())==[2357136044,2546248239,3071714933,2588848963,2340255427]
    assert list(labels.values())==[int(v)%1073741825 for v in raw.values()]
    b=n.UniformCategorical(5,0,1073741825);assert n.equal_bits(raw,b.raw)
    assert not raw.requires_grad() and n.max(n.UniformCategorical(8,6338,1).labels).item()==0
    with pytest.raises(RuntimeError):n.UniformCategorical(0,1)


def test_census_actual_skips_target_only_g_and_image_only_d_are_pure():
    g,d,x,o,t=inputs();out=n.StarGan(g,d,x,o,t)
    assert (g.count(),g.size(),d.count(),d.size())==(1609,10,688,8)
    before=(list(g.flatten()),list(d.flatten()));gen=n.StarGanGenerate(g,x,t);critic=n.StarGanDiscriminate(d,x)
    assert list(gen.input.shape())==[2,28,28,4] and list(gen.joined14.shape())==[2,14,14,8] and list(gen.joined28.shape())==[2,28,28,6]
    assert n.equal_bits(gen.image,out.fake) and n.equal_bits(critic.source,out.real_source)
    assert n.equal_bits(n.StarGan(g,d,x,t,t).fake,out.fake)
    assert n.max(n.abs(n.sub(n.StarGanGenerate(g,x,o).image,gen.image))).item()>1e-5
    assert (list(g.flatten()),list(d.flatten()))==before


def test_stable_domain_ce_live_higher_paths_and_exact_loss_scales():
    logits=n.from_vector_double([1000,-1000,0],[1,3]).requires_grad_();ce=n.StarDomainLoss(logits,n.ones([1],n.DType.Int64))
    assert ce.item()==2000 and list(n.grad(ce,[logits])[0].values())==[1,-1,0]
    logits=n.zeros([1,3],n.DType.Float64).requires_grad_();graph=n.GradOptions();graph.create_graph=True;first=n.grad(n.StarDomainLoss(logits,n.zeros([1],n.DType.Int64)),[logits],graph)[0]
    assert n.max(n.abs(n.grad(n.sum(n.square(first)),[logits])[0])).item()>0
    g,d,x,o,t=inputs();n.StarGan(g,d,x,o,t)
    for name in d.names():d.set(name,n.zeros(d.at(name).shape(),d.dtype()))
    out=n.StarGan(g,d,x,o,t)
    assert out.source_loss_g.item()==1 and out.source_loss_d.item()==.5
    assert out.discriminator_loss.item()==pytest.approx(.5+1.0986122886681098)
    assert out.generator_loss.item()==pytest.approx(1+1.0986122886681098+10*out.reconstruction_loss.item())


def test_reconstruction_partial_includes_both_shared_generator_uses():
    g,d,x,o,t=inputs();out=n.StarGan(g,d,x,o,t);full=n.grad(out.reconstruction_loss,g.parameters())
    stopped=n.StarGanGenerate(g,out.fake.detach(),o).image;wrong=n.grad(n.mean(n.abs(n.sub(stopped,x))),g.parameters())
    assert max(n.max(n.abs(n.sub(a,b))).item() for a,b in zip(full,wrong))>1e-10


def test_explicit_d_then_fresh_g_changes_partial_and_preserves_owners():
    g,d,x,o,t=inputs();out=n.StarGan(g,d,x,o,t);keep=n.GradOptions();keep.retain_graph=True;stale=n.grad(out.generator_loss,g.parameters(),keep);dd=n.grad(out.discriminator_loss,d.parameters())
    config=n.OptimizerConfig();config.kind=n.OptimizerKind.Adam;config.beta1=.5;go,do=n.Optimizer(g,.0002,config),n.Optimizer(d,.0002,config);before=list(g.flatten());do.step(dd);assert list(g.flatten())==before
    after=list(d.flatten());gg=n.grad(n.StarGan(g,d,x,o,t).generator_loss,g.parameters());assert max(n.max(n.abs(n.sub(a,b))).item() for a,b in zip(stale,gg))>1e-10
    go.step(gg);assert list(d.flatten())==after and go.iterations()==do.iterations()==1


def test_owned_fields_keep_higher_paths_and_bad_domains_fail_before_leaves():
    g,d,x,o,t=inputs();x.requires_grad_();out=n.StarGan(g,d,x,o,t);image=out.reconstruction;loss=out.generator_loss;prob=out.fake_domain_probability;del out,g,d,o,t;gc.collect();graph=n.GradOptions();graph.create_graph=True
    first=n.grad(loss,[x],graph)[0];assert n.max(n.abs(n.grad(n.sum(n.square(first)),[x])[0])).item()>0
    assert list(image.shape())==list(x.shape()) and list(prob.shape())==[2,3]
    g,d,x,o,t=inputs()
    for args in [(g,g,x,o,t),(g,d,x,n.full([2],-1,n.DType.Int64),t),(g,d,x,o,n.full([2],3,n.DType.Int64)),(g,d,x,o,n.ones([2])),(g,d,x,o,n.ones([1],n.DType.Int64))]:
        with pytest.raises(RuntimeError):n.StarGan(*args)
    assert g.size()==d.size()==0
