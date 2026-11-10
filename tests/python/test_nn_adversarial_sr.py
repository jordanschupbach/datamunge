"""Native frozen-feature SR games, live relativistic means, explicit BN/Adam protocol."""
import gc
import pytest
from pydatamunge import nn as n

NAMES=['residual0/bn1','residual0/bn2','residual1/bn1','residual1/bn2','body_bn']

def inputs():
    t=n.DType.Float64
    return [n.ParameterStore(s,t) for s in [7001,7013,0]],n.full([2,16,16,3],.2,t),n.full([2,32,32,3],.4,t),n.full([2,32,32,3],.2,t),n.full([3,3,3,64],.001,t),n.zeros([64],t)


def game(p,lr,hr,bic,w,b,enhanced=False,phase=True):return n.AdversarialSR(*p,lr,hr,bic,w,b,enhanced,phase)


def test_unequal_relativistic_pools_have_live_cross_means_and_common_shift():
    r=n.from_vector_double([-2,0,3],[3,1]).requires_grad_();f=n.from_vector_double([-1,2],[2,1]).requires_grad_();o=n.RelativisticAverageGan(r,f);graph=n.GradOptions();graph.create_graph=True;g=n.grad(o.generator,[r,f],graph)
    assert n.sum(g[0]).item()==pytest.approx(-n.sum(g[1]).item(),abs=1e-14)
    assert n.max(n.abs(n.grad(n.sum(n.square(g[0])),[f])[0])).item()>0
    shifted=n.RelativisticAverageGan(n.add(r.detach(),100),n.add(f.detach(),100))
    assert shifted.generator.item()==pytest.approx(o.generator.item(),abs=1e-14)
    with pytest.raises(RuntimeError):n.RelativisticAverageGan(n.ones([2,2]),n.ones([2,1]))


def test_frozen_features_reverse_rgb_and_stop_all_weight_and_bias_paths():
    t=n.DType.Float64;data=[0.]*(3*3*3*64)
    for c in range(3):data[((1*3+1)*3+c)*64+c]=1
    w=n.from_vector_double(data,[3,3,3,64]).requires_grad_();b=n.zeros([64],t).requires_grad_();x=n.from_vector_double([1,0,.5],[1,1,1,3]).requires_grad_();out=n.AdversarialSRFeatures(x,w,b)
    assert list(out.values())[:3]==pytest.approx([127.5-103.93900299072266,-116.77899932861328,255-123.68000030517578])
    unused=n.GradOptions();unused.allow_unused=True;partial=n.grad(n.sum(out),[x,w,b],unused)
    assert list(partial[0].values())==[255]*3
    assert all(not p.defined() or n.sum(n.abs(p)).item()==0 for p in partial[1:])


@pytest.mark.parametrize('enhanced',[False,True])
def test_source_census_paths_canary_and_rounded_objective_scales(enhanced):
    p,lr,hr,bic,w,b=inputs();out=game(p,lr,hr,bic,w,b,enhanced)
    assert [q.count() for q in p]==([3511,1672,0] if enhanced else [1591,1673,40])
    assert [q.size() for q in p]==([40,7,0] if enhanced else [25,8,10])
    before=[list(q.flatten()) for q in p];changed=game(p,lr,n.zeros(hr.shape(),hr.dtype()),bic,w,b,enhanced)
    assert n.equal_bits(out.prediction,changed.prediction) and n.equal_bits(out.prediction_training,changed.prediction_training)
    assert not out.target_feature.requires_grad()
    warm=game(p,lr,hr,bic,w,b,enhanced,False);assert n.equal_bits(warm.g_loss,warm.pixel_loss)
    rounded=lambda value:n.full([1],value,n.DType.Float32).item()
    assert out.g_loss.item()==pytest.approx(out.content_loss.item()+rounded(.005 if enhanced else .001)*out.adversarial_loss.item()+(rounded(.01)*out.pixel_loss.item() if enhanced else 0),abs=1e-13,rel=0)
    assert [list(q.flatten()) for q in p]==before


def test_bn_capture_is_detached_and_only_inference_reads_saved_values():
    p,lr,hr,bic,w,b=inputs();out=game(p,lr,hr,bic,w,b);before=list(p[2].flatten())
    capture=n.AdversarialSRGenerate(p[0],p[2],lr,bic,False,True);assert list(capture.moments_mean.shape())==[5,4] and not capture.moments_mean.requires_grad();assert list(p[2].flatten())==before
    for name in NAMES:
        p[2].set(name+'/moving_mean',n.ones([4],lr.dtype()));p[2].set(name+'/moving_var',n.full([4],2,lr.dtype()))
    batch=n.AdversarialSRGenerate(p[0],p[2],lr,bic,False,True);saved=n.AdversarialSRGenerate(p[0],p[2],lr,bic)
    assert n.equal_bits(batch.image,capture.image) and n.max(n.abs(n.sub(saved.image,out.prediction))).item()>1e-7
    assert not saved.moments_mean.defined()


def test_g_weight_write_then_same_capture_bn_commit_differs_from_recomputation():
    p,lr,hr,bic,w,b=inputs();out=game(p,lr,hr,bic,w,b,False,False);means=out.moments_mean;var=out.moments_unbiased_variance;before=list(p[2].flatten());cfg=n.OptimizerConfig();cfg.kind=n.OptimizerKind.Adam;cfg.beta1=.9;optimizer=n.Optimizer(p[0],.0001,cfg);optimizer.step(n.grad(out.g_loss,p[0].parameters()));assert list(p[2].flatten())==before
    wrong=n.AdversarialSRGenerate(p[0],p[2],lr,bic,False,True);assert n.max(n.abs(n.sub(wrong.moments_mean,means))).item()>1e-8
    for i,name in enumerate(NAMES):
        candidate=n.batch_norm_moving_values(p[2].at(name+'/moving_mean'),p[2].at(name+'/moving_var'),n.select(means,0,i),n.select(var,0,i));p[2].set(name+'/moving_mean',candidate.mean);p[2].set(name+'/moving_var',candidate.variance)
    assert list(p[2].flatten())!=before


def test_explicit_warmup_reset_d_then_fresh_g_and_owner_isolation():
    p,lr,hr,bic,w,b=inputs();game(p,lr,hr,bic,w,b,True);cfg=n.OptimizerConfig();cfg.kind=n.OptimizerKind.Adam;cfg.beta1=.9;go,do=n.Optimizer(p[0],.0001,cfg),n.Optimizer(p[1],.0001,cfg);db=list(p[1].flatten())
    go.step(n.grad(game(p,lr,hr,bic,w,b,True,False).g_loss,p[0].parameters()));assert list(p[1].flatten())==db and do.iterations()==0 and go.iterations()==1
    before=[list(q.flatten()) for q in p];go,do=n.Optimizer(p[0],.0001,cfg),n.Optimizer(p[1],.0001,cfg);assert go.iterations()==do.iterations()==0 and [list(q.flatten()) for q in p]==before
    out=game(p,lr,hr,bic,w,b,True);keep=n.GradOptions();keep.retain_graph=True;stale=n.grad(out.g_loss,p[0].parameters(),keep);dd=n.grad(out.d_loss,p[1].parameters());go_before=list(p[0].flatten());do.step(dd);assert list(p[0].flatten())==go_before
    after=list(p[1].flatten());gg=n.grad(game(p,lr,hr,bic,w,b,True).g_loss,p[0].parameters());assert max(n.max(n.abs(n.sub(a,b))).item() for a,b in zip(stale,gg))>1e-13;go.step(gg);assert list(p[1].flatten())==after and p[2].size()==0


@pytest.mark.parametrize('enhanced',[False,True])
def test_owned_fields_keep_higher_lr_paths_and_bad_calls_initialize_no_leaves(enhanced):
    p,lr,hr,bic,w,b=inputs();lr.requires_grad_();out=game(p,lr,hr,bic,w,b,enhanced);image=out.prediction;loss=n.mean(n.square(out.prediction_training));del out,p,hr,bic,w,b;gc.collect();graph=n.GradOptions();graph.create_graph=True;first=n.grad(loss,[lr],graph)[0];assert n.max(n.abs(n.grad(n.sum(n.square(first)),[lr])[0])).item()>0 and list(image.shape())==[2,32,32,3]
    p,lr,hr,bic,w,b=inputs()
    with pytest.raises(RuntimeError):n.AdversarialSR(p[0],p[0],p[2],lr,hr,bic,w,b)
    with pytest.raises(RuntimeError):game(p,lr,hr,bic,n.zeros([3,3,3,63],lr.dtype()),b)
    with pytest.raises(RuntimeError):game(p,lr,n.zeros([1,32,32,3],lr.dtype()),bic,w,b)
    assert sum(q.size() for q in p)==0
