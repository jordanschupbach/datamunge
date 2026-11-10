#!/usr/bin/env python3
"""Optional independent NumPy oracle for saved two-scale VQ-VAE-2 artifacts.

Never trains or prepares data. Checks all surrogate gradients, both dictionaries,
spatial hierarchy/interventions, checkpoint selection and both raster priors.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from check_adversarial_autoencoder import arrays, exact
from check_convolutional_autoencoder import convolution, convolution_adjoint, pool, repeat
from check_factor_tc_variational_autoencoder import first_update
from check_vector_quantized_autoencoder import quantize, factor_data, markov, markov_nll


def top_decode(p, codes):
    """Decode coarse vectors to a 14x14 feature map, retaining adjoint inputs."""
    enlarged = repeat(codes.reshape(-1, 7, 7, 4))
    return np.maximum(convolution(enlarged, p['top_decoder/W'], p['top_decoder/b']), 0), enlarged


def decode(p, bottom, feature):
    """Decode only supplied fine vectors and coarse features."""
    combined = np.concatenate([bottom.reshape(-1, 14, 14, 4), feature], axis=3)
    hidden = np.maximum(convolution(combined, p['decoder/W'], p['decoder/b']), 0)
    enlarged = repeat(hidden)
    logits = convolution(enlarged, p['output/W'], p['output/b']).reshape(-1, 784)
    return 1/(1+np.exp(-logits)), (combined, hidden, enlarged, logits)


def forward(p, images, override=None):
    """Independent coarse encoding, top-conditioned fine encoding and two losses."""
    images = images.astype(np.float64)
    c1 = np.maximum(convolution(images, p['encoder1/W'], p['encoder1/b']), 0);h = pool(c1)
    c2 = np.maximum(convolution(h, p['encoder2/W'], p['encoder2/b']), 0)
    encoded_top = pool(c2).reshape(-1, 4);top = quantize(encoded_top, p['top_codebook'])
    if override is not None:
        top['indices'] = np.asarray(override, dtype=np.int32).ravel()
        top['quantized'] = top['straight_through'] = p['top_codebook'][top['indices']]
        top['codebook_loss'] = top['commitment_loss'] = ((encoded_top-top['quantized'])**2).mean()
    feature, top_enlarged = top_decode(p, top['quantized'])
    projection_input = np.concatenate([h, feature], axis=3)
    encoded_bottom = (projection_input@p['bottom_projection/W'][0,0]+p['bottom_projection/b']).reshape(-1, 4)
    bottom = quantize(encoded_bottom, p['bottom_codebook'])
    probability, cache = decode(p, bottom['quantized'], feature)
    nll = (np.logaddexp(0, cache[-1])-images.reshape(-1,784)*cache[-1]).sum(1)
    out = dict(encoded_top=encoded_top, encoded_bottom=encoded_bottom, top_feature=feature,
               probability=probability, nll=nll, reconstruction_loss=nll.mean())
    for label, values in [('top',top),('bottom',bottom)]:
        out.update({label+'_'+k:v for k,v in values.items()})
    out['loss'] = nll.mean()+top['codebook_loss']+bottom['codebook_loss']+.25*(top['commitment_loss']+bottom['commitment_loss'])
    return out, (c1,h,c2,projection_input,top_enlarged,cache)


def derivatives(p, images):
    """Explicit two-scale surrogate reverse pass, including fine-to-coarse coupling."""
    images = images.astype(np.float64)
    o,(c1,h,c2,projection_input,top_enlarged,(combined,hidden,enlarged,_)) = forward(p,images)
    expected = {};delta = ((o['probability']-images.reshape(-1,784))/len(images)).reshape(images.shape)
    g,expected['output/W'],expected['output/b'] = convolution_adjoint(enlarged,p['output/W'],delta)
    g = 4*pool(g)*(hidden>0)
    g,expected['decoder/W'],expected['decoder/b'] = convolution_adjoint(combined,p['decoder/W'],g)
    fine = g[...,:4].reshape(-1,4);coarse_feature = g[...,4:]
    residual = o['encoded_bottom']-o['bottom_quantized']
    fine = fine+.5*residual/residual.size
    expected['bottom_codebook'] = np.zeros_like(p['bottom_codebook'])
    np.add.at(expected['bottom_codebook'],o['bottom_indices'],-2*residual/residual.size)
    fine = fine.reshape(-1,14,14,4)
    expected['bottom_projection/W'] = np.einsum('nhwc,nhwo->co',projection_input,fine)[None,None]
    expected['bottom_projection/b'] = fine.sum((0,1,2))
    g = fine@p['bottom_projection/W'][0,0].T
    direct_h = g[...,:4];coarse_feature = coarse_feature+g[...,4:]
    g,expected['top_decoder/W'],expected['top_decoder/b'] = convolution_adjoint(top_enlarged,p['top_decoder/W'],coarse_feature*(o['top_feature']>0))
    g = (4*pool(g)).reshape(-1,4)
    residual = o['encoded_top']-o['top_quantized'];g += .5*residual/residual.size
    expected['top_codebook'] = np.zeros_like(p['top_codebook'])
    np.add.at(expected['top_codebook'],o['top_indices'],-2*residual/residual.size)
    g = repeat(g.reshape(-1,7,7,4))/4*(c2>0)
    g,expected['encoder2/W'],expected['encoder2/b'] = convolution_adjoint(h,p['encoder2/W'],g)
    g = repeat(g+direct_h)/4*(c1>0)
    _,expected['encoder1/W'],expected['encoder1/b'] = convolution_adjoint(images,p['encoder1/W'],g)
    # Independent d mean(encoded_bottom^2)/d encoded_top: no commitment or decoder term.
    projection_delta = (2*o['encoded_bottom']/o['encoded_bottom'].size).reshape(-1,14,14,4)@p['bottom_projection/W'][0,0].T
    hierarchy,_,_ = convolution_adjoint(top_enlarged,p['top_decoder/W'],projection_delta[...,4:]*(o['top_feature']>0))
    return expected,(4*pool(hierarchy)).reshape(-1,4)


def check(directory):
    """Verify frozen source or report evidence and return measured error bounds."""
    data = arrays(directory/'data.npz');pixels = factor_data(data)
    initial = arrays(directory/'initial_arrays.npz');p = {k:v.astype(np.float64) for k,v in initial.items()}
    batch = arrays(directory/'first_batch.npz')['images'];actual = arrays(directory/'first_gradients.npz')
    expected,hierarchy = derivatives(p,batch);assert set(expected)==set(actual)
    error = 0.
    for k,v in expected.items():
        np.testing.assert_allclose(actual[k],v,atol=8e-6,rtol=3e-4);error=max(error,float(np.max(np.abs(v-actual[k]))))
    first_update(initial,actual,arrays(directory/'first_update_arrays.npz'))
    probe = arrays(directory/'first_hierarchy_gradients.npz')
    np.testing.assert_allclose(probe['bottom_encoded_square_dtop_encoded'],hierarchy,atol=2e-9,rtol=2e-4)
    for name in ['reconstruction_dtop_book','reconstruction_dbottom_book']:exact(probe[name],np.zeros_like(probe[name]))
    fields = values = 0;maximum = 0.
    def endpoints(out,saved):
        nonlocal fields,values,maximum
        assert set(out)==set(saved)
        for k,v in out.items():
            v=np.asarray(v)
            if k.endswith('indices'):exact(saved[k],v)
            else:
                np.testing.assert_allclose(saved[k],v,atol=6e-5,rtol=2e-5)
                maximum=max(maximum,float(np.max(np.abs(v-saved[k]))))
            fields+=1;values+=v.size
    out,_=forward(p,batch);endpoints(out,arrays(directory/'first_forward.npz'))
    selected=arrays(directory/'model_arrays.npz');p={k:v.astype(np.float64) for k,v in selected.items()}
    saved=arrays(directory/'outputs.npz')
    for split in ['fit','validation','test']:
        out,_=forward(p,data[split+'_images']);endpoints(out,{k:saved[split+'_'+k] for k in out})
        for scale,tokens in [('top',49),('bottom',196)]:
            ids=out[scale+'_indices'].reshape(-1,tokens);exact(saved[split+'_'+scale+'_index_grid'],ids)
            exact(saved[split+'_'+scale+'_code_counts'],np.bincount(ids.ravel(),minlength=16).astype(np.int32))
    intervention=arrays(directory/'top_conditioning_intervention.npz')
    for label,code in [('zero',0),('one',1)]:
        out,_=forward(p,intervention['images'],np.full((1,49),code,dtype=np.int32))
        np.testing.assert_allclose(intervention[label+'_top_encoded_bottom'],out['encoded_bottom'],atol=4e-6,rtol=2e-5)
        exact(intervention[label+'_top_bottom_indices'],out['bottom_indices'])
    history=np.genfromtxt(directory/'history.tsv',delimiter='\t',names=True);criteria=[]
    for epoch in range(1,4):
        ep={k:v.astype(np.float64) for k,v in arrays(directory/f'epoch{epoch}_arrays.npz').items()}
        out,_=forward(ep,data['validation_images']);criteria.append(out['reconstruction_loss'])
        for column,key in [('validation_nll','reconstruction_loss'),('validation_objective','loss')]+[('validation_'+scale+'_'+term,scale+'_'+term) for scale in ['top','bottom'] for term in ['codebook_loss','commitment_loss']]:
            np.testing.assert_allclose(history[column][epoch-1],out[key],atol=6e-5,rtol=2e-5)
    meta=json.loads((directory/'metrics.json').read_text());chosen=int(np.argmin(criteria))+1
    assert meta['selected_epoch']==chosen and meta['updates']==21 and meta['epochs']==3
    for k,v in arrays(directory/f'epoch{chosen}_arrays.npz').items():exact(v,selected[k])
    batches=np.genfromtxt(directory/'batches.tsv',delimiter='\t',names=True,dtype=np.int64)
    assert len(batches)==630
    for epoch in range(1,4):
        rows=batches[batches['epoch']==epoch];exact(np.sort(rows['row']),np.arange(210))
        exact(rows['source_id'],data['fit_source_ids'][rows['row']].astype(np.int64))
        exact(rows['step'],np.repeat(np.arange((epoch-1)*7,epoch*7),[32]*6+[18]))
    for split in ['fit','test']:
        for scale in ['top','bottom']:
            counts=saved[split+'_'+scale+'_code_counts'];v=counts[counts>0]/counts.sum()
            np.testing.assert_allclose(meta[split+'_'+scale+'_perplexity'],np.exp(-(v*np.log(v)).sum()),atol=1e-14,rtol=1e-14)
            if split=='fit':assert meta['fit_'+scale+'_used']==int((counts>0).sum())
    test=data['test_images'].reshape(-1,784).astype(np.float64)
    np.testing.assert_allclose(meta['test_mse'],((saved['test_probability']-test)**2).mean(),atol=5e-13,rtol=1e-14)
    fitmean=data['fit_images'].reshape(-1,784).astype(np.float64).mean(0).astype(np.float32)
    np.testing.assert_allclose(meta['fit_mean_test_mse'],((fitmean-test)**2).mean(),atol=5e-13,rtol=1e-14)
    np.testing.assert_allclose(meta['test_nll'],saved['test_reconstruction_loss'],atol=0,rtol=0)
    np.testing.assert_allclose(meta['test_objective'],saved['test_loss'],atol=0,rtol=0)
    zero_feature,_=top_decode(p,np.broadcast_to(p['top_codebook'][0],(49,4)))
    one_feature,_=top_decode(p,np.broadcast_to(p['top_codebook'][1],(49,4)))
    zero,_=decode(p,np.broadcast_to(p['bottom_codebook'][0],(196,4)),zero_feature)
    changed_top,_=decode(p,np.broadcast_to(p['bottom_codebook'][0],(196,4)),one_feature)
    changed_bottom,_=decode(p,np.broadcast_to(p['bottom_codebook'][1],(196,4)),zero_feature)
    for name,changed in [('top_symbol_max_effect',changed_top),('bottom_symbol_max_effect',changed_bottom)]:
        np.testing.assert_allclose(meta[name],np.abs(changed-zero).max(),atol=3e-6,rtol=1e-5)
    effect=np.abs(intervention['zero_top_encoded_bottom'].astype(np.float64)-intervention['one_top_encoded_bottom']).max()
    np.testing.assert_allclose(meta['top_conditioning_bottom_encoded_max_effect'],effect,atol=1e-14,rtol=1e-14)
    prior=arrays(directory/'prior.npz');top=saved['fit_top_index_grid'];bottom=saved['fit_bottom_index_grid']
    top_counts,top_p=markov(top);exact(prior['top_counts'],top_counts);exact(prior['top_probabilities'],top_p)
    counts=np.zeros((16,17,16),dtype=np.int32)
    parent=np.array([(i//14//2)*7+(i%14)//2 for i in range(196)])
    for b,row in enumerate(bottom):
        previous=16
        for i,code in enumerate(row):counts[top[b,parent[i]],previous,code]+=1;previous=int(code)
    probabilities=(counts.astype(np.float64)+.5)/(counts.sum(2,keepdims=True)+8.)
    exact(prior['bottom_counts'],counts);exact(prior['bottom_probabilities'],probabilities)
    for scale,tokens,seed in [('top',49,2827),('bottom',196,2833)]:
        rng=np.random.RandomState(seed);u=(rng.randint(0,2**32,(20,tokens),dtype=np.uint32).astype(np.float64)+.5)/2**32
        exact(prior[scale+'_uniform'],u);ids=np.empty((20,tokens),dtype=np.int32)
        for b in range(20):
            previous=16
            for t in range(tokens):
                q=top_p[previous] if scale=='top' else probabilities[prior['sample_top_ids'][b,parent[t]],previous]
                ids[b,t]=min(int(np.searchsorted(np.cumsum(q),u[b,t],side='right')),15);previous=ids[b,t]
        exact(prior['sample_'+scale+'_ids'],ids)
    feature,_=top_decode(p,p['top_codebook'][prior['sample_top_ids'].ravel()])
    prediction,_=decode(p,p['bottom_codebook'][prior['sample_bottom_ids'].ravel()],feature)
    np.testing.assert_allclose(prior['probability'],prediction,atol=5e-6,rtol=2e-5)
    for split in ['fit','test']:
        np.testing.assert_allclose(meta['prior_top_'+split+'_nll'],markov_nll(top_p,saved[split+'_top_index_grid']),atol=1e-12,rtol=1e-12)
        top=saved[split+'_top_index_grid'];bottom=saved[split+'_bottom_index_grid'];total=0.
        for b,row in enumerate(bottom):
            previous=16
            for i,code in enumerate(row):total-=np.log(probabilities[top[b,parent[i]],previous,code]);previous=int(code)
        np.testing.assert_allclose(meta['prior_bottom_'+split+'_nll'],total/len(bottom),atol=1e-12,rtol=1e-12)
    return dict(passed=True,parameter_derivatives_checked=sum(v.size for v in expected.values()),first_gradient_max_error=error,
                first_Adam_update_bit_exact=True,lattice_pixels_checked=pixels,endpoint_fields_checked=fields,endpoint_values_checked=values,
                maximum_endpoint_error=maximum,hierarchy_derivatives_checked=hierarchy.size,reconstruction_codebook_gradients_zero=True,
                top_conditioning_interventions_checked=2,eligible_validation_checkpoints_checked=3,selected_epoch=chosen,batch_history_rows_checked=630,
                prior_count_cells_checked=counts.size+top_counts.size,explicit_uniforms_checked=20*(49+196),unfiltered_prior_images_checked=20)


def main():
    """Check completed artifacts and optionally save a compact verification receipt."""
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory',type=Path);parser.add_argument('--json',type=Path)
    args=parser.parse_args();text=json.dumps(check(args.directory),indent=2)+'\n'
    if args.json:args.json.write_text(text)
    print(text,end='')


if __name__=='__main__':
    main()
