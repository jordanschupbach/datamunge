#!/usr/bin/env python3
"""Independent NumPy unpaired CycleGAN forwards, coupled adjoints and four Adam states.

Reads C++ artifacts only; never prepares data or fits parameters. Checks disjoint
per-class domains, rotation/inversion, both genuine U-Net skips, LS/L1 grouping,
all captured critics/joint generators, wrong sequential control and192 writes.
"""
import argparse,json
from pathlib import Path
import numpy as np
from check_pix2pix import arrays,exact,generate as pix_generate,layer


def transform(image,inverse=False):
    """Source out(y,x)=1-in(H-1-x,y), with the documented inverse orientation."""
    rotated=np.swapaxes(image[:,:,::-1,:],1,2) if inverse else np.swapaxes(image[:,::-1,:,:],1,2)
    return np.ones((),dtype=image.dtype)-rotated


def generate(p,group,image):
    """Reuse independent two-skip adjoints under either generator's source prefix."""
    local={'0/'+k[2:]:v for k,v in p.items() if k.startswith(str(group)+'/')};out,back=pix_generate(local,image)
    def backward(gradient):
        dx,g=back(gradient);return dx,{str(group)+k[1:]:v for k,v in g.items()}
    return out,backward


def discriminate(p,group,image):
    """Unconditional1-channel patch critic; both leaky maps retain raw source arithmetic."""
    first,b1=layer(p,group,'down14',image);h1=np.where(first>0,first,.2*first)
    second,b2=layer(p,group,'down7',h1);h2=np.where(second>0,second,.2*second);score,b3=layer(p,group,'output',h2)
    def backward(gradient):
        out={};delta=b3(gradient,out)*np.where(second>0,1.,.2);delta=b2(delta,out)*np.where(first>0,1.,.2);return b1(delta,out),out
    return score,backward


def add(out,other):
    """Accumulate all repeated-generator and shared-critic contributions before writes."""
    for key,value in other.items():out[key]=out.get(key,0)+value


def forward(p,a,b):
    """Nineteen source fields, shared joint G loss and each selected critic loss."""
    fa,bfa=generate(p,1,b);fb,bfb=generate(p,0,a);ca,bca=generate(p,1,fb);cb,bcb=generate(p,0,fa);ia,bia=generate(p,1,a);ib,bib=generate(p,0,b)
    ra,bra=discriminate(p,2,a);rb,brb=discriminate(p,3,b);sa,bsa=discriminate(p,2,fa);sb,bsb=discriminate(p,3,fb)
    aa=((sa-1)**2).mean();ab=((sb-1)**2).mean();la=np.abs(ca-a).mean();lb=np.abs(cb-b).mean();ida=np.abs(ia-a).mean();idb=np.abs(ib-b).mean()
    out=dict(fake_a=fa,fake_b=fb,cycle_a=ca,cycle_b=cb,identity_a=ia,identity_b=ib,real_a_score=ra,real_b_score=rb,fake_a_score=sa,fake_b_score=sb,g_loss=(aa+ab)+(10*(la+lb)+5*(ida+idb)),d_a_loss=.5*(((ra-1)**2).mean()+(sa*sa).mean()),d_b_loss=.5*(((rb-1)**2).mean()+(sb*sb).mean()),adversarial_a=aa,adversarial_b=ab,cycle_loss_a=la,cycle_loss_b=lb,identity_loss_a=ida,identity_loss_b=idb)
    partials={}
    for back,gradient in [(bra,(ra-1)/ra.size),(bsa,sa/sa.size),(brb,(rb-1)/rb.size),(bsb,sb/sb.size)]:add(partials,back(gradient)[1])
    dfa,_=bsa(2*(sa-1)/sa.size);dfb,_=bsb(2*(sb-1)/sb.size)
    dfb_cycle,term=bca(10*np.sign(ca-a)/ca.size);add(partials,term);dfa_cycle,term=bcb(10*np.sign(cb-b)/cb.size);add(partials,term)
    for back,gradient in [(bfa,dfa+dfa_cycle),(bfb,dfb+dfb_cycle),(bia,5*np.sign(ia-a)/ia.size),(bib,5*np.sign(ib-b)/ib.size)]:add(partials,back(gradient)[1])
    return out,partials


def check(root,prepared):
    """Verify all source ownership, paired-free rows, analytic derivatives and exact Adam writes."""
    output=root/'original-output';data=arrays(output/'data.npz');raw=arrays(prepared);initial=arrays(output/'initial_arrays.npz');final=arrays(output/'final.npz')
    assert len(initial)==32 and sum(v.size for v in initial.values())==4102
    original_b_pools={}
    for split in ['fit','validation','test']:
        image=raw[split+'_images'].astype(np.float32)/np.float32(255);labels=raw[split+'_labels'];ids=raw[split+'_source_ids'];half=len(image)//20;seen=np.zeros(10,int);rows=[[],[]]
        for i,c in enumerate(labels):rows[int(seen[c]>=half)].append(i);seen[c]+=1
        assert np.array_equal(seen,np.full(10,len(image)//10));assert not set(ids[rows[0]])&set(ids[rows[1]])
        original_b_pools[split]=image[rows[1]]
        for name,index in [('a',0),('b',1)]:
            expected=image[rows[index]];expected=transform(expected) if index else expected
            exact(expected,data[split+'_'+name]);exact(ids[rows[index]],data[split+'_ids_'+name])
    records=np.genfromtxt(output/'batches.tsv',names=True,delimiter='\t',dtype=np.int64);assert len(records)==1500
    before=initial;first={k:np.zeros_like(v) for k,v in initial.items()};second={k:np.zeros_like(v) for k,v in initial.items()};iterations=[0]*4
    values=forwards=0;gerror=ferror=0.;batch=0;wrong_error=0.
    def compare(native,expected):
        nonlocal forwards,ferror
        assert set(native)==set(expected)
        for key,value in expected.items():np.testing.assert_allclose(native[key],value,atol=1e-5,rtol=5e-4,err_msg=key);forwards+=np.size(value);ferror=max(ferror,float(np.max(abs(native[key]-value))))
    def compare_grad(native,expected,label):
        nonlocal values,gerror
        assert set(native)==set(expected)
        for key,value in expected.items():np.testing.assert_allclose(native[key],value,atol=2e-5,rtol=3e-3,err_msg=label+'/'+key);values+=value.size;gerror=max(gerror,float(np.max(abs(native[key]-value))))
    def apply(group,gradient,after):
        nonlocal before
        iterations[group]+=1;t=iterations[group]
        for key,value in before.items():
            if key[0]!=str(group):exact(value,after[key]);continue
            g=gradient[key];first[key]=first[key]*np.float32(.5)+g*np.float32(.5);second[key]=second[key]*np.float32(.999)+(g*g)*np.float32(.001)
            mean=first[key]/np.float32(1-.5**t);variance=second[key]/np.float32(1-.999**t);direction=mean/(np.sqrt(variance)+np.float32(1e-8));exact(value-direction*np.float32(.0002),after[key])
        before=after
    first_input=arrays(root/'probe-first-input.npz');compare(arrays(output/'first_forward.npz'),forward({k:v.astype(float) for k,v in initial.items()},first_input['a'].astype(float),first_input['b'].astype(float))[0])
    for epoch in range(1,4):
        visits=records[records['epoch']==epoch]
        for name in ['a','b']:exact(np.sort(visits[name+'_row']),np.arange(500));np.testing.assert_array_equal(visits[name+'_source_id'],data['fit_ids_'+name][visits[name+'_row']])
        for step in range(16):
            rows=visits[visits['step']==batch];assert len(rows)==(20 if step==15 else 32)
            a,b=[data['fit_'+name][rows[name+'_row']].astype(float) for name in ['a','b']]
            if batch==0:
                exact(first_input['a'],data['fit_a'][rows['a_row']]);exact(first_input['b'],data['fit_b'][rows['b_row']]);assert not np.array_equal(rows['a_row'],rows['b_row'])
            for group in [2,3]:
                native=arrays(root/f'phases/batch{batch}-d{group}-gradient.npz');expected=forward({k:v.astype(float) for k,v in before.items()},a,b)[1];expected={k:v for k,v in expected.items() if k[0]==str(group)};compare_grad(native,expected,f'{batch}-d{group}')
                apply(group,native,arrays(root/f'phases/batch{batch}-d{group}-after.npz'))
            joint=arrays(root/f'phases/batch{batch}-g-gradient.npz');expected=forward({k:v.astype(float) for k,v in before.items()},a,b)[1];compare_grad(joint,{k:v for k,v in expected.items() if k[0] in '01'},f'{batch}-joint')
            for group in [0,1]:
                apply(group,joint,arrays(root/f'phases/batch{batch}-g{group}-after.npz'))
                if batch==0 and group==0:
                    expected=forward({k:v.astype(float) for k,v in before.items()},a,b)[1];wrong=arrays(root/'probe-wrong-sequential-gradient.npz');compare_grad(wrong,{k:v for k,v in expected.items() if k[0]=='1'},'wrong-sequential')
                    wrong_error=max(float(np.max(abs(wrong[k]-joint[k]))) for k in wrong)
            batch+=1
        checkpoint=arrays(root/f'probe-epoch{epoch}.npz')
        for key,value in checkpoint.items():exact(before[key],value)
        for split in ['validation','test']:compare(arrays(root/f'probe-epoch{epoch}-{split}.npz'),forward({k:v.astype(float) for k,v in checkpoint.items()},data[split+'_a'].astype(float),data[split+'_b'].astype(float))[0])
    assert iterations==[48]*4 and batch==48
    for key,value in final.items():exact(before[key],value)
    for p,name in [(initial,'initial_outputs.npz'),(final,'outputs.npz')]:compare(arrays(output/name),forward({k:v.astype(float) for k,v in p.items()},data['test_a'].astype(float),data['test_b'].astype(float))[0])
    p={k:v.astype(float) for k,v in final.items()};a,b=first_input['a'].astype(float),first_input['b'].astype(float);compare_grad(arrays(output/'final_gradients.npz'),forward(p,a,b)[1],'final')
    meta=json.loads((output/'metrics.json').read_text());assert wrong_error>1e-8;np.testing.assert_equal(meta['wrong_sequential_generator_gradient_max_difference'],wrong_error)
    diagnostic=arrays(output/'diagnostic_known_mapping.npz');exact(diagnostic['b_to_a_target'],original_b_pools['test']);exact(diagnostic['a_to_b_target'],transform(data['test_a']));np.testing.assert_allclose(diagnostic['b_to_a_target'],transform(data['test_b'],True),atol=4e-8,rtol=2e-5)
    predictions=arrays(output/'outputs.npz')
    for key in ['g_loss','d_a_loss','d_b_loss','cycle_loss_a','cycle_loss_b','identity_loss_a','identity_loss_b']:assert meta['test_'+key]==predictions[key].item()
    history=np.genfromtxt(output/'history.tsv',names=True,delimiter='\t')
    for row in history:
        fields=arrays(root/f"probe-epoch{int(row['epoch'])}-validation.npz")
        for key in history.dtype.names[1:]:assert float(row[key])==fields[key].item()
    for direction,field,target in [('a_to_b','fake_b','a_to_b_target'),('b_to_a','fake_a','b_to_a_target')]:np.testing.assert_allclose(meta['diagnostic_mapping_'+direction+'_mse'],((predictions[field].astype(float)-diagnostic[target].astype(float))**2).mean(),atol=2e-15)
    # Representative final scalar differences cover every named leaf of all four stores.
    a,b=a[:2],b[:2];partials=forward(p,a,b)[1];differences=0
    for key,gradient in partials.items():
        for index in sorted({0,gradient.size//2,gradient.size-1}):
            old=p[key].flat[index];h=1e-6;field='g_loss' if key[0] in '01' else 'd_a_loss' if key[0]=='2' else 'd_b_loss';p[key].flat[index]=old+h;plus=forward(p,a,b)[0][field];p[key].flat[index]=old-h;minus=forward(p,a,b)[0][field];p[key].flat[index]=old
            np.testing.assert_allclose((plus-minus)/(2*h),gradient.flat[index],atol=1e-6,rtol=5e-3,err_msg=key);differences+=1
    return dict(passed=True,native_parameter_derivatives_checked=values,maximum_derivative_error=gerror,forward_values_checked=forwards,maximum_forward_error=ferror,independent_final_scalar_finite_differences=differences,all_192_Adam_updates_bit_exact=True,all_192_phase_ownership_checks_pass=True,optimizer_iterations=iterations,batch_row_pairs_checked=1500,source_domain_values_and_disjoint_ids_exact=True,wrong_sequential_generator_gradient_max_difference=wrong_error,known_mapping_diagnostics_only=True,original_b_diagnostic_pixels_bit_exact=True,all_validation_history_and_final_test_scalar_records_exact=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,default=Path('build/nn-report-conversion/cycle_gan'));parser.add_argument('--data',type=Path,default=Path('build/nn-report-conversion/datasets/saliency/saliency_data.npz'));parser.add_argument('--json',type=Path);args=parser.parse_args();result=check(args.root,args.data);text=json.dumps(result,indent=2)+'\n';print(text,end='')
    if args.json:args.json.write_text(text)

if __name__=='__main__':main()
