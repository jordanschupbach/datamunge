#!/usr/bin/env python3
"""Independent NumPy/PCM oracle for the WaveNet teaching report, never a fit path."""
from pathlib import Path
import argparse,json,wave,struct,math
import numpy as np

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('data',type=Path);p.add_argument('rollouts',type=Path);a=p.parse_args()
 d=np.load(a.data/'wavenet_stack.npz');manifest=json.loads((a.data/'manifest.json').read_text());clips={};checked=0
 for record in manifest['records']:
  with wave.open(str(a.data/record['path']))as w:
   assert w.getframerate()==8000 and w.getsampwidth()==2 and w.getnchannels()==1
   samples=np.asarray(struct.unpack('<1025h',w.readframes(1025)),dtype=np.float64)/32768.
  codes=np.clip(np.floor((np.sign(samples)*np.log1p(255*np.abs(samples))/math.log(256.)+1)*127.5+.5),0,255).astype(np.int32);clips[record['source_id']]=codes
  split=record['split']
  for row in np.flatnonzero(d[split+'_source_id']==record['source_id']):
   start=d[split+'_start'][row];np.testing.assert_array_equal(d[split+'_input'][row],codes[start:start+256]);np.testing.assert_array_equal(d[split+'_target'][row],codes[start+1:start+257]);checked+=512
 for row,source in enumerate(d['prompt_source_ids']):np.testing.assert_array_equal(d['prompts'][row],clips[int(source)][:384])
 r=np.load(a.rollouts);logits=r['step_logits'].astype(np.float64);q=np.exp(logits-logits.max(-1,keepdims=True));q/=q.sum(-1,keepdims=True);cdf=np.cumsum(q,axis=-1);draws=r['uniform_draws'];sampled=np.minimum((cdf<=draws[:,:,None]).sum(-1),255)
 np.testing.assert_array_equal(sampled,r['codes'][:,128:]);np.testing.assert_array_equal(r['codes'][:,:128],d['prompts'][:,:128]);np.testing.assert_array_equal(r['reference'],d['prompts'])
 state=547;expected=np.empty((3,256))
 for t in range(256):
  for row in range(3):state=(48271*state)%2147483647;expected[row,t]=(state-1)/2147483646.
 np.testing.assert_array_equal(draws,expected)
 print(json.dumps({'independent_pcm_input_target_values':checked,'original_prompt_values':1152,'inverse_cdf_samples':768,'exact_seeded_uniforms':768,'all_equal':True},indent=2))
if __name__=='__main__':main()
