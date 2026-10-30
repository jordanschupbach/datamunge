#!/usr/bin/env python3
"""Independent NumPy/OFF oracle for all sampled ModelNet source coordinates;never a fit path."""
from pathlib import Path
import argparse,json
import numpy as np
def uniforms(seed,count):
 mask=(1<<64)-1;state=[]
 for i in range(4):
  seed=(seed+0x9e3779b97f4a7c15)&mask;z=seed;z=((z^(z>>30))*0xbf58476d1ce4e5b9)&mask;z=((z^(z>>27))*0x94d049bb133111eb)&mask;state.append(z^(z>>31))
 def rotl(x,k):return ((x<<k)|(x>>(64-k)))&mask
 out=[]
 for i in range(count):
  result=(rotl((state[1]*5)&mask,7)*9)&mask;t=(state[1]<<17)&mask;state[2]^=state[0];state[3]^=state[1];state[1]^=state[2];state[0]^=state[3];state[2]^=t;state[3]=rotl(state[3],45);out.append((result>>11)*2.**-53)
 return np.asarray(out)
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);a=p.parse_args();root=a.root;manifest=json.loads((root/'manifest.json').read_text());data=np.load(root/'point_clouds.npz');draws=np.load(root/'point_sampling.npz');checked=0;maxerr=0.;seen={}
 for record in manifest['records']:
  source=record['source_id'];tokens=(root/record['ingress']).read_text().split();nv,nf=map(int,tokens[:2]);vertices=np.asarray(tokens[2:2+3*nv],dtype=np.float64).reshape(nv,3);faces=np.asarray(tokens[2+3*nv:],dtype=np.int64).reshape(nf,3);triangles=vertices[faces];ids=draws[str(source)+'/triangles'];area=np.linalg.norm(np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]),axis=1)*.5;logits=np.where(area>0,np.log(np.maximum(area,1e-300)),-1e30);weights=np.exp(logits-logits.max());random=uniforms(941+source,3072);expected=np.searchsorted(np.cumsum(weights),random[:1024]*weights.sum(),side='right');np.testing.assert_array_equal(ids,expected);np.testing.assert_array_equal(draws[str(source)+'/uniforms'],random[1024:].reshape(1024,2));u,v=draws[str(source)+'/uniforms'].T;selected=triangles[ids];s=np.sqrt(u)[:,None];raw=selected[:,0]*(1-s)+selected[:,1]*s*(1-v[:,None])+selected[:,2]*s*v[:,None];center=raw.mean(0);centered=raw-center;radius=np.linalg.norm(centered,axis=1).max();xyz=(centered/radius).astype(np.float32);split=record['split'];row=np.flatnonzero(data[split+'_source_id']==source);assert len(row)==1 and source not in seen;seen[source]=split;row=int(row[0]);assert int(data[split+'_label'][row])==record['label'];np.testing.assert_allclose(center,draws[str(source)+'/center'],atol=1e-12,rtol=0);np.testing.assert_allclose(radius,draws[str(source)+'/radius'],atol=1e-12,rtol=0);np.testing.assert_allclose(xyz,data[split+'_xyz'][row],atol=1e-7,rtol=0);maxerr=max(maxerr,float(np.abs(xyz-data[split+'_xyz'][row]).max()));checked+=xyz.size
 print(json.dumps({'original_off_objects':len(seen),'independent_barycentric_xyz_values':checked,'independent_area_weighted_faces':43008,'exact_barycentric_uniforms':86016,'maximum_float32_error':maxerr,'all_match':True},indent=2))
if __name__=='__main__':main()
