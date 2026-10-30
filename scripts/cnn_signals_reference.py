#!/usr/bin/env python3
"""Independent NumPy source oracle: original TXT/ARFF versus prepared TS-ingress tensors.

Run only for validation;report preprocessing and fitting use datamunge tensors.
Uses the isolated reference environment already pinned for CNN asset audits.
"""
from pathlib import Path
import argparse,io,json,zipfile
import numpy as np

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);args=p.parse_args();root=args.root
 for name,folder,offset in [('GunPoint','time_series',50),('ECG200','ecg_sensor',100),('BasicMotions','ecg_sensor',40)]:
  base=root/folder;cache=base if name=='GunPoint'else base/name;manifest=json.loads((cache/'manifest.json').read_text());arrays=np.load(cache/('time_series.npz'if name=='GunPoint'else'signals.npz'));raw=[];truth=[]
  with zipfile.ZipFile(base/(name+'.zip'))as archive:
   for part in ['TRAIN','TEST']:
    if name!='BasicMotions':
     table=np.loadtxt(io.BytesIO(archive.read(name+'_'+part+'.txt')),dtype=np.float64);raw.append(table[:,1:,None].astype(np.float32));truth.extend(manifest['class_names'].index(str(int(x)))for x in table[:,0])
    else:
     channels=[];labels=None
     for channel in range(1,7):
      lines=archive.read('BasicMotionsDimension'+str(channel)+'_'+part+'.arff').decode().splitlines();active=False;rows=[];current=[]
      for line in lines:
       if line.strip().lower()=='@data':active=True;continue
       if not active or not line.strip()or line.lstrip().startswith('%'):continue
       values=line.strip().split(',');rows.append([float(x)for x in values[:-1]]);current.append(values[-1].strip(" '\""))
      if labels is None:labels=current
      else:assert current==labels
      channels.append(np.array(rows,dtype=np.float32))
     raw.append(np.stack(channels,axis=2));truth.extend(manifest['class_names'].index(x)for x in labels)
  values=np.concatenate(raw,axis=0);truth=np.array(truth,dtype=np.int32);seen=set();checked=0
  for split in ['fit','validation','test']:
   ids=arrays[split+'_source_ids'];np.testing.assert_array_equal(arrays[split+'_signals'],values[ids]);np.testing.assert_array_equal(arrays[split+'_labels'],truth[ids]);assert not seen.intersection(ids.tolist());seen.update(ids.tolist());checked+=int(arrays[split+'_signals'].size)
  assert seen==set(range(values.shape[0]));print(name+':every '+str(checked)+' source TXT/ARFF samples and all original labels/IDs match TS-ingress tensors',flush=True)
if __name__=='__main__':main()
