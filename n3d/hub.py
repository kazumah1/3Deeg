"""Fetch the private packed dataset, verify hashes, and smoke-test the CPU loader."""
from __future__ import annotations
import argparse,hashlib,json,os,re
from pathlib import Path

REPO_ID='kazumah/3deeg-packed'
DEFAULT_REVISION='0107f64d1d6077f9b36b8e6fc688bf285e8246d4'

def required_files(participants,geometry='point_cloud_simple'):
 if geometry not in ('point_cloud_simple','point_cloud'):raise ValueError('Unknown geometry')
 if not participants or any(not re.fullmatch(r'sub(?:0[1-9]|1[0-2])',s) for s in participants):raise ValueError('Use sub01 through sub12')
 return ['packed/objects.json','packed/clip_video.npy','packed/color_label.npy',f'packed/geometry_{geometry}.npy']+[f'packed/{s}_eeg.npz' for s in dict.fromkeys(participants)]

def verify_files(root,manifest,names):
 root=Path(root);verified=[]
 for name in names:
  path=Path(name)
  if path.is_absolute() or '..' in path.parts or path.parts[0]!='packed':raise ValueError('Unsafe manifest path')
  p=root/path;expected=manifest['files'][name]
  if not p.is_file() or p.stat().st_size!=expected['bytes']:raise ValueError(f'Missing or wrong size: {name}')
  with p.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
  if digest!=expected['sha256']:raise ValueError(f'SHA256 mismatch: {name}')
  verified.append(name)
 return verified

def fetch(participants=('sub01',),geometry='point_cloud_simple',data_root=None,revision=DEFAULT_REVISION):
 from huggingface_hub import hf_hub_download,snapshot_download
 if not re.fullmatch(r'[0-9a-f]{40}',revision):raise ValueError('A full pinned commit hash is required')
 names=required_files(participants,geometry)
 manifest=json.loads(Path(hf_hub_download(REPO_ID,'manifest.json',repo_type='dataset',revision=revision)).read_text())
 for name in names:
  if name not in manifest['files']:raise ValueError(f'File absent from manifest: {name}')
 root=Path(snapshot_download(REPO_ID,repo_type='dataset',revision=revision,allow_patterns=['manifest.json']+names,local_dir=data_root,max_workers=2))
 verify_files(root,manifest,names)
 return root

def validate_arrays(root,participants,geometry='point_cloud_simple'):
 import numpy as np
 p=Path(root)/'packed';meta=json.loads((p/'objects.json').read_text());objects=meta['objects'];cats=meta['categories']
 if len(objects)!=720 or len(set(objects))!=720 or len(cats)!=72:raise ValueError('Object/category metadata invalid')
 details={}
 for subject in participants:
  with np.load(p/f'{subject}_eeg.npz',allow_pickle=False) as z:
   if set(z.files)!={'dyn','stat','label'}:raise ValueError('Unexpected EEG archive members')
   dyn,stat,labels=z['dyn'],z['stat'],z['label']
   if dyn.shape[1:]!=(64,600) or stat.shape[1:]!=(64,250) or len(dyn)!=len(stat) or len(labels)!=len(dyn):raise ValueError('EEG dimensions inconsistent')
   if set(labels.tolist())!=set(objects) or not np.isfinite(dyn).all() or not np.isfinite(stat).all():raise ValueError('EEG values or labels invalid')
   details[subject]={'dyn':list(dyn.shape),'stat':list(stat.shape),'labels':len(labels),'finite':True}
 for name,shape in [(f'geometry_{geometry}.npy',(720,8192,6)),('clip_video.npy',(720,1024)),('color_label.npy',(720,))]:
  arr=np.load(p/name,mmap_mode='r',allow_pickle=False)
  if arr.shape!=shape or not np.isfinite(arr).all():raise ValueError(f'Invalid shared array: {name}')
 return {'objects':len(objects),'categories':len(cats),'subjects':details}

def smoke_loader(root,subject='sub01',geometry='point_cloud_simple'):
 import numpy as np
 from .data import load_bundle,make_splits,EEG3D
 b=load_bundle(Path(root),participants=[subject],geometry=geometry);splits=make_splits(b)
 if {k:len(v) for k,v in splits.items()}!={'train':504,'val':72,'test':144}:raise ValueError('Split changed')
 for a in splits:
  for c in splits:
   if a!=c and set(splits[a])&set(splits[c]):raise ValueError('Object split overlap')
 ds=EEG3D(b,splits['test'],train=False,aug=False);sample=ds[0]
 if len(ds)!=144 or tuple(sample['dyn'].shape)!=(64,600) or tuple(sample['stat'].shape)!=(64,250):raise ValueError('Dataset sample changed')
 for key in ('dyn','stat','points','clip'):
  if not np.isfinite(sample[key].numpy()).all():raise ValueError('Nonfinite loader sample')
 return {'subject':subject,'split_objects':{k:len(v) for k,v in splits.items()},'test_items':len(ds),'sample_shapes':{k:list(sample[k].shape) for k in ('dyn','stat','points','clip')},'device':str(sample['dyn'].device),'training_started':False}

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--participants',nargs='+',default=['sub01']);p.add_argument('--geometry',choices=['point_cloud_simple','point_cloud'],default='point_cloud_simple');p.add_argument('--data-root',type=Path);p.add_argument('--revision',default=DEFAULT_REVISION);p.add_argument('--smoke-test',action='store_true');p.add_argument('--report',type=Path);a=p.parse_args()
 root=fetch(a.participants,a.geometry,a.data_root,a.revision);result={'repo':REPO_ID,'revision':a.revision,'data_root':str(root),'integrity':'SHA256 verified','arrays':validate_arrays(root,a.participants,a.geometry)}
 if a.smoke_test:result['loader']=smoke_loader(root,a.participants[0],a.geometry)
 if a.report:a.report.write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result,indent=2));print(f'N3D_DATA={root}')
if __name__=='__main__':main()
