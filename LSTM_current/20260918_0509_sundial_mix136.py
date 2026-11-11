"""Matched mix136 Sundial seven-input adapter; same protocol as completed mix244."""
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK']='0'
os.environ['MPLCONFIGDIR']='/private/tmp/sundial136-mpl'
from pathlib import Path
import json,hashlib,time,sys,shutil,platform,gzip,io
import numpy as np,pandas as pd
H=Path(__file__).resolve().parent
BASE=H/'20260912_2134_Python_OPNET_Simulator/runs/mix_2w_10s/20260916_0835_mix136_comparison'
DATA=BASE.parent/'20260916_0712'
P=H/'vendor/sundial_20260917'
O=BASE.parent/'20260918_0509_sundial_mix136'
REFERENCE=BASE.parent/'20260917_1313_sundial_mix244'
def sha(p):
 with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(n,v):(O/n).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def source_bytes(n):
 p=BASE/n
 return p.read_bytes() if p.exists() else gzip.decompress(Path(str(p)+'.gz').read_bytes())
def source_sha(n):return hashlib.sha256(source_bytes(n)).hexdigest()
def prepare():
 O.mkdir(exist_ok=True)
 assert not (O/'protocol.json').exists()
 protocol=json.loads((BASE/'protocol.json').read_text());hashes=json.loads((BASE/'artifact_hashes.json').read_text())
 names=['samples.npz','lstm_predictions.npz','timesfm3_predictions.npz','metrics.csv','lstm_metadata.json','timesfm3_metadata.json','train_origins.csv','val_origins.csv','test_origins.csv']
 for n in names:assert source_sha(n)==hashes[n],n
 assert sha(BASE/'samples.npz')==protocol['samples_sha256'] and protocol['scenario']['mix']=='136'
 validation=json.loads((BASE/'final_validation.json').read_text());assert validation['status']=='PASS' and validation['raw_windows_reconstructed']==40960
 raw_current=0
 for r in protocol['runs']:
  d=DATA/r['run'];assert sha(d/'manifest.json')==r['manifest_sha256']
  if (d/'allocation.npz').exists():assert sha(d/'allocation.npz')==r['allocation_sha256'];raw_current+=1
 a=np.load(BASE/'samples.npz')
 for split,seed,n in [('train',128,20480),('val',129,10240),('test',130,10240)]:
  f=pd.read_csv(io.BytesIO(source_bytes(f'{split}_origins.csv')));assert len(f)==n and set(f.seed)=={seed} and not f.duplicated(['run','cycle','onu']).any();assert f.time.between(3,10,inclusive='left').all()
  assert a[split+'_x'].shape==(n,50,7) and a[split+'_y'].shape==(n,)
  assert set(f.load)==set(range(10,101,10)) and set(f.onu)==set(range(64))
  for _,g in f.groupby(['run','onu']):assert g.time.is_monotonic_increasing
  x=a[split+'_x'];assert np.isfinite(x).all() and (x>=0).all()
  assert np.array_equal(a[split+'_y'],np.minimum(x[:,-1,1],np.maximum(0,46875-x[:,-1,0])))
  (O/f'{split}_origins.csv').write_bytes(source_bytes(f'{split}_origins.csv'))
 ref=json.loads((REFERENCE/'protocol.json').read_text())
 save('protocol.json',dict(status='PREPARED',scenario=protocol['scenario'],source=str(BASE),source_hashes={n:source_sha(n) for n in names},source_compression='CSV sources are gzip-compressed; hashes verified on decompressed bytes against original artifact manifest',source_protocol=protocol,target='AF GRANT(t)',split='128 train /129 validation /130 test',adapter=ref['adapter'],adapter_protocol_reference=str(REFERENCE),native_sampling_executed=False,adapter_only_control_executed=False,reason='User confirmed five methods and requested shortest completion time; only missing matched seven-input adapter run is executed',inference_in_simulated_time=False,new_qos_simulation=False,runner_sha256=sha(__file__),raw_archive_recheck_available=raw_current,raw_archive_limit='Raw allocation archives were removed earlier; current samples/origins/manifest hashes verified against saved validated source; no new raw-window reconstruction claimed',prior_raw_validation=validation,python=sys.executable,python_version=sys.version,platform=platform.platform(),mps_fallback=os.environ['PYTORCH_ENABLE_MPS_FALLBACK']))
 print('DATA_HASH_AND_ORIGIN_PASS',flush=True)
def load_model():
 import torch,transformers
 from transformers import AutoModelForCausalLM
 assert torch.backends.mps.is_available() and transformers.__version__=='4.40.1'
 torch.set_num_threads(2)
 manifest=json.loads((P/'checkpoint_manifest.json').read_text());expected=next(v['lfs']['sha256'] for v in manifest['siblings'] if v['rfilename']=='model.safetensors');assert sha(P/'model.safetensors')==expected
 model=AutoModelForCausalLM.from_pretrained(str(P),trust_remote_code=True,local_files_only=True).to('mps').eval();model.requires_grad_(False)
 assert {str(p.device) for p in model.parameters()}=={'mps:0'}
 observed={'linear_calls':0}
 def hook(m,args,out):
  assert args[0].device.type=='mps' and out.device.type=='mps';observed['linear_calls']+=1
 for m in model.modules():
  if isinstance(m,torch.nn.Linear):m.register_forward_hook(hook)
 save('model_provenance.json',dict(model='thuml/sundial-base-128m',revision=manifest['sha'],checkpoint_sha256=expected,source_hashes={p.name:sha(p) for p in P.glob('*.py')},torch=torch.__version__,transformers=transformers.__version__,device='mps',cpu_ml_fallback=False,parameters=sum(p.numel() for p in model.parameters()),cpu_support='Official sampler constructs noise and scalar time on CPU then transfers to MPS; neural computation GPU'))
 return torch,model,observed

def features():
 torch,m,observed=load_model();a=np.load(BASE/'samples.npz');timings={}
 def extract(v):
  # Shape n,50,7; each scalar series retains chronological order.
  v=v.copy();v[:,:,:6]=np.log1p(v[:,:,:6]);mu=v.mean(axis=1);sd=v.std(axis=1);z=(v-mu[:,None,:])/np.maximum(sd[:,None,:],1e-5)
  inp=torch.from_numpy(z.transpose(0,2,1).reshape(-1,50).copy()).to('mps')
  with torch.inference_mode():emb=m.model(input_ids=inp,use_cache=False,return_dict=True).last_hidden_state[:,-1,:]
  assert emb.device.type=='mps' and not emb.requires_grad
  return np.concatenate([emb.cpu().numpy().reshape(len(v),-1),v[:,-1,:],mu,sd],axis=1)
 for s in ['train','val','test']:
  x=a[s+'_x'];extract(x[:2]);torch.mps.synchronize();start=time.perf_counter();parts=[]
  for i in range(0,len(x),16):
   parts.append(extract(x[i:i+16]))
   if i%4096==0:print('features',s,i,len(x),flush=True)
  torch.mps.synchronize();timings[s]=time.perf_counter()-start;f=np.concatenate(parts);assert f.shape==(len(x),5397) and np.isfinite(f).all();np.save(O/f'{s}_features.npy',f)
  assert np.allclose(extract(x[:16]),f[:16],atol=1e-5,rtol=1e-5)
 save('features_metadata.json',dict(status='COMPLETED',device='mps',**observed,extraction_seconds=timings,feature_dimension=5397,batch_size=16,backbone_frozen=True,replay='PASS'))
 print('FEATURES_COMPLETED',timings,flush=True)

def train():
 import torch
 from sklearn.preprocessing import StandardScaler
 assert torch.backends.mps.is_available();torch.set_num_threads(2);a=np.load(BASE/'samples.npz');history=[];predictions={};meta={}
 for name in ['Sundial_adapter']:
  raw={}
  for s in ['train','val','test']:
   if name=='Sundial_adapter':raw[s]=np.load(O/f'{s}_features.npy')
   else:
    v=a[s+'_x'].copy();v[:,:,:6]=np.log1p(v[:,:,:6]);raw[s]=v.reshape(len(v),-1)
  sx=StandardScaler().fit(raw['train']);sy=StandardScaler().fit(a['train_y'][:,None])
  np.savez(O/f'{name}_scalers.npz',x_mean=sx.mean_,x_scale=sx.scale_,y_mean=sy.mean_,y_scale=sy.scale_)
  x={s:torch.from_numpy(sx.transform(v).astype('f')) for s,v in raw.items()};y={s:torch.from_numpy(sy.transform(a[s+'_y'][:,None]).astype('f')) for s in raw}
  def build():return torch.nn.Sequential(torch.nn.Linear(x['train'].shape[1],64),torch.nn.ReLU(),torch.nn.Dropout(.1),torch.nn.Linear(64,1)).to('mps')
  torch.manual_seed(42);m=build();opt=torch.optim.Adam(m.parameters(),lr=.001);best=float('inf');stale=0;start=time.perf_counter();gradient_checked=False
  for epoch in range(50):
   m.train();total=0
   for i in range(0,len(x['train']),256):
    xx=x['train'][i:i+256].to('mps');yy=y['train'][i:i+256].to('mps');opt.zero_grad();out=m(xx);assert out.device.type=='mps';loss=torch.nn.functional.mse_loss(out,yy);loss.backward()
    if not gradient_checked:assert all(p.grad is not None and p.grad.device.type=='mps' for p in m.parameters());gradient_checked=True
    opt.step();total+=loss.item()*len(xx)
   m.eval();val=0
   with torch.inference_mode():
    for i in range(0,len(x['val']),256):val+=torch.nn.functional.mse_loss(m(x['val'][i:i+256].to('mps')),y['val'][i:i+256].to('mps'),reduction='sum').item()
   val/=len(x['val']);history.append(dict(model=name,epoch=epoch+1,train_mse=total/len(x['train']),val_mse=val));pd.DataFrame(history).to_csv(O/'history.csv',index=False)
   if val<best:best=val;stale=0;best_epoch=epoch+1;torch.save(m.state_dict(),O/f'{name}.pt')
   else:stale+=1
   print(name,'epoch',epoch+1,'val',val,flush=True)
   if stale>=7:break
  torch.mps.synchronize();elapsed=time.perf_counter()-start;m.load_state_dict(torch.load(O/f'{name}.pt',map_location='mps',weights_only=True));m.eval()
  def predict(model,v):
   with torch.inference_mode():r=np.concatenate([model(v[i:i+256].to('mps')).cpu().numpy() for i in range(0,len(v),256)])
   return sy.inverse_transform(r.astype('d')).ravel().clip(min=0)
  times={};p={}
  for s in raw:
   start=time.perf_counter();p[s]=predict(m,x[s]);torch.mps.synchronize();times[s]=time.perf_counter()-start;assert np.isfinite(p[s]).all()
  fresh=build();fresh.load_state_dict(torch.load(O/f'{name}.pt',map_location='mps',weights_only=True));fresh.eval();replay=predict(fresh,x['test'][:256]);assert np.allclose(replay,p['test'][:256],atol=.02,rtol=1e-5)
  np.savez_compressed(O/f'{name}_predictions.npz',**p)
  meta[name]=dict(device='mps',torch=torch.__version__,epochs=epoch+1,best_epoch=best_epoch,training_seconds=elapsed,inference_seconds=times,gradient_device_verified=gradient_checked,parameter_devices=sorted({str(p.device) for p in m.parameters()}),parameters=sum(p.numel() for p in m.parameters()),replay_max_difference_bits=float(abs(replay-p['test'][:256]).max()),seed=42,batch=256,optimizer='Adam0.001',loss='train-standardized target MSE')
  save('adapter_metadata.json',meta);print(name,'COMPLETED',flush=True)
if __name__=='__main__':
 if len(sys.argv)>1:globals()[sys.argv[1]]()
 else:
  prepare();features();train()
