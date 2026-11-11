"""Frozen TimesFM-3, matched AF GRANT(t), causal known-at-decision covariates."""
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK']='0'
import sys,json,hashlib,time,subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import torch
H=Path(__file__).resolve().parent
sys.path.insert(0,str(H/'vendor/timesfm3/src'))
from timesfm3 import TimesFM3Forecaster
ROOT=H/'20260912_2134_Python_OPNET_Simulator/runs/casebase_grant_lstm_informer'
BASE=ROOT/'20260915_0631_sliding';AF=ROOT/'20260915_0722_af_only'
OUT=ROOT/'20260916_0704_timesfm3_af'
def sha(p):
 with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(n,v):(OUT/n).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def score(y,p):
 e=np.asarray(p,dtype='d')-y;return dict(mae_bits=float(abs(e).mean()),rmse_bits=float(np.sqrt((e**2).mean())),r2=float(1-(e**2).sum()/((y-y.mean())**2).sum()))
def main():
 assert torch.backends.mps.is_available(),'GPU required'
 torch.set_num_threads(2);torch.manual_seed(42)
 OUT.mkdir(exist_ok=False)
 parent=json.loads((AF/'metadata.json').read_text());source=json.loads((BASE/'metadata.json').read_text())
 assert sha(BASE/'samples.npz')==parent['source_samples_sha256']
 raw=np.load(BASE/'samples.npz');x=raw['test_x'];y=raw['test_y'][:,1].astype('d')
 assert x.shape==(10000,50,7) and np.isfinite(x).all()
 checkpoint=next((H/'vendor/timesfm3_checkpoints/models--google--timesfm-3.0-pytorch/snapshots').iterdir())
 for weight in checkpoint.glob('*.safetensors'):
  assert sha(weight)==weight.resolve().name,'Checkpoint SHA256 mismatch'
 meta=dict(status='RUNNING',model='google/timesfm-3.0-pytorch',checkpoint_revision=checkpoint.name,checkpoint_hashes={p.name:sha(p) for p in checkpoint.iterdir() if p.is_file()},source_commit=subprocess.check_output(['git','-C',str(H/'vendor/timesfm3'),'rev-parse','HEAD'],text=True).strip(),runner_sha256=sha(__file__),device='mps',framework=torch.__version__,cpu_ml_fallback=False,seed=42,target='AF GRANT(t)',domain_horizon=0,model_api_horizon=1,window=50,target_history=49,features=parent['features'],alignment='Target history x[1:,4] = AF GRANT(t-49..t-1); seven known covariates x[:,:].T = REPORT(t-49..t), lagged GRANT(t-50..t-1), observed interval(t-49..t). All covariates known at decision t.',source_samples_sha256=sha(BASE/'samples.npz'),source_metadata_sha256=sha(BASE/'metadata.json'),sample_count=10000,split=parent['split'],scenario_runs=source['runs'],target_measurement_interval=source['target_measurement_interval'],batch_size=16,training='none; frozen zero-shot',preprocessing='model causal normalization; no fitted external scaler',point='median P50; nonnegative clipping',inference_in_simulated_time=False,new_network_simulation=False)
 save('metadata.json',meta)
 model=TimesFM3Forecaster.from_pretrained(str(checkpoint),device='mps',per_core_batch_size=16)
 devices=sorted(set(str(p.device) for p in model.model.parameters()));assert devices==['mps:0'],devices
 observed=[]
 decode=model.model.decode
 def checked_decode(**kwargs):
  assert kwargs['target'].device.type=='mps' and kwargs['past_future_covariates'].device.type=='mps'
  out=decode(**kwargs);assert out.device.type=='mps';observed.append(str(out.device));return out
 model.model.decode=checked_decode
 def infer(a):
  return np.array([o.forecast.item() for o in model.predict_batch([v[1:,4] for v in a],horizon=1,past_future_covariates=[v.T for v in a],make_positive=True,use_symmetric_averaging=False,return_quantiles=False)],dtype='d')
 probe=infer(x[:2]);torch.mps.synchronize();assert np.isfinite(probe).all()
 print('GPU_SMOKE_PASS',devices,probe,flush=True)
 start=time.perf_counter();pred=[]
 for i in range(0,len(x),16):
  pred.extend(infer(x[i:i+16]));
  if i%800==0:print('progress',i,'/',len(x),flush=True)
 p=np.asarray(pred);torch.mps.synchronize();elapsed=time.perf_counter()-start
 replay=infer(x[:16]);assert np.allclose(replay,p[:16],rtol=1e-5,atol=.05)
 assert observed and set(observed)=={'mps:0'};assert np.isfinite(p).all() and (p>=0).all()
 np.savez_compressed(OUT/'predictions.npz',actual=y,TimesFM3=p)
 origins=pd.read_csv(AF/'test_origins.csv');origins['actual_af_bits']=y;origins['timesfm3_af_bits']=p;origins.to_csv(OUT/'predictions.csv',index=False)
 rows=[dict(model='TimesFM-3 zero-shot',outputs='AF only',**score(y,p))]
 for path,label,names in [(BASE,'EF+AF+BE',['LSTM','Informer']),(AF,'AF only',['LSTM','Informer']),(ROOT/'20260916_0655_patchtst_af_only','AF only',['PatchTST'])]:
  z=np.load(path/'predictions.npz');yy=z['actual']; yy=yy[:,1] if yy.shape[1]==3 else yy[:,0];assert np.array_equal(y,yy)
  for n in names:
   q=z[n];q=q[:,1] if q.shape[1]==3 else q[:,0];rows.append(dict(model=n,outputs=label,**score(y,q)))
 for name,q in [('Train mean',np.full_like(y,raw['train_y'][:,1].mean())),('Persistence',x[:,-1,4]),('Current AF REPORT',x[:,-1,1])]:rows.append(dict(model=name,outputs='AF only',**score(y,q)))
 pd.DataFrame(rows).to_csv(OUT/'comparison.csv',index=False)
 meta.update(status='COMPLETED',parameter_devices=devices,forward_input_devices=sorted(set(observed)),gpu_smoke='PASS',inference_seconds=elapsed,replay_max_difference_bits=float(abs(replay-p[:16]).max()))
 save('metadata.json',meta);print(pd.DataFrame(rows).to_string(index=False),flush=True)
if __name__=='__main__':main()
