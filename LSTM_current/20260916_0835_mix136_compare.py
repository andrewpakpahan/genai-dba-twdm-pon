"""Seed-held-out, two-wavelength current AF GRANT comparison; GPU mandatory."""
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK']='0'
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ['MPLCONFIGDIR']='/private/tmp/mix136-mpl'
from pathlib import Path
import json,hashlib,time,sys,subprocess
import numpy as np
import pandas as pd
H=Path(__file__).resolve().parent
DATA=H/'20260912_2134_Python_OPNET_Simulator/runs/mix_2w_10s/20260916_0712'
O=DATA.parent/'20260916_0835_mix136_comparison'
def sha(p):
 with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(n,v):(O/n).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def prepare():
 O.mkdir(exist_ok=False); arrays={}; provenance=[]
 for split,seed,count in [('train',128,32),('val',129,16),('test',130,16)]:
  xx=[]; yy=[]; origins=[]
  for load in range(10,101,10):
   d=DATA/f'mix136_load{load:03d}_seed{seed}';m=json.loads((d/'manifest.json').read_text());c=json.loads((d/'settings.json').read_text())
   assert m['status']=='COMPLETED' and m['seed']==seed and m['mix']=='136' and m['load_pct']==load
   assert c['channels']==2 and c['onu_count']==64 and c['duration']==10 and c['base_cap']==46875 and c['channel_rates']==[1e9,1e9]
   for n in ['allocation.npz','settings.json']:assert sha(d/n)==m['artifacts'][n]
   a=np.load(d/'allocation.npz');r=a['report'];g=a['grant'];p=a['previous_grant'];t=a['time'];cy=a['cycle'];dt=a['report_interval']
   assert r.shape==g.shape==p.shape==(len(t),64,3) and np.array_equal(p[1:],g[:-1]) and not p[0].any()
   assert (np.diff(cy)==1).all() and (np.diff(t)>0).all() and np.isfinite(r).all() and (r>=0).all() and (g>=0).all()
   eligible=np.flatnonzero((t>=3)&(t<10)&(np.arange(len(t))>=49));indices=eligible[np.linspace(0,len(eligible)-1,count,dtype=int)]
   assert len(np.unique(indices))==count
   for idx in indices:
    for onu in range(64):
     x=np.concatenate([r[idx-49:idx+1,onu],p[idx-49:idx+1,onu],dt[idx-49:idx+1,onu,None]],axis=1).astype('f')
     xx.append(x);yy.append(g[idx,onu,1]);origins.append(dict(run=d.name,load=load,seed=seed,onu=onu,index=int(idx),cycle=int(cy[idx]),time=float(t[idx])))
   provenance.append(dict(run=d.name,split=split,eligible_origins=len(eligible)*64,selected=count*64,allocation_sha256=sha(d/'allocation.npz'),manifest_sha256=sha(d/'manifest.json')))
   print('prepared',d.name,flush=True)
  arrays[split+'_x']=np.asarray(xx);arrays[split+'_y']=np.asarray(yy,dtype='d');pd.DataFrame(origins).to_csv(O/f'{split}_origins.csv',index=False)
 np.savez_compressed(O/'samples.npz',**arrays)
 save('protocol.json',dict(status='PREPARED',target='AF GRANT(t)',features=['EF REPORT','AF REPORT','BE REPORT','previous EF GRANT','previous AF GRANT','previous BE GRANT','report interval seconds'],window=50,split='independent seed128 train /129 validation /130 test; no cross-run or ONU windows',sampling='32 train and16 validation/test evenly spaced eligible target cycles per ONU per load, chronological within run',target_window=[3,10],scenario=dict(mix='136',loads=list(range(10,101,10)),onus=64,wavelengths=2,rate_per_wavelength=1e9,data_budget_seconds=.0015,duration_seconds=10),runs=provenance,samples_sha256=sha(O/'samples.npz'),runner_sha256=sha(__file__),inference_in_simulated_time=False,new_qos_simulation=False))
def lstm():
 import tensorflow as tf
 assert tf.config.list_physical_devices('GPU')
 tf.config.set_soft_device_placement(True);tf.keras.utils.set_random_seed(42)
 a=np.load(O/'samples.npz');x={s:a[s+'_x'].copy() for s in ['train','val','test']};y={s:np.log1p(a[s+'_y'])[:,None] for s in x}
 for v in x.values():v[:,:,:6]=np.log1p(v[:,:,:6])
 def build():
  m=tf.keras.Sequential([tf.keras.layers.Input((50,7)),tf.keras.layers.LSTM(32,return_sequences=True),tf.keras.layers.Dropout(.1),tf.keras.layers.LSTM(32),tf.keras.layers.Dropout(.1),tf.keras.layers.Dense(1)])
  m.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=.001),loss='mse',jit_compile=False);return m
 with tf.device('/GPU:0'):
  m=build();probe=m(tf.convert_to_tensor(x['train'][:2]));assert 'GPU:0' in probe.device
  devices=[v.value.handle.device for v in m.weights];assert all('GPU:0' in v for v in devices)
 tf.debugging.set_log_device_placement(True)
 with tf.device('/GPU:0'):
  smoke=build();smoke.train_on_batch(x['train'][:4],y['train'][:4]);del smoke
 tf.debugging.set_log_device_placement(False);tf.config.set_soft_device_placement(True)
 tf.keras.utils.set_random_seed(42)
 with tf.device('/GPU:0'):m=build()
 hist=[]
 class History(tf.keras.callbacks.Callback):
  def on_epoch_end(self,epoch,logs=None):
   hist.append(dict(epoch=epoch+1,**{k:float(v) for k,v in logs.items()}));pd.DataFrame(hist).to_csv(O/'history.csv',index=False);print('epoch',epoch+1,logs,flush=True)
 start=time.perf_counter()
 with tf.device('/GPU:0'):
  m.fit(x['train'],y['train'],validation_data=(x['val'],y['val']),epochs=50,batch_size=256,shuffle=False,verbose=0,callbacks=[tf.keras.callbacks.EarlyStopping(patience=7,restore_best_weights=True),History()])
 train=time.perf_counter()-start;m.save_weights(O/'lstm.weights.h5')
 def predict(model,v):
  with tf.device('/GPU:0'):
   return np.maximum(0,np.expm1(np.concatenate([model(tf.convert_to_tensor(v[i:i+256]),training=False).numpy() for i in range(0,len(v),256)]).ravel().astype('d')))
 pred={};timings={}
 for s in x:
  start=time.perf_counter();pred[s]=predict(m,x[s]);timings[s]=time.perf_counter()-start
 with tf.device('/GPU:0'):fresh=build();fresh.load_weights(O/'lstm.weights.h5')
 replay=predict(fresh,x['test'][:256]);assert np.allclose(replay,pred['test'][:256],atol=.01,rtol=1e-5)
 np.savez_compressed(O/'lstm_predictions.npz',**pred)
 save('lstm_metadata.json',dict(device=probe.device,weight_devices=devices,framework=tf.__version__,seed=42,architecture='Keras LSTM32/dropout0.1/LSTM32/dropout0.1/Dense1',optimizer='RMSprop lr0.001 rho0.9 epsilon1e-7',preprocessing='log1p first6features and target; interval unchanged; no fitted scaler',epochs=len(hist),best_epoch=min(hist,key=lambda v:v['val_loss'])['epoch'],training_seconds=train,inference_seconds=timings,batch_size=256,parameters=m.count_params(),reload_max_difference_bits=float(abs(replay-pred['test'][:256]).max()),cpu_ml_fallback=False))
def timesfm():
 import torch
 assert torch.backends.mps.is_available();torch.set_num_threads(2);torch.manual_seed(42)
 sys.path.insert(0,str(H/'vendor/timesfm3/src'));from timesfm3 import TimesFM3Forecaster
 checkpoint=H/'vendor/timesfm3_checkpoints/models--google--timesfm-3.0-pytorch/snapshots/43046b85ec22d584a13f8098c2ed39c889e129c2'
 for p in checkpoint.glob('*.safetensors'):assert sha(p)==p.resolve().name
 model=TimesFM3Forecaster.from_pretrained(str(checkpoint),device='mps',per_core_batch_size=16)
 devices=sorted(set(str(p.device) for p in model.model.parameters()));assert devices==['mps:0'];observed=[];decode=model.model.decode
 def checked(**kwargs):
  assert kwargs['target'].device.type==kwargs['past_future_covariates'].device.type=='mps'
  out=decode(**kwargs);assert out.device.type=='mps';observed.append(str(out.device));return out
 model.model.decode=checked
 def infer(x):return np.array([o.forecast.item() for o in model.predict_batch([v[1:,4] for v in x],horizon=1,past_future_covariates=[v.T for v in x],make_positive=True,use_symmetric_averaging=False,return_quantiles=False)],dtype='d')
 a=np.load(O/'samples.npz');infer(a['test_x'][:2]);torch.mps.synchronize();pred={};timings={}
 for s in ['test']:
  x=a[s+'_x'];start=time.perf_counter();parts=[]
  for i in range(0,len(x),16):
   parts.append(infer(x[i:i+16]))
   if i%1600==0:print('TimesFM3',s,i,len(x),flush=True)
  torch.mps.synchronize();timings[s]=time.perf_counter()-start;pred[s]=np.concatenate(parts);assert np.isfinite(pred[s]).all()
  np.savez_compressed(O/'timesfm3_predictions.npz',**pred)
 replay=infer(a['test_x'][:16]);assert np.allclose(replay,pred['test'][:16],atol=.05,rtol=1e-5)
 save('timesfm3_metadata.json',dict(model='google/timesfm-3.0-pytorch',checkpoint_revision=checkpoint.name,checkpoint_hashes={p.name:sha(p) for p in checkpoint.iterdir() if p.is_file()},source_commit=subprocess.check_output(['git','-C',str(H/'vendor/timesfm3'),'rev-parse','HEAD'],text=True).strip(),device='mps',parameter_devices=devices,forward_devices=sorted(set(observed)),framework=torch.__version__,cpu_ml_fallback=False,training='none; frozen pretrained',target_history='49 AF grants t-49..t-1',covariates='all seven50-step causal features t-49..t; lagged grants t-50..t-1',point='P50; nonnegative',api_horizon=1,domain_horizon=0,batch_size=16,inference_seconds=timings,replay_max_difference_bits=float(abs(replay-pred['test'][:16]).max())))
if __name__=='__main__':globals()[sys.argv[1]]()
