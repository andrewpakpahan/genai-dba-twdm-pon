"""Historical Keras LSTM and ProbSparse Informer, GPU AF-only current-GRANT training."""
from pathlib import Path
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK']='0';os.environ['TF_CPP_MIN_LOG_LEVEL']='2';os.environ['MPLCONFIGDIR']='/tmp/casebase-sliding-mpl'
import json,time,random,hashlib,gc,sys
import numpy as np,pandas as pd
H=Path(__file__).resolve().parent
BASE=H/'20260912_2134_Python_OPNET_Simulator/runs/casebase_grant_lstm_informer/20260915_0631_sliding'
O=BASE.parent/'20260915_0722_af_only'
def save(name,v):(O/name).write_text(json.dumps(v,indent=2))
def sha(p):
 with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def metrics(y,p,name,group='all'):
 rows=[]
 for i,q in enumerate(['AF']):
  err=p[:,i]-y[:,i];den=np.square(y[:,i]-y[:,i].mean()).sum()
  rows.append(dict(model=name,group=group,traffic_class=q,samples=len(y),mae_bits=float(np.abs(err).mean()),rmse_bits=float(np.sqrt(np.square(err).mean())),r2=float(1-np.square(err).sum()/den) if den>0 else None))
 return rows
def main():
 import tensorflow as tf
 import torch
 from importlib import import_module
 from sklearn.preprocessing import StandardScaler
 inf=import_module('20260730_2000_run_informer_grant_7features')
 assert torch.backends.mps.is_available();assert tf.config.list_physical_devices('GPU')
 # TensorFlow Metal requires CPU kernels for seed/shape bookkeeping. Audit actual neural-kernel placement below.
 tf.config.set_soft_device_placement(True);tf.debugging.set_log_device_placement(True);torch.set_num_threads(2)
 a=np.load(BASE/'samples.npz');raw={k:(a[k][:,1:2] if k.endswith('_y') else a[k]) for k in a.files};m=json.loads((O/'metadata.json').read_text());m['status']='TRAINING'
 m['training']=dict(model_seed=42,maximum_epochs=50,patience=7,batch_size=256,shuffle=False,gradient_clipping=False,latest_report_included=True,LSTM=dict(framework=tf.__version__,device='/GPU:0',optimizer='Keras RMSprop default rho=.9 epsilon=1e-7',lr=.001,preprocessing='log1p first6features and targets; timing untouched',architecture='Keras LSTM32/dropout.1/LSTM32/dropout.1/Dense1'),Informer=dict(framework=torch.__version__,device='mps',optimizer='Adam',lr=.001,preprocessing='train-only StandardScaler',architecture='ProbSparse,d_model64,heads4,FF128,enc2,dec1,dropout.1',evaluation_attention_seed=4242))
 m['tensorflow_support_ops']='CPU seed and shape bookkeeping permitted; explicit GPU scope and neural-kernel placement audited; weights/output GPU verified'
 m['trainer_source_sha256']=sha(Path(__file__));m['vendor_source_sha256']={str(p.relative_to(H)):sha(p) for p in sorted((H/'vendor/PatchTST/PatchTST_supervised').glob('**/*.py'))};save('metadata.json',m)
 hist=[];timings={};preds={};verification={}
 # Historical LSTM uses actual Keras implementation, initializer and RMSprop math.
 def build_lstm():
  model=tf.keras.Sequential([tf.keras.layers.Input((50,7)),tf.keras.layers.LSTM(32,return_sequences=True),tf.keras.layers.Dropout(.1),tf.keras.layers.LSTM(32),tf.keras.layers.Dropout(.1),tf.keras.layers.Dense(1)])
  model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=.001),loss='mse',jit_compile=False);return model
 x={s:raw[s+'_x'].copy() for s in ['train','val','test']}
 for v in x.values():v[:,:,:6]=np.log1p(v[:,:,:6])
 y={s:np.log1p(raw[s+'_y']) for s in x}
 tf.keras.utils.set_random_seed(42)
 with tf.device('/GPU:0'):
  model=build_lstm();probe=model(tf.convert_to_tensor(x['train'][:2]));assert 'GPU:0' in probe.device
  weight_devices=[v.value.handle.device for v in model.weights];assert all('GPU:0' in d for d in weight_devices)
  # Backprop smoke occurs on a separate model, so actual initialization is unaffected.
  smoke=build_lstm();smoke.train_on_batch(x['train'][:4],y['train'][:4]);del smoke
 tf.debugging.set_log_device_placement(False);tf.config.set_soft_device_placement(True)
 tf.keras.utils.set_random_seed(42)
 with tf.device('/GPU:0'):model=build_lstm()
 class History(tf.keras.callbacks.Callback):
  def on_epoch_end(self,epoch,logs=None):
   row=dict(model='LSTM',epoch=epoch+1,train_loss=float(logs['loss']),val_loss=float(logs['val_loss']));hist.append(row);pd.DataFrame(hist).to_csv(O/'history.csv',index=False);print('LSTM',epoch+1,'val',row['val_loss'],flush=True)
 start=time.perf_counter()
 with tf.device('/GPU:0'):
  model.fit(x['train'],y['train'],validation_data=(x['val'],y['val']),epochs=50,batch_size=256,shuffle=False,verbose=0,callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=7,restore_best_weights=True),History()])
  _=model(tf.convert_to_tensor(x['val'][:1])).numpy()
 train_seconds=time.perf_counter()-start
 model.save_weights(O/'lstm.weights.h5')
 def keras_predict(model):
  with tf.device('/GPU:0'):p=np.concatenate([model(tf.convert_to_tensor(x['test'][i:i+256]),training=False).numpy() for i in range(0,len(x['test']),256)])
  return np.expm1(p.astype('float64')).clip(min=0)
 start=time.perf_counter();preds['LSTM']=keras_predict(model);infer=time.perf_counter()-start
 with tf.device('/GPU:0'):fresh=build_lstm();fresh.load_weights(O/'lstm.weights.h5')
 replay=keras_predict(fresh);err=float(np.max(np.abs(preds['LSTM']-replay)));assert np.allclose(preds['LSTM'],replay,rtol=1e-5,atol=.01)
 verification['LSTM_reload_max_difference_bits']=err;verification['LSTM_tensor_device']=probe.device;verification['LSTM_weight_devices']=weight_devices
 h=[r for r in hist if r['model']=='LSTM'];timings['LSTM']=dict(training_seconds=train_seconds,test_inference_seconds=infer,epochs=len(h),best_epoch=min(h,key=lambda v:v['val_loss'])['epoch'],device='TensorFlow Metal GPU:0',parameters=model.count_params());save('timings.json',timings)
 del model,fresh,x,y;tf.keras.backend.clear_session();gc.collect()
 # Historical Informer factory only: never reuse later-edited t+1 dataset code.
 device=torch.device('mps');sx=StandardScaler().fit(raw['train_x'].reshape(-1,7));sy=StandardScaler().fit(raw['train_y']);np.savez(O/'scalers.npz',x_mean=sx.mean_,x_scale=sx.scale_,y_mean=sy.mean_,y_scale=sy.scale_)
 xx={s:torch.from_numpy(sx.transform(raw[s+'_x'].reshape(-1,7)).reshape(raw[s+'_x'].shape).astype('f')) for s in ['train','val','test']};yy={s:torch.from_numpy(sy.transform(raw[s+'_y']).astype('f')) for s in xx}
 def build_inf():return inf.build_model(feature_count=7,target_count=1,pred_len=1,d_model=64,n_heads=4,d_ff=128,e_layers=2,d_layers=1,dropout=.1).to(device)
 def forward(model,x):return inf._batch_predict(model,x,label_len=25,device=device)
 def predict(model,split):
  rng=torch.get_rng_state();gen=torch.Generator().manual_seed(4242);torch.set_rng_state(gen.get_state());model.eval()
  with torch.no_grad():p=np.concatenate([forward(model,v.to(device)).cpu().numpy() for v in xx[split].split(256)])
  torch.set_rng_state(rng);return p
 random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.mps.manual_seed(42);model=build_inf();opt=torch.optim.Adam(model.parameters(),lr=.001);best=float('inf');bad=0;torch.mps.synchronize();start=time.perf_counter()
 for ep in range(1,51):
  model.train();losses=[]
  for xb,yb in zip(xx['train'].split(256),yy['train'].split(256)):
   opt.zero_grad();p=forward(model,xb.to(device));loss=(p-yb.to(device)).square().mean();assert torch.isfinite(loss);loss.backward();opt.step();losses.append(float(loss.item()))
  p=predict(model,'val');vl=float(np.mean([np.square(p[i:i+256]-yy['val'][i:i+256].numpy()).mean() for i in range(0,len(p),256)]));hist.append(dict(model='Informer',epoch=ep,train_loss=float(np.mean(losses)),val_loss=vl));pd.DataFrame(hist).to_csv(O/'history.csv',index=False);print('Informer',ep,'val',vl,flush=True)
  if vl<best:best=vl;bad=0;bestepoch=ep;torch.save(model.state_dict(),O/'informer.pt')
  else:bad+=1
  if bad>=7:break
 torch.mps.synchronize();train_seconds=time.perf_counter()-start;model.load_state_dict(torch.load(O/'informer.pt',map_location=device,weights_only=True));start=time.perf_counter();preds['Informer']=sy.inverse_transform(predict(model,'test')).clip(min=0).astype('float64');torch.mps.synchronize();infer=time.perf_counter()-start
 fresh=build_inf();fresh.load_state_dict(torch.load(O/'informer.pt',map_location=device,weights_only=True));replay=sy.inverse_transform(predict(fresh,'test')).clip(min=0);err=float(np.abs(preds['Informer']-replay).max());assert np.allclose(preds['Informer'],replay,atol=.01,rtol=1e-5)
 verification['Informer_reload_max_difference_bits']=err;timings['Informer']=dict(training_seconds=train_seconds,test_inference_seconds=infer,best_epoch=bestepoch,epochs=ep,device=str(next(model.parameters()).device),parameters=sum(v.numel() for v in model.parameters()))
 y=raw['test_y'].astype('float64');r=raw['test_x'][:,-1,:3].astype('float64');orig=pd.read_csv(O/'test_origins.csv');cap=orig.cap_bits.to_numpy()[:,None];preds['Persistence']=raw['test_x'][:,-1,4:5].astype('float64');preds['Training mean']=np.broadcast_to(raw['train_y'].mean(0,dtype=np.float64),y.shape);preds['Known-rule allocator']=np.minimum(r,np.maximum(cap-np.cumsum(r,axis=1)+r,0))[:,1:2]
 rows=[]
 for name,p in preds.items():
  assert np.isfinite(p).all();rows+=metrics(y,p,name)
  for case,ids in orig.groupby('case').groups.items():rows+=metrics(y[ids],p[ids],name,case)
 np.savez_compressed(O/'predictions.npz',actual=y,**preds);pd.DataFrame(rows).to_csv(O/'metrics.csv',index=False);save('timings.json',timings);save('verification.json',verification);m['status']='COMPLETED';m['timings']=timings;save('metadata.json',m);print('COMPLETED',O,flush=True)
if __name__=='__main__':main()
