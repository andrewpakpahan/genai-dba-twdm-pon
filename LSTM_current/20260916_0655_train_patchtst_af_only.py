"""Matched current-cycle AF GRANT regression with vendored PatchTST; GPU only."""
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'
import hashlib, json, random, sys, time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler

H = Path(__file__).resolve().parent
ROOT = H/'20260912_2134_Python_OPNET_Simulator/runs/casebase_grant_lstm_informer'
BASE = ROOT/'20260915_0631_sliding'
AF = ROOT/'20260915_0722_af_only'
OUT = ROOT/'20260916_0655_patchtst_af_only'
VENDOR = H/'vendor/PatchTST/PatchTST_supervised'
sys.path.insert(0, str(VENDOR))
from models.PatchTST import Model
CONFIG = dict(enc_in=7, seq_len=50, pred_len=1, e_layers=2, n_heads=4,
              d_model=64, d_ff=128, dropout=.1, fc_dropout=.1, head_dropout=0.,
              individual=False, patch_len=10, stride=5, padding_patch='end',
              revin=False, affine=False, subtract_last=False, decomposition=False, kernel_size=25)
class PatchTSTAF(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = Model(SimpleNamespace(**CONFIG)).float()
        self.target_head = nn.Linear(7, 1)
    def forward(self, x):
        return self.target_head(self.backbone(x)[:, 0, :])
def sha(p):
    with open(p, 'rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()
def save(name, obj): (OUT/name).write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')
def scores(y,p):
    y=np.asarray(y,dtype=np.float64);p=np.asarray(p,dtype=np.float64)
    e=p-y;den=((y-y.mean())**2).sum()
    return dict(mae_bits=float(abs(e).mean()),rmse_bits=float(np.sqrt((e**2).mean())),r2=float(1-(e**2).sum()/den) if den>0 else None)
def main():
    if not torch.backends.mps.is_available(): raise RuntimeError('MPS GPU required; CPU fallback prohibited')
    OUT.mkdir(exist_ok=False)
    device=torch.device('mps');torch.set_num_threads(2)
    parent=json.loads((AF/'metadata.json').read_text())
    sample_hash=sha(BASE/'samples.npz')
    assert sample_hash==parent['source_samples_sha256']
    raw=np.load(BASE/'samples.npz')
    sx=StandardScaler().fit(raw['train_x'].reshape(-1,7));sy=StandardScaler().fit(raw['train_y'][:,1:2])
    np.savez(OUT/'scalers.npz',x_mean=sx.mean_,x_scale=sx.scale_,y_mean=sy.mean_,y_scale=sy.scale_)
    xx={s:torch.from_numpy(sx.transform(raw[s+'_x'].reshape(-1,7)).reshape(-1,50,7).astype('f')) for s in ['train','val','test']}
    yy={s:torch.from_numpy(sy.transform(raw[s+'_y'][:,1:2]).astype('f')) for s in xx}
    assert [len(xx[s]) for s in xx]==[20000,10000,10000]
    for s in xx:
        assert torch.isfinite(xx[s]).all() and torch.isfinite(yy[s]).all()
        (OUT/(s+'_origins.csv')).write_bytes((AF/(s+'_origins.csv')).read_bytes())
    # Probe forward/backward on GPU, then reset seeds and instantiate actual model.
    probe=PatchTSTAF().to(device);z=probe(xx['train'][:4].to(device))
    assert z.shape==(4,1) and z.device.type=='mps'
    z.square().mean().backward()
    grads=[p.grad for p in probe.parameters() if p.requires_grad]
    assert all(g is not None and g.device.type=='mps' and torch.isfinite(g).all() for g in grads)
    assert sum(g.abs().sum().item() for g in grads)>0
    del probe,z,grads
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.mps.manual_seed(42)
    model=PatchTSTAF().to(device);opt=torch.optim.Adam(model.parameters(),lr=.001)
    meta=dict(status='TRAINING',target='AF GRANT(t)',latest_report_included=True,
              source_samples=str(BASE/'samples.npz'),source_samples_sha256=sample_hash,
              source_metadata=str(AF/'metadata.json'),source_metadata_sha256=sha(AF/'metadata.json'),
              scenario_source='Same 64 saved Python Case_Base runs as AF-only LSTM/Informer',
              raw_alignment=parent['raw_alignment'],features=parent['features'],split=parent['split'],
              sample_counts={s:len(xx[s]) for s in xx},window=50,candidate_stride=1,
              sampling='Inherited capped evenly spaced samples, not all stride-one candidates',
              model='Vendored PatchTST channel-independent backbone with learned 7-to-1 AF head',config=CONFIG,
              seed=42,batch_size=256,max_epochs=50,patience=7,shuffle=False,gradient_clipping=False,
              optimizer='Adam',learning_rate=.001,preprocessing='Training-only StandardScaler on inputs and AF target',
              framework=torch.__version__,device=str(device),gpu_probe='forward and backward PASS',cpu_ml_fallback=False,
              parameters=sum(p.numel() for p in model.parameters()),runner_sha256=sha(Path(__file__)),
              vendor_sha256={str(p.relative_to(H)):sha(p) for p in [VENDOR/'models/PatchTST.py',*sorted((VENDOR/'layers').glob('PatchTST*.py')),VENDOR/'layers/RevIN.py']},
              inference_in_simulated_time=False,new_network_simulation=False)
    save('metadata.json',meta)
    def predict(m,split):
        m.eval()
        with torch.no_grad():return np.concatenate([m(b.to(device)).cpu().numpy() for b in xx[split].split(256)])
    history=[];best=float('inf');bad=0;best_epoch=0
    torch.mps.synchronize();start=time.perf_counter()
    for ep in range(1,51):
        model.train();losses=[]
        for xb,yb in zip(xx['train'].split(256),yy['train'].split(256)):
            opt.zero_grad(set_to_none=True);p=model(xb.to(device));loss=(p-yb.to(device)).square().mean()
            assert torch.isfinite(loss);loss.backward();opt.step();losses.append(loss.item())
        p=predict(model,'val')
        # Same validation batch-mean objective as previous Informer run.
        vl=float(np.mean([np.square(p[i:i+256]-yy['val'][i:i+256].numpy()).mean() for i in range(0,len(p),256)]))
        history.append(dict(epoch=ep,train_loss=float(np.mean(losses)),val_loss=vl))
        pd.DataFrame(history).to_csv(OUT/'history.csv',index=False)
        print(f'epoch={ep} val_loss={vl:.8f}',flush=True)
        if vl<best:
            best=vl;bad=0;best_epoch=ep;torch.save(model.state_dict(),OUT/'patchtst.pt')
        else:bad+=1
        if bad>=7:break
    torch.mps.synchronize();train_sec=time.perf_counter()-start
    model.load_state_dict(torch.load(OUT/'patchtst.pt',map_location=device,weights_only=True))
    start=time.perf_counter();p=sy.inverse_transform(predict(model,'test')).clip(min=0).astype('float64');torch.mps.synchronize();infer_sec=time.perf_counter()-start
    fresh=PatchTSTAF().to(device);fresh.load_state_dict(torch.load(OUT/'patchtst.pt',map_location=device,weights_only=True))
    replay=sy.inverse_transform(predict(fresh,'test')).clip(min=0)
    diff=float(np.max(abs(replay-p)));assert diff==0
    actual=raw['test_y'][:,1:2].astype('float64');assert np.isfinite(p).all()
    np.savez_compressed(OUT/'predictions.npz',actual=actual,PatchTST=p)
    origins=pd.read_csv(OUT/'test_origins.csv');rows=[dict(model='PatchTST',group='all',traffic_class='AF',samples=len(p),**scores(actual,p))]
    for case,ids in origins.groupby('case').groups.items():rows.append(dict(model='PatchTST',group=case,traffic_class='AF',samples=len(ids),**scores(actual[ids],p[ids])))
    pd.DataFrame(rows).to_csv(OUT/'metrics.csv',index=False)
    comparison=[]
    for path,label in [(BASE,'EF+AF+BE'),(AF,'AF only')]:
        prev=np.load(path/'predictions.npz');yy0=prev['actual'];yy0=yy0[:,1:2] if yy0.shape[1]==3 else yy0
        assert np.array_equal(yy0,actual)
        for name in ['LSTM','Informer']:
            pp=prev[name];pp=pp[:,1:2] if pp.shape[1]==3 else pp
            comparison.append(dict(model=name,outputs=label,**scores(actual,pp)))
    comparison.append(dict(model='PatchTST',outputs='AF only',**scores(actual,p)))
    pd.DataFrame(comparison).to_csv(OUT/'comparison.csv',index=False)
    save('verification.json',dict(sample_hash_match=True,all_comparator_targets_equal=True,gpu_forward_backward=True,
         parameter_devices=sorted(set(str(v.device) for v in model.parameters())),checkpoint_replay_max_difference_bits=diff,finite_predictions=True))
    meta.update(status='COMPLETED',epochs=ep,best_epoch=best_epoch,best_val_loss=best,training_seconds=train_sec,test_inference_seconds=infer_sec)
    save('metadata.json',meta)
    print(pd.DataFrame(comparison).to_string(index=False),flush=True)
    print('COMPLETED',OUT,flush=True)
if __name__=='__main__':main()
