"""GPU models calibrated to actual post-REPORT AF bits at a known planned horizon.

TimesFM's causal proxy remains an imperfect observable input. Its train-only
GPU affine calibration and the supervised LSTM use identical actual labels.
The planned horizon is known at the decision; actual packet futures are labels
only and cannot enter this module's runtime predictor API.
"""
from __future__ import annotations
import argparse
import csv
import importlib
import json
import math
import os
from pathlib import Path
import shutil
import time
import numpy as np

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'
COMMON = importlib.import_module('20260917_1246_future_af_predictor')
SIM, DATA = COMMON.SIM, COMMON.DATA
OUT = SIM / 'runs/ponsim_future_af/20260917_1329/model'
PREVIOUS = SIM / 'runs/ponsim_future_af/20260917_1315/model'
ORDER_OFFSETS = [0,1,4,8,16,24,28,31]
RIDGES = [0., .01, .1, 1., 10.]
sha, write, stamp, gpu = COMMON.sha, COMMON.write, COMMON.stamp, COMMON.gpu


def prepare():
    OUT.mkdir(parents=True, exist_ok=False)
    old = dict(np.load(PREVIOUS/'samples.npz'))
    old_protocol = json.loads((PREVIOUS/'protocol.json').read_text())
    arrays, provenance, validation = {}, [], []
    for split, seed, per_onu in [('train',128,16),('val',129,8),('test',130,8)]:
        d = DATA / f'mix136_load090_seed{seed}'
        manifest = json.loads((d/'manifest.json').read_text())
        cfg = json.loads((d/'settings.json').read_text())
        assert manifest['status']=='COMPLETED' and manifest['seed']==seed and manifest['mix']=='136'
        assert manifest['load_pct']==90 and cfg['channels']==2 and cfg['onu_count']==64
        hashes = {}
        previous_source = next(z for z in old_protocol['sources'] if z['split']==split)
        for name in ['allocation.npz','offered.npz','control.npz','settings.json']:
            hashes[name] = sha(d/name)
            assert hashes[name] == manifest['artifacts'][name] == previous_source['hashes'][name]
        a,ctl,offered = (dict(np.load(d/name)) for name in ['allocation.npz','control.npz','offered.npz'])
        r,p,g,dt,t = (a[k] for k in ['report','previous_grant','grant','report_interval','time'])
        n = len(t)
        assert np.array_equal(a['cycle'],np.arange(1,n+1)) and np.array_equal(p[1:],g[:-1])
        reconstructed=np.zeros_like(g);left=np.full((n,64),cfg['base_cap'])
        for q in range(3):
            reconstructed[:,:,q]=np.minimum(r[:,:,q],left);left-=reconstructed[:,:,q]
        assert np.allclose(reconstructed,g,rtol=0,atol=1e-7)
        report_map={(int(z['cycle']),int(z['onu'])):z for z in ctl['reports']}
        gates={(int(z['cycle'])-1,int(z['onu'])):z for z in ctl['gates']}
        assert np.allclose(ctl['dba_times'][:,0],t,rtol=0,atol=1e-12)
        proxy=COMMON.proxy_rates(r,p,dt)
        oldrows=list(csv.DictReader((PREVIOUS/f'{split}_origins.csv').open()))
        assert len(oldrows)==64*per_onu
        ordinal=np.zeros(64,int);rows=[];cache={};fallback=0
        def schedule(index,offset):
            key=(index,offset)
            if key not in cache:
                starts=np.empty(64)
                for channel in [0,1]:
                    order=list(range(channel*32,(channel+1)*32))
                    order=order[offset:]+order[:offset]
                    cursor=float(ctl['dba_times'][index,1])+cfg['channel_offsets'][channel]
                    rate=cfg['channel_rates'][channel]
                    for onu in order:
                        cursor=math.ceil(cursor*1e9+.5)/1e9
                        starts[onu]=cursor+reconstructed[index,onu,0]/rate
                        cursor+=reconstructed[index,onu].sum()/rate+cfg['report_tx_time']+cfg['guard_time']
                cache[key]=starts
            return cache[key]
        for j,row in enumerate(oldrows):
            i,u=int(row['allocation_index']),int(row['onu'])
            rr=report_map[(i,u)];begin=float(rr['time'])
            assert rr['receive_time']<=t[i]+1e-12 and begin==float(row['report_snapshot'])
            assert np.array_equal(rr['bits'],r[i,u])
            baseline=float(schedule(i,0)[u])
            assert abs(baseline-float(gates[(i,u)]['starts'][1]))<3e-12
            requested=ORDER_OFFSETS[int(ordinal[u])%len(ORDER_OFFSETS)];ordinal[u]+=1
            used=requested;end=float(schedule(i,used)[u])
            if end>=cfg['duration']:
                used=0;end=baseline;fallback+=1
            assert 3<=begin<end<10 and end>t[i]
            mask=(offered['onu']==u)&(offered['queue']==1)&(offered['time']>begin)&(offered['time']<=end)
            y=float(offered['bits'][mask].sum())
            xx=np.column_stack([r[i-49:i+1,u],p[i-49:i+1,u],dt[i-49:i+1,u]]).astype('f')
            pp=proxy[i-49:i+1,u].astype('f')
            np.testing.assert_array_equal(xx,old[split+'_x'][j])
            np.testing.assert_array_equal(pp,old[split+'_proxy'][j])
            rows.append(dict(seed=seed,onu=u,allocation_index=i,cycle=int(a['cycle'][i]),decision_time=t[i],
                report_snapshot=begin,planned_af_start=end,horizon_seconds=end-begin,actual_AF_bits=y,
                requested_order_offset=requested,order_offset=used,source_window_fallback=used!=requested,
                baseline_planned_af_start=baseline))
        assert (ordinal==per_onu).all()
        h=np.asarray([z['horizon_seconds'] for z in rows]);y=np.asarray([z['actual_AF_bits'] for z in rows])
        arrays.update({split+'_x':old[split+'_x'],split+'_proxy':old[split+'_proxy'],split+'_h':h,split+'_y_bits':y})
        with (OUT/f'{split}_origins.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        provenance.append(dict(split=split,seed=seed,path=str(d),hashes=hashes,manifest_sha256=sha(d/'manifest.json')))
        validation.append(dict(split=split,samples=len(rows),all_64_onus=True,per_onu=per_onu,
            report_alignment=True,previous_grant_alignment=True,current_grants_reconstructed_from_report=True,
            causal_counterfactual_schedule_checked=True,all_labels_direct_event_sums=True,
            old_x_proxy_exact_equal=True,old_origin_rows_exact_equal=True,source_window_fallback_count=fallback,
            min_horizon_seconds=float(h.min()),max_horizon_seconds=float(h.max()),horizon_std_seconds=float(h.std()),
            requested_order_counts={str(k):sum(z['requested_order_offset']==k for z in rows) for k in ORDER_OFFSETS},
            effective_order_counts={str(k):sum(z['order_offset']==k for z in rows) for k in ORDER_OFFSETS},
            label_ends_after_decision=True))
        print('Prepared',split,len(rows),'horizon',float(h.min()),float(h.max()),'fallback',fallback,flush=True)
    np.savez_compressed(OUT/'samples.npz',**arrays)
    x,y,h=arrays['train_x'].astype('d'),arrays['train_y_bits'],arrays['train_h']
    mean_rate=float(y.sum()/h.sum())
    np.savez(OUT/'scalers.npz',x_mean=x.mean(axis=(0,1)),x_std=np.maximum(x.std(axis=(0,1)),1e-12),
        h_mean=h.mean(),h_std=max(float(h.std()),1e-12),y_mean=y.mean(),y_std=max(float(y.std()),1.),mean_rate=mean_rate)
    # Raw TimesFM depends only on x/proxy, not horizon or labels. Exact equality
    # above proves the previously GPU-evaluated frozen forecasts are reusable.
    shutil.copy2(PREVIOUS/'timesfm_raw_rates.npz',OUT/'timesfm_raw_rates.npz')
    raw_meta=json.loads((PREVIOUS/'timesfm_raw_metadata.json').read_text())
    raw_meta['reuse']=dict(source=str(PREVIOUS/'timesfm_raw_rates.npz'),sha256=sha(PREVIOUS/'timesfm_raw_rates.npz'),
        previous_samples_sha256=sha(PREVIOUS/'samples.npz'),new_samples_sha256=sha(OUT/'samples.npz'),
        exact_all_x_proxy_and_origin_rows_verified=True,forecasts_reexecuted=False,
        reason='Only known planned horizons and offline labels changed; frozen neural x/proxy inputs are byte-identical.')
    write(OUT/'timesfm_raw_metadata.json',raw_meta)
    protocol=dict(status='PREPARED',created_at=stamp(),target='Actual AF offered bits in (REPORT snapshot, causally planned AF start]',
        inputs='50 cycles x7 causal REPORT/previous applied GRANT/REPORT interval features, plus known planned service horizon',
        horizon='Counterfactual cyclic ONU schedules use currentREPORT IPACT grants and known DBAavailable time; same packet-free schedule equation as runtime.',
        sample_counts=dict(train=1024,val=512,test=512),seeds=dict(train=128,val=129,diagnostic_test=130,untouched_closed_loop=131),
        window_seconds=[3,10],sample_selection='Exact1315originrows and50x7/proxy inputs; perONU chronological ordinal cycles across8fixed cyclic order offsets',
        counterfactual_order_offsets=ORDER_OFFSETS,source_window_boundary='If chosen AFstart>=10s use valid identity plan for same origin; no truncated label.',
        sources=provenance,validation=validation,mean_rate_bps=mean_rate,
        timesfm='Frozen proxy-rate forecast then GPU train-only affine calibration to same actual bits and known horizon as LSTM',
        raw_timesfm_reused=raw_meta['reuse'],calibration_features=['raw_timesfm_rate * current_known_horizon','training_mean_rate * current_known_horizon'],
        calibration_candidates=dict(ridge=RIDGES,nonnegative_output=True,mean_fallback=True),
        selection='Validation RMSE of actual horizon bits only; diagnostic test seed130 never selects weights/settings',
        test_scope='Seed130 inspected earlier; descriptive diagnostic, not new untouched holdout',
        structural_change='Baseline-only horizons had near-zero variation. Counterfactual causal orders expand horizon support before any new neural closed-loop outcomes. All model and selection settings preserved.',
        limits=['Target starts at older REPORT snapshot, not strictly future after DBA.',
            'Baseline states with counterfactual schedules do not remove closed-loop queue-state distribution shift.',
            'TimesFM history is an imperfect REPORT/grant-balance proxy. Calibration aligns labels, not input-target semantics.',
            'LSTM supervised; TimesFM frozen plus fitted affine head; not an architecture-only comparison.',
            'Single mix136/load90 and1024training/512validation origins; bounded POC only.'],
        inference_in_simulated_time=False,gpu_required=True,source_hashes={str(Path(__file__)):sha(__file__),str(Path(COMMON.__file__)):sha(COMMON.__file__)},
        samples_sha256=sha(OUT/'samples.npz'),scalers_sha256=sha(OUT/'scalers.npz'))
    write(OUT/'protocol.json',protocol)
    shutil.copy2(__file__,OUT/Path(__file__).name)


def make_network(torch):
    class HorizonLSTM(torch.nn.Module):
        def __init__(self):
            super().__init__();self.first=torch.nn.LSTM(7,32,batch_first=True)
            self.second=torch.nn.LSTM(32,32,batch_first=True);self.drop=torch.nn.Dropout(.1)
            self.out=torch.nn.Linear(33,1)
        def forward(self,x,h):
            z,_=self.first(x);z,_=self.second(self.drop(z))
            return self.out(torch.cat([self.drop(z[:,-1]),h[:,None]],dim=1)).squeeze(-1)
    return HorizonLSTM()


def scores(y,p):
    e=np.asarray(p)-y;variance=np.sum((y-y.mean())**2)
    return dict(mae_bits=float(np.mean(abs(e))),rmse_bits=float(np.sqrt(np.mean(e*e))),
                r2=float(1-np.sum(e*e)/variance) if variance else None,sample_count=len(y),unit='bits')


def train_lstm():
    torch,device=gpu();torch.manual_seed(42);np.random.seed(42)
    a,s=dict(np.load(OUT/'samples.npz')),dict(np.load(OUT/'scalers.npz'))
    xs={k:torch.tensor(((a[k+'_x']-s['x_mean'])/s['x_std']).astype('f'),device=device) for k in ['train','val','test']}
    hs={k:torch.tensor(((a[k+'_h']-s['h_mean'])/s['h_std']).astype('f'),device=device) for k in xs}
    ys={k:torch.tensor(((a[k+'_y_bits']-s['y_mean'])/s['y_std']).astype('f'),device=device) for k in xs}
    m=make_network(torch).to(device);opt=torch.optim.RMSprop(m.parameters(),lr=.001,alpha=.9,eps=1e-7)
    best=float('inf');wait=0;hist=[];start=time.perf_counter()
    for epoch in range(1,51):
        m.train();losses=[]
        for i in range(0,len(xs['train']),256):
            opt.zero_grad(set_to_none=True);yp=m(xs['train'][i:i+256],hs['train'][i:i+256])
            assert yp.device.type=='mps';loss=((yp-ys['train'][i:i+256])**2).mean();loss.backward()
            assert all(p.grad is None or p.grad.device.type=='mps' for p in m.parameters())
            opt.step();losses.append(float(loss.detach().cpu()))
        m.eval()
        with torch.inference_mode():
            yp=m(xs['val'],hs['val']);value=float(((yp-ys['val'])**2).mean().cpu())
        hist.append(dict(epoch=epoch,training_loss=float(np.mean(losses)),validation_loss=value));write(OUT/'lstm_history.json',hist)
        print('LSTM',epoch,value,flush=True)
        if value<best:
            best,best_epoch,wait=value,epoch,0;state={k:v.detach().clone() for k,v in m.state_dict().items()}
        else:wait+=1
        if wait>=7:break
    torch.mps.synchronize();seconds=time.perf_counter()-start
    torch.save({k:v.cpu() for k,v in state.items()},OUT/'lstm.weights.pt')
    m.load_state_dict(state);m.eval();pred={}
    with torch.inference_mode():
        for k in xs:
            yp=m(xs[k],hs[k]);assert yp.device.type=='mps'
            pred[k]=np.maximum(0,yp.cpu().numpy().astype('d')*s['y_std']+s['y_mean'])
    np.savez_compressed(OUT/'lstm_predictions.npz',**pred)
    meta=dict(status='COMPLETED',created_at=stamp(),device='mps:0',parameter_devices=sorted(set(str(p.device) for p in m.parameters())),
        forward_backward_gpu_verified=True,cpu_ml_fallback=False,framework=torch.__version__,epochs=epoch,best_epoch=best_epoch,
        optimizer='RMSprop lr0.001 alpha0.9 eps1e-7',architecture='LSTM32/dropout0.1/LSTM32/dropout0.1/concat_known_horizon/Dense1',
        objective='MSE of standardized actual AF bits through planned service horizon',seed=42,shuffle=False,batch_size=256,
        training_seconds=seconds,weights_sha256=sha(OUT/'lstm.weights.pt'),source_sha256=sha(__file__),
        mean_training_rate_bps=float(s['mean_rate']),metrics={k:scores(a[k+'_y_bits'],pred[k]) for k in xs})
    write(OUT/'lstm_metadata.json',meta)
    print('LSTM TRAINED',json.dumps(meta['metrics']),flush=True)


def raw_timesfm():
    a=dict(np.load(OUT/'samples.npz'));model=COMMON.FutureAFPredictor('timesfm3');pred={};timings={}
    for split in ['train','val','test']:
        x,p=a[split+'_x'],a[split+'_proxy'];parts=[];start=time.perf_counter()
        for i in range(0,len(x),16):
            parts.append(model.predict_rate(x[i:i+16],p[i:i+16]))
            if i%128==0:print('TimesFM',split,i,len(x),flush=True)
        pred[split]=np.concatenate(parts);timings[split]=time.perf_counter()-start
        np.savez_compressed(OUT/'timesfm_raw_rates.npz',**pred)
    write(OUT/'timesfm_raw_metadata.json',dict(status='COMPLETED',created_at=stamp(),metadata=model.meta,
        forward_devices=sorted(set(model.observed_devices)),seconds=timings,
        feature_limit='Proxy history is observable but not actual arrival history; all covariates past-only',
        source_hashes={str(Path(__file__)):sha(__file__),str(Path(COMMON.__file__)):sha(COMMON.__file__)}))


def calibrate():
    torch,device=gpu();torch.manual_seed(42)
    a,s=dict(np.load(OUT/'samples.npz')),dict(np.load(OUT/'scalers.npz'))
    raw=dict(np.load(OUT/'timesfm_raw_rates.npz'))
    features={k:np.column_stack([raw[k]*a[k+'_h'],s['mean_rate']*a[k+'_h']]) for k in raw}
    fm,fs=features['train'].mean(axis=0),np.maximum(features['train'].std(axis=0),1.)
    xx={k:torch.tensor(((v-fm)/fs).astype('f'),device=device) for k,v in features.items()}
    yy=torch.tensor(((a['train_y_bits']-s['y_mean'])/s['y_std']).astype('f'),device=device)
    candidates=[];stored=[];start=time.perf_counter()
    for ridge in RIDGES:
        model=torch.nn.Linear(2,1).to(device)
        with torch.no_grad():model.weight.zero_();model.bias.zero_()
        opt=torch.optim.Adam(model.parameters(),lr=.03)
        for step in range(600):
            opt.zero_grad(set_to_none=True);out=model(xx['train']).squeeze(-1)
            assert out.device.type=='mps'
            loss=((out-yy)**2).mean()+ridge*model.weight.square().sum()
            loss.backward();assert all(p.grad.device.type=='mps' for p in model.parameters());opt.step()
        with torch.inference_mode():
            p=np.maximum(0,model(xx['val']).squeeze(-1).cpu().numpy().astype('d')*s['y_std']+s['y_mean'])
        weight=model.weight.detach().cpu().numpy().ravel().astype('d');bias=float(model.bias.detach().cpu()[0])
        metric=scores(a['val_y_bits'],p)
        candidates.append(dict(mode='affine',ridge=ridge,validation=metric,weight_z=weight.tolist(),bias_z=bias))
        stored.append((weight,bias))
    fallback=scores(a['val_y_bits'],s['mean_rate']*a['val_h'])
    candidates.append(dict(mode='mean_fallback',ridge=None,validation=fallback))
    selected=min(range(len(candidates)),key=lambda i:candidates[i]['validation']['rmse_bits'])
    choice=candidates[selected]
    torch.mps.synchronize()
    # Store selected coefficients; runtime applies this head on MPS as well.
    np.savez(OUT/'calibration.npz',feature_mean=fm,feature_std=fs,
             weight_z=np.asarray(choice.get('weight_z',[0.,0.])),bias_z=choice.get('bias_z',0.))
    meta=dict(status='COMPLETED',created_at=stamp(),device='mps:0',framework=torch.__version__,
        forward_backward_gpu_verified=True,cpu_ml_fallback=False,training_seconds=time.perf_counter()-start,
        training_labels='Same1024actualAFbitlabels andknownhorizons asLSTM',selection_partition='seed129validationonly',
        selected=choice,candidates=candidates,head='2-feature affine standardized bit regressor, ridge on weights, nonnegative bit output',
        input_features=['raw_timesfm_proxy_forecast_rate*known_horizon','training_mean_rate*known_horizon'],
        mean_training_rate_bps=float(s['mean_rate']),source_sha256=sha(__file__),weights_sha256=sha(OUT/'calibration.npz'),
        feature_limit='Calibrated output now targets actual AF bits, but proxy input and pretrained model differ from LSTM')
    write(OUT/'calibration_metadata.json',meta)
    print('CALIBRATION SELECTED',json.dumps(choice),flush=True)


class MatchedHorizonPredictor:
    def __init__(self,mode,model_directory=OUT):
        self.mode,self.directory=mode,Path(model_directory);self.seconds=0.;self.points=0
        self._last_raw_x=None;self._last_raw_proxy=None;self._last_raw_rates=None
        self.scalers=dict(np.load(self.directory/'scalers.npz'))
        self.metadata=dict(mode=mode,target='Actual AF offered bits from latest REPORT snapshot to supplied causally planned AFservice start',
            returned_unit='bits/second',point_objective='actual horizon bits MSE',horizon_conditioned=True,
            mean_training_rate_bps=float(self.scalers['mean_rate']),source_sha256=sha(__file__),
            source_dependencies={str(Path(COMMON.__file__)):sha(COMMON.__file__)},cpu_ml_fallback=False,
            raw_forecast_calls=0,raw_forecast_cache_hits=0,
            cache_policy='One-entry exact x/proxy value cache; every changed causal history invalidates it; known horizon recalibrated each call')
        self.meta=self.metadata
        if mode=='mean':self.metadata['device']='CPU supporting constant-rate control'
        elif mode=='lstm':
            self.torch,self.device=gpu();self.model=make_network(self.torch).to(self.device)
            self.model.load_state_dict(self.torch.load(self.directory/'lstm.weights.pt',map_location=self.device,weights_only=True));self.model.eval()
            self.metadata.update(device='mps:0',framework=self.torch.__version__,weights_sha256=sha(self.directory/'lstm.weights.pt'))
            lm=json.loads((self.directory/'lstm_metadata.json').read_text())
            self.metadata.update(validation_mae_bits=lm['metrics']['val']['mae_bits'],validation_rmse_bits=lm['metrics']['val']['rmse_bits'])
        elif mode in ['timesfm3','timesfm3_raw']:
            self.raw_model=COMMON.FutureAFPredictor('timesfm3');self.torch,self.device=self.raw_model.torch,self.raw_model.device
            self.metadata.update(device='mps:0',raw_model=self.raw_model.metadata)
            if mode=='timesfm3':
                self.calibration=dict(np.load(self.directory/'calibration.npz'))
                cm=json.loads((self.directory/'calibration_metadata.json').read_text());self.choice=cm['selected']
                self.metadata.update(calibration=cm,validation_mae_bits=self.choice['validation']['mae_bits'],
                    validation_rmse_bits=self.choice['validation']['rmse_bits'])
                self.cal_tensors={k:self.torch.tensor(v,dtype=self.torch.float32,device=self.device) for k,v in self.calibration.items()}
                self.metadata['forecast_used_in_calibrated_output']=self.choice['mode']!='mean_fallback'
                self.metadata['mean_fallback_skips_neural_forward']=self.choice['mode']=='mean_fallback'
        else:raise ValueError(mode)
        if (self.directory/'quality.json').exists():
            quality=json.loads((self.directory/'quality.json').read_text())
            baseline=quality['validation_training_mean']['rmse_bits']
            self.metadata['mean_validation_rmse_bits']=baseline
            if 'validation_rmse_bits' in self.metadata:
                self.metadata['validation_rmse_improvement_vs_mean_fraction']=1-self.metadata['validation_rmse_bits']/baseline

    def predict_rate(self,x,proxy_history,horizons):
        x,h=np.asarray(x),np.asarray(horizons,dtype='d')
        assert x.shape==(len(h),50,7) and np.isfinite(x).all() and np.isfinite(h).all() and (h>=0).all()
        start=time.perf_counter();s=self.scalers
        if self.mode=='mean':rates=np.full(len(h),float(s['mean_rate']))
        elif self.mode=='lstm':
            z=((x-s['x_mean'])/s['x_std']).astype('f');zh=((h-s['h_mean'])/s['h_std']).astype('f')
            with self.torch.inference_mode():
                out=self.model(self.torch.tensor(z,device=self.device),self.torch.tensor(zh,device=self.device))
                assert out.device.type=='mps';bits=np.maximum(0,out.cpu().numpy().astype('d')*s['y_std']+s['y_mean'])
            rates=np.divide(bits,h,out=np.zeros_like(h),where=h>0);self.torch.mps.synchronize()
        elif self.mode=='timesfm3' and self.choice['mode']=='mean_fallback':
            rates=np.full(len(h),float(s['mean_rate']))
        else:
            proxy_history=np.asarray(proxy_history)
            assert proxy_history.shape==(len(h),50) and np.isfinite(proxy_history).all()
            if (self._last_raw_x is not None and np.array_equal(x,self._last_raw_x)
                    and np.array_equal(proxy_history,self._last_raw_proxy)):
                raw=self._last_raw_rates.copy()
                self.metadata['raw_forecast_cache_hits']+=1
            else:
                raw=self.raw_model.predict_rate(x,proxy_history)
                self._last_raw_x=x.copy();self._last_raw_proxy=proxy_history.copy();self._last_raw_rates=raw.copy()
                self.metadata['raw_forecast_calls']+=1
            if self.mode=='timesfm3_raw':rates=raw
            else:
                features=np.column_stack([raw*h,s['mean_rate']*h]).astype('f')
                with self.torch.inference_mode():
                    ft=self.torch.tensor(features,device=self.device)
                    c=self.cal_tensors
                    z=(ft-c['feature_mean'])/c['feature_std']
                    bits=self.torch.clamp((z@c['weight_z']+c['bias_z'])*float(s['y_std'])+float(s['y_mean']),min=0)
                    assert bits.device.type=='mps';bits=bits.cpu().numpy().astype('d')
                rates=np.divide(bits,h,out=np.zeros_like(h),where=h>0)
            self.torch.mps.synchronize()
        rates=np.maximum(0,rates);rates[h==0]=0
        assert rates.shape==h.shape and np.isfinite(rates).all()
        self.seconds+=time.perf_counter()-start;self.points+=len(h)
        return rates

    predict_rates=predict_rate


def report():
    torch,device=gpu()
    a,s=dict(np.load(OUT/'samples.npz')),dict(np.load(OUT/'scalers.npz'))
    raw=dict(np.load(OUT/'timesfm_raw_rates.npz'));lstm=dict(np.load(OUT/'lstm_predictions.npz'))
    c=dict(np.load(OUT/'calibration.npz'));cm=json.loads((OUT/'calibration_metadata.json').read_text())
    choice=cm['selected'];rows=[];predictions={}
    for k in ['train','val','test']:
        h,y=a[k+'_h'],a[k+'_y_bits'];f=np.column_stack([raw[k]*h,s['mean_rate']*h])
        if choice['mode']=='affine':
            with torch.inference_mode():
                ft=torch.tensor(f.astype('f'),device=device)
                fm=torch.tensor(c['feature_mean'],dtype=torch.float32,device=device)
                fs=torch.tensor(c['feature_std'],dtype=torch.float32,device=device)
                w=torch.tensor(c['weight_z'],dtype=torch.float32,device=device)
                b=torch.tensor(c['bias_z'],dtype=torch.float32,device=device)
                out=torch.clamp((((ft-fm)/fs)@w+b)*float(s['y_std'])+float(s['y_mean']),min=0)
                assert out.device.type=='mps';calibrated=out.cpu().numpy().astype('d')
        else:
            calibrated=s['mean_rate']*h
        predictions[k+'_calibrated_bits']=calibrated
        for name,p in [('LSTM horizon conditioned',lstm[k]),('TimesFM raw proxy',raw[k]*h),
                       ('TimesFM calibrated actual AF',calibrated),('Training mean rate',s['mean_rate']*h),
                       ('REPORT balance persistence',a[k+'_proxy'][:,-1]*h),('Zero reserve',np.zeros(len(h)))]:
            rows.append(dict(model=name,partition=k,target='AF actualofferedbits postREPORTtoPlannedAFstart',
                traffic_class='AF',horizon='per-origin known planned service horizon',**scores(y,p)))
    np.savez_compressed(OUT/'calibrated_predictions.npz',**predictions)
    with (OUT/'metrics.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    quality=dict(status='READY_FOR_GPU_REPLAY',mean_rate_bps=float(s['mean_rate']),
        calibration_inference_device='mps:0' if choice['mode']=='affine' else 'scalar mean control',
        validation_training_mean=scores(a['val_y_bits'],s['mean_rate']*a['val_h']),
        validation_lstm=scores(a['val_y_bits'],lstm['val']),calibration_selected=choice,
        calibration_uses_forecast=choice['mode']!='mean_fallback',metrics=rows)
    write(OUT/'quality.json',quality)
    print('REPORT',json.dumps([r for r in rows if r['partition']=='test']),flush=True)


def replay():
    a=dict(np.load(OUT/'samples.npz'));old=dict(np.load(OUT/'lstm_predictions.npz'))
    checks={}
    for mode in ['lstm','timesfm3']:
        model=MatchedHorizonPredictor(mode);p=model.predict_rate(a['test_x'][:16],a['test_proxy'][:16],a['test_h'][:16])*a['test_h'][:16]
        expected=old['test'][:16] if mode=='lstm' else np.load(OUT/'calibrated_predictions.npz')['test_calibrated_bits'][:16]
        assert np.allclose(p,expected,atol=.02,rtol=1e-5)
        checks[mode]=dict(max_abs_difference_bits=float(abs(p-expected).max()),device=model.metadata['device'],metadata=model.metadata)
        if mode=='timesfm3' and model.choice['mode']=='mean_fallback':
            assert model.metadata['raw_forecast_calls']==0
        del model
    raw=MatchedHorizonPredictor('timesfm3_raw')
    x,pr,h=a['test_x'][:16].copy(),a['test_proxy'][:16].copy(),a['test_h'][:16]
    first=raw.predict_rate(x,pr,h)
    second=raw.predict_rate(x,pr,h*.9)
    assert np.array_equal(first,second) and raw.metadata['raw_forecast_calls']==1 and raw.metadata['raw_forecast_cache_hits']==1
    x[0,0,0]+=1.
    raw.predict_rate(x,pr,h)
    assert raw.metadata['raw_forecast_calls']==2 and raw.metadata['raw_forecast_cache_hits']==1
    checks['timesfm_raw_cache']=dict(status='PASS',different_horizon_reuses_exact_raw_forecast=True,
        changed_history_invalidates_cache=True,raw_forecast_calls=2,cache_hits=1,
        observed_forward_devices=sorted(set(raw.raw_model.observed_devices)))
    write(OUT/'gpu_replay.json',dict(status='PASS',created_at=stamp(),checks=checks))
    protocol=json.loads((OUT/'protocol.json').read_text());protocol['status']='READY_FOR_CLOSED_LOOP';write(OUT/'protocol.json',protocol)
    print('GPU REPLAY PASS',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','train_lstm','raw_timesfm','calibrate','report','replay'])
    globals()[parser.parse_args().command]()
