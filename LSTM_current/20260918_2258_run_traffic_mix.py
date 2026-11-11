"""New traffic mixes with the unchanged frozen adaptive policy; PONSIM_MIX required."""
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK']='0'
from pathlib import Path
from importlib import import_module
from dataclasses import replace
from datetime import datetime
from zoneinfo import ZoneInfo
import argparse,json,sys,time,hashlib,traceback,shutil
import numpy as np
HERE=Path(__file__).resolve().parent
S=HERE/'20260912_2134_Python_OPNET_Simulator';sys.path.insert(0,str(S))
import os
MIX=os.environ['PONSIM_MIX']
assert MIX in ('244','163'), MIX
OUT=S/'runs/ponsim_traffic_mixes/20260918_2258'/('mix'+MIX)
DATA=S/'runs/mix_2w_10s/20260916_0712'
LOADS=tuple(range(10,101,10))
MODELS=OUT/'models'
LABELS={'baseline':'IPACT','ipact_flex':'IPACT + flexible service','noforecast':'Adaptive scheduler without prediction','mean':'Adaptive + mean','lstm':'Adaptive + LSTM','timesfm3':'Guarded TimesFM 3 (validation-selected)' ,'timesfm3_raw':'Adaptive + raw TimesFM 3','oracle':'Adaptive + oracle'}
def stamp():return datetime.now(ZoneInfo('Asia/Jakarta')).isoformat()
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,obj):
    p=Path(p);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');t.replace(p)
def prepare(load,seed):
    assert load in LOADS and seed in (131,132)
    folder=OUT/'traffic'/f'load{load:03d}'/f'seed{seed}'
    if (folder/'manifest.json').exists():
        m=json.loads((folder/'manifest.json').read_text())
        assert m['status']=='COMPLETED' and m['load_percent']==load and m['seed']==seed and m['mix']==MIX
        assert all(sha(folder/n)==v for n,v in m['artifacts'].items())
        return folder
    folder.mkdir(parents=True,exist_ok=False)
    settings_source=DATA/f'mix{MIX}_load{load:03d}_seed130'/'settings.json'
    cfg=json.loads(settings_source.read_text());cfg['duration']=4.
    assert cfg['base_cap']==46875 and cfg['channels']==2 and cfg['onu_count']==64
    assert cfg['channel_rates']==[1e9,1e9]
    write(folder/'settings.json',cfg)
    gen=import_module('20260913_0538_supfrp_workload')
    rates=2e9*(load/100)*(np.array([int(c) for c in MIX])/10)/64
    data,traffic=gen.generate_offered(560/rates[0],rates[1]/6400,rates[2]/6400,duration_s=4.,n_onu=64,seed=seed,hurst=.7,onset_s=1.)
    keep=data['time']<4.;data={k:v[keep] for k,v in data.items()}
    np.savez_compressed(folder/'offered.npz',**data);write(folder/'traffic.json',traffic)
    write(folder/'manifest.json',dict(status='COMPLETED',created_at=stamp(),mix=MIX,load_percent=load,seed=seed,duration=4.,
        provenance=dict(kind='fresh per-load confirmation traffic',generator=str(Path(gen.__file__)),generator_sha256=sha(gen.__file__),settings_source=str(settings_source),settings_source_sha256=sha(settings_source),duration_prefix_invariant=True),
        artifacts={p.name:sha(p) for p in folder.iterdir()}))
    print('PREPARED',load,seed,len(data['time']),flush=True);return folder
class TimedPredictor:
    def __init__(self,model):self.model=model;self.latencies=[]
    def predict_rate(self,x,proxy,horizons):
        st=time.perf_counter();y=self.model.predict_rate(x,proxy,horizons);self.latencies.append(time.perf_counter()-st)
        if len(self.latencies)%40==0:print('PREDICTION_CALLS',len(self.latencies),'MODEL_SECONDS',round(sum(self.latencies),2),flush=True)
        return y
def run(load,mode,seed,duration=3.1):
    assert load in LOADS and mode in LABELS and seed in (131,132)
    source=prepare(load,seed)
    MODEL=MODELS/f'load{load:03d}'
    core=import_module('20260912_2134_simulator');perf=import_module('20260912_2134_performance');engine=import_module('20260917_1320_adaptive_predictive_engine')
    folder=OUT/f'load{load:03d}'/f'seed{seed}'/mode;folder.mkdir(parents=True,exist_ok=False)
    assert 3.<duration<=4.
    cfg=replace(core.SimulationConfig(**json.loads((source/'settings.json').read_text())),duration=duration)
    with np.load(source/'offered.npz') as z:offered={k:z[k] for k in z.files}
    keep=offered['time']<duration;offered={k:v[keep] for k,v in offered.items()}
    predictor=None;model_meta=None
    if mode in ('mean','lstm','timesfm3','timesfm3_raw'):
        p=import_module('20260917_1329_order_horizon_predictor');model=p.MatchedHorizonPredictor(mode,model_directory=MODEL);predictor=TimedPredictor(model);model_meta=model.metadata
    arrays=[]
    if mode=='oracle':
        for u in range(64):
            mask=(offered['onu']==u)&(offered['queue']==1);arrays.append((offered['time'][mask],np.r_[0.,np.cumsum(offered['bits'][mask])]))
    def oracle(snapshot,horizon,onu):
        tt,bb=arrays[onu];lo,hi=np.searchsorted(tt,[snapshot,snapshot+horizon],side='right');return float((bb[hi]-bb[lo])/horizon) if horizon>0 else 0.
    if mode=='ipact_flex':
        flex=import_module('20260917_1310_flexible_burst_engine')
        sim=flex.FlexibleBurstSimulator(cfg,mode='baseline',flexible=True,activation_time=3.,window=50)
    else:
        sim=engine.AdaptivePredictiveSimulator(cfg,mode='timesfm3' if mode=='timesfm3_raw' else mode,predictor=predictor,flexible=True,credit_fraction=.25,be_floor_fraction=.75,be_max_wait_rounds=2,activation_time=3.,window=50,max_packet_bits=12144.,oracle_rate_query=oracle if mode=='oracle' else None)
    paths=[Path(__file__)]+[S/(x+'.py') for x in ['20260917_1320_adaptive_predictive_engine','20260917_1310_flexible_burst_engine','20260917_1246_predictive_rotation_engine','20260913_0858_rotating_be_engine','20260913_0822_bounded_af_engine','20260912_2134_simulator','20260912_2134_performance','20260913_0538_supfrp_workload','20260912_2346_supfrp_source_math']]
    if predictor:paths.extend(S/(x+'.py') for x in ['20260917_1329_order_horizon_predictor','20260917_1246_future_af_predictor'])
    meta=dict(status='RUNNING',started_at=stamp(),mode=mode,label=LABELS[mode],mix=MIX,load_percent=load,seed=seed,source=str(source),source_traffic_sha256=sha(source/'offered.npz'),duration=duration,activation=3.,measurement=[3.,duration],deadline_policy='same-state full IPACT envelope',original_data_budget_us=1500,base_cap_bits=46875.,be_floor_fraction=.75,max_AF_borrow_fraction=.25,model=model_meta,model_directory=str(MODEL),inference_in_simulated_time=False,source_hashes={str(p):sha(p) for p in paths},model_comparison='Matched actual AF arrival bits through planned AF service. Guarded TimesFM calibration selected on validation; rawTimesFM is a separately labelled active-forecast diagnostic. Per-load training128/validation129; horizon support covers valid rotated ONU orders. Diagnostic130 previously inspected; closed-loop131/132 not used for tuning. Same frozen policy and hyperparameters across loads.')
    if predictor:meta['model_asset_hashes']={p.name:sha(p) for p in MODEL.iterdir() if p.is_file() and p.suffix in ('.json','.npz','.pt')}
    write(folder/'manifest.json',meta);shutil.copyfile(__file__,folder/'executed_runner.py')
    print('START',load,seed,mode,stamp(),flush=True);start=time.perf_counter()
    try:
        result=sim.run(offered)
        np.savez_compressed(folder/'packets.npz',services=result['services'],drops=result['drops'])
        np.savez_compressed(folder/'control.npz',gates=result['gates'],reports=result['reports'],dba_times=result['dba_times'])
        np.savez_compressed(folder/'decisions.npz',decisions=sim.decision_array(),cycles=sim.cycle_array())
        np.savez_compressed(folder/'flexible.npz',bursts=sim.burst_array(),services=sim.service_audit_array())
        unused=np.array(result['unused_events'],dtype=[('time','f8'),('onu','i2'),('queue','i1'),('duration','f8'),('reason','U6')]);np.savez_compressed(folder/'unused.npz',events=unused)
        write(folder/'summary.json',result['summary']);write(folder/'performance.json',perf.summarize(result,offered,begin=3.,end=duration,bin_width=.01,run_name=LABELS[mode]))
        for q in range(3):
            om=offered['queue']==q;s=result['services'];d=result['drops'];sm=s['queue']==q;dm=d['queue']==q
            assert om.sum()==sm.sum()+dm.sum()+result['summary']['queued_packets_end'][q]
            assert np.isclose(offered['bits'][om].sum(),s['bits'][sm].sum()+d['bits'][dm].sum()+result['summary']['queued_bits_end'][q],atol=1e-3,rtol=0)
        latency=predictor.latencies if predictor else []
        meta.update(status='COMPLETED',finished_at=stamp(),wall_seconds=time.perf_counter()-start,inference_seconds=sum(latency),inference_calls=len(latency),inference_mean_ms=float(np.mean(latency)*1000) if latency else None,engine_metadata=sim.metadata,model_final=model.metadata if predictor else None,artifacts={p.name:sha(p) for p in folder.iterdir() if p.name!='manifest.json'})
        assert all(sha(p)==v for p,v in meta['source_hashes'].items())
        write(folder/'manifest.json',meta);print('COMPLETED',load,seed,mode,meta['wall_seconds'],flush=True)
    except Exception:
        meta.update(status='FAILED',error=traceback.format_exc());write(folder/'manifest.json',meta);raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--load',type=int,choices=LOADS,required=True);p.add_argument('--seed',type=int,default=131);p.add_argument('--mode',choices=LABELS);p.add_argument('--prepare-only',action='store_true');p.add_argument('--duration',type=float,default=3.1);a=p.parse_args()
    if a.prepare_only:prepare(a.load,a.seed)
    elif a.mode:run(a.load,a.mode,a.seed,a.duration)
    else:p.error('--mode is required unless --prepare-only')
