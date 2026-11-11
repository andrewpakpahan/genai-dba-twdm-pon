"""Independent report readback, provenance packaging and explicit visual-review gate."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from importlib import import_module
import argparse,hashlib,json,shutil,zipfile
import numpy as np
import pandas as pd
H=Path(__file__).resolve().parent
R=import_module('20260918_2306_report_traffic_mixes')
OUT=R.OUT;S=R.S;LOADS=R.LOADS;MIXES=R.MIXES;MODES=R.MODES
read=R.read;write=R.write;sha=R.sha;stamp=R.stamp
def eq(a,b):
 if b is None:assert pd.isna(a)
 elif isinstance(b,bool):assert str(a).lower()==str(b).lower()
 elif isinstance(b,(int,float)):assert np.isclose(float(a),b,atol=1e-9,rtol=1e-11,equal_nan=True),(a,b)
 else:assert str(a)==str(b)
def artifacts(folder,pins):
 for n,h in pins.items():assert sha(Path(folder)/n)==h,str(Path(folder)/n)
def main(reviewed=False):
 assert reviewed,'Supply --visual-review-complete only after inspecting all four PNGs'
 for role in ['inputs','controls','models','audit']:assert read(OUT/(role+'_JOB.json'))['status']=='COMPLETED'
 protocol=read(OUT/'protocol.json');assert protocol['new_cases']==320 and protocol['reused_cases']==160
 for p,h in protocol['source_hashes'].items():assert sha(p)==h
 validation=read(OUT/'report_validation.json');assert validation['status'] in ['PASS_PENDING_VISUAL_REVIEW','PASS']
 assert validation['reporter_sha256']==sha(R.__file__)
 for p,h in validation['source_hashes'].items():assert sha(p)==h
 total_checks=0;data_origins=0;restore_checks=0;horizon_cases=0;forecast_cases=0;poststop_gates=0;new_manifest_hashes={}
 model_evidence=[];horizon_evidence=[];audit_evidence={};asset_hashes={}
 for mix in ['244','163']:
  root=OUT/('mix'+mix);audit=read(root/'independent_verification.json')
  assert audit['status']=='PASS' and audit['expected_runs']==audit['completed_verified_runs']==160
  assert not audit['pending'] and not audit['failures'] and len(audit['runs'])==160
  assert {(x['load_percent'],x['seed'],x['mode']) for x in audit['runs']}=={(l,s,m) for l in LOADS for s in [131,132] for m in MODES}
  assert all(x['status']=='PASS' and all(c['passed'] for c in x['checks']) for x in audit['runs'])
  assert sum(x['check_count'] for x in audit['runs'])==audit['check_count'];total_checks+=audit['check_count']
  for p,h in audit['constituent_audits'].items():assert sha(p)==h
  audit_evidence[str(root/'independent_verification.json')]=sha(root/'independent_verification.json')
  for load in LOADS:
   model=R.modelroot(mix,load);frozen=read(model/'FROZEN_MANIFEST.json');assert frozen['status']=='FROZEN_READY_FOR_CLOSED_LOOP';artifacts(model,frozen['artifacts'])
   model_protocol=read(model/'protocol.json');assert model_protocol['mix']==mix and model_protocol['load_pct']==load
   assert model_protocol['seeds']==dict(train=128,val=129,diagnostic_test=130,untouched_closed_loop=[131,132])
   assert read(model/'gpu_replay.json')['status']=='PASS'
   gpu=read(model/'gpu_probe.json');assert gpu['device'] in ['mps','mps:0'] and gpu['actual_tensor_device']=='mps:0' and gpu['cpu_ml_fallback'] is False
   raw=read(model/'timesfm_raw_metadata.json');assert raw['forward_devices']==['mps:0']
   checkpoint=raw['metadata'];artifacts(checkpoint['checkpoint'],checkpoint['checkpoint_hashes'])
   for filename in ['lstm_metadata.json','calibration_metadata.json']:
    meta=read(model/filename);assert meta['device']=='mps:0' and meta['forward_backward_gpu_verified'] and meta['cpu_ml_fallback'] is False
   cal=read(model/'calibration_metadata.json');assert cal['selected']==min(cal['candidates'],key=lambda x:x['validation']['rmse_bits'])
   data=read(root/f'load{load:03d}/predictor_data_independent_verification.json')
   assert data['status']=='PASS' and data['verified_origins']==2048;data_origins+=2048
   for p,h in data['artifact_hashes'].items():assert sha(p)==h
   for seed in [128,129,130]:
    p=root/'training_sources'/f'mix{mix}_load{load:03d}_seed{seed}';rest=read(p/'restoration_verification.json')
    assert rest['status']=='PASS' and all(rest['exact_historical_input_hashes'].values())
    assert sha(rest['historical_manifest'])==rest['historical_manifest_sha256'];historic=read(rest['historical_manifest'])
    for n in ['allocation.npz','control.npz','offered.npz','settings.json']:assert sha(p/n)==historic['artifacts'][n];restore_checks+=1
   for phase in ['controls','neural']:
    hpath=root/f'load{load:03d}/horizon_coverage_{phase}_verification.json';ha=read(hpath);assert ha['status']=='PASS'
    for p,h in ha['artifact_hashes'].items():assert sha(p)==h
    if phase=='neural':
     assert len(ha['cases'])==6
     for c in ha['cases']:
      assert c['all_planned_horizons_reconstructed'] and c['no_missing_gate_receipts_within_simulation'] and c['final_horizon_matches_dispatched_gate']
      assert c['observed_gate_count']+c['unreceived_final_gates_after_stop']==c['points']
      poststop_gates+=c['unreceived_final_gates_after_stop'];horizon_cases+=1
     horizon_evidence.append(dict(mix=mix,load_percent=load,totals=ha['totals']))
   for seed in [131,132]:
    for mode in MODES:
     p=R.root(mix,load)/f'seed{seed}'/mode;j=read(p/'manifest.json');artifacts(p,j['artifacts']);assert j['mix']==mix
     assert j['measurement']==[3.,3.1] and j['inference_in_simulated_time'] is False
     if mode in ['lstm','timesfm3','timesfm3_raw']:assert j['model_final']['device']=='mps:0' and not j['model_final']['cpu_ml_fallback']
     new_manifest_hashes[str(p.relative_to(OUT)/'manifest.json')]=sha(p/'manifest.json')
   if cal['selected']['mode']=='mean_fallback':
    for seed in [131,132]:
     base=R.root(mix,load)/f'seed{seed}'
     for filename in ['packets.npz','control.npz','flexible.npz','unused.npz']:assert sha(base/'mean'/filename)==sha(base/'timesfm3'/filename)
   model_evidence.append(dict(mix=mix,load_percent=load,frozen_manifest_sha256=sha(model/'FROZEN_MANIFEST.json'),selection=cal['selected']))
  forecast=read(root/'runtime_forecast_metrics.json');assert forecast['status']=='PASS' and forecast['completed_cases']==forecast['expected_cases']==100
  assert not forecast['pending'];forecast_cases+=100
  assert sha(root/'runtime_forecast_metrics.csv')==forecast['generated_csv']['sha256']
  for p,h in forecast['artifact_hashes'].items():assert sha(p)==h
 assert data_origins==40960 and restore_checks==240 and horizon_cases==120 and forecast_cases==200
 # Reconstruct every report export from saved per-case metrics, independently of plotting aggregation.
 exported=pd.read_csv(OUT/'qos_metrics.csv',dtype={'mix':str}).set_index(['mix','load_percent','seed','mode','traffic_class'])
 wavelengths=pd.read_csv(OUT/'qos_by_wavelength.csv',dtype={'mix':str,'wavelength':str}).set_index(['mix','load_percent','seed','mode','traffic_class','wavelength'])
 assert len(exported)==1920 and len(wavelengths)==5760 and not exported.index.duplicated().any() and not wavelengths.index.duplicated().any()
 case_export=pd.read_csv(OUT/'case_comparison.csv',dtype={'mix':str}).set_index(['mix','load_percent','seed','mode']);assert len(case_export)==480 and not case_export.index.duplicated().any()
 runtime_export=pd.read_csv(OUT/'runtime.csv',dtype={'mix':str}).set_index(['mix','load_percent','seed','mode']);assert len(runtime_export)==480 and not runtime_export.index.duplicated().any()
 records=[];numeric=0
 for mix in MIXES:
  for load in LOADS:
   for seed in [131,132]:
    for mode in MODES:
     p=R.root(mix,load)/f'seed{seed}'/mode;m=read(p/'manifest.json');assert sha(p/'performance.json')==m['artifacts']['performance.json']
     if mix=='136':assert sha(p/'manifest.json')==protocol['reused_run_manifest_hashes'][str(p/'manifest.json')]
     runtime=runtime_export.loc[(mix,load,seed,mode)]
     for field in ['wall_seconds','inference_seconds','inference_calls']:eq(runtime[field],m[field]);numeric+=1
     eq(runtime.calendar_elapsed_seconds,(datetime.fromisoformat(m['finished_at'])-datetime.fromisoformat(m['started_at'])).total_seconds())
     assert runtime.elapsed_clock=='perf_counter';eq(runtime.inference_in_simulated_time,False)
     perf=read(p/'performance.json');block={}
     for row in perf['primary']['rows']:
      key=(mix,load,seed,mode,row['traffic_class'],str(row['wavelength']))
      selected=wavelengths.loc[key]
      for field,value in row.items():
       if field in ['traffic_class','wavelength']:continue
       eq(selected[field],value);numeric+=1
      if row['wavelength']=='all':
       block[row['traffic_class']]=row
       for field,value in row.items():
        if field=='traffic_class':continue
        eq(exported.loc[key[:5],field],value);numeric+=1
     weighted=sum(block[c]['delay_mean_ms']*block[c]['completed_packets'] for c in ['ef','af','be'])/sum(block[c]['completed_packets'] for c in ['ef','af','be'])
     eq(weighted,block['total']['delay_mean_ms'])
     record=dict(mix=mix,load_percent=load,seed=seed,mode=mode,EF_ms=block['ef']['delay_mean_ms'],AF_ms=block['af']['delay_mean_ms'],BE_ms=block['be']['delay_mean_ms'],ETE_ms=weighted,total_Gbps=block['total']['packet_throughput_gbps'],BE_loss_fraction=block['be']['dropped_packet_fraction'])
     for field in ['EF_ms','AF_ms','BE_ms','ETE_ms','total_Gbps','BE_loss_fraction']:eq(case_export.loc[(mix,load,seed,mode),field],record[field]);numeric+=1
     records.append(record)
 cases=pd.DataFrame(records);keys=['mix','load_percent','mode'];columns=['EF_ms','AF_ms','BE_ms','ETE_ms','total_Gbps','BE_loss_fraction']
 ag=cases.groupby(keys)[columns].agg(['mean','min','max']);ag.columns=['_'.join(x) for x in ag.columns]
 saved=pd.read_csv(OUT/'comparison_seed_means_ranges.csv',dtype={'mix':str}).set_index(keys);assert len(saved)==240
 for key,row in ag.iterrows():
  for field,v in row.items():eq(saved.loc[key,field],float(v));numeric+=1
 plot=pd.read_csv(OUT/'plotted_values.csv',dtype={'mix':str});assert len(plot)==360 and not plot.duplicated(['metric','mix','mode','load_percent']).any()
 for row in plot.itertuples():
  key=(row.mix,row.load_percent,row.mode);assert row.mode in R.MAIN
  for field,label in [('mean','mean'),('minimum','min'),('maximum','max')]:eq(getattr(row,field),float(ag.loc[key,row.metric+'_'+label]));numeric+=1
 criterion=pd.read_csv(OUT/'relative_to_IPACT_and_criterion.csv',dtype={'mix':str}).set_index(['mix','load_percent','seed','mode']);assert len(criterion)==480
 for mix in MIXES:
  for load in LOADS:
   for seed in [131,132]:
    block=cases[(cases.mix==mix)&(cases.load_percent==load)&(cases.seed==seed)].set_index('mode');base=block.loc['baseline']
    for mode in MODES:
     row=block.loc[mode];checks=[row.total_Gbps>base.total_Gbps,row.EF_ms<=base.EF_ms,row.AF_ms<=base.AF_ms,row.BE_ms<=base.BE_ms,row.BE_loss_fraction<=base.BE_loss_fraction]
     eq(criterion.loc[(mix,load,seed,mode),'overall_criterion_met'],bool(all(checks)))
     for field in columns:
      col='BE_loss_change_percentage_points' if field=='BE_loss_fraction' else field+'_change_percent'
      value=100*(row[field]-base[field]) if field=='BE_loss_fraction' else 100*(row[field]/base[field]-1)
      eq(criterion.loc[(mix,load,seed,mode),col],float(value));numeric+=1
 # Archive authored postprocessing as well as frozen execution sources.
 for p in [Path(__file__),Path(R.__file__)]:shutil.copyfile(p,OUT/'source_snapshot'/p.name)
 reviewed_hashes={n+'.png':sha(OUT/(n+'.png')) for n in validation['plot_names']}
 validation.update(status='PASS',completed_at=stamp(),visual_review='All four PNGs inspected before flag',reviewed_plot_hashes=reviewed_hashes,independent_export_checks=numeric,finalizer_sha256=sha(__file__))
 write(OUT/'report_validation.json',validation)
 write(OUT/'verification_summary.json',dict(status='PASS',new_cases=320,reused_cases=160,independent_run_checks=total_checks,prepared_model_origins=data_origins,restored_input_hash_checks=restore_checks,neural_horizon_cases=horizon_cases,runtime_forecast_cases=forecast_cases,final_gate_receipts_after_stop=poststop_gates,audit_files=audit_evidence,model_evidence=model_evidence,horizon_coverage=horizon_evidence))
 report=OUT/'RESULTS.md';s=report.read_text();marker='<!-- FINAL VERIFICATION -->'
 if marker in s:s=s[:s.index(marker)]
 s+=f'\n\n{marker}\n\n## Final validation\n\nAll320 new cases passed **{total_checks:,} independent saved-output checks**. All40,960 prepared model origins,240 exact historical input hashes,120 neural horizon cases and200 realized forecast cases passed. All four plots were visually reviewed; {numeric:,} exported numeric/metric values reconciled independently. The160 historical136 cases retain their original manifests and measured performance hashes.\n\n[Verification summary](verification_summary.json) · [Report validation](report_validation.json) · [Plot/data bundle](Traffic_Mix_136_244_163.zip).\n'
 retention=OUT/'support_packet_removal_receipt.json'
 if retention.exists():
  receipt=read(retention)
  assert receipt['status']=='COMPLETED' and receipt['deleted_count']==60
  assert len(receipt['files'])==60 and sum(row['bytes'] for row in receipt['files'])==receipt['bytes']
  for row in receipt['files']:
   p=Path(row['path']);rel=p.relative_to(OUT)
   assert len(rel.parts)==4 and rel.parts[0] in ['mix244','mix163'] and rel.parts[1]=='training_sources' and rel.parts[3]=='packets.npz'
   assert not p.exists() and read(Path(row['manifest']))['artifacts']['packets.npz']==row['sha256']
  s+='\n## Supporting-data retention\n\nUser-approved storage cleanup removed60 regenerated supporting-dataset packet arrays (6.97GiB). Required training inputs and every confirmation-comparison raw output remain. Original supporting manifests retain creation-time artifact hashes; packet-level reanalysis of those supporting datasets requires regeneration. [Exact removal receipt](support_packet_removal_receipt.json).\n'
 report.write_text(s)
 bundle=[OUT/(n+suffix) for n in validation['plot_names'] for suffix in ['.png','.pdf']]
 bundle += list(OUT.glob('*.csv'))+[report,OUT/'protocol.json',OUT/'verification_summary.json',OUT/'report_validation.json']
 if retention.exists():bundle += [retention,OUT/'storage_cleanup_proposal.json']
 with zipfile.ZipFile(OUT/'Traffic_Mix_136_244_163.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
  for p in bundle:z.write(p,p.name)
 with zipfile.ZipFile(OUT/'Traffic_Mix_136_244_163.zip') as z:
  assert z.testzip() is None
  for p in bundle:assert hashlib.sha256(z.read(p.name)).hexdigest()==sha(p)
 write(OUT/'manifest.json',dict(status='COMPLETED',completed_at=stamp(),new_case_count=320,reused_case_count=160,total_case_count=480,independent_run_checks=total_checks,protocol_sha256=sha(OUT/'protocol.json'),new_run_manifest_hashes=new_manifest_hashes,reused_run_manifest_hashes=protocol['reused_run_manifest_hashes'],report_validation_sha256=sha(OUT/'report_validation.json'),verification_summary_sha256=sha(OUT/'verification_summary.json'),inference_in_simulated_time=False))
 hashes={str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p!=OUT/'artifact_hashes.json'}
 write(OUT/'artifact_hashes.json',hashes);artifacts(OUT,hashes)
 print(json.dumps(dict(status='COMPLETED',new_cases=320,reused_cases=160,checks=total_checks,hashed_files=len(hashes))))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--visual-review-complete',action='store_true');a=p.parse_args();main(a.visual_review_complete)
