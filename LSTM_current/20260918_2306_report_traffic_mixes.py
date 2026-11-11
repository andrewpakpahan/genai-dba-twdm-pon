"""Render measured three-mix PON comparisons; saved outputs only, no ML/DES."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib,json,os,zipfile
os.environ['MPLCONFIGDIR']='/private/tmp/ponsim-mix-charts'
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter,LogLocator,NullFormatter
H=Path(__file__).resolve().parent;S=H/'20260912_2134_Python_OPNET_Simulator'
OUT=S/'runs/ponsim_traffic_mixes/20260918_2258'
OLD=S/'runs/ponsim_full_loads/20260917_2213';PRIOR=S/'runs/ponsim_adaptive_loads/20260917_1933';NINETY=S/'runs/ponsim_adaptive/20260917_1321'
MIXES=['136','244','163'];LOADS=list(range(10,101,10));SEEDS=[131,132]
MODES=['baseline','ipact_flex','noforecast','mean','lstm','timesfm3','timesfm3_raw','oracle']
MAIN=['baseline','lstm','timesfm3_raw']
LABELS={'baseline':'IPACT','lstm':'LSTM','timesfm3_raw':'GenAI (raw TimesFM 3)','timesfm3':'Calibrated TimesFM pipeline','ipact_flex':'IPACT + flexible service','noforecast':'Adaptive, no forecast','mean':'Adaptive + mean','oracle':'Adaptive + oracle'}
COLORS={'baseline':'#535f6c','lstm':'#c66b12','timesfm3_raw':'#1e5faf'}
MARKERS={'baseline':'o','lstm':'s','timesfm3_raw':'D'}
STYLES={'136':'-','244':'--','163':':'}
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,j):Path(p).write_text(json.dumps(j,indent=2,allow_nan=False)+'\n')
def stamp():return datetime.now(ZoneInfo('Asia/Jakarta')).isoformat()
def root(mix,load):
 if mix!='136':return OUT/('mix'+mix)/f'load{load:03d}'
 return NINETY if load==90 else (PRIOR if load in [60,80,100] else OLD)/f'load{load:03d}'
def modelroot(mix,load):
 if mix!='136':return OUT/('mix'+mix)/f'models/load{load:03d}'
 return S/'runs/ponsim_future_af/20260917_1329/model' if load==90 else (PRIOR if load in [60,80,100] else OLD)/f'models/load{load:03d}'
def main():
 for role in ['inputs','controls','models','audit']:assert read(OUT/(role+'_JOB.json'))['status']=='COMPLETED',role
 protocol=read(OUT/'protocol.json');assert protocol['main_plot_methods']==MAIN
 rows=[];waves=[];runtime=[];selection=[];sources={};case_rows=[];packet_weight_checks=0
 for mix in MIXES:
  for load in LOADS:
   qpath=modelroot(mix,load)/'quality.json';quality=read(qpath);sources[str(qpath)]=sha(qpath)
   selected=quality['calibration_selected']
   selection.append(dict(mix=mix,load_percent=load,selection=selected['mode'],uses_active_forecast=quality['calibration_uses_forecast'],ridge=selected['ridge'],validation_rmse_bits=selected['validation']['rmse_bits']))
   for seed in SEEDS:
    for mode in MODES:
     d=root(mix,load)/f'seed{seed}'/mode;m=read(d/'manifest.json');assert m['status']=='COMPLETED' and m['mode']==mode and m['seed']==seed
     assert m['measurement']==[3.,3.1] and m['inference_in_simulated_time'] is False
     if 'mix' in m:assert m['mix']==mix
     if mix=='136':assert sha(d/'manifest.json')==protocol['reused_run_manifest_hashes'][str(d/'manifest.json')]
     assert sha(d/'performance.json')==m['artifacts']['performance.json']
     sources[str(d/'manifest.json')]=sha(d/'manifest.json');sources[str(d/'performance.json')]=sha(d/'performance.json')
     p=read(d/'performance.json');block={r['traffic_class']:r for r in p['primary']['rows'] if r['wavelength']=='all'}
     for field in ['completed_packets','completed_packet_bits','packet_throughput_bps']:
      assert np.isclose(sum(block[c][field] for c in ['ef','af','be']),block['total'][field],rtol=1e-12,atol=1e-6)
     count=sum(block[c]['completed_packets'] for c in ['ef','af','be'])
     weighted=sum(block[c]['delay_mean_ms']*block[c]['completed_packets'] for c in ['ef','af','be'])/count
     assert np.isclose(weighted,block['total']['delay_mean_ms'],rtol=1e-11,atol=1e-10)
     packet_weight_checks+=1
     for r in p['primary']['rows']:
      row=dict(mix=mix,load_percent=load,seed=seed,mode=mode,method=LABELS[mode],**r);waves.append(row)
      if r['wavelength']=='all':rows.append(row)
     case_rows.append(dict(mix=mix,load_percent=load,seed=seed,mode=mode,EF_ms=block['ef']['delay_mean_ms'],AF_ms=block['af']['delay_mean_ms'],BE_ms=block['be']['delay_mean_ms'],ETE_ms=weighted,total_Gbps=block['total']['packet_throughput_gbps'],BE_loss_fraction=block['be']['dropped_packet_fraction']))
     runtime.append(dict(mix=mix,load_percent=load,seed=seed,mode=mode,wall_seconds=m['wall_seconds'],calendar_elapsed_seconds=(datetime.fromisoformat(m['finished_at'])-datetime.fromisoformat(m['started_at'])).total_seconds(),elapsed_clock='perf_counter',inference_seconds=m['inference_seconds'],inference_calls=m['inference_calls'],device=(m.get('model_final') or {}).get('device'),inference_in_simulated_time=False))
 df=pd.DataFrame(rows);assert len(df)==1920 and len(waves)==5760
 assert not df.duplicated(['mix','load_percent','seed','mode','traffic_class']).any()
 df.to_csv(OUT/'qos_metrics.csv',index=False);pd.DataFrame(waves).to_csv(OUT/'qos_by_wavelength.csv',index=False)
 pd.DataFrame(runtime).to_csv(OUT/'runtime.csv',index=False);pd.DataFrame(selection).to_csv(OUT/'model_selection.csv',index=False)
 cases=pd.DataFrame(case_rows);cases.to_csv(OUT/'case_comparison.csv',index=False)
 columns=['EF_ms','AF_ms','BE_ms','ETE_ms','total_Gbps','BE_loss_fraction']
 aggregate=cases.groupby(['mix','load_percent','mode'])[columns].agg(['mean','min','max'])
 aggregate.columns=['_'.join(c) for c in aggregate.columns];aggregate=aggregate.reset_index();aggregate.to_csv(OUT/'comparison_seed_means_ranges.csv',index=False)
 criteria=[]
 for mix in MIXES:
  for load in LOADS:
   for seed in SEEDS:
    b=cases[(cases.mix==mix)&(cases.load_percent==load)&(cases.seed==seed)].set_index('mode');base=b.loc['baseline']
    for mode in MODES:
     row=b.loc[mode];checks=dict(throughput_higher=bool(row.total_Gbps>base.total_Gbps),EF_no_worse=bool(row.EF_ms<=base.EF_ms),AF_no_worse=bool(row.AF_ms<=base.AF_ms),BE_no_worse=bool(row.BE_ms<=base.BE_ms),BE_loss_no_worse=bool(row.BE_loss_fraction<=base.BE_loss_fraction))
     item=dict(mix=mix,load_percent=load,seed=seed,mode=mode,**checks,overall_criterion_met=all(checks.values()))
     item.update({k+'_change_percent':100*(float(row[k])/float(base[k])-1) for k in columns if k!='BE_loss_fraction'})
     item['BE_loss_change_percentage_points']=100*(row.BE_loss_fraction-base.BE_loss_fraction);criteria.append(item)
 criteria=pd.DataFrame(criteria);criteria.to_csv(OUT/'relative_to_IPACT_and_criterion.csv',index=False)
 plotted=[];visual_checks=[];names=[]
 plt.rcParams.update({'font.family':'DejaVu Sans','text.color':'#192733','axes.labelcolor':'#33424f','xtick.color':'#33424f','ytick.color':'#33424f','font.size':12})
 for number,metric,title,log in [(1,'EF_ms','EF delay',False),(2,'AF_ms','AF delay',False),(3,'BE_ms','BE delay',True),(4,'ETE_ms','Average ETE delay',True)]:
  fig,ax=plt.subplots(figsize=(12.5,8));fig.subplots_adjust(left=.105,right=.98,bottom=.15,top=.73)
  fig.suptitle(title,fontsize=27,fontweight='bold',y=.98)
  for mode in MAIN:
   for mix in MIXES:
    b=aggregate[(aggregate.mix==mix)&(aggregate['mode']==mode)].set_index('load_percent').loc[LOADS]
    y=b[metric+'_mean'].to_numpy();lo=b[metric+'_min'].to_numpy();hi=b[metric+'_max'].to_numpy()
    assert np.isfinite(y).all() and (y>0).all()
    label=('GenAI' if mode=='timesfm3_raw' else LABELS[mode])+f' ({mix})'
    line,=ax.plot(LOADS,y,color=COLORS[mode],marker=MARKERS[mode],markersize=5.4,markerfacecolor=COLORS[mode] if mix=='136' else 'white',markeredgewidth=1.4,linestyle=STYLES[mix],linewidth=2.1,label=label)
    ax.errorbar(LOADS,y,yerr=np.array([y-lo,hi-y]),fmt='none',ecolor=COLORS[mode],alpha=.38,elinewidth=.75,capsize=1.6)
    np.testing.assert_array_equal(line.get_xdata(),LOADS);np.testing.assert_allclose(line.get_ydata(),y,rtol=0,atol=0)
    for load,mean,minimum,maximum in zip(LOADS,y,lo,hi):plotted.append(dict(graph=number,metric=metric,mix=mix,mode=mode,load_percent=load,mean=float(mean),minimum=float(minimum),maximum=float(maximum)))
  if log:
   ax.set_yscale('log');ax.yaxis.set_major_locator(LogLocator(base=10));ax.yaxis.set_major_formatter(ScalarFormatter());ax.yaxis.set_minor_formatter(NullFormatter())
  else:ax.set_ylim(bottom=0)
  ax.set_xlim(8,102);ax.set_xticks(LOADS);ax.set_xlabel('Traffic load (%)',fontsize=15)
  ax.set_ylabel(('Average ETE' if metric=='ETE_ms' else 'Mean delay')+(' (ms, log scale)' if log else ' (ms)'),fontsize=15)
  ax.grid(axis='y',which='major',color='#dde3ea',linewidth=.8);ax.set_axisbelow(True)
  for side in ['top','right']:ax.spines[side].set_visible(False)
  for side in ['bottom','left']:ax.spines[side].set_color('#909ca8')
  handles,labels=ax.get_legend_handles_labels();legend=fig.legend(handles,labels,ncol=3,loc='upper center',bbox_to_anchor=(.5,.91),frameon=False,fontsize=12,handlelength=2.6,columnspacing=2)
  fig.text(.5,.045,'Measured · 2 wavelengths · 3 s warm-up + 100 ms measurement · mean of 2 seeds; whiskers = range.\nGenAI = raw TimesFM 3. Inference latency excluded. Mix labels give EF:AF:BE ratios.',ha='center',fontsize=9,color='#53616d')
  fig.canvas.draw();renderer=fig.canvas.get_renderer()
  assert legend.get_window_extent(renderer).y0>ax.get_window_extent(renderer).y1,'Legend overlaps axes'
  assert len(handles)==9 and len(fig.axes)==1
  visual_checks.append(dict(graph=number,series=9,points=90,legend_clear_of_axes=True,one_axes=True,log_scale=log))
  name=f'{number:02d}_{title.replace(" ","_")}_136_244_163';names.append(name)
  fig.savefig(OUT/(name+'.png'),dpi=240);fig.savefig(OUT/(name+'.pdf'));plt.close(fig)
 assert len(plotted)==360
 pd.DataFrame(plotted).to_csv(OUT/'plotted_values.csv',index=False)
 success=criteria.groupby(['mix','mode']).overall_criterion_met.sum().unstack().reindex(index=MIXES,columns=MODES)
 table=aggregate[aggregate['mode'].isin(MAIN)][['mix','load_percent','mode']+[c+'_mean' for c in columns]]
 fallback=pd.DataFrame(selection);fallback=fallback[~fallback.uses_active_forecast]
 report=['# Measured PON comparison: traffic mixes 136, 244 and 163','',
 'Completed comparison:320 new cases for244/163 plus160 reused136 cases;480 total, eight methods, ten loads and two confirmation seeds per mix. The four main plots compare IPACT, LSTM and raw TimesFM3. Raw versus calibrated TimesFM was fixed in protocol before the new simulation outputs were available; calibrated and all control results remain in the exports.','',
 '| Mix | EF | AF | BE |','|---|---:|---:|---:|','|136|10%|30%|60%|','|244|20%|40%|40%|','|163|10%|60%|30%|','',
 '## Four separate comparison plots','']
 report += [f'- [{n.replace("_"," ")} PNG]({n}.png) · [PDF]({n}.pdf)' for n in names]
 report += ['', '## Overall improvement screen','',
 'Counts out of20 same-seed/load cases per mix. Requires strictly higher completed-packet throughput, no worse EF/AF/BE mean delay and no worse BE drop fraction than IPACT. This descriptive screen is not a significance test; IPACT itself cannot satisfy its strict improvement condition.', '',success.to_markdown(),'',
 '## Measured delay and throughput','',table.to_markdown(index=False,floatfmt='.6f'),'',
 '## Calibrated pipeline fallback','',fallback.to_markdown(index=False) if len(fallback) else 'No mean fallback selected.', '',
 'These fallback cells have no active TimesFM forecast in the calibrated pipeline. The separately reported raw TimesFM arm always uses the frozen forecast. Neither confirmation QoS nor these plots select a model or scheduler setting.','',
 '## Scenario and interpretation','',
 '64 ONUs; two1Gbit/s wavelengths;46,875-bit ONU cap;1.5ms payload budget. Actual full cycle lengths vary with load and control overhead. Identical confirmation traffic within each mix/load/seed across methods. Each case shares3s IPACT warm-up, followed by100ms intervention and measurement[3,3.1). This is a transient response, not steady state. Every-cycle inference uses the same two-query horizon pipeline; model runtime is excluded from simulated network time.','',
 'Models fit separately per mix and load:seed128 training,129 validation,130 diagnostic only; confirmation131/132. Same1024/512/512 origins,50x7 input, eight schedule rotations and frozen model hyperparameters. New training/inference requires MPS and records actual GPU evidence. Supporting10s training datasets are isolated and match recorded historical input hashes.','',
 'Predictors target actual AF arrivals after the latest REPORT snapshot and through the causally planned AF service start. This includes unreported arrivals before the DBA decision as well as future arrivals after it. TimesFM uses a causal REPORT/grant-balance proxy history; a matched target does not make this an architecture-only comparison.', '',
 'Runtime exports distinguish perf_counter elapsed time from calendar timestamp differences; pauses or system sleep can make them differ. Neither measure is added to the simulated network clock. A50s CPU-worker pause was used for a bounded contention probe; affected CPU elapsed times include it. See parallel_contention_probe.json.', '',
 'Delay = original offered arrival to wire completion, among packets completing within the measurement window. Average ETE pools completed packets across classes within each seed; it does not weight class delays by configured traffic percentages. Curves then average the two seed-level values equally. Whiskers show min/max of those two seeds, not confidence intervals. Warm-up backlog contributes to delay; unfinished packets are censored. BE/ETE axes use log scales.','',
 'Traffic mix ratios are nominal bit-rate shares. AF/BE arrival rates retain the historical6400-bit nominal size convention, while the generator mean is6328.5bits. Measured offered load can differ. Source hashes and manifest settings, not labels alone, establish the scenario.','',
 'IPACT gains combine scheduler changes and forecasting. Mean/no-forecast/flexible controls support attribution; better prediction alone does not guarantee better QoS. The oracle is a privileged input to the same constrained scheduler, not a globally optimal bound. Inherited nominal burst/REPORT-boundary behavior is recorded in independent audits; this work retains the prior engine for a matched comparison.','',
 'The attached style references had hypothetical LSTM/GenAI values; these new plots use measured simulation outputs only. Prior outputs are preserved.','',
 '## Evidence and exports','',
 '[All QoS](qos_metrics.csv) · [Per wavelength](qos_by_wavelength.csv) · [Case comparison](case_comparison.csv) · [Seed means and ranges](comparison_seed_means_ranges.csv) · [Plotted data](plotted_values.csv) · [IPACT changes and criterion](relative_to_IPACT_and_criterion.csv) · [Model selection](model_selection.csv) · [Runtime](runtime.csv) · [Protocol](protocol.json).']
 (OUT/'RESULTS.md').write_text('\n'.join(report)+'\n')
 write(OUT/'report_validation.json',dict(status='PASS_PENDING_VISUAL_REVIEW',created_at=stamp(),source_hashes=sources,reporter_sha256=sha(__file__),case_count=480,qos_rows=1920,wavelength_rows=5760,packet_weight_checks=packet_weight_checks,plotted_rows=360,render_checks=visual_checks,plot_names=names))
 print('REPORT_READY',len(rows),'QoS rows;',len(plotted),'plotted values');print(success.to_string())
if __name__=='__main__':main()
