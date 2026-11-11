"""Fixed historical AF-only Informer/PatchTST on exact saved two-wave samples.

No resampling, new simulator execution, hyperparameter search, or CPU ML fallback.
"""
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'
import argparse
import datetime
import gc
import hashlib
import importlib
import json
import random
import shutil
import sys
import time
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
H = next(p for p in HERE.parents if p.name == 'LSTM_current')
BASE = H / '20260912_2134_Python_OPNET_Simulator/runs/mix_2w_10s'
SOURCES = {'136': BASE/'20260916_0835_mix136_comparison',
           '244': BASE/'20260916_0848_mix244_comparison'}
EXPECTED = {'136': 'd7387288b934f2963df892b166c0c5eeb1dbb8fecd16c16f7942a663ef125057',
            '244': 'ad88895e8c9b13a0c6027fb201d30278941e0aff4f960a9215269e00548d98c2'}
sys.path.insert(0, str(H))
INF = importlib.import_module('20260730_2000_run_informer_grant_7features')
PATCH = importlib.import_module('20260916_0655_train_patchtst_af_only')
PARTS = ('train', 'val', 'test')
COUNTS = {'train': 20480, 'val': 10240, 'test': 10240}
DEVICE = torch.device('mps')
FIXED = dict(seed=42, batch_size=256, max_epochs=50, patience=7,
             optimizer='Adam', learning_rate=.001, shuffle=False, gradient_clipping=False,
             loss='standardized AF mean squared error',
             preprocessing='StandardScaler fitted on training input rows and training AF targets only')
INF_CONFIG = dict(feature_count=7, target_count=1, pred_len=1, d_model=64,
                  n_heads=4, d_ff=128, e_layers=2, d_layers=1, dropout=.1)

def stamp():
    return datetime.datetime.now(ZoneInfo('Asia/Jakarta')).isoformat()

def sha(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')

def scores(y, p):
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    p = np.asarray(p, dtype=np.float64).reshape(-1)
    e = p-y
    denom = np.square(y-y.mean()).sum()
    return dict(mae_bits=float(np.abs(e).mean()), rmse_bits=float(np.sqrt(np.square(e).mean())),
                r2=float(1-np.square(e).sum()/denom) if denom > 0 else None)

def build(kind):
    return (INF.build_model(**INF_CONFIG) if kind == 'Informer' else PATCH.PatchTSTAF()).to(DEVICE)

def forward(model, x, kind):
    assert x.device.type == 'mps'
    p = INF._batch_predict(model, x, label_len=25, device=DEVICE) if kind == 'Informer' else model(x)
    assert p.device.type == 'mps' and p.shape == (len(x), 1)
    return p

def predict(model, tensors, kind):
    # ProbSparse attention samples CPU indices even in eval. Preserve historical
    # fixed seed4242 per partition and restore training RNG after validation.
    state = torch.get_rng_state()
    if kind == 'Informer':
        gen = torch.Generator().manual_seed(4242)
        torch.set_rng_state(gen.get_state())
    model.eval()
    with torch.no_grad():
        p = np.concatenate([forward(model, x.to(DEVICE), kind).cpu().numpy()
                            for x in tensors.split(FIXED['batch_size'])])
    torch.set_rng_state(state)
    return p

def main(args):
    if not torch.backends.mps.is_available():
        raise RuntimeError('MPS GPU unavailable; CPU model fallback prohibited')
    torch.set_num_threads(2)
    source = SOURCES[args.mix]
    protocol = json.loads((source/'protocol.json').read_text())
    assert protocol['status'] == 'COMPLETED'
    assert sha(source/'samples.npz') == EXPECTED[args.mix] == protocol['samples_sha256']
    a = np.load(source/'samples.npz')
    raw = {k: a[k] for k in a.files}
    origins = {s: pd.read_csv(source/f'{s}_origins.csv.gz') for s in PARTS}
    for s, seed in zip(PARTS, [128, 129, 130]):
        assert raw[s+'_x'].shape == (COUNTS[s], 50, 7)
        assert raw[s+'_y'].shape == (COUNTS[s],)
        assert np.isfinite(raw[s+'_x']).all() and np.isfinite(raw[s+'_y']).all()
        assert (raw[s+'_x'] >= 0).all() and (raw[s+'_y'] >= 0).all()
        assert len(origins[s]) == COUNTS[s] and set(origins[s].seed) == {seed}
        assert set(origins[s].load) == set(range(10, 101, 10))
        assert set(origins[s].onu) == set(range(64))
        assert origins[s].groupby('load').size().eq(COUNTS[s]//10).all()
        assert not origins[s].duplicated(['run', 'cycle', 'onu']).any()
    out = HERE/f'mix{args.mix}_{args.model.lower()}'
    if args.preflight:
        print(json.dumps(dict(status='PREFLIGHT_PASS_NO_ML', device='mps',
                              framework=torch.__version__, counts=COUNTS,
                              sample_sha256=EXPECTED[args.mix], output=str(out))))
        return
    out.mkdir(exist_ok=False)
    sources = [Path(__file__), H/'20260915_0722_train_casebase_af_only.py',
               H/'20260730_2000_run_informer_grant_7features.py',
               H/'20260916_0655_train_patchtst_af_only.py']
    sources += sorted((H/'vendor/PatchTST/PatchTST_supervised').rglob('*.py'))
    source_hashes = {str(p): sha(p) for p in sources}
    snapshot = out/'source_snapshot'
    snapshot.mkdir()
    for p in sources:
        rel = p.relative_to(H)
        target = snapshot/rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, target)
    save(out/'source_hashes.json', source_hashes)
    shutil.copy2(source/'protocol.json', out/'source_protocol.json')
    shutil.copy2(source/'final_validation.json', out/'source_final_validation.json')
    for s in PARTS:
        shutil.copy2(source/f'{s}_origins.csv.gz', out/f'{s}_origins.csv.gz')
    meta = dict(status='PREPARING', started_at=stamp(), model=args.model, mix=args.mix,
                target='AF GRANT(t)', features=protocol['features'], window=50,
                scenario=protocol['scenario'], split=protocol['split'], sampling=protocol['sampling'],
                target_window=protocol['target_window'], counts=COUNTS, fixed_training=FIXED,
                architecture=INF_CONFIG if args.model == 'Informer' else PATCH.CONFIG,
                architecture_extra=('ProbSparse factor5, GELU, embed_type4, distillation, label_len25; eval CPU attention seed4242'
                                    if args.model == 'Informer' else 'Learned7-to1 AF output head; RevIN disabled'),
                historical_training_source=str(H/('20260915_0722_train_casebase_af_only.py'
                                      if args.model == 'Informer' else '20260916_0655_train_patchtst_af_only.py')),
                source_samples=str(source/'samples.npz'), source_samples_sha256=EXPECTED[args.mix],
                source_protocol_sha256=sha(source/'protocol.json'),
                origins_sha256={s:sha(source/f'{s}_origins.csv.gz') for s in PARTS},
                framework=torch.__version__, python=sys.executable, device='mps',
                mps_available=torch.backends.mps.is_available(), cpu_ml_fallback=False,
                fallback_environment=os.environ['PYTORCH_ENABLE_MPS_FALLBACK'],
                runner_sha256=sha(Path(__file__)), source_hashes=str(out/'source_hashes.json'),
                inference_in_simulated_time=False, new_network_simulation=False,
                comparability='Exact same samples as saved LSTM/TimesFM3; method-specific historical preprocessing and objectives differ. Not architecture-only controlled. Inference timings are local unsynchronized-workload observations.')
    save(out/'metadata.json', meta)
    try:
        sx = StandardScaler().fit(raw['train_x'].reshape(-1, 7))
        sy = StandardScaler().fit(raw['train_y'].reshape(-1, 1))
        np.savez(out/'scalers.npz', x_mean=sx.mean_, x_scale=sx.scale_, y_mean=sy.mean_, y_scale=sy.scale_)
        xx = {s:torch.from_numpy(sx.transform(raw[s+'_x'].reshape(-1, 7)).reshape(-1, 50, 7).astype('f')) for s in PARTS}
        yy = {s:torch.from_numpy(sy.transform(raw[s+'_y'].reshape(-1, 1)).astype('f')) for s in PARTS}
        # Separate GPU smoke model; reset all seeds before actual initialization.
        smoke = build(args.model)
        result = forward(smoke, xx['train'][:4].to(DEVICE), args.model)
        result.square().mean().backward()
        grads = [p.grad for p in smoke.parameters() if p.grad is not None]
        assert grads and all(g.device.type == 'mps' and torch.isfinite(g).all() for g in grads)
        assert sum(g.abs().sum().item() for g in grads) > 0
        gpu = dict(status='PASS', forward_device=str(result.device),
                   parameter_devices=sorted({str(p.device) for p in smoke.parameters()}),
                   backward_gradient_devices=sorted({str(g.device) for g in grads}),
                   parameter_count=sum(p.numel() for p in smoke.parameters()),
                   used_gradient_tensor_count=len(grads), fallback_disabled=True)
        save(out/'gpu_audit.json', gpu)
        del smoke, result, grads
        random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.mps.manual_seed(42)
        model = build(args.model)
        opt = torch.optim.Adam(model.parameters(), lr=.001)
        meta.update(status='TRAINING', gpu_audit=gpu)
        save(out/'metadata.json', meta)
        history=[];best=float('inf');bad=0;best_epoch=0
        torch.mps.synchronize();started=time.perf_counter()
        for epoch in range(1, 51):
            epoch_started=time.perf_counter();model.train();losses=[]
            for xb, yb in zip(xx['train'].split(256), yy['train'].split(256)):
                opt.zero_grad(set_to_none=True)
                pred=forward(model, xb.to(DEVICE), args.model)
                loss=(pred-yb.to(DEVICE)).square().mean()
                assert torch.isfinite(loss)
                loss.backward();opt.step();losses.append(loss.item())
            vp=predict(model, xx['val'], args.model)
            vl=float(np.mean([np.square(vp[i:i+256]-yy['val'][i:i+256].numpy()).mean()
                              for i in range(0, len(vp), 256)]))
            assert np.isfinite(vl)
            history.append(dict(epoch=epoch, train_loss=float(np.mean(losses)),
                                val_loss=vl, seconds=time.perf_counter()-epoch_started))
            pd.DataFrame(history).to_csv(out/'history.csv', index=False)
            print(f'mix={args.mix} model={args.model} epoch={epoch} val_loss={vl:.8f}', flush=True)
            if vl < best:
                best=vl;bad=0;best_epoch=epoch;torch.save(model.state_dict(), out/'model.pt')
            else:
                bad+=1
            if bad >= 7:
                break
        torch.mps.synchronize();train_seconds=time.perf_counter()-started
        model.load_state_dict(torch.load(out/'model.pt', map_location=DEVICE, weights_only=True))
        predictions={};timings={}
        for s in PARTS:
            torch.mps.synchronize();start=time.perf_counter()
            predictions[s]=sy.inverse_transform(predict(model, xx[s], args.model)).clip(min=0).astype('float64').reshape(-1)
            torch.mps.synchronize();timings[s]=time.perf_counter()-start
            assert np.isfinite(predictions[s]).all()
        np.savez_compressed(out/'predictions.npz', **predictions,
                            **{s+'_actual':raw[s+'_y'].astype('float64') for s in PARTS})
        fresh=build(args.model)
        fresh.load_state_dict(torch.load(out/'model.pt', map_location=DEVICE, weights_only=True))
        replay=sy.inverse_transform(predict(fresh, xx['test'], args.model)).clip(min=0).reshape(-1)
        diff=float(np.max(np.abs(predictions['test']-replay)))
        assert np.allclose(predictions['test'], replay, atol=.01, rtol=1e-5), diff
        rows=[]
        for s in PARTS:
            for load in ['all', *range(10, 101, 10)]:
                ids=np.arange(COUNTS[s]) if load=='all' else np.flatnonzero(origins[s].load.to_numpy()==load)
                rows.append(dict(model=args.model, mix=args.mix, partition=s, load=load,
                                 traffic_class='AF', target='GRANT(t)', samples=len(ids), unit='bits',
                                 **scores(raw[s+'_y'][ids], predictions[s][ids])))
        pd.DataFrame(rows).to_csv(out/'metrics.csv', index=False)
        controls={'Persistence':raw['test_x'][:,-1,4],
                  'Train mean':np.full(COUNTS['test'], raw['train_y'].mean(dtype=np.float64)),
                  'LSTM':np.load(source/'lstm_predictions.npz')['test'],
                  'TimesFM 3':np.load(source/'timesfm3_predictions.npz')['test']}
        pd.DataFrame([dict(model=k, **scores(raw['test_y'], p)) for k,p in controls.items()]).to_csv(out/'saved_controls.csv', index=False)
        assert all(sha(p)==v for p,v in source_hashes.items()), 'Source changed during run'
        verification=dict(status='PASS', input_archive_sha256_match=True,
                          exact_origin_files_preserved=True, count_load_seed_onu_checks=True,
                          train_only_scalers=True, gpu_forward_backward=True,
                          checkpoint_replay_max_difference_bits=diff,
                          checkpoint_replay_atol=.01, checkpoint_replay_rtol=1e-5,
                          finite_nonnegative_predictions=True, source_hashes_unchanged=True,
                          prior_raw_window_validation=str(source/'final_validation.json'))
        save(out/'verification.json', verification)
        meta.update(status='COMPLETED', completed_at=stamp(), epochs=epoch, best_epoch=best_epoch,
                    best_val_loss=best, training_seconds=train_seconds,
                    inference_seconds=timings, checkpoint_sha256=sha(out/'model.pt'),
                    predictions_sha256=sha(out/'predictions.npz'))
        save(out/'metadata.json', meta)
        save(out/'artifact_hashes.json', {str(p.relative_to(out)):sha(p) for p in out.iterdir() if p.is_file()})
        summary=[r for r in rows if r['partition']=='test' and r['load']=='all'][0]
        (out/'RESULTS.md').write_text(f'# {args.model}: mix{args.mix}, matched current AF GRANT(t)\n\n'
            f'Completed on MPS; all 10,240 saved test origins. No new network simulation.\n\n'
            f'MAE {summary["mae_bits"]:.9f} bits; RMSE {summary["rmse_bits"]:.9f} bits; R² {summary["r2"]:.9f}.\n\n'
            f'Trained {epoch} epochs; validation-selected epoch {best_epoch}; training {train_seconds:.3f} seconds; '
            f'test inference {timings["test"]:.3f} seconds. Checkpoint replay maximum difference {diff:.9f} bits.\n\n'
            'Same samples, splits, target and causal information as saved LSTM/TimesFM3. Historical method-specific '
            'preprocessing, objective and output architecture are retained; this is not an architecture-only controlled comparison. '
            'Offline grant error does not establish packet-delay, jitter, throughput or packet-loss improvement.\n')
        with (HERE/'RUN_LOG.md').open('a') as f:
            f.write(f'\n{stamp()} COMPLETED mix{args.mix} {args.model}: {json.dumps(summary)}; output {out.name}.\n')
        print('COMPLETED', json.dumps(summary), flush=True)
        del model, fresh, xx, yy
        gc.collect();torch.mps.empty_cache()
    except BaseException as exc:
        meta.update(status='FAILED', failed_at=stamp(), error=repr(exc))
        save(out/'metadata.json', meta)
        with (HERE/'RUN_LOG.md').open('a') as f:
            f.write(f'\n{stamp()} FAILED mix{args.mix} {args.model}: {repr(exc)}\n')
        raise

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--mix', choices=['136','244'], required=True)
    parser.add_argument('--model', choices=['Informer','PatchTST'], required=True)
    parser.add_argument('--preflight', action='store_true')
    main(parser.parse_args())
