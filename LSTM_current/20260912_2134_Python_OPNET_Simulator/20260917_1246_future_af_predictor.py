"""Causal post-REPORT AF rate models for a bounded PONSim experiment.

The LSTM target is offered AF bits in (REPORT snapshot, snapshot+1.5 ms].
TimesFM forecasts a causal REPORT-balance proxy, not a measured arrival history.
No future packet data is available through the deployment predictor interface.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'
import numpy as np

SIM = Path(__file__).resolve().parent
LSTM_ROOT = SIM.parent
DATA = SIM / 'runs/mix_2w_10s/20260916_0712'
OUT = SIM / 'runs/ponsim_future_af/20260917_1246/model'
HORIZON = .0015
FEATURES = ['REPORT_EF_bits', 'REPORT_AF_bits', 'REPORT_BE_bits',
            'previous_applied_GRANT_EF_bits', 'previous_applied_GRANT_AF_bits',
            'previous_applied_GRANT_BE_bits', 'REPORT_interval_seconds']


def stamp():
    return datetime.now(ZoneInfo('Asia/Jakarta')).isoformat()


def sha(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def proxy_rates(report, previous_grant, interval):
    """Observable proxy; allocated grant is not actual delivered service."""
    z = np.zeros(report.shape[:2], dtype=np.float64)
    new = np.maximum(0, report[1:, :, 1] - np.maximum(
        0, report[:-1, :, 1] - previous_grant[1:, :, 1]))
    z[1:] = np.divide(new, interval[1:], out=np.zeros_like(new),
                      where=interval[1:] > 0)
    return z


def prepare():
    OUT.mkdir(parents=True, exist_ok=False)
    arrays, sources, checks, bounds = {}, [], [], []
    for split, seed, per_onu in [('train', 128, 128), ('val', 129, 64), ('test', 130, 64)]:
        d = DATA / f'mix136_load090_seed{seed}'
        manifest = json.loads((d / 'manifest.json').read_text())
        cfg = json.loads((d / 'settings.json').read_text())
        assert manifest['status'] == 'COMPLETED' and manifest['seed'] == seed
        assert manifest['mix'] == '136' and manifest['load_pct'] == 90
        assert cfg['channels'] == 2 and cfg['onu_count'] == 64 and cfg['duration'] == 10
        source_hashes = {}
        for name in ['allocation.npz', 'control.npz', 'offered.npz', 'settings.json']:
            digest = sha(d / name)
            assert digest == manifest['artifacts'][name]
            source_hashes[name] = digest
        with np.load(d / 'allocation.npz') as a:
            r, p = a['report'], a['previous_grant']
            g, dt, t, cycle = a['grant'], a['report_interval'], a['time'], a['cycle']
        assert np.array_equal(cycle, np.arange(1, len(cycle) + 1))
        assert np.array_equal(p[1:], g[:-1]) and not p[0].any()
        with np.load(d / 'control.npz') as c:
            reports = c['reports']
        reports = reports[(reports['cycle'] >= 0) & (reports['cycle'] < len(t))]
        counts = np.zeros((len(t), 64), dtype=np.int16)
        snapshot = np.full((len(t), 64), np.nan)
        receive = np.full((len(t), 64), np.nan)
        ii, oo = reports['cycle'], reports['onu']
        np.add.at(counts, (ii, oo), 1)
        snapshot[ii, oo], receive[ii, oo] = reports['time'], reports['receive_time']
        assert (counts == 1).all()
        assert np.array_equal(r[ii, oo], reports['bits'])
        assert (snapshot <= receive).all() and (receive <= t[:, None] + 1e-12).all()
        proxy = proxy_rates(r, p, dt)
        with np.load(d / 'offered.npz') as a:
            times, onus, queues, bits = a['time'], a['onu'], a['queue'], a['bits']
        records = []
        for onu in range(64):
            take = (onus == onu) & (queues == 1)
            tt, bb = times[take], bits[take]
            assert (np.diff(tt) >= 0).all()
            cumulative = np.r_[0, np.cumsum(bb, dtype=np.int64)]
            eligible = np.flatnonzero((np.arange(len(t)) >= 50) &
                (snapshot[:, onu] >= 3) & (snapshot[:, onu] + HORIZON < 10))
            selected = eligible[np.linspace(0, len(eligible)-1, per_onu, dtype=int)]
            assert len(np.unique(selected)) == per_onu
            for idx in selected:
                start, end = snapshot[idx, onu], snapshot[idx, onu] + HORIZON
                lo, hi = np.searchsorted(tt, [start, end], side='right')
                label = int(cumulative[hi] - cumulative[lo])
                # Direct label recounts at a bounded, deterministic subset.
                if len(records) % 64 == 0:
                    assert label == int(bb[(tt > start) & (tt <= end)].sum())
                records.append((int(idx), onu, label, float(start), float(end)))
        records.sort(key=lambda z: (z[0], z[1]))
        x = np.stack([np.concatenate([r[i-49:i+1, o], p[i-49:i+1, o],
                  dt[i-49:i+1, o, None]], axis=1) for i, o, *_ in records]).astype('f')
        proxy_x = np.stack([proxy[i-49:i+1, o] for i, o, *_ in records]).astype('f')
        y_bits = np.asarray([z[2] for z in records], dtype='d')
        assert x.shape == (per_onu*64, 50, 7) and np.isfinite(x).all()
        assert np.isfinite(proxy_x).all() and (x >= 0).all()
        arrays.update({f'{split}_x': x, f'{split}_proxy': proxy_x,
                       f'{split}_y_rate': y_bits / HORIZON, f'{split}_y_bits': y_bits})
        with open(OUT / f'{split}_origins.csv', 'w') as f:
            w = csv.writer(f)
            w.writerow(['seed', 'onu', 'allocation_index', 'decision_cycle', 'decision_time',
                        'report_snapshot', 'target_end', 'actual_af_bits'])
            for i, o, y, a, b in records:
                w.writerow([seed, o, i, int(cycle[i]), t[i], a, b, y])
        delays = np.asarray([t[i] - a for i, o, y, a, b in records])
        bounds.append(dict(split=split, count=len(records), snapshot_to_decision_min=float(delays.min()),
                           snapshot_to_decision_max=float(delays.max()),
                           label_end_before_or_at_decision=int(np.sum(delays >= HORIZON))))
        sources.append(dict(split=split, seed=seed, path=str(d), hashes=source_hashes,
                            manifest_sha256=sha(d / 'manifest.json')))
        checks.append(dict(split=split, status='PASS', samples=len(records),
                           full_report_cycle_alignment=True, causal_reports=True,
                           previous_grants_aligned=True, balanced_per_onu=per_onu,
                           direct_label_recount=True, target_window_inside_3_to_10=True))
        print('Prepared', split, x.shape, flush=True)
    np.savez_compressed(OUT / 'samples.npz', **arrays)
    tx, ty = arrays['train_x'].astype('d'), arrays['train_y_rate']
    xm, xs = tx.mean(axis=(0, 1)), tx.std(axis=(0, 1))
    xs[xs < 1e-12] = 1
    np.savez(OUT / 'scalers.npz', x_mean=xm, x_std=xs, y_mean=ty.mean(),
             y_std=max(float(ty.std()), 1.0))
    protocol = dict(status='PREPARED', created_at=stamp(), target='AF offered arrival rate in (REPORT snapshot, snapshot+1.5ms]',
        target_bits_horizon_seconds=HORIZON, target_window_seconds=[3, 10], features=FEATURES,
        history_cycles=50, traffic_splits=dict(train=128, val=129, test=130), source_runs=sources,
        sample_selection='Deterministic evenly spaced eligible decisions per ONU; chronological ordering; disjoint traffic seeds.',
        sample_counts=dict(train=8192, val=4096, test=4096),
        lstm_target='Actual offered future-after-REPORT AF arrivals, offline packet labels only',
        timesfm_target_history='Causal REPORT balance proxy; not actual observed arrivals',
        proxy_equation='max(0,R_AF(t)-max(0,R_AF(t-1)-G_AF(t-1)))/observed_REPORT_interval(t)',
        proxy_limit='GRANT is allocated capacity, not actual transmission; proxy includes this measurement error.',
        timesfm_forecast='Next proxy-rate point with 50 proxy points and 50 seven-feature past-only covariates; no future covariates',
        label_timing=bounds, extrapolation='Fixed-horizon rate is multiplied by the allocator own causal service horizon. Stationary-rate approximation.',
        limitations=['Post-REPORT is not wholly future after DBA: earlier reports precede the decision.',
                    'LSTM predicts offered rate mean; frozen TimesFM emits proxy-rate P50. Pipelines have different targets and objectives.',
                    'One held-out traffic seed, one load, one mix, one LSTM initialization.',
                    'No QoS benefit or inference deadline is established by offline metrics.'],
        gpu_required=True, inference_in_simulated_time=False, source_sha256=sha(__file__),
        data_validation=checks, samples_sha256=sha(OUT / 'samples.npz'), scalers_sha256=sha(OUT / 'scalers.npz'))
    write(OUT / 'protocol.json', protocol)
    shutil.copy2(__file__, OUT / Path(__file__).name)


def gpu():
    import torch
    torch.set_num_threads(2)
    assert torch.backends.mps.is_available(), 'Apple MPS GPU is required; CPU fallback is forbidden'
    device = torch.device('mps')
    probe = torch.ones(2, device=device)
    assert str(probe.device) == 'mps:0' and probe.sum().item() == 2
    return torch, device


def network(torch):
    class AFRateLSTM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.first = torch.nn.LSTM(7, 32, batch_first=True)
            self.second = torch.nn.LSTM(32, 32, batch_first=True)
            self.dropout = torch.nn.Dropout(.1)
            self.output = torch.nn.Linear(32, 1)

        def forward(self, x):
            z, _ = self.first(x)
            z, _ = self.second(self.dropout(z))
            return self.output(self.dropout(z[:, -1])).squeeze(-1)
    return AFRateLSTM()


class FutureAFPredictor:
    def __init__(self, mode, model_directory=OUT):
        self.mode, self.directory = mode, Path(model_directory)
        self.seconds, self.points = 0.0, 0
        self.scalers = dict(np.load(self.directory / 'scalers.npz'))
        self.meta = dict(mode=mode, unit='bits/second', cpu_ml_fallback=False,
                         source_sha256=sha(__file__), model_directory=str(self.directory),
                         mean_training_rate_bps=float(self.scalers['y_mean']))
        self.metadata = self.meta
        if mode == 'mean':
            self.meta.update(device='CPU supporting scalar control', mean_rate=float(self.scalers['y_mean']))
        elif mode == 'lstm':
            self.torch, self.device = gpu()
            self.model = network(self.torch).to(self.device)
            self.model.load_state_dict(self.torch.load(self.directory / 'lstm.weights.pt',
                                                       map_location=self.device, weights_only=True))
            self.model.eval()
            assert all(p.device.type == 'mps' for p in self.model.parameters())
            self.meta.update(device='mps:0', framework=self.torch.__version__,
                             weights_sha256=sha(self.directory / 'lstm.weights.pt'))
        elif mode == 'timesfm3':
            self.torch, self.device = gpu()
            sys.path.insert(0, str(LSTM_ROOT / 'vendor/timesfm3/src'))
            from timesfm3 import TimesFM3Forecaster
            checkpoint = LSTM_ROOT / 'vendor/timesfm3_checkpoints/models--google--timesfm-3.0-pytorch/snapshots/43046b85ec22d584a13f8098c2ed39c889e129c2'
            self.model = TimesFM3Forecaster.from_pretrained(str(checkpoint), device='mps', per_core_batch_size=16)
            assert all(p.device.type == 'mps' for p in self.model.model.parameters())
            decode = self.model.model.decode
            self.observed_devices = []
            def checked(**kwargs):
                assert kwargs['target'].device.type == 'mps'
                assert kwargs['past_only_covariates'].device.type == 'mps'
                assert kwargs.get('past_future_covariates') is None
                output = decode(**kwargs)
                assert output.device.type == 'mps'
                self.observed_devices.append(str(output.device))
                return output
            self.model.model.decode = checked
            self.meta.update(device='mps:0', framework=self.torch.__version__, batch_size=16,
                             checkpoint=str(checkpoint), checkpoint_hashes={p.name:sha(p) for p in checkpoint.iterdir() if p.is_file()},
                             target_history='50 causal REPORT-balance proxy rates; not actual arrival history',
                             covariates='50 seven-feature points, past only', point='P50 nonnegative')
        else:
            raise ValueError(mode)

    def predict_rate(self, x, proxy_history=None):
        x = np.asarray(x)
        assert x.ndim == 3 and x.shape[1:] == (50, 7) and np.isfinite(x).all()
        started = time.perf_counter()
        if self.mode == 'mean':
            pred = np.full(len(x), float(self.scalers['y_mean']))
        elif self.mode == 'lstm':
            z = ((x - self.scalers['x_mean']) / self.scalers['x_std']).astype('f')
            with self.torch.inference_mode():
                y = self.model(self.torch.as_tensor(z, device=self.device))
                assert y.device.type == 'mps'
                pred = y.cpu().numpy().astype('d') * self.scalers['y_std'] + self.scalers['y_mean']
            self.torch.mps.synchronize()
        else:
            proxy = np.asarray(proxy_history)
            assert proxy.shape == (len(x), 50) and np.isfinite(proxy).all() and (proxy >= 0).all()
            with self.torch.inference_mode():
                output = self.model.predict_batch([v for v in proxy], horizon=1,
                    past_only_covariates=[v.T for v in x], make_positive=True,
                    use_symmetric_averaging=False, return_quantiles=False)
                pred = np.asarray([v.forecast.item() for v in output], dtype='d')
            self.torch.mps.synchronize()
        pred = np.maximum(0, pred)
        assert pred.shape == (len(x),) and np.isfinite(pred).all()
        self.seconds += time.perf_counter() - started
        self.points += len(x)
        return pred

    predict_rates = predict_rate


def metric_rows(model, split, actual, pred):
    rows = []
    for unit, scale in [('bits/second', 1.), ('bits per 1.5ms', HORIZON)]:
        y, p = actual * scale, pred * scale
        e, variance = p-y, np.sum((y-y.mean())**2)
        rows.append(dict(model=model, target='post_REPORT_AF_arrivals', traffic_class='AF',
            horizon_seconds=HORIZON, partition=split, sample_count=len(y), unit=unit,
            mae=float(np.mean(abs(e))), rmse=float(np.sqrt(np.mean(e*e))),
            r2=float(1-np.sum(e*e)/variance) if variance else None))
    return rows


def train():
    torch, device = gpu()
    torch.manual_seed(42)
    np.random.seed(42)
    a, s = dict(np.load(OUT / 'samples.npz')), dict(np.load(OUT / 'scalers.npz'))
    xx = {k:torch.tensor(((a[k+'_x']-s['x_mean'])/s['x_std']).astype('f'), device=device) for k in ['train','val','test']}
    yy = {k:torch.tensor(((a[k+'_y_rate']-s['y_mean'])/s['y_std']).astype('f'), device=device) for k in xx}
    m = network(torch).to(device)
    opt = torch.optim.RMSprop(m.parameters(), lr=.001, alpha=.9, eps=1e-7)
    assert all(p.device.type == 'mps' for p in m.parameters())
    best, state, waited, history = float('inf'), None, 0, []
    started = time.perf_counter()
    for epoch in range(1, 51):
        m.train(); losses = []
        for i in range(0, len(xx['train']), 256):
            opt.zero_grad(set_to_none=True)
            pred = m(xx['train'][i:i+256]); assert pred.device.type == 'mps'
            loss = ((pred-yy['train'][i:i+256])**2).mean()
            loss.backward(); assert all(p.grad is None or p.grad.device.type == 'mps' for p in m.parameters())
            opt.step(); losses.append(float(loss.detach().cpu()))
        m.eval()
        with torch.inference_mode():
            vp = torch.cat([m(xx['val'][i:i+256]) for i in range(0, len(xx['val']), 256)])
            val = float(((vp-yy['val'])**2).mean().cpu())
        history.append(dict(epoch=epoch, train_loss=float(np.mean(losses)), validation_loss=val))
        write(OUT / 'history.json', history)
        print('Epoch', epoch, 'val', val, flush=True)
        if val < best:
            best, waited, best_epoch = val, 0, epoch
            state = {k:v.detach().clone() for k,v in m.state_dict().items()}
        else:
            waited += 1
        if waited >= 7:
            break
    torch.mps.synchronize()
    training_seconds = time.perf_counter()-started
    m.load_state_dict(state); m.eval()
    torch.save({k:v.cpu() for k,v in state.items()}, OUT / 'lstm.weights.pt')
    predictor = FutureAFPredictor('lstm')
    rows, predictions = [], {}
    for split in ['train','val','test']:
        x, y = a[split+'_x'], a[split+'_y_rate']
        p = np.concatenate([predictor.predict_rate(x[i:i+256]) for i in range(0,len(x),256)])
        predictions[split] = p
        for label, pred in [('LSTM',p),('Training mean',np.full(len(y),float(s['y_mean']))),
                            ('Causal REPORT balance persistence',a[split+'_proxy'][:,-1]),('Zero reserve',np.zeros(len(y)))]:
            rows.extend(metric_rows(label,split,y,pred))
    np.savez_compressed(OUT / 'lstm_predictions.npz', **predictions)
    replay = predictor.predict_rate(a['test_x'][:64])
    assert np.allclose(replay, predictions['test'][:64], atol=2., rtol=1e-5)
    with open(OUT/'offline_metrics.csv','w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    meta = dict(status='COMPLETED', completed_at=stamp(), device='mps:0', framework=torch.__version__,
        parameter_devices=sorted(set(str(v.device) for v in m.parameters())), forward_backward_verified=True,
        cpu_ml_fallback=False, model_seed=42, architecture='LSTM32/dropout0.1/LSTM32/dropout0.1/Dense1',
        optimizer='RMSprop lr0.001 alpha0.9 eps1e-7', loss='standardized actual AF arrival rate MSE',
        training_epochs=epoch,best_epoch=best_epoch,maximum_epochs=50,patience=7,batch_size=256,shuffle=False,
        training_seconds=training_seconds,preprocessing='training-only per-feature and rate z scores',
        mean_training_rate_bps=float(s['y_mean']),weights_sha256=sha(OUT/'lstm.weights.pt'),
        scalers_sha256=sha(OUT/'scalers.npz'),source_sha256=sha(__file__),
        replay_max_difference_bps=float(abs(replay-predictions['test'][:64]).max()),
        metrics=rows, limitations='Offline prediction only; TimesFM target-history semantics differ; no QoS claim')
    write(OUT / 'lstm_metadata.json',meta)
    print('LSTM COMPLETE',json.dumps([r for r in rows if r['partition']=='test' and r['unit']=='bits per 1.5ms']),flush=True)


def timesfm_smoke():
    with np.load(OUT/'samples.npz') as a:
        x, proxy = a['test_x'][:2], a['test_proxy'][:2]
    model = FutureAFPredictor('timesfm3')
    p = model.predict_rate(x,proxy)
    write(OUT/'timesfm_smoke.json',dict(status='PASS',created_at=stamp(),metadata=model.meta,
        observed_forward_devices=model.observed_devices,points=2,predicted_rates_bps=p.tolist(),
        inference_seconds=model.seconds,scope='API/GPU smoke only; not an accuracy benchmark'))
    print('TIMESFM SMOKE PASS',p.tolist(),model.seconds,flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['prepare','train','timesfm_smoke'])
    args=parser.parse_args()
    globals()[args.command]()
