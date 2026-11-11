"""Isolated causal AF-arrival reserve with packet-capable rotating BE visits.

The historical simulator and rotation engine are unchanged. Before activation,
all arms execute the core IPACT scheduler exactly while retaining causal input
history. Learned policies receive REPORT / applied-GRANT history, never queue
state or future arrivals. The explicitly separate oracle callback is privileged.

For shared two-wavelength upstream serialization, the configured deadline is a
*planned scheduling target*, not a claimed exact next-REPORT bound. Actual next
DBA intervals are joined into the saved cycle array after execution. REPORT
snapshot timestamps are exact event timestamps carried through REPORT receipt.
"""
from collections import deque
from importlib import import_module
import math
import time
import numpy as np

core = import_module('20260912_2134_simulator')
rotation = import_module('20260913_0858_rotating_be_engine')

DECISION_DTYPE = np.dtype(rotation.DECISION_DTYPE.descr + [
    ('policy_active', '?'), ('oracle_only', '?'),
    ('snapshot_timestamp_exact', '?'), ('report_wire_queue_delay_s', 'f8'),
    ('proxy_af_rate_bps', 'f8'), ('prediction_seconds', 'f8'),
])
CYCLE_DTYPE = np.dtype(rotation.CYCLE_DTYPE.descr + [
    ('policy_active', '?'), ('prediction_seconds', 'f8'),
    ('actual_next_dba_time', 'f8'), ('actual_interval_us', 'f8'),
    ('actual_interval_available', '?'), ('actual_target_met', '?'),
])


class _CausalHistory:
    def __init__(self, window):
        self.window = window
        self.reset()

    def reset(self):
        self.features = deque(maxlen=self.window)
        self.times = deque(maxlen=self.window)
        self.proxies = deque(maxlen=self.window)
        self.previous_reports = None
        self.last_cycle = 0
        self.active = False

    def observe(self, reports, previous, intervals, receive_times, now, cycle):
        if cycle != self.last_cycle + 1:
            raise AssertionError('Causal history must contain consecutive cycles')
        if not (np.isfinite(reports).all() and np.isfinite(previous).all()
                and np.isfinite(intervals).all()):
            raise AssertionError('Nonfinite causal input')
        if (reports < 0).any() or (previous < 0).any() or (intervals < 0).any():
            raise AssertionError('Negative causal input')
        if (receive_times > now + 1e-12).any():
            raise AssertionError('Unreceived REPORT used in current decision')
        proxy = np.zeros(len(reports))
        if self.previous_reports is not None:
            bits = np.maximum(0., reports[:, 1] - np.maximum(
                0., self.previous_reports[:, 1] - previous[:, 1]))
            np.divide(bits, intervals, out=proxy, where=intervals > 0)
        self.features.append(np.column_stack((reports, previous, intervals)))
        self.times.append(receive_times.copy())
        self.proxies.append(proxy)
        self.previous_reports = reports.copy()
        self.last_cycle = cycle
        return (len(self.features) == self.window
                and all((v[:, 6] > 0).all() for v in self.features))


class PredictiveRotatingBESimulator(rotation.RotatingBESimulator):
    """A future-AF-rate adapter plus protected-EF/AF rotating-BE scheduler.

    ``predictor.predict_rate(x, proxy_history)`` receives raw x[N,window,7]
    and causal REPORT-balance proxy rates[N,window], and returns bit/s[N].
    This proxy is not measured true arrivals: past grants can be partly unused.
    ``oracle_rate_query(snapshot, horizon, onu=u)`` is accepted only in mode ``oracle``;
    it is intentionally excluded from the learned-policy call path.
    """
    MODES = ('baseline', 'ipact', 'rotation', 'short', 'rate', 'mean',
             'lstm', 'timesfm3', 'predictor', 'oracle')

    def __init__(self, config, mode='baseline', predictor=None,
                 target_cycle_us=1500., credit_fraction=.5, activation_time=3.,
                 window=50, be_visits_per_cycle=12, max_packet_bits=12144.,
                 oracle_rate_query=None):
        if mode not in self.MODES:
            raise ValueError('Unknown predictive rotation mode: ' + str(mode))
        if mode in ('mean', 'lstm', 'timesfm3', 'predictor') and predictor is None:
            raise ValueError('Learned/mean mode requires an explicit predictor')
        if mode == 'oracle' and oracle_rate_query is None:
            raise ValueError('Oracle mode requires a separately labelled callback')
        if mode != 'oracle' and oracle_rate_query is not None:
            raise ValueError('Future-arrival oracle cannot enter a causal policy')
        if mode == 'oracle' and predictor is not None:
            raise ValueError('Oracle and learned predictors are separate arms')
        super().__init__(config, mode='short', predictor=None,
                         target_cycle_us=target_cycle_us,
                         credit_fraction=credit_fraction,
                         activation_time=activation_time, window=window,
                         be_visits_per_cycle=be_visits_per_cycle,
                         max_packet_bits=max_packet_bits)
        self.mode = 'baseline' if mode == 'ipact' else 'rotation' if mode == 'short' else mode
        self.predictor = predictor
        self.oracle_rate_query = oracle_rate_query
        self.activation_time = float(activation_time)
        self.window = int(window)
        self.policy = _CausalHistory(window)
        self.metadata = {
            'role': 'future_AF_arrival_rate_to_bounded_current_GRANT',
            'baseline_warmup_seconds': self.activation_time,
            'target_cycle_us': float(target_cycle_us),
            'deadline_semantics': 'planned next REPORT scheduling target; actual next DBA intervals saved',
            'shared_multiwavelength_timing_bound_verified': False,
            'report_snapshot_timestamp': 'exact REPORT generation event, carried through receipt',
            'predictor_input': 'current REPORT3, prior applied GRANT3, REPORT receipt interval; 50-cycle history',
            'proxy_target_history': 'causal REPORT balance / interval; grants are not actual transmitted bits',
            'inference_in_simulated_time': False,
            'oracle_only': self.mode == 'oracle',
            'oracle_horizon': 'exact future arrivals to earliest reserved-BE AF start; optional BE can delay actual AF start, so not an outcome upper bound',
            'learned_horizon_assumption': 'fixed 1.5ms trained arrival rate extrapolated across current causal reserve horizon',
            'protected_AF': 'never below baseline current reported AF grant',
            'packet_capability': 'every positive BE grant >= min(current BE REPORT, configured max packet)',
        }

    def run(self, traffic, external_gates=None):
        if external_gates is not None:
            raise ValueError('This experiment requires autonomous closed-loop REPORTs')
        self.report_snapshot_times = np.full(self.config.onu_count, np.nan)
        self.policy.reset()
        self.rotation[:] = 0
        self.decision_rows = []
        self.cycle_rows = []
        self.predictor_calls = 0
        self.prediction_seconds = 0.
        self.oracle_query_seconds = 0.
        return super().run(traffic)

    def _report(self, now, u, gate):
        if (self._cmp(gate.starts[3], now) == 1
                or self._cmp(gate.starts[3] + gate.lengths[3], now) != 1):
            return
        # Exactly the inherited ONU snapshot and serialization behavior. The
        # timestamp adds causal control metadata; it does not expose queue state.
        snapshot = self.occupancy[u].copy()
        received = self._wire_completion(
            now, gate.lengths[3] * self.channel_rates[0], 0) + (self.config.report_propagation or 0)
        if self.config.record_control:
            self.report_rows.append((now, received, u, gate.cycle, snapshot))
        self._push(received, 'report_rx', (u, snapshot, now))

    def _report_rx(self, now, u, snapshot, snapshot_time):
        if snapshot_time > now + 1e-12:
            raise AssertionError('REPORT snapshot is later than receipt')
        self.report_snapshot_times[u] = snapshot_time
        return core.Simulator._report_rx(self, now, u, snapshot)

    def decision_array(self):
        return np.concatenate(self.decision_rows) if self.decision_rows else np.empty(0, DECISION_DTYPE)

    def cycle_array(self):
        out = np.asarray(self.cycle_rows, dtype=CYCLE_DTYPE)
        if len(out) > 1:
            out['actual_next_dba_time'][:-1] = out['decision_time'][1:]
            out['actual_interval_us'][:-1] = np.diff(out['decision_time']) * 1e6
            out['actual_interval_available'][:-1] = True
            out['actual_target_met'][:-1] = out['actual_interval_us'][:-1] <= self.target_cycle_us + 1e-6
        return out

    def _predict(self, snapshots):
        n = self.config.onu_count
        rates = np.zeros(n)
        if self.mode in ('baseline', 'rotation'):
            return rates, 0.
        if self.mode == 'oracle':
            # The privileged callback is queried only after the reserve
            # scheduler determines each ONU's causal candidate horizon.
            return rates, 0.
        x = np.stack(self.policy.features, axis=1)
        proxy = np.stack(self.policy.proxies, axis=1)
        started = time.perf_counter()
        if self.mode == 'rate':
            duration = x[:, :, 6].sum(axis=1)
            rates = np.divide((proxy*x[:, :, 6]).sum(axis=1), duration,
                              out=np.zeros(n), where=duration > 0)
        else:
            rates = self.predictor.predict_rate(x, proxy)
        elapsed = time.perf_counter() - started
        rates = np.asarray(rates, dtype=float).reshape(-1)
        if rates.shape != (n,) or not np.isfinite(rates).all() or (rates < 0).any():
            raise ValueError('Predictor must return finite nonnegative AF rate bit/s[N]')
        self.predictor_calls += 1
        self.prediction_seconds += elapsed
        return rates, elapsed

    def _reserved_credit(self, baseline, rates, snapshots, reserved,
                         credit_caps, ch, available, scale):
        if self.mode != 'oracle':
            return super()._reserved_credit(baseline, rates, snapshots, reserved,
                                            credit_caps, ch, available, scale)
        cfg = self.config
        group = cfg.onu_count // cfg.channels
        cursor = available + cfg.get_channel_offsets()[ch]
        credits, horizons = np.zeros(group), np.zeros(group)
        for j, u in enumerate(range(ch*group, (ch+1)*group)):
            cursor = math.ceil(cursor*1e9+.5)/1e9
            cursor += baseline[u, 0]/self.channel_rates[ch]
            horizons[j] = max(0., cursor-snapshots[u])
            started = time.perf_counter()
            rate = (float(self.oracle_rate_query(float(snapshots[u]),
                                                float(horizons[j]), onu=u))
                    if scale > 0 and horizons[j] > 0 else 0.)
            self.oracle_query_seconds += time.perf_counter()-started
            if not np.isfinite(rate) or rate < 0:
                raise ValueError('Oracle must return nonnegative arrival rate bit/s')
            rates[u] = rate
            credits[j] = scale*min(rate*horizons[j], credit_caps[u])
            cursor += (baseline[u, 1]+credits[j]+reserved[u])/self.channel_rates[ch]
            report_end = cursor+cfg.report_tx_time
            cursor += cfg.report_tx_time+cfg.guard_time
        return credits, horizons, report_end+cfg.report_propagation

    def _dba(self, now):
        cfg = self.config
        n = cfg.onu_count
        group = n // cfg.channels
        cycle = self.cycle + 1
        reports, previous = self.reports.copy(), self.previous_grants.copy()
        ready = self.policy.observe(reports, previous, self.intervals.copy(),
                                    self.report_receive_times.copy(), now, cycle)
        snapshots = self.report_snapshot_times.copy()
        if not np.isfinite(snapshots).all() or (snapshots > self.report_receive_times + 1e-12).any():
            raise AssertionError('Each allocation requires a received exact-timestamp REPORT')
        active = self.mode != 'baseline' and now >= self.activation_time and ready
        self.policy.active = active
        base = self._baseline(reports, now, previous, self.intervals, cycle)
        oracle_before = self.oracle_query_seconds
        rates, elapsed = self._predict(snapshots) if active else (np.zeros(n), 0.)
        available = float(np.max(self.next_available))
        if available < now:
            available = now + cfg.gate_tx_time
        available += cfg.gate_tx_time
        deadline = now + self.target_cycle_us*1e-6
        grants = base.copy()
        reserved, selected = np.zeros(n), np.zeros(n, bool)
        rank = np.full(n, -1, dtype=int)
        rejected, deferred = np.zeros(n, bool), np.zeros(n, bool)
        minimum = np.minimum(reports[:, 2], self.max_packet_bits)
        eligible = (minimum > 0) & (base[:, 2] >= minimum - 1e-7)
        credits, horizons = np.zeros(n), np.zeros(n)
        scales, infeasible = np.zeros(cfg.channels), np.zeros(cfg.channels, bool)
        rotation_start = self.rotation.copy()
        counts = np.zeros(cfg.channels, dtype=int)
        credit_caps = np.minimum(self.credit_fraction*base[:, 2], base[:, 2])
        goal = min(self.be_visits_per_cycle, group)
        if not active:
            # The historical core scheduler is called verbatim, preserving all
            # warmup GATE timestamps, class grants and REPORT feedback.
            core.Simulator._dba(self, now)
            available = self.dba_times[-1][1]
        else:
            grants[:, 2] = 0.
            for ch in range(cfg.channels):
                lo = ch*group
                sl = slice(lo, lo+group)
                order = lo + (self.rotation[ch]+np.arange(group)) % group
                infeasible[ch] = self._channel_schedule(grants, ch, available)[0] > deadline+1e-12
                last_selected = None
                if not infeasible[ch]:
                    for u in order:
                        if counts[ch] >= goal:
                            break
                        if not eligible[u]:
                            continue
                        grants[u, 2] = minimum[u]
                        if self._channel_schedule(grants, ch, available)[0] <= deadline:
                            reserved[u], selected[u] = minimum[u], True
                            rank[u] = counts[ch]
                            counts[ch] += 1
                            last_selected = int(u-lo)
                        else:
                            grants[u, 2] = 0.
                            rejected[u] = True
                    credit_caps[sl] = np.minimum(self.credit_fraction*base[sl, 2],
                                                np.maximum(0., base[sl, 2]-reserved[sl]))
                    c, h, finish = self._reserved_credit(base, rates, snapshots, reserved,
                                                         credit_caps, ch, available, 1.)
                    scale = 1.
                    if finish > deadline:
                        low, high = 0., 1.
                        for _ in range(30):
                            mid = (low+high)/2
                            if self._reserved_credit(base, rates, snapshots, reserved,
                                                     credit_caps, ch, available, mid)[2] <= deadline:
                                low = mid
                            else:
                                high = mid
                        scale = low
                        c, h, _ = self._reserved_credit(base, rates, snapshots, reserved,
                                                        credit_caps, ch, available, scale)
                    credits[sl], horizons[sl], scales[ch] = c, h, scale
                    grants[sl, 1] += c
                    for u in order:
                        candidate = max(0., base[u, 2]-credits[u])
                        if not eligible[u] or candidate < minimum[u]-1e-7:
                            continue
                        if candidate <= grants[u, 2]+1e-7:
                            continue
                        saved = grants[u, 2]
                        grants[u, 2] = candidate
                        if self._channel_schedule(grants, ch, available)[0] > deadline:
                            grants[u, 2], deferred[u] = saved, True
                else:
                    _, horizons[sl], _ = self._reserved_credit(base, rates, snapshots, reserved,
                                                              credit_caps, ch, available, 0.)
                self.rotation[ch] = ((last_selected+1) % group if last_selected is not None
                                     else (self.rotation[ch]+1) % group)
            positive = grants[:, 2] > 1e-7
            if (not np.isfinite(grants).all() or (grants < 0).any()
                    or not np.array_equal(grants[:, 0], base[:, 0])
                    or (grants[:, 1] < base[:, 1]-1e-7).any()
                    or (grants[:, 1] > base[:, 1]+credits+1e-7).any()
                    or (credits+grants[:, 2] > base[:, 2]+1e-7).any()
                    or (grants.sum(axis=1) > np.asarray(cfg.get_base_caps())+1e-7).any()
                    or (grants[selected, 2] < reserved[selected]-1e-7).any()
                    or (grants[positive, 2] < minimum[positive]-1e-7).any()):
                raise AssertionError('Protected service/cap/BE packet capability violation')
        starts, lengths = np.zeros((n, 4)), np.zeros((n, 4))
        next_receives = np.zeros(cfg.channels)
        wavelengths = np.arange(n)//group
        for ch in range(cfg.channels):
            sl = slice(ch*group, (ch+1)*group)
            next_receives[ch], cursor, starts[sl], lengths[sl] = self._channel_schedule(
                grants, ch, available, output=True)
            if active:
                self.next_available[ch] = cursor
        if active:
            self.cycle = cycle
            self.previous_grants[:] = grants
            self.dba_times.append((now, available))
            self._dispatch(now, starts, lengths, wavelengths, cycle)
        verified = not cfg.shared_upstream_serialization or cfg.channels == 1
        oracle_elapsed = self.oracle_query_seconds-oracle_before
        elapsed += oracle_elapsed
        self.prediction_seconds += oracle_elapsed
        row = np.zeros(n, DECISION_DTYPE)
        fields = dict(cycle=cycle, onu=np.arange(n), wavelength=wavelengths, decision_time=now,
            report_snapshot_time=snapshots, report_receive_time=self.report_receive_times,
            report_interval_s=self.intervals, window_start_time=self.policy.times[0],
            window_end_time=self.policy.times[-1], window_count=len(self.policy.features),
            prediction_active=active and self.mode != 'rotation', reports=reports,
            previous_grants=previous, baseline=base, final=grants, cap_bits=cfg.get_base_caps(),
            predicted_af_rate_bps=rates, credit_horizon_s=horizons,
            actual_af_horizon_s=np.maximum(0., starts[:, 1]-snapshots), af_credit_bits=credits,
            be_deferred_bits=base[:, 2]-grants[:, 2]-credits, credit_scale=scales[wavelengths],
            target_cycle_us=self.target_cycle_us, planned_interval_us=(next_receives[wavelengths]-now)*1e6,
            target_infeasible=infeasible[wavelengths], timing_bound_verified=verified,
            be_selected=selected, be_eligible=eligible, be_reserved_bits=reserved,
            credit_cap_bits=credit_caps, be_rotation_start=rotation_start[wavelengths],
            be_rotation_next=self.rotation[wavelengths], be_selection_rank=rank,
            be_selection_budget_rejected=rejected, be_optional_full_grant_deferred=deferred,
            be_reserved_visits_target=goal if active else 0, be_reserved_visits_actual=counts[wavelengths],
            max_packet_bits=self.max_packet_bits, policy_active=active, oracle_only=self.mode == 'oracle',
            snapshot_timestamp_exact=True, report_wire_queue_delay_s=np.maximum(0.,
                self.report_receive_times-snapshots-cfg.report_tx_time-cfg.report_propagation),
            proxy_af_rate_bps=self.policy.proxies[-1], prediction_seconds=elapsed)
        for name, value in fields.items():
            row[name] = value
        self.decision_rows.append(row)
        requested = goal*cfg.channels if active else 0
        self.cycle_rows.append((cycle, now, float(next_receives.max()),
            float((next_receives.max()-now)*1e6), self.target_cycle_us,
            bool(infeasible.any()), verified, float(credits.sum()),
            float(row['be_deferred_bits'].sum()), int(selected.sum()), int(eligible.sum()),
            requested, requested-int(selected.sum()), float(reserved.sum()), active,
            elapsed, np.nan, np.nan, False, False))


PredictiveRotationSimulator = PredictiveRotatingBESimulator
