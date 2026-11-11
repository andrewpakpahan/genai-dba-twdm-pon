"""Isolated bounded AF reserve experiment; original simulator is unchanged.

The original offline all-REPORT barrier, EF->AF->BE service, whole-packet FIFO,
GATE dispatch, REPORTs, guard and source rounding are inherited. ``baseline``
executes the original _dba verbatim. The other modes protect the original
Limited EF/AF grants and shorten a round by deferring BE. A predictive AF reserve
is funded only by that ONU's baseline BE grant, never by increasing Wmax.

Prediction is an auxiliary post-REPORT AF rate. The simple rate control uses a
REPORT-balance proxy, while a supplied trained predictor must document its own
labels (the new scenario trains on actual accepted post-REPORT AF arrivals).
The resulting current-cycle GRANT(t) combines REPORT(t) with this explicitly
bounded reserve. No queue or future traffic is read at an allocation decision.
New experiments must compare simple rate and learned controls.
"""
from collections import deque
from importlib import import_module
import math
import numpy as np

core = import_module('20260912_2134_simulator')

DECISION_DTYPE = np.dtype([
    ('cycle', 'i8'), ('onu', 'i2'), ('wavelength', 'i2'),
    ('decision_time', 'f8'), ('report_snapshot_time', 'f8'),
    ('report_receive_time', 'f8'), ('report_interval_s', 'f8'),
    ('window_start_time', 'f8'), ('window_end_time', 'f8'),
    ('window_count', 'i2'), ('prediction_active', '?'),
    ('reports', 'f8', (3,)), ('previous_grants', 'f8', (3,)),
    ('baseline', 'f8', (3,)), ('final', 'f8', (3,)),
    ('cap_bits', 'f8'), ('predicted_af_rate_bps', 'f8'),
    ('credit_horizon_s', 'f8'), ('actual_af_horizon_s', 'f8'),
    ('af_credit_bits', 'f8'), ('be_deferred_bits', 'f8'),
    ('credit_scale', 'f8'), ('target_cycle_us', 'f8'),
    ('planned_interval_us', 'f8'), ('target_infeasible', '?'),
    ('timing_bound_verified', '?'),
])
CYCLE_DTYPE = np.dtype([
    ('cycle', 'i8'), ('decision_time', 'f8'), ('planned_next_dba_time', 'f8'),
    ('planned_interval_us', 'f8'), ('target_cycle_us', 'f8'),
    ('target_infeasible', '?'), ('timing_bound_verified', '?'),
    ('af_credit_bits', 'f8'), ('be_deferred_bits', 'f8'),
])


class BoundedAFPolicy:
    """Reusable causal-history/rate adapter, with no scheduling or queue access.

Inputs in each window are current EF/AF/BE REPORT, previous *applied* EF/AF/BE
GRANT, and the raw per-ONU REPORT interval in seconds. The simple rate control
uses sum(max(0, R_AF(t)-max(0,R_AF(t-1)-G_AF(t-1)))) / sum(interval) over
the same history length. A predictor's predict(N,window,7) returns N rates in
bits/second. Histories are neither clipped nor scaled here.
"""
    def __init__(self, mode='short', predictor=None, activation_time=1., window=50):
        if mode not in ('baseline', 'short', 'rate', 'lstm'):
            raise ValueError('Unknown bounded experiment mode')
        if mode == 'lstm' and predictor is None:
            raise ValueError('LSTM mode requires an explicit trained predictor')
        if not isinstance(window, int) or isinstance(window, bool) or window < 1:
            raise ValueError('window must be a positive integer')
        if not np.isfinite(activation_time) or activation_time < 0:
            raise ValueError('activation_time must be finite and nonnegative')
        self.mode, self.predictor = mode, predictor
        self.activation_time, self.window = float(activation_time), window
        self.reset()

    def reset(self):
        self.features = deque(maxlen=self.window)
        self.times = deque(maxlen=self.window)
        self.proxies = deque(maxlen=self.window)
        self.proxy_intervals = deque(maxlen=self.window)
        self.previous_reports = None
        self.last_cycle = 0
        self.last_time = -1.

    def observe(self, reports, previous_grants, intervals, receive_times, now, cycle):
        reports = np.asarray(reports, dtype=float)
        previous_grants = np.asarray(previous_grants, dtype=float)
        intervals = np.asarray(intervals, dtype=float)
        receive_times = np.asarray(receive_times, dtype=float)
        n = len(reports)
        if (reports.shape != (n, 3) or previous_grants.shape != (n, 3)
                or intervals.shape != (n,) or receive_times.shape != (n,)
                or not np.isfinite(reports).all() or not np.isfinite(previous_grants).all()
                or not np.isfinite(intervals).all() or (reports < 0).any()
                or (previous_grants < 0).any() or (intervals < 0).any()):
            raise ValueError('Invalid causal REPORT, previous GRANT or interval')
        if cycle != self.last_cycle + 1 or now < self.last_time:
            raise ValueError('Causal history requires consecutive increasing cycles')
        if (np.isfinite(receive_times) & (receive_times > now + 1e-12)).any():
            raise ValueError('REPORT feature timestamp is in the future')
        feature = np.column_stack((reports, previous_grants, intervals))
        proxy = np.zeros(n)
        valid_interval = np.zeros(n)
        if self.previous_reports is not None:
            valid = intervals > 0
            proxy[valid] = np.maximum(0., reports[valid, 1] - np.maximum(
                0., self.previous_reports[valid, 1] - previous_grants[valid, 1]))
            valid_interval[valid] = intervals[valid]
        self.features.append(feature)
        self.times.append(receive_times.copy())
        self.proxies.append(proxy)
        self.proxy_intervals.append(valid_interval)
        self.previous_reports = reports.copy()
        self.last_cycle, self.last_time = cycle, now
        ready = len(self.features) == self.window and all((x[:, 6] > 0).all() for x in self.features)
        active = self.mode in ('rate', 'lstm') and now >= self.activation_time and ready
        rates = np.zeros(n)
        if active and self.mode == 'rate':
            denominator = np.sum(self.proxy_intervals, axis=0)
            rates = np.divide(np.sum(self.proxies, axis=0), denominator,
                              out=np.zeros(n), where=denominator > 0)
        elif active:
            # A fresh array prevents callbacks mutating retained causal history.
            windows = np.stack(self.features, axis=1)
            rates = np.asarray(self.predictor.predict(windows), dtype=float).reshape(-1)
            if rates.shape != (n,) or not np.isfinite(rates).all() or (rates < 0).any():
                raise ValueError('Predictor must return finite nonnegative N AF rates in bit/s')
        self.active, self.rates = bool(active), rates
        return rates


class BoundedSimulator(core.Simulator):
    """New scenarios only; the source Simulator and its baseline are untouched.

``target_cycle_us`` bounds prospective current DBA -> next all-REPORT receipt,
including the inherited scheduling lag, guard slots, REPORT wire time and
propagation. It is a ceiling, never an idle-time pad. If EF+reported AF alone
cannot fit, they remain intact, BE is zero on that wavelength, and infeasibility
is recorded. AF reserve itself is reduced if necessary before BE is allocated.

Prediction horizon uses an earliest feasible AF start with *zero BE*. Earlier
AF reserves are included sequentially. Retaining BE afterwards can only move the
actual AF start later, so there is no circular horizon claim or future oracle.

The prospective next-REPORT time is exact for one shared upstream channel, or
for independent upstream serialization. With shared serialization and >1
channel, inherited REPORTs use wire 0; target bounds are explicitly unverified
and the actual packet/control event simulation remains authoritative.
"""
    def __init__(self, config, mode='baseline', predictor=None, target_cycle_us=1200.,
                 credit_fraction=.5, activation_time=1., window=50):
        super().__init__(config)
        if not np.isfinite(target_cycle_us) or target_cycle_us <= 0:
            raise ValueError('target_cycle_us must be finite and positive')
        if not np.isfinite(credit_fraction) or not 0 <= credit_fraction <= 1:
            raise ValueError('credit_fraction must be between zero and one')
        self.mode, self.target_cycle_us = mode, float(target_cycle_us)
        self.credit_fraction = float(credit_fraction)
        self.policy = BoundedAFPolicy(mode, predictor, activation_time, window)
        self.decision_rows, self.cycle_rows = [], []

    def decision_array(self):
        return np.concatenate(self.decision_rows) if self.decision_rows else np.empty(0, DECISION_DTYPE)

    def cycle_array(self):
        return np.array(self.cycle_rows, dtype=CYCLE_DTYPE)

    def _channel_schedule(self, grants, channel, available, *, output=False):
        cfg = self.config
        group = cfg.onu_count // cfg.channels
        cursor = available + cfg.get_channel_offsets()[channel]
        starts = np.zeros((group, 4)) if output else None
        lengths = np.zeros((group, 4)) if output else None
        report_end = cursor
        for j, u in enumerate(range(channel*group, (channel+1)*group)):
            cursor = math.ceil(cursor*1e9 + .5)/1e9
            for q in range(3):
                length = grants[u, q] / self.channel_rates[channel]
                if output:
                    starts[j, q], lengths[j, q] = cursor, length
                cursor += length
            if output:
                starts[j, 3], lengths[j, 3] = cursor, cfg.report_tx_time
            report_end = cursor + cfg.report_tx_time
            cursor += cfg.report_tx_time + cfg.guard_time
        return report_end + cfg.report_propagation, cursor, starts, lengths

    def _channel_credit(self, baseline, rates, snapshots, channel, available, scale):
        cfg = self.config
        group = cfg.onu_count // cfg.channels
        cursor = available + cfg.get_channel_offsets()[channel]
        credit, horizon = np.zeros(group), np.zeros(group)
        for j, u in enumerate(range(channel*group, (channel+1)*group)):
            cursor = math.ceil(cursor*1e9 + .5)/1e9
            cursor += baseline[u, 0]/self.channel_rates[channel]
            horizon[j] = max(0., cursor-snapshots[u]) if np.isfinite(snapshots[u]) else 0.
            credit[j] = scale * min(rates[u]*horizon[j], self.credit_fraction*baseline[u, 2])
            cursor += (baseline[u, 1]+credit[j])/self.channel_rates[channel]
            report_end = cursor+cfg.report_tx_time
            cursor += cfg.report_tx_time+cfg.guard_time
        return credit, horizon, report_end+cfg.report_propagation

    def _dba(self, now):
        cfg = self.config
        if self.cycle == 0:
            self.policy.reset()
            self.decision_rows, self.cycle_rows = [], []
        reports, previous = self.reports.copy(), self.previous_grants.copy()
        cycle = self.cycle+1
        rates = self.policy.observe(reports, previous, self.intervals.copy(),
                                    self.report_receive_times.copy(), now, cycle)
        baseline = self._baseline(reports, now, previous, self.intervals, cycle)
        # REPORT generation reads occupancy at its slot start. No queue is read
        # here: for c1/independent serialization its timestamp is exactly causal
        # receipt minus configured REPORT wire time and propagation.
        snapshots = self.report_receive_times - cfg.report_tx_time - cfg.report_propagation
        available = float(np.max(self.next_available))
        if available < now:
            available = now+cfg.gate_tx_time
        available += cfg.gate_tx_time
        group = cfg.onu_count//cfg.channels
        wavelengths = np.arange(cfg.onu_count)//group
        grants = baseline.copy()
        credits, horizons = np.zeros(cfg.onu_count), np.zeros(cfg.onu_count)
        scales = np.zeros(cfg.channels)
        infeasible = np.zeros(cfg.channels, dtype=bool)
        deadline = now+self.target_cycle_us*1e-6
        if self.mode == 'baseline':
            super()._dba(now)
            available = self.dba_times[-1][1]
        else:
            grants[:, 2] = 0.
            for ch in range(cfg.channels):
                sl = slice(ch*group, (ch+1)*group)
                minimum_end = self._channel_schedule(grants, ch, available)[0]
                infeasible[ch] = minimum_end > deadline+1e-12
                if not infeasible[ch]:
                    c, h, credit_end = self._channel_credit(baseline, rates, snapshots, ch, available, 1.)
                    scale = 1.
                    if credit_end > deadline:
                        lo, hi = 0., 1.
                        for _ in range(30):
                            mid = (lo+hi)/2
                            if self._channel_credit(baseline, rates, snapshots, ch, available, mid)[2] <= deadline:
                                lo = mid
                            else:
                                hi = mid
                        scale = lo
                        c, h, _ = self._channel_credit(baseline, rates, snapshots, ch, available, scale)
                    credits[sl], horizons[sl], scales[ch] = c, h, scale
                    grants[sl, 1] += c
                    eligible = np.maximum(0., baseline[sl, 2]-c)
                    grants[sl, 2] = eligible
                    if self._channel_schedule(grants, ch, available)[0] > deadline:
                        lo, hi = 0., 1.
                        for _ in range(30):
                            mid = (lo+hi)/2
                            grants[sl, 2] = eligible*mid
                            if self._channel_schedule(grants, ch, available)[0] <= deadline:
                                lo = mid
                            else:
                                hi = mid
                        grants[sl, 2] = eligible*lo
                else:
                    # EF and already-reported AF are never reduced to claim a
                    # target that the physical protected service cannot meet.
                    _, horizons[sl], _ = self._channel_credit(baseline, rates, snapshots, ch, available, 0.)
            caps = np.asarray(cfg.get_base_caps())
            if (not np.isfinite(grants).all() or (grants < 0).any()
                    or not np.array_equal(grants[:, 0], baseline[:, 0])
                    or (grants[:, 1] < baseline[:, 1]-1e-7).any()
                    or (grants[:, 1] > baseline[:, 1]+credits+1e-7).any()
                    or (grants[:, 2] > baseline[:, 2]-credits+1e-7).any()
                    or (grants.sum(axis=1) > caps+1e-7).any()):
                raise AssertionError('Bounded Limited grant or protected EF/AF contract violated')
            self.cycle = cycle
        starts, lengths = np.zeros((cfg.onu_count, 4)), np.zeros((cfg.onu_count, 4))
        next_receives = np.zeros(cfg.channels)
        for ch in range(cfg.channels):
            sl = slice(ch*group, (ch+1)*group)
            next_receives[ch], cursor, starts[sl], lengths[sl] = self._channel_schedule(grants, ch, available, output=True)
            if self.mode != 'baseline':
                self.next_available[ch] = cursor
        if self.mode != 'baseline':
            self.previous_grants[:] = grants
            self.dba_times.append((now, available))
            self._dispatch(now, starts, lengths, wavelengths, cycle)
        bound_verified = not cfg.shared_upstream_serialization or cfg.channels == 1
        row = np.zeros(cfg.onu_count, dtype=DECISION_DTYPE)
        row['cycle'], row['onu'], row['wavelength'] = cycle, np.arange(cfg.onu_count), wavelengths
        row['decision_time'], row['report_snapshot_time'] = now, snapshots
        row['report_receive_time'], row['report_interval_s'] = self.report_receive_times, self.intervals
        row['window_start_time'], row['window_end_time'] = self.policy.times[0], self.policy.times[-1]
        row['window_count'], row['prediction_active'] = len(self.policy.features), self.policy.active
        row['reports'], row['previous_grants'] = reports, previous
        row['baseline'], row['final'] = baseline, grants
        row['cap_bits'], row['predicted_af_rate_bps'] = cfg.get_base_caps(), rates
        row['credit_horizon_s'] = horizons
        row['actual_af_horizon_s'] = np.maximum(0., starts[:, 1]-snapshots)
        row['af_credit_bits'], row['be_deferred_bits'] = credits, baseline[:, 2]-grants[:, 2]-credits
        row['credit_scale'], row['target_cycle_us'] = scales[wavelengths], self.target_cycle_us
        row['planned_interval_us'] = (next_receives[wavelengths]-now)*1e6
        row['target_infeasible'], row['timing_bound_verified'] = infeasible[wavelengths], bound_verified
        self.decision_rows.append(row)
        self.cycle_rows.append((cycle, now, float(next_receives.max()),
                               float((next_receives.max()-now)*1e6), self.target_cycle_us,
                               bool(infeasible.any()), bound_verified,
                               float(credits.sum()), float(row['be_deferred_bits'].sum())))
