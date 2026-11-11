"""Causal demand ordering and protected predictive reserve in flexible bursts.

The full same-state IPACT grant envelope is retained. Forecast AF credits replace
at most 25% of an ONU's baseline BE allowance; BE reclaims abandoned AF capacity
inside the unchanged burst. The allocator sees REPORTs and prior grants only.
"""
from importlib import import_module
import math
import time
import numpy as np

flex = import_module('20260917_1310_flexible_burst_engine')
predictive = flex.predictive
core = predictive.core

DECISION_DTYPE = np.dtype(predictive.DECISION_DTYPE.descr + [
    ('schedule_rank', 'i2'), ('forecast_horizon_reference_s', 'f8'),
    ('same_state_baseline_report_end_s', 'f8'), ('planned_deadline_s', 'f8'),
    ('be_floor_bits', 'f8'), ('be_report_occupancy_fraction', 'f8'),
    ('reserve_confidence_scale', 'f8'), ('be_opportunity_age_before', 'i4'),
    ('be_opportunity_age_after', 'i4'), ('be_age_urgent', '?'),
    ('be_age_constraint_infeasible', '?'), ('forecast_order_score_bits', 'f8'),
    ('ordering_prediction_rate_bps', 'f8'),
])
CYCLE_DTYPE = predictive.CYCLE_DTYPE


class AdaptivePredictiveSimulator(flex.FlexibleBurstSimulator):
    """No runtime queue/future access except the explicitly isolated oracle arm.

    Predictor API: ``predict_rate(x[N,50,7], proxy[N,50], horizons[N])``.
    Horizons are REPORT-snapshot to AF start in the same-state IPACT schedule.
    Ordering then changes this horizon; prediction is evaluated again for the
    final AF start. Total ONU grant is conserved, so no credit/horizon iteration
    is necessary. TimesFM can cache its raw forecast for this calibration-only
    second call. Oracle re-queries its privileged callback at that final start.
    """
    def __init__(self, config, mode='baseline', predictor=None, flexible=True,
                 credit_fraction=.25, be_floor_fraction=.75,
                 be_max_wait_rounds=2, **kwargs):
        if mode == 'noforecast':
            mode = 'rotation'
        if not 0 <= credit_fraction <= .25:
            raise ValueError('Frozen policy permits at most 25% baseline BE credit')
        if not .75 <= be_floor_fraction <= 1:
            raise ValueError('BE floor must preserve at least 75% baseline BE')
        if be_max_wait_rounds != 2:
            raise ValueError('Frozen BE opportunity urgency threshold is two rounds')
        super().__init__(config, mode=mode, predictor=predictor,
                         flexible=bool(flexible and mode not in ('baseline', 'ipact')),
                         credit_fraction=credit_fraction, **kwargs)
        self.be_floor_fraction = float(be_floor_fraction)
        self.be_max_wait_rounds = int(be_max_wait_rounds)
        self.be_age = np.zeros(config.onu_count, dtype=np.int64)
        self.metadata.update(
            scheduler='adaptive demand ordering within static wavelength groups',
            scheduler_modes={'baseline': 'unmodified IPACT', 'rotation': 'flexible no-forecast report-based ordering'},
            credit_fraction=credit_fraction, be_floor_fraction=be_floor_fraction,
            selective_reserve='credit cap times max(0, 1 - BE REPORT / configured BE queue cap); no outcome-dependent tuning',
            reserve_confidence='calibration occurs in predictor; fixed congestion suppression, no model-specific scheduler advantage',
            learned_horizon_assumption='first forecast uses same-state IPACT snapshot-to-AF-start target for ordering; second forecast uses final reordered AF-start horizon for reserve',
            oracle_horizon='exact privileged arrivals between REPORT snapshot and final scheduled AF start; not a global optimum',
            deadline_semantics='same-state full IPACT report-end envelope plus at most one nanosecond per ONU for reordered rounding',
            configured_cycle_budget_us=self.target_cycle_us,
            shortened_round_budget=False,
            ordering_score='2*reported EF + reported AF + predicted AF at reference horizon; two-round BE opportunity urgency first; rotating ties',
            be_max_wait_rounds=be_max_wait_rounds,
            be_wait_guarantee='Every baseline-packet-eligible ONU retains a packet-capable BE grant every round; urgency threshold two rounds tracks grant opportunities, not observed packet service. Baseline insufficient capacity is recorded as infeasible.',
            actual_wire_service_timing_bound_verified=False,
            gate_dispatch='active GATEs serialize by earliest scheduled service globally across wavelengths; common warmup keeps original ONU-ID order',
            gate_readiness='all positive EF and contiguous AF/BE windows and REPORT starts checked against physical GATE receive time',
            OLT_queue_access=False,
        )

    def run(self, traffic, external_gates=None):
        self.be_age[:] = 0
        return super().run(traffic, external_gates=external_gates)

    def decision_array(self):
        return np.concatenate(self.decision_rows) if self.decision_rows else np.empty(0, DECISION_DTYPE)

    def cycle_array(self):
        out = super().cycle_array()
        if len(out) > 1:
            out['actual_target_met'][:-1] = out['actual_interval_us'][:-1] <= out['target_cycle_us'][:-1]+1e-6
        return out

    def _ordered_schedule(self, grants, ch, available, order):
        cfg, group = self.config, self.config.onu_count//self.config.channels
        lo = ch*group
        if sorted(map(int, order)) != list(range(lo, lo+group)):
            raise AssertionError('Ordering must be a permutation within one static wavelength')
        cursor = available+cfg.get_channel_offsets()[ch]
        starts, lengths = np.zeros((group, 4)), np.zeros((group, 4))
        report_end = cursor
        for u in order:
            cursor = math.ceil(cursor*1e9+.5)/1e9
            j = u-lo
            for q in range(3):
                starts[j, q], lengths[j, q] = cursor, grants[u, q]/self.channel_rates[ch]
                cursor += lengths[j, q]
            starts[j, 3], lengths[j, 3] = cursor, cfg.report_tx_time
            report_end = cursor+cfg.report_tx_time
            cursor += cfg.report_tx_time+cfg.guard_time
        return report_end+cfg.report_propagation, cursor, starts, lengths

    def _dispatch(self, now, starts, lengths, wavelengths, cycle):
        if self.mode == 'baseline' or not self.policy.active or now < self.activation_time:
            return core.Simulator._dispatch(self, now, starts, lengths, wavelengths, cycle)
        cfg = self.config
        earliest = np.asarray([min(starts[u, q] for q in range(4)
                                  if lengths[u, q] > cfg.tolerance)
                               for u in range(cfg.onu_count)])
        order = np.argsort(earliest, kind='stable')
        wire_time = cfg.gate_wire_tx_time if cfg.gate_wire_tx_time is not None else cfg.gate_tx_time
        for j, u in enumerate(order):
            rx = now+wire_time*(j+1 if cfg.gate_serialization else 1)+cfg.gate_propagation
            required = [starts[u, 3]]
            if lengths[u, 0] > cfg.tolerance:
                required.append(starts[u, 0])
            if lengths[u, 1]+lengths[u, 2] > cfg.tolerance:
                required.append(starts[u, 1])
            if min(required) < rx-cfg.tolerance:
                raise AssertionError('Scheduled data/REPORT starts before serialized GATE receipt')
            gate = core.Gate(int(u), rx, tuple(starts[u]), tuple(lengths[u]), int(wavelengths[u]), cycle)
            self._push(rx, 'gate', gate)

    def _rates(self, x, proxy, horizons, snapshots):
        n = self.config.onu_count
        if self.mode == 'rotation':
            return np.zeros(n), 0.
        started = time.perf_counter()
        if self.mode == 'oracle':
            rates = np.asarray([float(self.oracle_rate_query(float(snapshots[u]), float(horizons[u]), onu=u))
                                if horizons[u] > 0 else 0. for u in range(n)])
        else:
            rates = np.asarray(self.predictor.predict_rate(x, proxy, horizons), dtype=float).reshape(-1)
        elapsed = time.perf_counter()-started
        if rates.shape != (n,) or not np.isfinite(rates).all() or (rates < 0).any():
            raise ValueError('Predictor must return finite nonnegative AF rate bit/s[N]')
        self.predictor_calls += 1
        self.prediction_seconds += elapsed
        return rates, elapsed

    def _credits(self, base, reports, rates, horizons, age):
        minimum = np.minimum(reports[:, 2], self.max_packet_bits)
        eligible = (minimum > 0) & (base[:, 2] >= minimum-1e-7)
        floor = np.maximum(self.be_floor_fraction*base[:, 2], np.where(eligible, minimum, base[:, 2]))
        caps = np.minimum(self.credit_fraction*base[:, 2], np.maximum(0., base[:, 2]-floor))
        occupancy = np.clip(reports[:, 2]/self.config.queue_caps[2], 0., 1.)
        confidence = 1.-occupancy
        confidence[age >= self.be_max_wait_rounds] = 0.
        credits = np.minimum(rates*horizons, caps)*confidence
        return credits, caps, floor, minimum, eligible, confidence, occupancy

    def _order(self, reports, rates, horizons, age, ch):
        group = self.config.onu_count//self.config.channels
        lo = ch*group
        rotated = lo+(self.rotation[ch]+np.arange(group)) % group
        score = 2.*reports[:, 0]+reports[:, 1]+rates*horizons
        # Python's stable sort preserves the rotating order when scores tie.
        order = sorted(map(int, rotated), key=lambda u: (-int(age[u] >= self.be_max_wait_rounds), -score[u]))
        return np.asarray(order), score

    def _augment_inactive_row(self):
        old = self.decision_rows[-1]
        row = np.zeros(len(old), DECISION_DTYPE)
        for name in old.dtype.names:
            row[name] = old[name]
        group = self.config.onu_count//self.config.channels
        row['schedule_rank'] = np.arange(len(row)) % group
        row['forecast_horizon_reference_s'] = row['actual_af_horizon_s']
        row['same_state_baseline_report_end_s'] = row['decision_time']+row['planned_interval_us']*1e-6
        row['planned_deadline_s'] = row['same_state_baseline_report_end_s']
        row['be_floor_bits'] = row['baseline'][:, 2]
        self.decision_rows[-1] = row

    def _dba(self, now):
        cfg, n = self.config, self.config.onu_count
        # Check readiness without appending history twice. The delegated original
        # adapter retains bit-exact IPACT decisions throughout common warm-up.
        history = list(self.policy.features)[-(self.window-1):] if self.window > 1 else []
        ready = (len(history) >= self.window-1 and (self.intervals > 0).all()
                 and all((f[:, 6] > 0).all() for f in history))
        if self.mode == 'baseline' or now < self.activation_time or not ready:
            super()._dba(now)
            self._augment_inactive_row()
            return
        group, cycle = n//cfg.channels, self.cycle+1
        reports, previous = self.reports.copy(), self.previous_grants.copy()
        if not self.policy.observe(reports, previous, self.intervals.copy(),
                                   self.report_receive_times.copy(), now, cycle):
            raise AssertionError('Readiness precheck disagreed with causal history')
        self.policy.active = True
        snapshots = self.report_snapshot_times.copy()
        if not np.isfinite(snapshots).all() or (snapshots > self.report_receive_times+1e-12).any():
            raise AssertionError('Each allocation requires exact received REPORT timestamps')
        base = self._baseline(reports, now, previous, self.intervals, cycle)
        available = float(np.max(self.next_available))
        if available < now:
            available = now+cfg.gate_tx_time
        available += cfg.gate_tx_time
        reference_starts, reference_ends = np.zeros((n, 4)), np.zeros(cfg.channels)
        for ch in range(cfg.channels):
            sl = slice(ch*group, (ch+1)*group)
            reference_ends[ch], _, reference_starts[sl], _ = self._ordered_schedule(base, ch, available, np.arange(sl.start, sl.stop))
        reference_horizon = np.maximum(0., reference_starts[:, 1]-snapshots)
        x, proxy = np.stack(self.policy.features, axis=1), np.stack(self.policy.proxies, axis=1)
        rates, elapsed = self._rates(x, proxy, reference_horizon, snapshots)
        ordering_rates = rates.copy()
        before = self.be_age.copy()
        starts, lengths = np.zeros((n, 4)), np.zeros((n, 4))
        next_receives, cursors = np.zeros(cfg.channels), np.zeros(cfg.channels)
        rank, scores = np.zeros(n, dtype=int), np.zeros(n)
        orders, rotation_start = [], self.rotation.copy()
        for ch in range(cfg.channels):
            sl = slice(ch*group, (ch+1)*group)
            order, score = self._order(reports, rates, reference_horizon, before, ch)
            orders.append(order)
            rank[order], scores[sl] = np.arange(group), score[sl]
            next_receives[ch], cursors[ch], starts[sl], lengths[sl] = self._ordered_schedule(base, ch, available, order)
        horizons = np.maximum(0., starts[:, 1]-snapshots)
        if self.mode != 'rotation':
            # Same causal x/proxy, changed horizon only. Model calibration can
            # reuse the raw TimesFM forecast. Oracle remains an isolated arm.
            rates, extra = self._rates(x, proxy, horizons, snapshots)
            elapsed += extra
        credits, caps, floor, minimum, eligible, confidence, occupancy = self._credits(base, reports, rates, horizons, before)
        grants = base.copy()
        grants[:, 1] += credits
        grants[:, 2] -= credits
        deadlines = reference_ends+group*1e-9
        for ch, order in enumerate(orders):
            sl = slice(ch*group, (ch+1)*group)
            finish, cursor, new_starts, new_lengths = self._ordered_schedule(grants, ch, available, order)
            if not np.allclose(new_starts[:, 1], starts[sl, 1], rtol=0., atol=1e-10):
                raise AssertionError('AF/BE credit transfer changed the final AF-start horizon')
            if finish > deadlines[ch]+1e-10:
                raise AssertionError('Schedule exceeded its same-state IPACT envelope')
            next_receives[ch], cursors[ch], starts[sl], lengths[sl] = finish, cursor, new_starts, new_lengths
            self.rotation[ch] = (self.rotation[ch]+1) % group
        if (not np.array_equal(grants[:, 0], base[:, 0])
                or (grants[:, 1] < base[:, 1]-1e-7).any()
                or not np.allclose(grants.sum(axis=1), base.sum(axis=1), rtol=0., atol=1e-7)
                or (grants.sum(axis=1) > np.asarray(cfg.get_base_caps())+1e-7).any()
                or (grants[:, 2] < floor-1e-7).any()
                or (credits > self.credit_fraction*base[:, 2]+1e-7).any()):
            raise AssertionError('Adaptive policy violated protected class service or grant cap')
        opportunity = (minimum > 0) & (grants[:, 2] >= minimum-1e-7)
        self.be_age = np.where(reports[:, 2] <= 0, 0, np.where(opportunity, 0, before+1))
        infeasible = (self.be_age >= self.be_max_wait_rounds) & ~eligible
        wavelengths = np.arange(n)//group
        self.cycle = cycle
        self.previous_grants[:] = grants
        self.next_available[:] = cursors
        self.dba_times.append((now, available))
        if self.flexible:
            self.flexible_cycles.add(cycle)
        self._dispatch(now, starts, lengths, wavelengths, cycle)
        row = np.zeros(n, DECISION_DTYPE)
        fields = dict(cycle=cycle, onu=np.arange(n), wavelength=wavelengths, decision_time=now,
            report_snapshot_time=snapshots, report_receive_time=self.report_receive_times,
            report_interval_s=self.intervals, window_start_time=self.policy.times[0], window_end_time=self.policy.times[-1],
            window_count=len(self.policy.features), prediction_active=self.mode != 'rotation', reports=reports,
            previous_grants=previous, baseline=base, final=grants, cap_bits=cfg.get_base_caps(),
            predicted_af_rate_bps=rates, credit_horizon_s=horizons, actual_af_horizon_s=horizons,
            af_credit_bits=credits, be_deferred_bits=np.zeros(n), credit_scale=confidence,
            target_cycle_us=(deadlines[wavelengths]-now)*1e6,
            planned_interval_us=(next_receives[wavelengths]-now)*1e6,
            target_infeasible=False, timing_bound_verified=not cfg.shared_upstream_serialization or cfg.channels == 1,
            be_selected=opportunity, be_eligible=eligible, be_reserved_bits=floor,
            credit_cap_bits=caps, be_rotation_start=rotation_start[wavelengths], be_rotation_next=self.rotation[wavelengths],
            be_selection_rank=rank, be_selection_budget_rejected=False, be_optional_full_grant_deferred=False,
            be_reserved_visits_target=group, be_reserved_visits_actual=np.bincount(wavelengths, weights=opportunity, minlength=cfg.channels)[wavelengths],
            max_packet_bits=self.max_packet_bits, policy_active=True, oracle_only=self.mode == 'oracle',
            snapshot_timestamp_exact=True, report_wire_queue_delay_s=np.maximum(0., self.report_receive_times-snapshots-cfg.report_tx_time-cfg.report_propagation),
            proxy_af_rate_bps=self.policy.proxies[-1], prediction_seconds=elapsed,
            schedule_rank=rank, forecast_horizon_reference_s=reference_horizon,
            same_state_baseline_report_end_s=reference_ends[wavelengths], planned_deadline_s=deadlines[wavelengths],
            be_floor_bits=floor, be_report_occupancy_fraction=occupancy, reserve_confidence_scale=confidence,
            be_opportunity_age_before=before, be_opportunity_age_after=self.be_age,
            be_age_urgent=before >= self.be_max_wait_rounds, be_age_constraint_infeasible=infeasible,
            forecast_order_score_bits=scores, ordering_prediction_rate_bps=ordering_rates)
        for name, value in fields.items():
            row[name] = value
        self.decision_rows.append(row)
        self.cycle_rows.append((cycle, now, float(next_receives.max()), float((next_receives.max()-now)*1e6),
            float((deadlines.max()-now)*1e6), False,
            not cfg.shared_upstream_serialization or cfg.channels == 1,
            float(credits.sum()), 0., int(opportunity.sum()), int(eligible.sum()), n, n-int(opportunity.sum()),
            float(floor.sum()), True, elapsed, np.nan, np.nan, False, False))


AdaptiveSimulator = AdaptivePredictiveSimulator
