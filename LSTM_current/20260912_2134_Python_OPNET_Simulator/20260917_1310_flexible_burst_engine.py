"""Isolated ONU-side AF-to-BE capacity borrowing inside a fixed data burst.

Allocation, EF service, REPORT timing, guard time, and the inherited shared-wire
serializer are preserved. After AF finishes or stops, BE may use the contiguous
remaining AF+BE time up to the unchanged REPORT start. AF never exceeds its
announced grant. Only the ONU's own live FIFO heads are examined; no queue state
is added to the OLT/model input. Stopped service is not resumed by new arrivals.
"""
from importlib import import_module
import numpy as np

predictive = import_module('20260917_1246_predictive_rotation_engine')
core = predictive.core

BURST_DTYPE = np.dtype([
    ('cycle', 'i8'), ('onu', 'i2'), ('wavelength', 'i2'),
    ('receive_time', 'f8'), ('af_start', 'f8'), ('original_af_end', 'f8'),
    ('original_be_start', 'f8'), ('report_start', 'f8'),
    ('announced_af_bits', 'f8'), ('announced_be_bits', 'f8'),
    ('af_served_bits', 'f8'), ('be_served_bits', 'f8'),
    ('released_af_bits', 'f8'), ('be_bits_beyond_announced_grant', 'f8'),
    ('be_bits_in_released_af_time', 'f8'), ('unused_burst_bits', 'f8'),
    ('af_release_time', 'f8'), ('service_stop_time', 'f8'),
    ('af_stop_reason', 'U12'), ('be_stop_reason', 'U12'),
    ('window_complete', '?'), ('service_stopped', '?'),
    ('service_packets', 'i8'), ('max_wire_overrun_s', 'f8'),
])
SERVICE_AUDIT_DTYPE = np.dtype([
    ('cycle', 'i8'), ('onu', 'i2'), ('queue', 'i1'), ('event_id', 'i8'),
    ('time', 'f8'), ('completion_time', 'f8'), ('wire_completion_time', 'f8'),
    ('bits', 'f8'), ('original_af_end', 'f8'), ('report_start', 'f8'),
    ('borrowed_af_time_bits', 'f8'), ('wire_overrun_s', 'f8'),
])


class FlexibleBurstSimulator(predictive.PredictiveRotatingBESimulator):
    """Same allocation modes, optionally sharing abandoned AF time with BE.

    Flexible service activates only for GATEs whose DBA decision is at or after
    activation_time. The first three seconds therefore remain bit-exact IPACT.
    ``burst_array`` records announced versus realized allocation and borrowing;
    ``service_audit_array`` records the actual single packet event chain.
    """
    def __init__(self, config, mode='baseline', flexible=True, **kwargs):
        super().__init__(config, mode=mode, **kwargs)
        self.flexible = bool(flexible)
        self.flexible_cycles = set()
        self.burst_states = {}
        self.service_audit_rows = []
        self.metadata.update(
            flexible_burst=self.flexible,
            flexible_activation='DBA decision time >= activation_time; fixed REPORT boundary',
            flexible_service='AF honors announced grant; BE borrows abandoned contiguous AF remainder',
            arrival_resume=False,
            borrow_accounting='BE transmitted beyond announced BE grant and capacity used before original BE start are distinct',
            unused_event_semantics='only physical time left unused after flexible BE stops; released AF is recorded in burst audit',
            wire_boundary_semantics='nominal packet completion is bounded by REPORT start; inherited shared-wire serialization can complete later and overruns are recorded',
            numerical_capacity_tolerance='inherited time tolerance times channel rate, plus 1e-5 bit floating-point allowance',
            OLT_queue_access=False)

    def run(self, traffic, external_gates=None):
        self.flexible_cycles = set()
        self.burst_states = {}
        self.service_audit_rows = []
        result = super().run(traffic, external_gates=external_gates)
        result['summary']['flexible_burst'] = {
            'enabled': self.flexible,
            'active_bursts': len(self.burst_states),
            'borrowed_af_time_bits': float(sum(s['borrowed'] for s in self.burst_states.values())),
            'BE_bits_beyond_announced_grant': float(sum(max(0., s['be_served']-s['be_bits']) for s in self.burst_states.values())),
            'AF_released_bits': float(sum(s['released'] for s in self.burst_states.values())),
            'wire_overrun_packets': int(sum(row[-1] > self.config.tolerance for row in self.service_audit_rows)),
            'arrival_resume': False,
        }
        return result

    def _dba(self, now):
        if self.flexible and now >= self.activation_time:
            self.flexible_cycles.add(self.cycle+1)
        return super()._dba(now)

    def _receive_gate(self, now, gate):
        if not self.flexible or gate.cycle not in self.flexible_cycles:
            return core.Simulator._receive_gate(self, now, gate)
        u = gate.onu
        primary = self.primary[u]
        old_end = 0 if primary is None else primary.starts[3]+primary.lengths[3]
        if now > old_end:
            self.primary[u] = gate
        else:
            self.secondary[u] = gate
        self._record_gate(gate)
        # EF retains the original independent event chain and grant boundaries.
        if self._cmp(gate.lengths[0], 0) == 1 and self._cmp(gate.starts[0], now) != -1:
            self._push(max(now, gate.starts[0]), 'service', (u, 0))
        elif gate.lengths[0] > self.config.tolerance and gate.starts[0] < now-self.config.tolerance:
            self.late_gates += 1
        rate = self.channel_rates[gate.wavelength]
        af_start = float(gate.starts[1])
        af_end = af_start+gate.lengths[1]
        be_start = float(gate.starts[2])
        end = float(gate.starts[3])
        be_end = be_start+gate.lengths[2]
        if abs(af_end-be_start) > 1e-10 or abs(be_end-end) > 1e-10:
            raise AssertionError('Flexible service requires contiguous AF/BE slots ending at REPORT')
        state = dict(gate=gate, rate=rate, af_start=af_start, af_end=af_end,
                     be_start=be_start, end=end, af_bits=float(gate.lengths[1]*rate),
                     be_bits=float(gate.lengths[2]*rate), af_served=0., be_served=0.,
                     released=0., borrowed=0., release_time=np.nan, stop_time=np.nan,
                     af_reason='', be_reason='', phase=1, stopped=False,
                     pending=False, ticket=-1, next_time=np.nan, packets=0,
                     max_wire_overrun=0., unused=0.)
        key = (u, gate.cycle)
        if key in self.burst_states:
            raise AssertionError('Duplicate activated GATE for an ONU/cycle')
        self.burst_states[key] = state
        if end > af_start+self.config.tolerance:
            if af_start < now-self.config.tolerance:
                self.late_gates += 1
                state.update(stopped=True, stop_time=now, af_reason='late_gate', be_reason='late_gate')
            else:
                self._schedule_burst(state, max(now, af_start))
        else:
            state.update(stopped=True, stop_time=end, af_reason='zero_budget', be_reason='zero_budget')
        # REPORT remains at its exact preannounced timestamp.
        report_end = gate.starts[3]+gate.lengths[3]
        required = self.config.report_tx_time or 0
        if (self._cmp(gate.lengths[3], required) != -1
                and self._cmp(gate.starts[3], now) != -1 and report_end > gate.starts[3]):
            self._push(max(now, gate.starts[3]), 'report', (u, gate))
        elif gate.starts[3] != 0:
            self.late_reports += 1

    def _schedule_burst(self, state, when):
        if state['pending'] or state['stopped']:
            raise AssertionError('A burst may have exactly one pending service event')
        state['ticket'] += 1
        state['pending'] = True
        state['next_time'] = float(when)
        gate = state['gate']
        self._push(when, 'service', (gate.onu, 1, gate.cycle, state['ticket']))

    def _service(self, now, u, q, burst_cycle=None, ticket=None):
        if burst_cycle is None:
            return core.Simulator._service(self, now, u, q)
        state = self.burst_states[(u, burst_cycle)]
        if (not state['pending'] or ticket != state['ticket']
                or abs(now-state['next_time']) > self.config.tolerance):
            raise AssertionError('Duplicate or stale flexible burst event')
        state['pending'] = False
        if state['stopped']:
            raise AssertionError('Service event after a burst stopped')
        gate, rate, end = state['gate'], state['rate'], state['end']
        if now < state['af_start']-self.config.tolerance or now > end+self.config.tolerance:
            raise AssertionError('Flexible service event is outside its ONU data burst')
        if state['phase'] == 1:
            queue = self.queues[u][1]
            af_end = state['af_end']
            if self._cmp(now, af_end) != -1:
                reason = 'budget_done'
            elif not queue:
                reason = 'empty'
            elif self._cmp(now+queue[0][1]/rate, af_end) == 1:
                reason = 'nonfit'
            else:
                reason = None
            if reason is not None:
                state['phase'] = 2
                state['released'] = max(0., state['af_bits']-state['af_served'])
                state['release_time'] = now
                state['af_reason'] = reason
        q = state['phase']
        queue = self.queues[u][q]
        limit = state['af_end'] if q == 1 else end
        if self._cmp(now, limit) != -1:
            self._stop_burst(state, now, 'budget_done')
            return
        if not queue:
            self._stop_burst(state, now, 'empty')
            return
        arrival, bits, event_id, stored = queue[0]
        completion = now+bits/rate
        if self._cmp(completion, limit) == 1:
            self._stop_burst(state, now, 'nonfit')
            return
        queue.popleft()
        self.occupancy[u, q] -= stored
        if abs(self.occupancy[u, q]) < 1e-7:
            self.occupancy[u, q] = 0.
        if self.occupancy[u, q] < -1e-7:
            raise AssertionError('Negative ONU queue occupancy')
        wire = self._wire_completion(now, bits, gate.wavelength)
        sink = np.nan
        if self.config.data_propagation is not None and self.config.olt_to_sink_delay is not None:
            sink = wire+self.config.data_propagation+self.config.olt_to_sink_delay
        grant_start = state['af_start'] if q == 1 else state['release_time']
        self.service_rows.append((now, completion, wire, u, q, bits, now-arrival,
            arrival, event_id, grant_start, limit, gate.cycle, gate.wavelength, sink))
        self.served_bits[q] += bits
        self.served_count[q] += 1
        state['af_served' if q == 1 else 'be_served'] += bits
        state['packets'] += 1
        borrowed = (max(0., min(completion, state['af_end'])-now)*rate if q == 2 else 0.)
        state['borrowed'] += borrowed
        overrun = max(0., wire-end)
        state['max_wire_overrun'] = max(state['max_wire_overrun'], overrun)
        self.service_audit_rows.append((gate.cycle, u, q, event_id, now, completion,
            wire, bits, state['af_end'], end, borrowed, overrun))
        # The inherited packet-fit comparison is time based (default 1 ps).
        # At 1 Gb/s this permits 0.001 bit of rounding at a slot boundary.
        # Capacity diagnostics must use that same numerical tolerance.
        bit_tolerance = max(1e-6, self.config.tolerance*rate+1e-5)
        if (state['af_served'] > state['af_bits']+bit_tolerance
                or state['af_served']+state['be_served'] > state['af_bits']+state['be_bits']+bit_tolerance
                or state['borrowed'] > state['released']+bit_tolerance):
            raise AssertionError('Flexible burst exceeded its announced capacity')
        if self._cmp(completion, end) == -1:
            self._schedule_burst(state, completion)
        else:
            self._stop_burst(state, completion, 'budget_done')

    def _stop_burst(self, state, now, reason):
        state['stopped'] = True
        state['stop_time'] = now
        state['be_reason'] = reason
        remaining = max(0., state['end']-now)
        state['unused'] = remaining*state['rate']
        if remaining > self.config.tolerance and reason in ('empty', 'nonfit'):
            if reason == 'empty':
                self.empty_slots += 1
            else:
                self.nonfit_slots += 1
            self.unused_events.append((now, state['gate'].onu, 2, remaining, reason))

    def burst_array(self):
        rows = []
        for state in self.burst_states.values():
            g = state['gate']
            rows.append((g.cycle, g.onu, g.wavelength, g.receive_time,
                state['af_start'], state['af_end'], state['be_start'], state['end'],
                state['af_bits'], state['be_bits'], state['af_served'], state['be_served'],
                state['released'], max(0., state['be_served']-state['be_bits']), state['borrowed'],
                state['unused'], state['release_time'], state['stop_time'], state['af_reason'],
                state['be_reason'], state['end'] <= self.config.duration,
                state['stopped'], state['packets'], state['max_wire_overrun']))
        return np.asarray(rows, dtype=BURST_DTYPE)

    def service_audit_array(self):
        return np.asarray(self.service_audit_rows, dtype=SERVICE_AUDIT_DTYPE)
