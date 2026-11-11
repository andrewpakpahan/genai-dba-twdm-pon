"""Packet-level ONU/OLT DES translated from saved OPNET behavior.

Native event replay is an ONU-only verification mode, never an independent OLT
result. Autonomous mode recomputes REPORTs and GATEs. Physical timing parameters
must be supplied explicitly; missing source-to-sink timing prevents native ETE
claims. See SEMANTICS_ONU.md for source references and known equivalence gaps.
"""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass, asdict
from heapq import heappush, heappop
from itertools import count
import math
from typing import Callable, Mapping
import numpy as np


@dataclass(frozen=True)
class SimulationConfig:
    duration: float = 5.0
    onu_count: int = 64
    channels: int = 2
    line_rate: float = 1e9
    channel_rates: tuple | None = None
    channel_offsets: tuple | None = None
    base_caps: tuple | None = None
    queue_caps: tuple = (1e6, 4e6, 5e6)
    base_cap: float = 31250.0
    guard_time: float = 5e-6
    report_tx_time: float | None = None
    gate_tx_time: float | None = None
    gate_wire_tx_time: float | None = None
    report_propagation: float | None = None
    gate_propagation: float | None = None
    data_propagation: float | None = None
    olt_to_sink_delay: float | None = None
    initial_time: float = 0.0
    channel1_offset: float = 13e-6
    tolerance: float = 1e-12
    queue_size_mode: str = "exact"
    arrival_tie_order: str = "before_control"
    gate_serialization: bool = True
    shared_upstream_serialization: bool = False
    initial_report_only: bool = True
    record_control: bool = True
    record_queue_samples: bool = False
    timing_provenance: str = "unverified: explicit timing assumptions"

    def get_channel_rates(self):
        return tuple(self.channel_rates) if self.channel_rates is not None else (self.line_rate,)*self.channels

    def get_channel_offsets(self):
        if self.channel_offsets is not None:
            return tuple(self.channel_offsets)
        if self.channels == 1:
            return (0.0,)
        if self.channels == 2:
            return (0.0,self.channel1_offset)
        raise ValueError("More than two wavelengths requires explicit channel_offsets; native timing is unverified")

    def get_base_caps(self):
        return tuple(self.base_caps) if self.base_caps is not None else (self.base_cap,)*self.onu_count

    def validate(self, external_gates=False):
        if self.duration <= 0 or self.onu_count <= 0 or not isinstance(self.channels,int) or isinstance(self.channels,bool) or self.channels <= 0:
            raise ValueError("duration/ONU count and integer wavelength count must be positive")
        if self.onu_count % self.channels:
            raise ValueError("ONU count must divide evenly across channels")
        if not np.isfinite(self.line_rate) or self.line_rate <= 0 or len(self.queue_caps) != 3 or not np.isfinite(self.queue_caps).all() or min(self.queue_caps) <= 0:
            raise ValueError("Invalid rate or queue capacity")
        rates=self.get_channel_rates()
        if len(rates)!=self.channels or not np.isfinite(rates).all() or min(rates)<=0:
            raise ValueError("channel_rates must contain one finite positive bit/s rate per wavelength")
        if not external_gates or self.channel_offsets is not None:
            offsets=self.get_channel_offsets()
            if len(offsets)!=self.channels or not np.isfinite(offsets).all() or min(offsets)<0:
                raise ValueError("channel_offsets must contain one finite nonnegative second offset per wavelength")
        caps=self.get_base_caps()
        if len(caps)!=self.onu_count or not np.isfinite(caps).all() or min(caps)<0:
            raise ValueError("base_caps must contain one finite nonnegative bit cap per ONU")
        if self.queue_size_mode not in ("exact", "truncate_packet", "round_packet"):
            raise ValueError("Unknown queue-size quantization")
        if self.arrival_tie_order not in ("before_control", "after_control"):
            raise ValueError("Unknown equal-time arrival ordering")
        for name in ("report_tx_time", "gate_tx_time", "report_propagation", "gate_propagation"):
            value = getattr(self, name)
            if not external_gates and value is None:
                raise ValueError(f"{name} must be explicitly configured; no silent zero")
            if value is not None and value < 0:
                raise ValueError(f"{name} cannot be negative")


@dataclass(frozen=True)
class Gate:
    onu: int
    receive_time: float
    starts: tuple
    lengths: tuple
    wavelength: int = 0
    cycle: int = 0
    event_id: int = -1


SERVICE_DTYPE = np.dtype([
    ("time", "f8"), ("completion_time", "f8"), ("wire_completion_time", "f8"), ("onu", "i2"), ("queue", "i1"),
    ("bits", "f8"), ("queue_wait", "f8"), ("arrival_time", "f8"),
    ("event_id", "i8"), ("grant_start", "f8"), ("grant_end", "f8"),
    ("cycle", "i8"), ("wavelength", "i1"), ("sink_time_estimate", "f8"),
])
DROP_DTYPE = np.dtype([("time", "f8"), ("onu", "i2"), ("queue", "i1"), ("bits", "f8"), ("event_id", "i8")])
REPORT_DTYPE = np.dtype([("time", "f8"), ("receive_time", "f8"), ("onu", "i2"), ("cycle", "i8"), ("bits", "f8", (3,))])
GATE_DTYPE = np.dtype([("receive_time", "f8"), ("onu", "i2"), ("cycle", "i8"), ("starts", "f8", (4,)), ("lengths", "f8", (4,)), ("wavelength", "i1")])


class Simulator:
    """Allocation callback: (reports, now, previous_grants, intervals, cycle).

    Returns N x 3 grant bits or an object with `.final`. The callback must use
    only passed causal information and its previous state. Optional callback
    `.set_report_receive_times(times)` receives causal times before allocation.
    """
    def __init__(self, config: SimulationConfig, allocation: Callable | None = None):
        self.config = config
        self.allocation = allocation or self._baseline

    def _baseline(self, reports, now, previous_grants, intervals, cycle):
        out = np.zeros_like(reports)
        for u in range(len(out)):
            left = self.config.get_base_caps()[u]
            for q in range(3):
                out[u, q] = min(left, reports[u, q])
                left -= out[u, q]
        return out

    def _push(self, when, kind, payload):
        if not np.isfinite(when):
            raise ValueError("Non-finite event timestamp")
        heappush(self.heap, (float(when), next(self.sequence), kind, payload))

    def _cmp(self, a, b):
        if abs(a - b) <= self.config.tolerance:
            return 0
        return -1 if a < b else 1

    def _stored_bits(self, bits):
        mode = self.config.queue_size_mode
        if mode == "truncate_packet":
            return float(int(bits))
        if mode == "round_packet":
            return float(int(bits + .5))
        return bits

    def _record_gate(self, gate):
        if self.config.record_control:
            self.gate_rows.append((gate.receive_time, gate.onu, gate.cycle, gate.starts, gate.lengths, gate.wavelength))

    def _receive_gate(self, now, gate):
        u = gate.onu
        primary = self.primary[u]
        old_end = 0 if primary is None else primary.starts[3] + primary.lengths[3]
        if now > old_end:
            self.primary[u] = gate
        else:
            self.secondary[u] = gate
        self._record_gate(gate)
        for q in range(3):
            if self._cmp(gate.lengths[q], 0) == 1 and self._cmp(gate.starts[q], now) != -1:
                self._push(max(now, gate.starts[q]), "service", (u, q))
            elif gate.lengths[q] > self.config.tolerance and gate.starts[q] < now - self.config.tolerance:
                self.late_gates += 1
        report_end = gate.starts[3] + gate.lengths[3]
        required = self.config.report_tx_time or 0
        if self._cmp(gate.lengths[3], required) != -1 and self._cmp(gate.starts[3], now) != -1 and report_end > gate.starts[3]:
            self._push(max(now, gate.starts[3]), "report", (u, gate))
        elif gate.starts[3] != 0:
            self.late_reports += 1

    def _current_gate(self, u, q, now):
        gate = self.primary[u]
        if gate is not None and gate.starts[q] + gate.lengths[q] > now:
            return gate
        return self.secondary[u]

    def _wire_completion(self, now, bits, wavelength):
        rate = self.channel_rates[wavelength]
        if not self.config.shared_upstream_serialization:
            return now + bits/rate
        complete = max(now, self.wire_available[wavelength]) + bits/rate
        self.wire_available[wavelength] = complete
        return complete

    def _service(self, now, u, q):
        gate = self._current_gate(u, q, now)
        if gate is None:
            return
        start = gate.starts[q]
        end = start + gate.lengths[q]
        if self._cmp(start, now) == 1 or self._cmp(end, now) != 1:
            return
        queue = self.queues[u][q]
        if not queue:
            self.empty_slots += 1
            self.unused_events.append((now, u, q, end - now, "empty"))
            return
        arrival, bits, event_id, stored = queue[0]
        complete = now + bits / self.channel_rates[gate.wavelength]
        if self._cmp(complete, end) == 1:
            self.nonfit_slots += 1
            self.unused_events.append((now, u, q, end - now, "nonfit"))
            return
        queue.popleft()
        self.occupancy[u, q] -= stored
        if abs(self.occupancy[u, q]) < 1e-7:
            self.occupancy[u, q] = 0
        if self.occupancy[u, q] < -1e-7:
            raise AssertionError("Negative queue occupancy")
        wire_complete = self._wire_completion(now, bits, gate.wavelength)
        sink = np.nan
        if self.config.data_propagation is not None and self.config.olt_to_sink_delay is not None:
            sink = wire_complete + self.config.data_propagation + self.config.olt_to_sink_delay
        self.service_rows.append((now, complete, wire_complete, u, q, bits, now-arrival, arrival, event_id, start, end, gate.cycle, gate.wavelength, sink))
        self.served_bits[q] += bits
        self.served_count[q] += 1
        if self._cmp(complete, end) == -1:
            self._push(complete, "service", (u, q))

    def _arrival(self, now, u, q, bits, event_id):
        self.offered_count[q] += 1
        self.offered_bits[q] += bits
        if bits + self.occupancy[u, q] < self.config.queue_caps[q]:
            stored = self._stored_bits(bits)
            self.queues[u][q].append((now, bits, event_id, stored))
            self.occupancy[u, q] += stored
            self.admitted_count[q] += 1
        else:
            self.drop_rows.append((now, u, q, bits, event_id))
        if self.config.record_queue_samples:
            self.queue_rows.append((now, u, q, self.occupancy[u, q]))

    def _report(self, now, u, gate):
        if self._cmp(gate.starts[3], now) == 1 or self._cmp(gate.starts[3]+gate.lengths[3], now) != 1:
            return
        snapshot = self.occupancy[u].copy()
        received = self._wire_completion(now, gate.lengths[3]*self.channel_rates[0], 0) + (self.config.report_propagation or 0)
        if self.config.record_control:
            self.report_rows.append((now, received, u, gate.cycle, snapshot))
        if not self.external:
            self._push(received, "report_rx", (u, snapshot))

    def _report_rx(self, now, u, snapshot):
        if self.pending_reports[u]:
            self.duplicate_reports += 1
        self.reports[u] = snapshot
        self.intervals[u] = now-self.report_receive_times[u] if np.isfinite(self.report_receive_times[u]) else 0
        self.report_receive_times[u] = now
        self.pending_reports[u] = True
        if self.pending_reports.all():
            self.pending_reports[:] = False
            self._push(now, "dba", None)

    def _dispatch(self, now, starts, lengths, wavelengths, cycle):
        for u in range(self.config.onu_count):
            wire_time = self.config.gate_wire_tx_time if self.config.gate_wire_tx_time is not None else self.config.gate_tx_time
            tx = wire_time * (u+1 if self.config.gate_serialization else 1)
            rx = now + tx + self.config.gate_propagation
            gate = Gate(u, rx, tuple(starts[u]), tuple(lengths[u]), int(wavelengths[u]), cycle)
            self._push(rx, "gate", gate)

    def _initialize_reports(self):
        cfg = self.config
        starts = np.zeros((cfg.onu_count, 4))
        lengths = np.zeros_like(starts)
        available = cfg.initial_time
        for u in range(cfg.onu_count):
            available += cfg.gate_tx_time
            starts[u, 3] = available
            lengths[u, 3] = cfg.report_tx_time
            available += cfg.report_tx_time + cfg.guard_time
            available = math.ceil(available*1e10 + .5)/1e10
        self.next_available[0] = available
        self._dispatch(cfg.initial_time, starts, lengths, np.zeros(cfg.onu_count, dtype=int), 0)

    def _dba(self, now):
        cfg = self.config
        self.cycle += 1
        available = float(np.max(self.next_available))
        if available < now:
            available = now + cfg.gate_tx_time
        available += cfg.gate_tx_time
        starts = np.zeros((cfg.onu_count, 4))
        lengths = np.zeros_like(starts)
        if hasattr(self.allocation, "set_report_receive_times"):
            self.allocation.set_report_receive_times(self.report_receive_times.copy())
        result = self.allocation(self.reports.copy(), now, self.previous_grants.copy(), self.intervals.copy(), self.cycle)
        grants = np.asarray(result.final if hasattr(result, "final") else result, dtype=float)
        if grants.shape != (cfg.onu_count, 3) or not np.isfinite(grants).all() or (grants < 0).any():
            raise ValueError("Allocator returned invalid grants")
        if (grants > self.reports + 1e-5).any():
            raise ValueError("This simulator contract serves reported demand; speculative grants require explicit extension")
        per_channel = cfg.onu_count//cfg.channels
        wavelengths = np.arange(cfg.onu_count)//per_channel
        offsets = cfg.get_channel_offsets()
        for channel in range(cfg.channels):
            cursor = available + offsets[channel]
            for u in range(channel*per_channel, (channel+1)*per_channel):
                cursor = math.ceil(cursor*1e9 + .5)/1e9
                for q in range(3):
                    starts[u, q] = cursor
                    lengths[u, q] = grants[u, q]/self.channel_rates[channel]
                    cursor += lengths[u, q]
                starts[u, 3] = cursor
                lengths[u, 3] = cfg.report_tx_time
                cursor += cfg.report_tx_time + cfg.guard_time
            self.next_available[channel] = cursor
        self.previous_grants[:] = grants
        self.dba_times.append((now, available))
        self._dispatch(now, starts, lengths, wavelengths, self.cycle)

    def run(self, traffic: Mapping, external_gates=None):
        cfg = self.config
        self.external = external_gates is not None
        cfg.validate(self.external)
        self.channel_rates = cfg.get_channel_rates()
        time = np.asarray(traffic["time"], dtype=float)
        onu = np.asarray(traffic["onu"], dtype=int)
        queue = np.asarray(traffic["queue"], dtype=int)
        bits = np.asarray(traffic["bits"], dtype=float)
        event_id = np.asarray(traffic.get("event_id", np.arange(len(time))), dtype=np.int64)
        if not (len(time)==len(onu)==len(queue)==len(bits)==len(event_id)):
            raise ValueError("Traffic columns have different lengths")
        if not np.isfinite(time).all() or not np.isfinite(bits).all() or (time < 0).any() or (bits <= 0).any():
            raise ValueError("Invalid traffic time or packet size")
        if (np.diff(time)<0).any() or (onu<0).any() or (onu>=cfg.onu_count).any() or (queue<0).any() or (queue>2).any():
            raise ValueError("Traffic must be chronological with valid ONU/classes")
        self.heap=[]; self.sequence=count(); self.cycle=0
        self.queues=[[deque() for _ in range(3)] for _ in range(cfg.onu_count)]
        self.occupancy=np.zeros((cfg.onu_count,3))
        self.primary=[None]*cfg.onu_count; self.secondary=[None]*cfg.onu_count
        self.reports=np.zeros_like(self.occupancy); self.previous_grants=np.zeros_like(self.occupancy)
        self.pending_reports=np.zeros(cfg.onu_count,dtype=bool)
        self.report_receive_times=np.full(cfg.onu_count,np.nan); self.intervals=np.zeros(cfg.onu_count)
        self.next_available=np.zeros(cfg.channels)
        self.wire_available=np.zeros(cfg.channels)
        self.service_rows=[]; self.drop_rows=[]; self.gate_rows=[]; self.report_rows=[]; self.queue_rows=[]; self.unused_events=[]; self.dba_times=[]
        self.offered_count=np.zeros(3,dtype=np.int64); self.offered_bits=np.zeros(3)
        self.admitted_count=np.zeros(3,dtype=np.int64); self.served_count=np.zeros(3,dtype=np.int64); self.served_bits=np.zeros(3)
        self.empty_slots=0; self.nonfit_slots=0; self.late_gates=0; self.late_reports=0; self.duplicate_reports=0
        if self.external:
            for gate in external_gates:
                if not isinstance(gate, Gate):
                    gate = Gate(**gate)
                if gate.wavelength < 0 or gate.wavelength >= cfg.channels or gate.onu < 0 or gate.onu >= cfg.onu_count:
                    raise ValueError("External GATE ONU/wavelength outside configured resource range")
                self._push(gate.receive_time, "gate", gate)
        elif cfg.initial_report_only:
            self._initialize_reports()
        else:
            self._push(cfg.initial_time,"dba",None)
        i=0
        while i<len(time) or self.heap:
            arrival_time = float(time[i]) if i<len(time) else math.inf
            control_time = self.heap[0][0] if self.heap else math.inf
            arrival_first = arrival_time < control_time or (arrival_time == control_time and cfg.arrival_tie_order == "before_control")
            if min(arrival_time, control_time)>cfg.duration:
                break
            if arrival_first:
                self._arrival(arrival_time,int(onu[i]),int(queue[i]),float(bits[i]),int(event_id[i])); i+=1
            else:
                now,_,kind,payload=heappop(self.heap)
                if kind=="gate": self._receive_gate(now,payload)
                elif kind=="service": self._service(now,*payload)
                elif kind=="report": self._report(now,*payload)
                elif kind=="report_rx": self._report_rx(now,*payload)
                elif kind=="dba": self._dba(now)
                else: raise AssertionError(kind)
        summary={"config":asdict(cfg),"mode":"native_gate_onu_verification" if self.external else "autonomous_source_port", "native_sink_ete_verified":False,
            "offered_packets":self.offered_count.tolist(),"offered_bits":self.offered_bits.tolist(),"admitted_packets":self.admitted_count.tolist(),
            "served_packets":self.served_count.tolist(),"served_bits":self.served_bits.tolist(),"queued_bits_end":self.occupancy.sum(axis=0).tolist(),
            "queued_packets_end":[sum(len(self.queues[u][q]) for u in range(cfg.onu_count)) for q in range(3)],
            "cycles":self.cycle,"empty_stop_events":self.empty_slots,"nonfit_stop_events":self.nonfit_slots,"late_data_grants":self.late_gates,"late_reports":self.late_reports,"duplicate_reports":self.duplicate_reports,
            "event_tie_order_status":"explicit assumption; native equal-time order unverified", "sink_time_estimate_status":"requires supplied physical delays; native sink definition unverified"}
        return {"summary":summary,"services":np.array(self.service_rows,dtype=SERVICE_DTYPE),"drops":np.array(self.drop_rows,dtype=DROP_DTYPE),
            "gates":np.array(self.gate_rows,dtype=GATE_DTYPE),"reports":np.array(self.report_rows,dtype=REPORT_DTYPE),"dba_times":np.array(self.dba_times,dtype=float).reshape(-1,2),
            "queue_samples":self.queue_rows,"unused_events":self.unused_events}
