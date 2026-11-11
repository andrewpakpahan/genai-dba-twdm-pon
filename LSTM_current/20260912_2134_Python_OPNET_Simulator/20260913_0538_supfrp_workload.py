"""Standalone Case_Base arrivals from the recovered OPNET traffic-source laws.

This generates new Sup-FRP renewal realizations; it never reads recorded c2
packets. The transforms, stationary initialization, and packet-size law come
from OPNET source. PCG64 stream assignment is explicit and is NOT the OPNET
seed-to-stream mapping. Therefore this module establishes distribution/source
fidelity, not identical native packet traces for the same integer seed.
"""
from dataclasses import asdict
from functools import lru_cache
from importlib import import_module
from pathlib import Path
import hashlib
import math

import numpy as np

_math = import_module("20260912_2346_supfrp_source_math")
ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = ROOT / "validation/case_base/native_sources/136_1p0ms_canonical/Chris normal IPACT/DBA"
VERSION = "source_supfrp_pcg64_stationary_v1"


@lru_cache(maxsize=1)
def source_evidence():
    result = []
    for name, lines in (("EPON_rpg_mgr.pr.c", "148-235;1188-1198;1387-1396"),
                        ("EPON_rpg_source.pr.c", "134-140;220-221"),
                        ("EPON_AF_BE_rpg.pr.c", "189-245")):
        path = SOURCE_ROOT / name
        result.append({"path": str(path), "lines": lines,
                       "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    path = ROOT / "validation/case_base/live_support_source_readback.md"
    result.append({"path": str(path), "source": "OMS transforms and simple_source timing readback",
                   "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    return result


def _positive_uniform(rng, count):
    """Native transforms reject zero draws; retain that boundary behavior."""
    values = rng.random(count)
    zero = values == 0.0
    while np.any(zero):
        values[zero] = rng.random(int(zero.sum()))
        zero = values == 0.0
    return values


def _first(location, shape, uniform):
    transformed = (1.0 + (shape - 1.0) * math.exp(shape)) * uniform / shape
    body = transformed >= 1.0
    result = np.empty_like(uniform)
    u, v = uniform[body], transformed[body]
    result[body] = location * (-1.0) / shape * np.log(u * (shape*v - 1.0) / (shape*v - u))
    result[~body] = location * np.power(transformed[~body], 1.0 / (1.0 - shape))
    return result


def _ordinary(location, shape, uniform):
    body = uniform >= math.exp(-shape)
    result = np.empty_like(uniform)
    result[body] = -np.log(uniform[body]) * location / shape
    result[~body] = location * math.exp(-1.0) * np.power(uniform[~body], -1.0 / shape)
    return result


def _renewals(params, duration, seed, onu, queue):
    """Superpose M sources, beginning with stationary residual intervals.

    Component-round vectorization consumes every component's draws in every
    round, even after a component passes the horizon. Thus changing duration
    does not alter earlier packets. Size and interval streams are independent;
    this is an explicit Python stream design, not native event draw order.
    Temporary component arrays are released after each ONU/class is merged.
    """
    count, location, shape = params.source_count, params.location_seconds, params.shape
    arrivals = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, onu, queue, 0])))
    sizes_rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, onu, queue, 1])))
    now = _first(location, shape, _positive_uniform(arrivals, count))
    times, sizes = [], []
    rounds = 0
    while np.any(now <= duration):
        active = now <= duration
        # OPNET rpg_pk_send ceils the sampled size. Uniform(512,12144)
        # already satisfies the native rejection limits 1..12144 bits.
        round_sizes = np.ceil(sizes_rng.uniform(512.0, 12144.0, count)).astype(np.uint32)
        times.append(now[active].copy())
        sizes.append(round_sizes[active])
        increments = _ordinary(location, shape, _positive_uniform(arrivals, count))
        following = now + increments
        if np.any(following[active] <= now[active]):
            raise ValueError("A Sup-FRP interval does not advance binary64 time")
        now = following
        rounds += 1
    if not times:
        return np.empty(0, np.float64), np.empty(0, np.uint32), rounds
    return np.concatenate(times), np.concatenate(sizes), rounds


def ef_arrivals(interval_s, duration_s, n_onu, max_packets=20_000_000):
    """Start Time=0 packet, followed by recursive binary64 additions."""
    if interval_s <= 0 or not math.isfinite(interval_s):
        raise ValueError("EF interarrival must be positive finite seconds")
    if duration_s / interval_s * n_onu >= max_packets:
        raise ValueError("EF settings exceed the configured packet safety bound")
    result = []
    now = 0.0
    while now <= duration_s:
        result.append(now)
        following = now + interval_s
        if following <= now:
            raise ValueError("EF interval does not advance binary64 time")
        now = following
    return np.asarray(result, dtype=np.float64)


def generate_offered(ef_interval_s, af_rate_pps, be_rate_pps, *, duration_s=5.0,
                     n_onu=64, seed=128, hurst=0.7, onset_s=1.0,
                     max_packets=20_000_000, max_components_per_onu=1_000_000):
    """Return (offered arrays, metadata) suitable for existing Python DES.

    Keys are time(float64 seconds), onu(uint16, zero based), queue(uint8:
    EF=0/AF=1/BE=2), bits(uint32), event_id(uint64). Events are sorted by
    time with deterministic ties. Event IDs are synthetic, not native IDs.
    Duration is inclusive, consistent with the current DES input boundary.
    Zero AF or BE rate disables that class; EF remains a periodic source.
    """
    if not math.isfinite(duration_s) or duration_s <= 0:
        raise ValueError("Duration must be positive finite seconds")
    if isinstance(n_onu, bool) or int(n_onu) != n_onu or not 1 <= n_onu <= 65535:
        raise ValueError("ONU count must be an integer in 1..65535")
    if isinstance(seed, bool) or int(seed) != seed or seed < 0:
        raise ValueError("Seed must be a nonnegative integer")
    if not math.isfinite(hurst) or not 0.5 < hurst < 1 or not math.isfinite(onset_s) or onset_s <= 0:
        raise ValueError("Require 0.5 < H < 1 and a positive finite onset")
    if any(not math.isfinite(rate) or rate < 0 for rate in (af_rate_pps, be_rate_pps)):
        raise ValueError("AF and BE rates must be finite nonnegative packets/second")
    n_onu, seed = int(n_onu), int(seed)
    ef = ef_arrivals(ef_interval_s, duration_s, n_onu, max_packets)
    ef_count = len(ef) * n_onu
    parts = {"time": [np.repeat(ef, n_onu)],
             "onu": [np.tile(np.arange(n_onu, dtype=np.uint16), len(ef))],
             "queue": [np.zeros(ef_count, dtype=np.uint8)],
             "bits": [np.full(ef_count, 560, dtype=np.uint32)]}
    count = ef_count
    class_metadata = {"EF": {"packet_count": ef_count, "expected_packet_count": ef_count,
                              "packet_size_bits": 560, "interarrival_seconds": ef_interval_s}}
    for queue, label, rate in ((1, "AF", af_rate_pps), (2, "BE", be_rate_pps)):
        if rate == 0:
            class_metadata[label] = {"enabled": False, "packet_count": 0, "expected_packet_count": 0}
            continue
        params = _math.parameters(rate, hurst, onset_s)
        if params.source_count > max_components_per_onu:
            raise ValueError("Sup-FRP component count exceeds the configured safety bound")
        class_count, max_rounds = 0, 0
        for onu in range(n_onu):
            times, sizes, rounds = _renewals(params, duration_s, seed, onu, queue)
            number = len(times)
            count += number
            if count > max_packets:
                raise ValueError("Generated offered traffic exceeds the configured packet safety bound")
            parts["time"].append(times)
            parts["bits"].append(sizes)
            parts["onu"].append(np.full(number, onu, dtype=np.uint16))
            parts["queue"].append(np.full(number, queue, dtype=np.uint8))
            class_count += number
            max_rounds = max(max_rounds, rounds)
        expected = rate * duration_s * n_onu
        class_metadata[label] = {"enabled": True, "parameters": asdict(params),
                                "packet_count": class_count, "expected_packet_count": expected,
                                "count_relative_difference": class_count / expected - 1,
                                "actual_packets_per_second_per_onu": class_count / duration_s / n_onu,
                                "max_component_renewal_rounds": max_rounds}
    combined = {key: np.concatenate(value) for key, value in parts.items()}
    del parts
    order = np.argsort(combined["time"], kind="stable")
    output = {key: value[order] for key, value in combined.items()}
    output["event_id"] = np.arange(1, len(order)+1, dtype=np.uint64)
    del combined, order
    for queue, label in enumerate(("EF", "AF", "BE")):
        sizes = output["bits"][output["queue"] == queue]
        class_metadata[label].update({"offered_bits": int(sizes.sum()),
                                     "offered_bitrate_bps": float(sizes.sum()) / duration_s,
                                     "observed_mean_packet_bits": float(sizes.mean()) if len(sizes) else None})
    metadata = {
        "generator": VERSION, "mode": "source_supfrp",
        "validation_status": "SOURCE_LAWS_VERIFIED_NATIVE_RNG_PARITY_UNVERIFIED",
        "native_recorded_input_used": False, "native_seeded_trace_equivalence": False,
        "duration_seconds": duration_s, "onu_count": n_onu, "seed": seed,
        "rates": {"ef_interarrival_seconds": ef_interval_s,
                  "af_arrival_rate_pps": af_rate_pps, "be_arrival_rate_pps": be_rate_pps},
        "hurst": hurst, "fractal_onset_seconds": onset_s, "start_time_seconds": 0.0,
        "end_boundary": "inclusive", "rng": "NumPy Generator PCG64",
        "numpy_version": np.__version__,
        "stream_assignment": "SeedSequence([seed, zero_based_onu, queue, kind]); kind=0 arrivals, kind=1 sizes",
        "draw_order": "All M stationary first draws, then M packet-size and M ordinary-interval draws per component round; independent streams",
        "native_draw_order_gap": "OPNET uses process/event ordered draws; its RNG family, seed mapping, stream assignment and initialization ordering are not reproduced",
        "ef_timing": "First packet at zero; recursive binary64 interval additions",
        "af_be_initialization": "One stationary first residual interval for each of M independent renewal components per ONU/class",
        "af_be_packet_size": "ceil(uniform(512,12144)) bits",
        "af_be_analytic_mean_packet_bits": 6328.5,
        "event_id_semantics": "Unique synthetic IDs in sorted offered stream, not native packet IDs",
        "tie_order": "Stable time order; EF ONU ascending before AF and BE component-round order",
        "source_evidence": source_evidence(), "classes": class_metadata,
        "returned_packet_count": count, "offered_bits": int(output["bits"].sum()),
        "offered_bitrate_bps": float(output["bits"].sum()) / duration_s,
    }
    return output, metadata
