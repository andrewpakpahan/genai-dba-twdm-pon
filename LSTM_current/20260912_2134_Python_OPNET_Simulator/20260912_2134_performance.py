"""Packet completion throughput, delay spread, and per-flow packet delay variation.

This is separate from native complete-slot service accounting and native running
EF variance. Equality of delay means does not validate these jitter statistics.
"""
from __future__ import annotations

import csv
from decimal import Decimal
from pathlib import Path
import numpy as np

CLASSES = ("ef", "af", "be", "total")


def _delay_statistics(delay_sec):
    values = np.asarray(delay_sec, dtype=float) * 1000.
    out = {"delay_sample_count": len(values), "delay_mean_ms": None,
           "delay_std_population_ms": None, "delay_variance_population_ms2": None,
           "delay_p95_ms": None, "delay_p99_ms": None}
    if len(values):
        out.update(delay_mean_ms=float(values.mean()),
                   delay_std_population_ms=float(values.std(ddof=0)),
                   delay_variance_population_ms2=float(values.var(ddof=0)),
                   delay_p95_ms=float(np.quantile(values, .95)),
                   delay_p99_ms=float(np.quantile(values, .99)))
    else:
        out["delay_undefined_reason"] = "No packet completed in the window"
    return out


def _ipdv_statistics(signed_sec):
    signed = np.asarray(signed_sec, dtype=float) * 1000.
    absolute = np.abs(signed)
    out = {"ipdv_pair_count": len(signed), "ipdv_signed_mean_ms": None,
           "ipdv_absolute_mean_ms": None, "ipdv_absolute_p95_ms": None,
           "ipdv_absolute_p99_ms": None, "ipdv_rms_ms": None}
    if len(signed):
        out.update(ipdv_signed_mean_ms=float(signed.mean()),
                   ipdv_absolute_mean_ms=float(absolute.mean()),
                   ipdv_absolute_p95_ms=float(np.quantile(absolute, .95)),
                   ipdv_absolute_p99_ms=float(np.quantile(absolute, .99)),
                   ipdv_rms_ms=float(np.sqrt(np.mean(signed * signed))))
    else:
        out["ipdv_undefined_reason"] = "No adjacent same-ONU same-class pair has both completions in the window"
    return out


def _prepare(result, offered):
    services, drops = result["services"], result["drops"]
    required = ("time", "wire_completion_time", "arrival_time", "queue", "onu", "wavelength", "bits")
    if services.dtype.names is None or any(k not in services.dtype.names for k in required):
        raise ValueError(f"Services require structured fields {required}")
    if any(not np.isfinite(services[k]).all() for k in required):
        raise ValueError("Nonfinite packet service field")
    if (services["bits"] < 0).any() or (services["wire_completion_time"] < services["time"]).any() or (services["time"] < services["arrival_time"]).any():
        raise ValueError("Negative packet size or reversed packet timestamps")
    config = result["summary"]["config"]
    channels, onu_count = int(config["channels"]), int(config["onu_count"])
    if channels < 1 or onu_count < 1:
        raise ValueError("Invalid channel or ONU count")
    if np.any((services["queue"] < 0) | (services["queue"] > 2)) or np.any((services["onu"] < 0) | (services["onu"] >= onu_count)) or np.any((services["wavelength"] < 0) | (services["wavelength"] >= channels)):
        raise ValueError("Service class, ONU, or wavelength out of range")
    offered = {k: np.asarray(offered[k]) for k in ("time", "queue", "bits")}
    if any(not np.isfinite(v).all() for v in offered.values()) or len({len(v) for v in offered.values()}) != 1 or (offered["bits"] < 0).any() or ((offered["queue"] < 0) | (offered["queue"] > 2)).any():
        raise ValueError("Invalid offered packet fields")
    if any(not np.isfinite(drops[k]).all() for k in ("time", "queue", "bits")) or (drops["bits"] < 0).any() or ((drops["queue"] < 0) | (drops["queue"] > 2)).any():
        raise ValueError("Invalid dropped packet fields")
    delay = services["wire_completion_time"] - services["arrival_time"]
    # Stable row order breaks simultaneous service-start ties. Group by ONU and
    # class BEFORE differencing; interleaved ONU packets must never form pairs.
    order = np.lexsort((np.arange(len(services)), services["time"], services["queue"], services["onu"]))
    left, right = order[:-1], order[1:]
    same_flow = ((services["onu"][left] == services["onu"][right]) &
                 (services["queue"][left] == services["queue"][right]))
    left, right = left[same_flow], right[same_flow]
    return services, drops, offered, delay, left, right


def _edges(begin, end, width):
    # Decimal bin labels converted once to binary doubles: avoids accumulated
    # np.arange drift while retaining strict half-open packet inclusion.
    lo, hi, step = Decimal(str(begin)), Decimal(str(end)), Decimal(str(width))
    if step <= 0:
        raise ValueError("Bin width must be positive")
    count = int((hi - lo) / step)
    edges = [float(lo + i * step) for i in range(count + 1)]
    if edges[-1] < end:
        edges.append(float(end))
    else:
        edges[-1] = float(end)
    return np.asarray(edges)


def _window(prepared, channels, begin, end, bin_width, channel_rates=None):
    services, drops, offered, delay, left, right = prepared
    completed = (services["wire_completion_time"] >= begin) & (services["wire_completion_time"] < end)
    pair = completed[left] & completed[right]
    signed = delay[right] - delay[left]
    offered_window = (offered["time"] >= begin) & (offered["time"] < end)
    drop_window = (drops["time"] >= begin) & (drops["time"] < end)
    duration = end - begin
    edges = _edges(begin, end, bin_width)
    completed_indices = np.flatnonzero(completed)
    # Index only selected completions, so the upper endpoint cannot leak into
    # the last bin. A completion exactly on an internal edge enters the next bin.
    bin_index = np.searchsorted(edges, services["wire_completion_time"][completed_indices], side="right") - 1
    rows, bins = [], []
    for channel in ["all", *range(channels)]:
        capacity = None if channel_rates is None else float(sum(channel_rates) if channel == "all" else channel_rates[channel])
        channel_select = np.ones(len(services), dtype=bool) if channel == "all" else services["wavelength"] == channel
        channel_pairs = np.ones(len(left), dtype=bool) if channel == "all" else ((services["wavelength"][left] == channel) & (services["wavelength"][right] == channel))
        for q, name in enumerate(CLASSES):
            class_select = np.ones(len(services), dtype=bool) if name == "total" else services["queue"] == q
            selected = completed & channel_select & class_select
            pair_select = pair & channel_pairs & class_select[right]
            bits = float(services["bits"][selected].sum())
            row = {"traffic_class": name, "wavelength": channel,
                   "start_sec": begin, "end_exclusive_sec": end, "duration_sec": duration,
                   "completed_packets": int(selected.sum()), "completed_packet_bits": bits,
                   "packet_throughput_bps": bits / duration, "packet_throughput_gbps": bits / duration / 1e9,
                   "capacity_bps": capacity,
                   "payload_utilization": bits / duration / capacity if capacity is not None else None,
                   **_delay_statistics(delay[selected]), **_ipdv_statistics(signed[pair_select])}
            if channel == "all":
                om = offered_window & (True if name == "total" else offered["queue"] == q)
                dm = drop_window & (True if name == "total" else drops["queue"] == q)
                offered_count, drop_count = int(om.sum()), int(dm.sum())
                row.update(offered_packets=offered_count, offered_packet_bits=float(offered["bits"][om].sum()),
                           offered_rate_bps=float(offered["bits"][om].sum()) / duration,
                           offered_load_fraction=float(offered["bits"][om].sum()) / duration / capacity if capacity is not None else None,
                           dropped_packets=drop_count, dropped_packet_bits=float(drops["bits"][dm].sum()),
                           dropped_packet_fraction=drop_count / offered_count if offered_count else None)
                if drop_count > offered_count:
                    raise ValueError("Dropped count exceeds offered count in the same arrival window")
            else:
                row.update(offered_packets=None, offered_packet_bits=None, dropped_packets=None,
                           offered_rate_bps=None, offered_load_fraction=None,
                           dropped_packet_bits=None, dropped_packet_fraction=None,
                           offered_drop_channel_reason="Exogenous arrival and drop records have no authoritative wavelength assignment")
            rows.append(row)
            local = (channel_select & class_select)[completed_indices]
            count_by_bin = np.bincount(bin_index[local], minlength=len(edges) - 1)
            bits_by_bin = np.bincount(bin_index[local], weights=services["bits"][completed_indices[local]], minlength=len(edges) - 1)
            assert int(count_by_bin.sum()) == row["completed_packets"]
            for b in range(len(edges) - 1):
                width = float(edges[b + 1] - edges[b])
                bins.append({"traffic_class": name, "wavelength": channel,
                             "start_sec": float(edges[b]), "end_exclusive_sec": float(edges[b + 1]),
                             "duration_sec": width, "completed_packets": int(count_by_bin[b]),
                             "completed_packet_bits": float(bits_by_bin[b]),
                             "packet_throughput_bps": float(bits_by_bin[b]) / width,
                             "packet_throughput_gbps": float(bits_by_bin[b]) / width / 1e9,
                             "capacity_bps": capacity,
                             "payload_utilization": float(bits_by_bin[b]) / width / capacity if capacity is not None else None})
    return {"start_sec": begin, "end_exclusive_sec": end, "duration_sec": duration,
            "rows": rows, "throughput_bins": bins}


def summarize(result, offered, begin=4.10, end=5.0, *, bin_width=.05, run_name="python"):
    """JSON-ready full-run and primary throughput/delay/jitter statistics.

    Empty estimates are None with a reason; zero throughput over a positive
    duration is meaningful and remains numeric zero.
    """
    duration = float(result["summary"]["config"]["duration"])
    if not np.isfinite([begin, end, duration, bin_width]).all() or not 0 <= begin < end <= duration or duration <= 0 or bin_width <= 0:
        raise ValueError("Invalid primary/full window or bin width")
    prepared = _prepare(result, offered)
    channels = int(result["summary"]["config"]["channels"])
    config = result["summary"]["config"]
    rates = config.get("channel_rates")
    if rates is None and config.get("line_rate") is not None:
        rates = [config["line_rate"]] * channels
    if rates is not None and (len(rates) != channels or not np.isfinite(rates).all() or any(r <= 0 for r in rates)):
        raise ValueError("Capacity requires one finite positive bit rate per wavelength")
    return {"run": run_name, "primary": _window(prepared, channels, begin, end, bin_width, rates),
            "full_run": _window(prepared, channels, 0., duration, bin_width, rates),
            "native_throughput_statistic_verified": False, "native_jitter_statistic_verified": False,
            "definitions": {
                "packet_throughput": "Sum of packet bits whose wire completion falls in [begin,end), divided by window seconds. Excludes REPORT/GATE/control packets; packet bits may include protocol overhead and are not verified application payload goodput. Not equated with OPNET receiver statistic aggregation.",
                "capacity_bps": "Sum of configured wavelength rates for all-channel rows; the individual wavelength rate for per-channel rows. Class rows share the same physical denominator. Missing rates are unavailable, never assumed.",
                "payload_utilization": "Completed data-packet throughput divided by capacity_bps, without clipping. EF/AF/BE fractions partition the total data-packet utilization; packet bits are not verified application-payload bits. REPORT/GATE/control use is excluded.",
                "offered_load_fraction": "Offered packet bits per second divided by total configured capacity. Changing wavelength rates does not regenerate or rescale the recorded exogenous traffic.",
                "delay": "Wire completion time minus original offered packet arrival time, converted from seconds to milliseconds; packet-weighted mean/percentiles. Native 80% uniform mean agreement is separate evidence, not encoded as a universal validation flag.",
                "delay_population_spread": "Population standard deviation (ddof=0) and variance over all selected completed packet delays. Standard deviation in ms; variance in ms^2. Total pools packet observations across classes, not class means.",
                "ipdv": "Signed current-minus-previous packet delay, adjacent within the same ONU and class in service-start order; original row order breaks exact service-start ties. Both packet completions must lie in the window. No inter-ONU/class or boundary-crossing pairs. Absolute summary is abs(signed difference), in ms.",
                "ipdv_total": "Pool valid within-ONU/class differences across flows. Never difference packets from different flows. Per-wavelength rows include a pair only when both packets used that wavelength.",
                "ipdv_rms": "sqrt(mean(signed_ipdv_ms^2)), identical to RMS of absolute IPDV.",
                "percentiles": "numpy.quantile linear interpolation; sample/pair counts reported independently.",
                "native_EF_variance": "Native EF statistic uses oms_stat_sample_add(..., ete_delay*1000) and oms_stat_variance_obtain for a running statistic. These fixed-window population estimates are different and have not been validated against that native trace.",
                "bins": "Default 0.05-second non-overlapping half-open completion bins; final partial bin uses its actual duration. Full bins sum exactly to their window packet counts/bits.",
                "offered_and_drops": "All-channel records count arrival/drop times in the same [begin,end) window. Counts are not the source cohort of the completed packets; queued carryover may increase completions beyond window arrivals. Per-wavelength offered/drop counts unavailable without authoritative channel assignment.",
            }}


def write_csv_tables(summary, output_dir):
    """Write aggregate and time-bin CSVs without modifying simulation files."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = {}
    for table, key in (("performance", "rows"), ("throughput_bins", "throughput_bins")):
        rows = [{"run": summary["run"], "window": window, **row}
                for window in ("primary", "full_run") for row in summary[window][key]]
        fields = list(dict.fromkeys(k for row in rows for k in row))
        target = output / f"{table}.csv"
        with target.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        paths[table] = str(target)
    return paths
