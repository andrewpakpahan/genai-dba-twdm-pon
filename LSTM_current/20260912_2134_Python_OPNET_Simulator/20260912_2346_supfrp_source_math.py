"""Recovered OPNET Sup-FRP arithmetic, without RNG or event scheduling.

Manager formulas: archived EPON_rpg_mgr.pr.c lines 148-235.
OMS formulas: parent agent's live Windows readback of oms_dist_support.ex.c,
ordinary outcome around lines 1199-1215 and first outcome from line 1220.

Uniform inputs must be supplied explicitly. Native code rejects zero draws in
a loop; this module rejects zero rather than substituting another RNG. Passing
these arithmetic checks does not reproduce OPNET seeded packet streams.
"""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class SupFRPParameters:
    rate_packets_per_second: float
    hurst: float
    requested_onset_seconds: float
    shape: float
    unrounded_location_seconds: float
    unrounded_source_count: float
    source_count: int
    location_seconds: float
    adjusted_onset_seconds: float


def parameters(rate: float, hurst: float, onset_seconds: float) -> SupFRPParameters:
    """Compute native M and adjusted A for the finite ordinary research range.

    The native warning/clamp at OPC_INT_INFINITY is not reproduced because that
    platform constant is not resolved here. All audited Case_Base inputs are
    far below normal integer limits; callers must not infer extreme-rate parity.
    """
    if not all(math.isfinite(v) for v in (rate, hurst, onset_seconds)):
        raise ValueError("Parameters must be finite")
    if rate <= 0 or onset_seconds <= 0 or not 0.5 < hurst < 1:
        raise ValueError("Require positive rate/onset and 0.5 < H < 1")
    shape = 3.0 - 2.0 * hurst
    temp1 = 2.0 * math.pow(shape, 2.0) * (shape - 1.0) * math.exp(shape)
    temp1 *= 1 / ((2.0 - shape) * (3.0 - shape))
    temp2 = 1.0 + (shape - 1.0) * math.exp(shape)
    temp1 *= math.pow(temp2, -2.0)
    unrounded_location = math.pow(temp1, 1.0 / (2.0 - shape)) * onset_seconds
    unrounded_count = rate * (1.0 + 1.0 / ((shape - 1.0) * math.exp(shape))) * unrounded_location / shape
    count = int(unrounded_count + 0.5)
    if count == 0:
        raise ValueError("Native manager terminates when rounded source count M is zero")
    location = count * shape
    location /= (1.0 + 1.0 / ((shape - 1.0) * math.exp(shape))) * rate
    adjusted_onset = math.pow(temp1, -1.0 / (2.0 - shape)) * location
    return SupFRPParameters(rate, hurst, onset_seconds, shape, unrounded_location,
                            unrounded_count, count, location, adjusted_onset)


def _validate_draw(location: float, shape: float, uniform: float) -> None:
    if not all(math.isfinite(v) for v in (location, shape, uniform)):
        raise ValueError("Location, shape and uniform draw must be finite")
    if location <= 0 or not 1 < shape < 2 or not 0 < uniform <= 1:
        raise ValueError("Require A > 0, 1 < shape < 2 and 0 < uniform <= 1")


def ordinary_outcome(location: float, shape: float, uniform: float) -> float:
    """OMS smoothed-Pareto interarrival, given one accepted uniform draw."""
    _validate_draw(location, shape, uniform)
    if uniform >= math.exp(-shape):
        return -math.log(uniform) * location / shape
    return location * math.exp(-1.0) * math.pow(uniform, -1.0 / shape)


def first_outcome(location: float, shape: float, uniform: float) -> float:
    """OMS special stationary first interarrival, given one accepted draw."""
    _validate_draw(location, shape, uniform)
    transformed = (1.0 + (shape - 1.0) * math.exp(shape)) * uniform / shape
    if transformed >= 1.0:
        return location * (-1.0) / shape * math.log(
            uniform * (shape * transformed - 1.0) /
            (shape * transformed - uniform))
    return location * math.pow(transformed, 1.0 / (1.0 - shape))


def ordinary_mean(location: float, shape: float) -> float:
    """Analytic mean of the ordinary interarrival, used to check rate closure."""
    _validate_draw(location, shape, 0.5)
    return location / shape * (1.0 + math.exp(-shape) / (shape - 1.0))
