"""Tiny Temporal Fusion Transformer for current-cycle GRANT(t) allocation.

The input window includes REPORT(t), and the current REPORT is used to project
the IPACT-style GRANT(t).  Same-cycle model metrics are diagnostic only.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


REQUEST_COLUMNS = ["ef_req_bit", "af_req_bit", "be_req_bit"]
GRANT_COLUMNS = ["ef_grant_bit", "af_grant_bit", "be_grant_bit"]
FEATURE_COLUMNS = REQUEST_COLUMNS + ["cycle_time", "total_req_bit"]
ID_COLUMNS = ["cycle", "onu_id"]
DEFAULT_DATASET = "olt_lstm_dba_dataset_ipact_c1_load80_seed0_5s.csv"


@dataclass
class WindowSet:
    features: np.ndarray
    targets: np.ndarray
    input_end_cycle: np.ndarray
    target_cycle: np.ndarray
    onu_id: np.ndarray


def clean_frame(frame: pd.DataFrame) -> pd.DataFrame:
    required = ID_COLUMNS + REQUEST_COLUMNS + GRANT_COLUMNS + ["cycle_time"]
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    result = frame[required].copy()
    result[required] = result[required].apply(pd.to_numeric, errors="coerce")
    result = result.replace([np.inf, -np.inf], np.nan).dropna(subset=required)
    result = result.drop_duplicates()
    if result.duplicated(ID_COLUMNS).any():
        raise ValueError("Duplicate (cycle, onu_id) keys found")
    if (result[REQUEST_COLUMNS + GRANT_COLUMNS + ["cycle_time"]] < 0).any().any():
        raise ValueError("Negative required values found")
    result["total_req_bit"] = result[REQUEST_COLUMNS].sum(axis=1)
    return result.sort_values(ID_COLUMNS, kind="stable").reset_index(drop=True)


def split_cycles(cycles: np.ndarray) -> dict[str, np.ndarray]:
    cycles = np.sort(np.asarray(cycles, dtype=int))
    test_count = max(1, int(np.ceil(len(cycles) * 0.20)))
    test_start = len(cycles) - test_count
    validation_count = max(1, int(np.ceil(test_start * 0.10)))
    return {
        "train": cycles[: test_start - validation_count],
        "validation": cycles[test_start - validation_count : test_start],
        "test": cycles[test_start:],
    }


def _even_indices(length: int, limit: int | None) -> np.ndarray:
    if limit is None or length <= limit:
        return np.arange(length, dtype=int)
    return np.unique(np.linspace(0, length - 1, limit, dtype=int))


def build_windows(
    frame: pd.DataFrame,
    *,
    window_size: int = 50,
    target_cycles: np.ndarray,
    max_samples: int | None = None,
) -> WindowSet:
    target_set = set(np.asarray(target_cycles, dtype=int))
    candidates: list[tuple[np.ndarray, np.ndarray, int, int, int]] = []
    for onu_id, group in frame.groupby("onu_id", sort=True):
        group = group.sort_values("cycle", kind="stable").reset_index(drop=True)
        segments = group["cycle"].diff().ne(1).cumsum()
        for _, segment in group.groupby(segments, sort=False):
            segment = segment.reset_index(drop=True)
            values = segment[FEATURE_COLUMNS].to_numpy(np.float32)
            targets = segment[GRANT_COLUMNS].to_numpy(np.float32)
            for target_index in range(window_size - 1, len(segment)):
                target_cycle = int(segment.iloc[target_index]["cycle"])
                if target_cycle in target_set:
                    candidates.append((values, targets, target_index, target_cycle, int(onu_id)))
    selected = [candidates[index] for index in _even_indices(len(candidates), max_samples)]
    if not selected:
        raise ValueError("No valid consecutive windows found")
    features = np.stack([item[0][item[2] - window_size + 1 : item[2] + 1] for item in selected])
    targets_out = np.stack([item[1][item[2]] for item in selected]).astype(np.float32)
    ends = np.asarray([item[3] for item in selected], dtype=int)
    target_cycles_out = np.asarray([item[3] for item in selected], dtype=int)
    onus = np.asarray([item[4] for item in selected], dtype=int)
    return WindowSet(features, targets_out, ends, target_cycles_out, onus)


class TinyTFT(nn.Module):
    """Small TFT-shaped network: variable selection, LSTM, temporal attention."""

    def __init__(self, input_size: int, hidden_size: int = 32, quantiles=(0.5, 0.9)):
        super().__init__()
        self.quantiles = tuple(float(q) for q in quantiles)
        self.variable_score = nn.Linear(input_size, input_size)
        self.input_projection = nn.Linear(input_size, hidden_size)
        self.encoder = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=1, batch_first=True)
        self.norm = nn.LayerNorm(hidden_size)
        self.gate = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Sigmoid())
        self.output = nn.Linear(hidden_size, 3 * len(self.quantiles))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.variable_score(features), dim=-1)
        selected = features * weights
        encoded, _ = self.encoder(torch.relu(self.input_projection(selected)))
        attended, _ = self.attention(encoded, encoded, encoded, need_weights=False)
        fused = self.norm(encoded[:, -1] + self.gate(attended[:, -1]) * attended[:, -1])
        raw = self.output(fused).view(-1, 3, len(self.quantiles))
        # Quantile crossing would invalidate a P90 reserve.  The current
        # experiment uses P50/P90, so parameterize P90 as a positive increment.
        positive = torch.nn.functional.softplus(raw)
        if len(self.quantiles) == 2 and self.quantiles == (0.5, 0.9):
            positive[:, :, 1] = positive[:, :, 0] + positive[:, :, 1]
        return positive


def pinball_loss(prediction: torch.Tensor, target: torch.Tensor, quantiles=(0.5, 0.9)) -> torch.Tensor:
    target = target.unsqueeze(-1)
    errors = target - prediction
    losses = torch.maximum(torch.as_tensor(quantiles, device=prediction.device) * errors,
                           (torch.as_tensor(quantiles, device=prediction.device) - 1.0) * errors)
    return losses.mean()


def _metrics(actual: np.ndarray, predicted: np.ndarray, model: str) -> list[dict]:
    rows = []
    for index, name in enumerate(("ef", "af", "be")):
        rows.append({
            "model": model,
            "target": "GRANT(t)",
            "traffic_class": name,
            "horizon": "current_cycle",
            "partition": "test",
            "sample_count": int(len(actual)),
            "unit": "bits",
            "mae_bits": float(mean_absolute_error(actual[:, index], predicted[:, index])),
            "rmse_bits": float(np.sqrt(mean_squared_error(actual[:, index], predicted[:, index]))),
            "r2": float(r2_score(actual[:, index], predicted[:, index])),
        })
    return rows


def ipact_grant_from_report(report: np.ndarray, capacity_bits: np.ndarray) -> np.ndarray:
    """Project current REPORT demand into an EF -> AF -> BE grant."""
    demand = np.maximum(np.asarray(report, dtype=float), 0.0)
    capacity = np.broadcast_to(np.asarray(capacity_bits, dtype=float).reshape(-1, 1), demand.shape)
    remaining = capacity[:, 0].copy()
    grant = np.zeros_like(demand)
    for index in range(demand.shape[1]):
        grant[:, index] = np.minimum(demand[:, index], np.maximum(remaining, 0.0))
        remaining -= grant[:, index]
    return grant


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _predict(model, features, device, batch_size=256):
    model.eval()
    loader = DataLoader(TensorDataset(torch.from_numpy(features)), batch_size=batch_size)
    values = []
    with torch.no_grad():
        for (batch,) in loader:
            values.append(model(batch.to(device)).cpu().numpy())
    return np.concatenate(values, axis=0)


def run(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("mps" if args.device == "auto" and torch.backends.mps.is_available() else args.device if args.device != "auto" else "cpu")
    data_path = Path(args.data)
    frame = clean_frame(pd.read_csv(data_path))
    splits = split_cycles(frame["cycle"].unique())
    sets = {
        name: build_windows(frame, window_size=args.window_size, target_cycles=cycles, max_samples=limit)
        for name, cycles, limit in (
            ("train", splits["train"], args.max_train_samples),
            ("validation", splits["validation"], args.max_validation_samples),
            ("test", splits["test"], args.max_test_samples),
        )
    }
    # Log-transform only targets/features that represent bits; cycle_time remains raw.
    def transform(values):
        result = values.copy()
        result[..., [0, 1, 2, 4]] = np.log1p(result[..., [0, 1, 2, 4]])
        return result
    x_train = transform(sets["train"].features)
    x_val = transform(sets["validation"].features)
    x_test = transform(sets["test"].features)
    y_train = np.log1p(sets["train"].targets)
    y_val = np.log1p(sets["validation"].targets)
    y_test = sets["test"].targets
    model = TinyTFT(len(FEATURE_COLUMNS), args.hidden_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loader = DataLoader(TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)), batch_size=args.batch_size, shuffle=False)
    best_state, best_loss, stale = None, float("inf"), 0
    for _ in range(args.epochs):
        model.train()
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            loss = pinball_loss(model(batch_x.to(device)), batch_y.to(device), model.quantiles)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(pinball_loss(model(torch.from_numpy(x_val).to(device)), torch.from_numpy(y_val).to(device), model.quantiles).cpu())
        if val_loss < best_loss:
            best_loss, stale, best_state = val_loss, 0, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= args.patience:
                break
    model.load_state_dict(best_state)
    prediction_log = _predict(model, x_test, device)
    prediction = np.expm1(prediction_log).clip(min=0.0)
    train_mean = sets["train"].targets.mean(axis=0)
    current_report = sets["test"].features[:, -1, :3]
    persistence = sets["test"].features[:, -2, :3]
    rows = _metrics(y_test, prediction[:, :, 0], "tft_q50")
    rows += _metrics(y_test, train_mean[None, :].repeat(len(y_test), axis=0), "mean_baseline")
    direct_grant = ipact_grant_from_report(current_report, sets["test"].features[:, -1, 3] * 1_000_000_000.0 / max(int(frame["onu_id"].nunique()), 1))
    persistence_grant = ipact_grant_from_report(persistence, sets["test"].features[:, -1, 3] * 1_000_000_000.0 / max(int(frame["onu_id"].nunique()), 1))
    rows += _metrics(y_test, direct_grant, "current_report_ipact_grant")
    rows += _metrics(y_test, persistence_grant, "report_persistence_ipact_grant")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    predictions = pd.DataFrame({"onu_id": sets["test"].onu_id, "input_end_cycle": sets["test"].input_end_cycle, "target_cycle": sets["test"].target_cycle})
    capacity = sets["test"].features[:, -1, 3] * 1_000_000_000.0 / max(int(frame["onu_id"].nunique()), 1)
    current_grant = direct_grant
    for i, name in enumerate(REQUEST_COLUMNS):
        grant_name = GRANT_COLUMNS[i].replace("_grant_bit", "")
        predictions[f"actual_{grant_name}_grant_bit"] = y_test[:, i]
        predictions[f"tft_q50_{grant_name}_grant_bit"] = prediction[:, i, 0]
        predictions[f"tft_q90_{grant_name}_grant_bit"] = prediction[:, i, 1]
        predictions[f"direct_ipact_{grant_name}_grant_bit"] = current_grant[:, i]
    predictions["capacity_bits"] = capacity
    predictions.to_csv(output / "predictions.csv", index=False)
    pd.DataFrame(rows).to_csv(output / "metrics.csv", index=False)
    torch.save(model.state_dict(), output / "tft_state.pt")
    metadata = {
        "model": "tiny_tft",
        "task": "REPORT(t-k...t) current-cycle features -> GRANT(t)",
        "warning": "same-cycle REPORT-to-GRANT metrics are diagnostic; allocation projection is offline and not closed-loop DBA evidence",
        "data": str(data_path), "data_sha256": _sha256(data_path), "features": FEATURE_COLUMNS,
        "target": GRANT_COLUMNS, "window_size": args.window_size, "quantiles": [0.5, 0.9],
        "split_cycles": {key: [int(value[0]), int(value[-1])] for key, value in splits.items()},
        "sample_limits": {"train": args.max_train_samples, "validation": args.max_validation_samples, "test": args.max_test_samples},
        "device": str(device), "seed": args.seed, "best_validation_pinball_log1p": best_loss,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", default="20260730_1300_tft_report_arrival_artifacts")
    parser.add_argument("--window-size", type=int, default=50)
    parser.add_argument("--max-train-samples", type=int, default=20000)
    parser.add_argument("--max-validation-samples", type=int, default=10000)
    parser.add_argument("--max-test-samples", type=int, default=10000)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
