"""Train local Informer with seven causal features to predict GRANT(t+1)."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


common = import_module("20260730_1300_tft_report_arrival")
DEFAULT_DATASET = common.DEFAULT_DATASET
REQUEST_COLUMNS = common.REQUEST_COLUMNS
GRANT_COLUMNS = common.GRANT_COLUMNS
FEATURE_COLUMNS = REQUEST_COLUMNS + GRANT_COLUMNS + ["cycle_time"]
clean_frame = common.clean_frame
split_cycles = common.split_cycles
ipact_grant_from_report = common.ipact_grant_from_report

VENDOR_ROOT = Path(__file__).resolve().parent / "vendor" / "PatchTST" / "PatchTST_supervised"


@dataclass
class WindowSet:
    features: np.ndarray
    targets: np.ndarray
    input_end_cycle: np.ndarray
    target_cycle: np.ndarray
    onu_id: np.ndarray


def build_windows(frame: pd.DataFrame, *, window_size: int, target_cycles: np.ndarray, max_samples: int | None = None) -> WindowSet:
    target_set = set(np.asarray(target_cycles, dtype=int))
    candidates = []
    for onu_id, group in frame.groupby("onu_id", sort=True):
        group = group.sort_values("cycle", kind="stable").reset_index(drop=True)
        for _, segment in group.groupby(group["cycle"].diff().ne(1).cumsum(), sort=False):
            segment = segment.reset_index(drop=True)
            reports = segment[REQUEST_COLUMNS + ["cycle_time"]].to_numpy(np.float32)
            grants = segment[GRANT_COLUMNS].to_numpy(np.float32)
            for target_index in range(window_size + 1, len(segment)):
                target_cycle = int(segment.iloc[target_index]["cycle"])
                if target_cycle not in target_set:
                    continue
                report_window = reports[target_index - window_size : target_index]
                grant_history = grants[target_index - window_size - 1 : target_index - 1]
                features = np.concatenate([report_window[:, :3], grant_history, report_window[:, 3:4]], axis=1)
                input_end_cycle = int(segment.iloc[target_index - 1]["cycle"])
                candidates.append((features, grants[target_index], input_end_cycle, target_cycle, int(onu_id)))
    if not candidates:
        raise ValueError("No Informer t+1 windows found")
    if max_samples is not None and len(candidates) > max_samples:
        indices = np.unique(np.linspace(0, len(candidates) - 1, max_samples, dtype=int))
        candidates = [candidates[index] for index in indices]
    features = np.stack([row[0] for row in candidates]).astype(np.float32)
    targets = np.stack([row[1] for row in candidates]).astype(np.float32)
    input_end_cycles = np.asarray([row[2] for row in candidates], dtype=int)
    target_cycles = np.asarray([row[3] for row in candidates], dtype=int)
    onus = np.asarray([row[4] for row in candidates], dtype=int)
    return WindowSet(features, targets, input_end_cycles, target_cycles, onus)


def metric_rows(actual: np.ndarray, predicted: np.ndarray, model: str) -> list[dict]:
    rows = []
    for index, name in enumerate(("ef", "af", "be")):
        rows.append({
            "model": model, "target": "GRANT(t+1)", "traffic_class": name,
            "horizon": "t+1", "partition": "test", "sample_count": int(len(actual)), "unit": "bits",
            "mae_bits": float(mean_absolute_error(actual[:, index], predicted[:, index])),
            "rmse_bits": float(np.sqrt(mean_squared_error(actual[:, index], predicted[:, index]))),
            "r2": float(r2_score(actual[:, index], predicted[:, index])),
        })
    return rows


def _informer_class():
    if str(VENDOR_ROOT) not in sys.path:
        sys.path.insert(0, str(VENDOR_ROOT))
    from models.Informer import Model
    return Model


def build_model(*, feature_count: int, target_count: int, pred_len: int, d_model: int, n_heads: int, d_ff: int, e_layers: int, d_layers: int, dropout: float) -> nn.Module:
    config = SimpleNamespace(
        pred_len=pred_len, output_attention=False, enc_in=feature_count, dec_in=feature_count, c_out=target_count,
        d_model=d_model, n_heads=n_heads, d_ff=d_ff, e_layers=e_layers, d_layers=d_layers, factor=5,
        dropout=dropout, activation="gelu", embed_type=4, embed="timeF", freq="s", distil=True,
    )
    return _informer_class()(config)


def _decoder_input(features: torch.Tensor, label_len: int) -> torch.Tensor:
    return torch.cat([features[:, -label_len:, :], torch.zeros_like(features[:, :1, :])], dim=1)


def _batch_predict(model: nn.Module, features: torch.Tensor, *, label_len: int, device: torch.device) -> torch.Tensor:
    marks = torch.zeros((len(features), features.shape[1], 1), device=device)
    decoder = _decoder_input(features, label_len)
    decoder_marks = torch.zeros((len(features), decoder.shape[1], 1), device=device)
    return model(features, marks, decoder, decoder_marks)[:, 0, :]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _device(value: str) -> torch.device:
    if value != "auto":
        return torch.device(value)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _evaluate(model: nn.Module, loader: DataLoader, *, label_len: int, device: torch.device) -> float:
    loss = nn.MSELoss()
    values = []
    model.eval()
    with torch.no_grad():
        for features, targets in loader:
            prediction = _batch_predict(model, features.to(device), label_len=label_len, device=device)
            values.append(float(loss(prediction, targets.to(device)).item()))
    return float(np.mean(values))


def run(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = _device(args.device)
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
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
    feature_scaler = StandardScaler().fit(sets["train"].features.reshape(-1, len(FEATURE_COLUMNS)))
    target_scaler = StandardScaler().fit(sets["train"].targets)
    scaled_features = {
        name: feature_scaler.transform(values.features.reshape(-1, len(FEATURE_COLUMNS))).reshape(values.features.shape).astype(np.float32)
        for name, values in sets.items()
    }
    scaled_targets = {name: target_scaler.transform(values.targets).astype(np.float32) for name, values in sets.items()}
    loaders = {
        name: DataLoader(TensorDataset(torch.from_numpy(scaled_features[name]), torch.from_numpy(scaled_targets[name])), batch_size=args.batch_size, shuffle=False)
        for name in sets
    }
    label_len = args.window_size // 2
    model = build_model(feature_count=len(FEATURE_COLUMNS), target_count=len(GRANT_COLUMNS), pred_len=1, d_model=args.d_model, n_heads=args.n_heads, d_ff=args.d_ff, e_layers=args.e_layers, d_layers=args.d_layers, dropout=args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss = nn.MSELoss()
    best_state, best_loss, remaining = None, float("inf"), args.patience
    history = {"train_loss": [], "validation_loss": []}
    for epoch in range(args.epochs):
        model.train()
        train_losses = []
        for features, targets in loaders["train"]:
            optimizer.zero_grad()
            prediction = _batch_predict(model, features.to(device), label_len=label_len, device=device)
            batch_loss = loss(prediction, targets.to(device))
            batch_loss.backward()
            optimizer.step()
            train_losses.append(float(batch_loss.item()))
        validation_loss = _evaluate(model, loaders["validation"], label_len=label_len, device=device)
        history["train_loss"].append(float(np.mean(train_losses)))
        history["validation_loss"].append(validation_loss)
        print(f"epoch={epoch + 1} train_loss={history['train_loss'][-1]:.6f} validation_loss={validation_loss:.6f}", flush=True)
        if validation_loss < best_loss:
            best_loss, remaining = validation_loss, args.patience
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            remaining -= 1
            if remaining == 0:
                break
    model.load_state_dict(best_state)
    model.eval()
    outputs = []
    with torch.no_grad():
        for features, _ in loaders["test"]:
            outputs.append(_batch_predict(model, features.to(device), label_len=label_len, device=device).cpu().numpy())
    prediction = target_scaler.inverse_transform(np.concatenate(outputs)).clip(min=0.0)
    actual = sets["test"].targets
    capacity = sets["test"].features[:, -1, 6] * 1_000_000_000.0 / frame["onu_id"].nunique()
    current_report = sets["test"].features[:, -1, :3]
    previous_report = sets["test"].features[:, -2, :3]
    metrics = metric_rows(actual, prediction, "informer_7features")
    metrics += metric_rows(actual, np.repeat(sets["train"].targets.mean(axis=0)[None, :], len(actual), axis=0), "mean_baseline")
    metrics += metric_rows(actual, ipact_grant_from_report(current_report, capacity), "current_report_ipact_grant")
    metrics += metric_rows(actual, ipact_grant_from_report(previous_report, capacity), "report_persistence_ipact_grant")
    output.mkdir(parents=True)
    predictions = pd.DataFrame({"onu_id": sets["test"].onu_id, "input_end_cycle": sets["test"].input_end_cycle, "target_cycle": sets["test"].target_cycle})
    for index, name in enumerate(("ef", "af", "be")):
        predictions[f"actual_{name}_grant_bit"] = actual[:, index]
        predictions[f"informer_{name}_grant_bit"] = prediction[:, index]
    predictions.to_csv(output / "predictions.csv", index=False)
    pd.DataFrame(metrics).to_csv(output / "metrics.csv", index=False)
    torch.save(model.state_dict(), output / "informer.pt")
    (output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    metadata = {
        "model": "Informer (local PatchTST vendor implementation)", "attention": "ProbSparse", "device": str(device),
        "task": "seven causal features through t -> GRANT(t+1)", "data": str(data_path), "data_sha256": _sha256(data_path),
        "features": FEATURE_COLUMNS, "feature_semantics": "REPORT(t-window+1...t), GRANT(t-window...t-1), cycle_time(t)",
        "target": GRANT_COLUMNS, "window_size": args.window_size, "label_len": label_len, "horizon": 1,
        "sample_limits": {"train": args.max_train_samples, "validation": args.max_validation_samples, "test": args.max_test_samples},
        "model_parameters": {"d_model": args.d_model, "n_heads": args.n_heads, "d_ff": args.d_ff, "e_layers": args.e_layers, "d_layers": args.d_layers, "dropout": args.dropout, "learning_rate": args.learning_rate},
        "epochs_run": len(history["train_loss"]), "best_validation_loss": best_loss, "seed": args.seed, "metric_units": "bits",
        "warning": "strict one-step grant forecast; offline prediction only, not a closed-loop DBA claim",
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(pd.DataFrame(metrics).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", default="20260730_2000_informer_grant_7features_t1_artifacts")
    parser.add_argument("--window-size", type=int, default=50)
    parser.add_argument("--max-train-samples", type=int, default=20_000)
    parser.add_argument("--max-validation-samples", type=int, default=10_000)
    parser.add_argument("--max-test-samples", type=int, default=10_000)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=128)
    parser.add_argument("--e-layers", type=int, default=2)
    parser.add_argument("--d-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
