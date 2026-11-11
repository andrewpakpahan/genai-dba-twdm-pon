# Prediction-Assisted DBA for TWDM-PON

Simulation code, configuration files and results for the paper:

> A. F. Pakpahan and I.-S. Hwang, "A GenAI-Assisted Framework for Predictive
> Bandwidth Allocation in TWDM-PON," *Photonics*, 2026.

## What is here

- **PONSim**: a packet-level Python simulator of a TWDM-PON upstream
  (64 ONUs, two 1 Gbit/s wavelengths, EF/AF/BE traffic classes).
- **DBA schemes**: IPACT, and the proposed scheduler that orders ONUs by demand
  and reserves bounded AF bandwidth, with LSTM or TimesFM 3 as the forecaster.
- **Prediction experiments**: current-cycle AF grant prediction with LSTM,
  Informer, PatchTST, Sundial and TimesFM 3, and AF-arrival forecasting.
- **Configurations and seeds** for every run reported in the paper.
- **Raw results** behind the tables and figures.

## Main results

- Mean AF packet delay about 17.3% lower than IPACT, with throughput preserved
  and no BE packet loss up to 80% offered load.
- TimesFM 3, used without training on PON data, gave the lowest mean absolute
  error among the compared models.

## Repository layout

File and folder names keep the date prefixes used during the study
(`YYYYMMDD_HHMM_name`), so that the scripts find each other unchanged.

| Path | Contents |
|---|---|
| `LSTM_current/20260912_2134_Python_OPNET_Simulator/` | PONSim core (`..._simulator.py`, `..._performance.py`), traffic workload, the DBA engines and the two forecasting modules |
| `LSTM_current/20260918_2258_run_traffic_mix.py` | Runner for the network-level experiments (traffic mixes 2:4:4 and 1:6:3) |
| `LSTM_current/20260917_2213_run_full_loads.py` | Runner for traffic mix 1:3:6 |
| `LSTM_current/20260918_2306_report_traffic_mixes.py`, `..._2310_finalize_traffic_mixes.py` | Aggregation and reporting of the network-level results |
| `LSTM_current/20260916_0835_mix136_compare.py`, `..._0848_mix244_compare.py` | LSTM and TimesFM 3 grant-prediction experiments |
| `LSTM_current/20260917_1313_sundial_mix244.py`, `20260918_0509_sundial_mix136.py` | Sundial with adapter |
| `.../runs/mix_2w_10s/20260918_0509_informer_patchtst_matched/` | Informer and PatchTST runs |
| `.../runs/ponsim_traffic_mixes/20260918_2258/`, `.../runs/ponsim_full_loads/20260917_2213/` | Protocol, manifest and result tables of the network-level runs |
| `.../runs/mix_2w_10s/` | Protocols, settings and metrics of the grant-prediction experiments |
| `paper_results/case_metrics.csv` | Delay, throughput and loss for every traffic mix, load, seed and scheduler |
| `paper_results/held_out_prediction_metrics.csv` | AF-arrival forecasting errors on held-out data |

## Experiment settings

- 64 ONUs, two upstream wavelengths at 1 Gbit/s, 1.5 ms cycle budget,
  per-ONU grant limit 46,875 bits.
- Offered load 10% to 100% in steps of 10%; EF:AF:BE mixes 1:3:6, 2:4:4, 1:6:3.
- Network-level runs: seeds 131 and 132, 3 s warm-up followed by a 100 ms
  measurement window.
- Prediction data: 10 s runs, seeds 128 (training), 129 (validation),
  130 (testing).
- Inference time is recorded per run but is not charged to simulated time.

## What is not included

- Raw packet traces and per-case arrays (tens of gigabytes). They can be
  regenerated from the seeds and protocols above.
- Trained model weight files.
- Third-party code and weights. Obtain them from their authors:
  TimesFM (`google-research/timesfm`, checkpoint `google/timesfm-3.0-pytorch`),
  Sundial (`thuml/Sundial`, `thuml/sundial-base-128m`),
  PatchTST and Informer (`yuqinie98/PatchTST`). The scripts expect them under
  `LSTM_current/vendor/`.

## Requirements

Python 3.12 and the packages in `requirements.txt`. The experiments were run
on a MacBook Air with an Apple M4 chip (10-core CPU, 10-core GPU, 24 GB
memory), using the GPU through PyTorch MPS and TensorFlow Metal.

## License

MIT; see `LICENSE`.

## Citation

[BibTeX entry to be added on publication]
