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

## Requirements

Python 3.x, PyTorch, TensorFlow. Pretrained TimesFM 3 and Sundial weights are
downloaded from their original authors and are not included here.

## Citation

[BibTeX entry to be added on publication]
