# spectral-comparison

Research code for comparing regional spectral organization across intracranial EEG and source-reconstructed high-density EEG.

This repository supports analyses for Chapter 2 on mapping oscillatory and aperiodic spectral features across cortical regions and recording modalities. The current workflow focuses on:

- Frauscher 2018 intracranial EEG atlas
- Mantini 2018 source-reconstructed HD-EEG
- Nencki-Symfonia HD-EEG replication pipeline in progress

## Project goals

The main questions in this project are:

- Whether regional spectral profiles are preserved across recording modalities
- How similar aperiodic parameters such as exponent and offset are between iEEG and HD-EEG
- How `specparam` and `IRASA` compare when applied to the same data
- How much aperiodic correction changes the detected oscillatory peaks

## Repository structure

- `pesco/` — reusable Python utilities for preprocessing, spectral estimation, spectral parameterization, IRASA, statistics, clustering, and plotting
- `notebooks/00_preprocessing/` — early preprocessing, artifact inspection, and data-loading experiments
- `notebooks/01_clustering/` — clustering and regional/lobar interval analyses
- `notebooks/02_fooof/` — main spectral parameterization workflow using `specparam` and `IRASA`
- `notebooks/03_viz/` — visualization scripts
- `notebooks/04_raport_Dante/` — report materials and figures
- `nencki/` — scripts and tests for the Nencki-Symfonia extension
- `data/` — local datasets and generated outputs
- `changes/` — dated development notes and analysis decisions

## Environment setup

This project uses [`uv`](https://docs.astral.sh/uv/) and a workspace package for `pesco`.

```bash
uv sync --all-extras
make kernel
