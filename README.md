# PONG: Test-Time Ping-Pong Propagation for 3D Object Retrieval

PONG is a gradient-free test-time refinement method for zero-shot 3D object
retrieval. It alternates query-to-gallery and gallery-to-query propagation over
dynamically rebuilt cross-set neighborhoods. The implementation operates on
pre-extracted multi-view features and does not update the vision backbone.

## Repository Layout

```text
PONG_v1/
|-- configs/                 # LODO parameter-selection protocol
|-- data/                    # Bundled ESB sample features (OpenCLIP)
|-- dataset/                 # Multi-view feature extraction
|-- methods/                 # Canonical PONG implementation
|-- experiments/             # Evaluation, analysis, and benchmarks
|-- results/                 # Released CSV, JSON, and figure artifacts
|-- tests/                   # Data-free protocol tests
|-- third_party/             # Bundled benchmark dataset loaders (TeDA, Apache-2.0)
|-- utils/                   # Retrieval metrics
|-- docs/                    # Experiment and reproducibility details
|-- run_pong.py              # Main retrieval entry point
`-- requirements.txt
```

## Environment

Python 3.9 or later is recommended.

```bash
conda create -n pong python=3.9 -y
conda activate pong
pip install -r requirements.txt
```

Feature extraction additionally requires the selected backbone implementation
(`clip`, `open_clip`, or DINOv2) and the benchmark dataset loaders described in
`docs/DATA.md`.

## Data and Features

A small ESB sample (OpenCLIP ViT-L/14) is bundled under `data/`, so
`python run_pong.py --dataset esb --backbone openclip` works out of the box.
The other datasets and backbones require cached features or raw data; cached
features use the following layout:

```text
data/
|-- clip_official_L14/
|-- openclip_L14/
`-- dino_feats/
```

Each directory contains query and target feature arrays and label arrays for
`esb`, `ntu`, `abo`, and `mn40`. See `docs/DATA.md` for filenames and feature
extraction commands.

## Quick Start

Run PONG on one cached feature setting:

```bash
python run_pong.py --dataset esb --backbone openclip
```

The default configuration is `k=5`, `tau=0.2`,
`lambda_min=0.0`, `lambda_max=0.7`, and two propagation rounds. Parameters can
be overridden from the command line.

## Reproduce Experiments

```bash
# Leave-one-dataset-out parameter selection and held-out evaluation
python experiments/select_lodo.py --output-dir results/lodo

# Propagation direction and round ablations
python experiments/evaluate_ablations.py

# Query-buffer, reset, persistence, and composition diagnostics
python experiments/evaluate_query_buffers.py

# Comparison with iterative AQE+DBA
python experiments/compare_aqe_dba.py

# Reduced-view, descriptor-noise, and low-purity evaluations
python experiments/evaluate_robustness.py

# Embedding diagnostics and analysis figures
python experiments/analyze_embeddings.py

# Runtime benchmarks
python experiments/benchmark_runtime_cpu.py
python experiments/benchmark_runtime_gpu.py

# Qualitative retrieval figure
python experiments/visualize_retrieval.py
```

The released outputs are grouped by experiment under `results/`. Detailed
inputs, random seeds, and output files are listed in `docs/EXPERIMENTS.md`.

## Protocol Notes

- PONG is label-free during refinement. Labels are used only for evaluation or
  post-hoc visualization.
- Each query-buffer episode starts from immutable gallery descriptors. The
  refined gallery is discarded after ranking.
- Cross-set neighborhoods are rebuilt before each directional update.
- The implementation uses exact dense similarities and does not maintain a
  persistent ANN index.
- Fold-specific parameters are selected only from the other three datasets and
  are applied unchanged to the held-out dataset.

## Tests

The protocol tests do not require benchmark data:

```bash
python -m unittest discover -s tests
```

See `docs/REPRODUCIBILITY.md` for the complete evaluation contract and runtime
measurement scope.
