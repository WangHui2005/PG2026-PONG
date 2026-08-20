# Data and Cached Features

PONG runs on pre-extracted multi-view features for four open-set 3D object
retrieval benchmarks: ESB, NTU, ABO, and ModelNet40. Features live under
`data/` and are grouped by backbone.

## Bundled Sample Features

The repository ships a small sample so the method can be run without
downloading anything: ESB query/target features for the OpenCLIP ViT-L/14
backbone, under `data/openclip_L14/`. Reproduce the ESB result directly:

```bash
python run_pong.py --dataset esb --backbone openclip
```

The remaining datasets and backbones require cached features or raw data (see
below). Full feature files can be regenerated from the raw datasets with
`dataset/extract_features.py`.

## Expected Feature Layout

```text
data/
|-- clip_official_L14/
|   |-- esb_query_feats_ViT-L_14_official.npy
|   |-- esb_query_labels_ViT-L_14_official.npy
|   |-- esb_target_feats_ViT-L_14_official.npy
|   `-- esb_target_labels_ViT-L_14_official.npy
|-- openclip_L14/
|   |-- esb_query_feats.npy
|   |-- esb_query_labels.npy
|   |-- esb_target_feats.npy
|   `-- esb_target_labels.npy
`-- dino_feats/
    |-- esb_query_feats.npy
    |-- esb_query_labels.npy
    |-- esb_target_feats.npy
    `-- esb_target_labels.npy
```

The same filename pattern is used for `ntu`, `abo`, and `mn40`. Feature arrays
contain view-level descriptors with shape `(N, V, D)`; label arrays contain one
integer label per object.

## Feature Extraction

The extraction script reads the raw benchmark data through the bundled dataset
loaders in `third_party/dataset/` (adapted from TeDA, Apache-2.0). The raw
datasets are public benchmarks; place them under `data/OS-<NAME>-core` with the
layout expected by those loaders.

Feature extraction additionally requires the chosen vision backbone and its
dependencies:

- `clip`  — the OpenAI CLIP package,
- `openclip` — `open_clip` (ViT-L-14, LAION-2B),
- `dinov2` — a local `torch.hub` cache of the DINOv2 model.

The loaders also import `torchvision`; `open3d` is only needed for the
point-cloud and voxel modalities, not for multi-view extraction.

```bash
python dataset/extract_features.py --backbone openclip --dataset esb
```

Supported backbones are `clip`, `openclip`, and `dinov2`; supported dataset
arguments are `esb`, `ntu`, `abo`, `mn40`, and `all`.
