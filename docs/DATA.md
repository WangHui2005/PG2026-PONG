# Data and Cached Features

PONG uses the four open-set 3D object retrieval benchmarks distributed through
the HGM2R/TeDA data pipeline: OS-ESB-core, OS-NTU-core, OS-ABO-core, and
OS-MN40-core. Place the datasets and cached features under `data/`, or create a
symbolic link named `data` that points to an existing cache.

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

The extraction script reuses the benchmark dataset loaders from a local
HGM2R/TeDA-style data repository. By default, it looks for `3dosr_fv` next to
the PONG repository. Override this path with `--dataset-code-root` or the
`PONG_DATASET_CODE_ROOT` environment variable.

```bash
python dataset/extract_features.py \
  --dataset-code-root /path/to/3dosr_fv \
  --backbone openclip \
  --dataset esb
```

Supported backbones are `clip`, `openclip`, and `dinov2`; supported dataset
arguments are `esb`, `ntu`, `abo`, `mn40`, and `all`.
