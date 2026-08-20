# Third-party dataset loaders

The files in `dataset/` are the benchmark dataset loaders used by
`dataset/extract_features.py` to read the ESB, NTU, ModelNet40, and ABO
multi-view data. They are adapted from the
[TeDA](https://github.com/3DObjectRepresentation/TeDA) data pipeline and are
distributed under the Apache License 2.0 (see `LICENSE`).

Modifications from the original:

- The top-level `import open3d as o3d` was moved into the `__read_vox` and
  `__read_pointcloud` methods. Multi-view feature extraction only exercises the
  `mv` modality, so `open3d` is no longer a hard dependency for extraction.
- `feat_dataset.py` and `fv_dataset.py` (TeDA training utilities) are not
  included; they are not needed by the PONG extraction path.
