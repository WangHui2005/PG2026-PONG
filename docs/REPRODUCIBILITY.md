# Reproducibility

## Evaluation Contract

All reported PONG results use pre-extracted multi-view features. The standard
preprocessing sequence is view averaging, `tanh` saturation, and L2
normalization. PONG then performs two alternating cross-set propagation rounds
unless an experiment explicitly varies the round count.

Labels and retrieval metrics are not accessed during refinement. They are used
only after the final state for mAP, NDCG@100, ANMRR, robustness subsets, and
embedding diagnostics.

## Parameter Selection

`configs/lodo_protocol.json` defines the 48-configuration search space. For
each held-out dataset, a configuration is selected by mean development mAP on
the other three datasets and all three backbones. The selected configuration is
then applied unchanged to the three held-out settings.

The selected configurations and per-setting held-out results are stored in
`results/lodo/`.

## Released Results

- `results/lodo/`: LODO selection and held-out retrieval results.
- `results/ablations/`: propagation direction and round analyses.
- `results/embedding_analysis/`: geometry diagnostics and generated figures.
- `results/query_buffer/`: buffer size, state policy, and composition tests.
- `results/aqe_dba/`: iterative AQE+DBA selection and comparison.
- `results/robustness/`: reduced-view, descriptor-noise, and low-purity tests.
- `results/runtime_cpu/`: CPU stage-level exact-search timings.
- `results/runtime_gpu/`: same-GPU PONG and TeDA timing with full ranking.
- `results/qualitative/`: qualitative retrieval figure and metadata.

## Randomness

- Buffer-size experiments with `B=5` and `B=10` use five random partitions.
- Persistent-state evaluation uses five arrival orders over a fixed partition.
- Buffer-composition experiments use five seeds.
- Descriptor-noise experiments use seeds 11, 29, and 47.
- t-SNE uses seed 42 and deterministic class-balanced sampling.
- LODO selection, propagation ablations, and reduced-view subsets are
  deterministic.

## Runtime Scope

The GPU benchmark uses preloaded float32 CLIP ViT-L/14 descriptors, three
warm-up runs, and the median of five CUDA-synchronized measurements. PONG
timing includes dense cross-set similarity, exact top-k neighborhood rebuilding,
two alternating rounds, final scoring, and full ranking. The reproduced TeDA
timing includes its 2,000-step optimization core and final ranking.

Both methods exclude rendering, feature extraction, disk loading, labels,
external multimodal language-model generation, and CPU metric evaluation. The
benchmark therefore measures the test-time retrieval cores rather than an
end-to-end data-processing pipeline or a persistent ANN deployment.

## Validation

Run the data-free protocol tests with:

```bash
python -m unittest discover -s tests
```

Each experiment writes its outputs to the corresponding directory under
`results/`. Use a separate `--output-dir` when retaining multiple runs.
