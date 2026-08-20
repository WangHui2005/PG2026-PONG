# Experiment Reference

All paths are relative to the repository root. The released result directories
contain the exact CSV, JSON, and PDF artifacts used for analysis.

| Experiment | Script | Output | Protocol summary |
| --- | --- | --- | --- |
| LODO selection | `experiments/select_lodo.py` | `results/lodo/` | Search 48 configurations on three development datasets x three backbones; evaluate the selected configuration unchanged on the held-out dataset. |
| Propagation ablations | `experiments/evaluate_ablations.py` | `results/ablations/` | Compare query-only, target-only, simultaneous, fixed alternating, and PONG updates; separately sweep zero to five rounds. |
| Embedding analysis | `experiments/analyze_embeddings.py` | `results/embedding_analysis/` | Compute neighborhood purity, target-occurrence Gini, Fisher ratio, development marginals, and t-SNE from fold-selected final states. |
| Query-buffer analysis | `experiments/evaluate_query_buffers.py` | `results/query_buffer/` | Evaluate buffer sizes, episodic reset versus persistent state, and controlled buffer composition. |
| AQE+DBA comparison | `experiments/compare_aqe_dba.py` | `results/aqe_dba/` | Give iterative AQE+DBA an independent 32-configuration LODO search before held-out comparison. |
| Robustness analysis | `experiments/evaluate_robustness.py` | `results/robustness/` | Evaluate 24, 12, and 6 views; descriptor noise; and frozen low-purity query subsets. |
| CPU runtime | `experiments/benchmark_runtime_cpu.py` | `results/runtime_cpu/` | Report exact-search stages over all 12 backbone-dataset settings. |
| GPU runtime | `experiments/benchmark_runtime_gpu.py` | `results/runtime_gpu/` | Compare PONG and the reproduced TeDA optimization core on one RTX 4090 with full ranking. |
| Qualitative retrieval | `experiments/visualize_retrieval.py` | `results/qualitative/` | Build frozen-versus-PONG top-5 retrieval examples on OS-ABO-core with OpenCLIP. |

## Shared Settings

- Datasets: OS-ESB-core, OS-NTU-core, OS-ABO-core, and OS-MN40-core.
- Backbones: CLIP ViT-L/14, OpenCLIP ViT-L/14, and DINOv2 ViT-B/14.
- Main metrics: mAP, NDCG@100, and ANMRR.
- Main protocol: fold-selected LODO parameters and two propagation rounds.
- State policy: reset gallery descriptors at the start of every query-buffer
  episode and discard the refined gallery after ranking.

## Result Interpretation

The reduced-view and descriptor-noise evaluations are controlled feature-input
stress tests. Descriptor perturbation does not represent corrupted raw meshes
or scans. Low-purity evaluation uses the same frozen-defined query subset for
both frozen retrieval and PONG. Persistent gallery refinement is a diagnostic
condition and is not the PONG retrieval protocol.
