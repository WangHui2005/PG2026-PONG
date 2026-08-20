"""Extract CLIP, OpenCLIP, or DINOv2 features for the four benchmarks.

Adapted from TeDA's clip_feats_extract.py.
View-level features are saved under ``data/`` with shape ``(N, V, D)``.

Usage:
    # CLIP ViT-L/14 (official)
    python dataset/extract_features.py --backbone clip --dataset esb

    # OpenCLIP ViT-L/14
    python dataset/extract_features.py --backbone openclip --dataset esb

    # DINOv2 ViT-B/14
    python dataset/extract_features.py --backbone dinov2 --dataset esb

    # All datasets
    python dataset/extract_features.py --backbone clip --dataset all
"""

import argparse
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_CODE_ROOT = Path(
    os.environ.get("PONG_DATASET_CODE_ROOT", PROJECT_ROOT / "third_party")
)


def load_dataset_classes(dataset_code_root: Path):
    """Load the benchmark dataset classes from the HGM2R/TeDA data code."""
    if not dataset_code_root.exists():
        raise FileNotFoundError(
            f"Dataset code not found at {dataset_code_root}. Pass "
            "--dataset-code-root or set PONG_DATASET_CODE_ROOT."
        )
    sys.path.insert(0, str(dataset_code_root))
    from dataset.abo_core import ABOCoreDataset
    from dataset.esb_core import ESBCoreDataset
    from dataset.mn40_core import MN40CoreDataset
    from dataset.ntu_core import NTUCoreDataset

    return {
        "esb": ESBCoreDataset,
        "ntu": NTUCoreDataset,
        "mn40": MN40CoreDataset,
        "abo": ABOCoreDataset,
    }


def setup_seed(seed=2022):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_clip():
    import clip
    model, _ = clip.load("ViT-L/14")
    return model.cuda().eval(), 768


def build_openclip():
    import open_clip
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="laion2b_s32b_b82k"
    )
    return model.cuda().eval(), 768


def build_dinov2():
    # Load from local cache (bypass GitHub — network blocked on this server)
    cache_dir = os.path.expanduser('~/.cache/torch/hub/facebookresearch_dinov2_main')
    sys.path.insert(0, cache_dir)
    import hubconf
    model = hubconf.dinov2_vitb14(pretrained=True)
    sys.path.remove(cache_dir)
    # Clean up cached module so it doesn't interfere
    for key in list(sys.modules.keys()):
        if 'dinov2' in key or key == 'hubconf':
            del sys.modules[key]
    return model.cuda().eval(), 768


@torch.no_grad()
def extract_features(model, data_loader, backbone):
    """Extract view-level features: returns (N, 24, D)."""
    model.eval()
    feats, labels = [], []

    for batch in tqdm(data_loader, desc="Extracting"):
        # TeDA-style loaders may return (imgs, label) or (imgs, label, instance_path).
        mv_imgs, category, *_ = batch
        mv_imgs = mv_imgs.cuda()
        bz, n, c, h, w = mv_imgs.size()
        mv_imgs = mv_imgs.view(-1, c, h, w)

        if backbone in ('clip', 'openclip'):
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                mv_feat = model.encode_image(mv_imgs.half())
        else:
            # DINOv2 runs in float32
            mv_feat = model(mv_imgs)

        mv_feat = mv_feat.view(bz, n, -1)
        feats.append(mv_feat.detach().cpu())
        labels.append(category.detach().cpu())

    return torch.cat(feats, dim=0), torch.cat(labels, dim=0)


def get_dataset(mapping, name, data_dir, split, n_view=24):
    if name == 'abo':
        # ABO has only 4 views per object; pass n_view=4 to avoid division by zero
        return mapping[name](data_dir, split, modality='mv', n_view=4)
    return mapping[name](data_dir, split, modality='mv', n_view=n_view)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', required=True,
                        choices=['clip', 'openclip', 'dinov2'])
    parser.add_argument('--dataset', default='all',
                        choices=['all', 'esb', 'ntu', 'mn40', 'abo'])
    parser.add_argument('--n_view', type=int, default=24)
    parser.add_argument(
        '--dataset-code-root',
        type=Path,
        default=DEFAULT_DATASET_CODE_ROOT,
        help='Path containing the benchmark dataset Python package.',
    )
    args = parser.parse_args()

    setup_seed()
    dataset_classes = load_dataset_classes(args.dataset_code_root)

    # Build model
    builders = {'clip': build_clip, 'openclip': build_openclip, 'dinov2': build_dinov2}
    model, dim = builders[args.backbone]()
    print(f"Backbone: {args.backbone}  dim={dim}")

    # Output paths
    save_dirs = {
        'clip':     PROJECT_ROOT / 'data/clip_official_L14',
        'openclip': PROJECT_ROOT / 'data/openclip_L14',
        'dinov2':   PROJECT_ROOT / 'data/dino_feats',
    }
    save_dir = save_dirs[args.backbone]
    os.makedirs(save_dir, exist_ok=True)

    # File naming: match PONG convention
    if args.backbone == 'clip':
        suffix = '_ViT-L_14_official'
    else:
        suffix = ''

    datasets = ['esb', 'ntu', 'mn40', 'abo'] if args.dataset == 'all' else [args.dataset]

    for ds in datasets:
        data_dir = PROJECT_ROOT / (f'data/OS-{ds.upper()}-core' if ds != 'mn40' else 'data/OS-MN40-core')
        if ds == 'ntu':
            data_dir = PROJECT_ROOT / 'data/OS-NTU-core'

        print(f"\n{'='*50}")
        print(f"  {ds} — data_dir={data_dir}")
        print(f"{'='*50}")

        for split in ['query', 'target']:
            ds_obj = get_dataset(dataset_classes, ds, data_dir, split, args.n_view)
            loader = torch.utils.data.DataLoader(
                ds_obj, batch_size=1, shuffle=False, num_workers=0
            )

            t0 = time.time()
            feats, labels = extract_features(model, loader, args.backbone)
            elapsed = time.time() - t0

            feat_path = save_dir / f"{ds}_{split}_feats{suffix}.npy"
            label_path = save_dir / f"{ds}_{split}_labels{suffix}.npy"

            np.save(feat_path, feats.numpy())
            np.save(label_path, labels.numpy())

            print(f"  {split:>6s}: {feats.shape}  {elapsed:.1f}s  → {feat_path}")

    print(f"\nDone! Features saved to {save_dir}/")


if __name__ == '__main__':
    main()
