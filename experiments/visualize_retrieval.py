"""Build a reproducible frozen-versus-PONG qualitative retrieval figure.

The figure uses real OS-ABO query RGB images and rendered gallery views. Examples
are selected by the largest increase in top-5 label matches under PONG.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from methods.pong import l2_normalize, refine


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA = PROJECT_ROOT / "data/OS-ABO-core"
FEATURES = PROJECT_ROOT / "data/openclip_L14"
OUT_DIR = PROJECT_ROOT / "results/qualitative"
OUT = OUT_DIR / "qualitative_abo_openclip.png"
OUT_PDF = OUT_DIR / "qualitative_abo_openclip.pdf"
META = OUT_DIR / "qualitative_abo_openclip.json"

TILE = 86
QUERY_TILE = 96
MARGIN = 16
GAP = 6
GROUP_GAP = 14
HEADER = 34
GREEN = (38, 135, 70)
RED = (207, 51, 51)
BLUE = (36, 102, 181)
BLACK = (25, 25, 25)
GRAY = (90, 90, 90)


def load_split(split: str):
    entries = []
    for line in (DATA / f"{split}_label.txt").read_text().splitlines():
        object_id, label = line.strip().split(",")
        entries.append((object_id, label))
    return entries


def load_descriptor(split: str) -> np.ndarray:
    features = np.load(FEATURES / f"abo_{split}_feats.npy").astype(np.float32)
    return l2_normalize(np.tanh(features.mean(axis=1)))


def image_for(split: str, object_id: str, prefer_rgb: bool) -> Image.Image:
    root = DATA / split / object_id
    candidates = []
    if prefer_rgb:
        candidates.extend(sorted((root / "real_image").glob("main_*.jpg")))
        candidates.extend(sorted((root / "real_image").glob("*.jpg")))
    candidates.extend(sorted((root / "image").glob("h_0.jpg")))
    candidates.extend(sorted((root / "image").glob("h_*.jpg")))
    if not candidates:
        raise FileNotFoundError(f"No view found for {root}")
    return Image.open(candidates[0]).convert("RGB")


def fit(image: Image.Image, size: int) -> Image.Image:
    scale = max(size / image.width, size / image.height)
    width, height = round(image.width * scale), round(image.height * scale)
    image = image.resize((width, height), Image.Resampling.LANCZOS)
    left, top = (width - size) // 2, (height - size) // 2
    return image.crop((left, top, left + size, top + size))


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    filename = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    return ImageFont.truetype(filename, size)


def draw_tile(canvas: Image.Image, xy: tuple[int, int], image: Image.Image, rank: int, correct: bool) -> None:
    x, y = xy
    canvas.paste(fit(image, TILE), (x, y))
    draw = ImageDraw.Draw(canvas)
    color = GREEN if correct else RED
    draw.rectangle((x, y, x + TILE - 1, y + TILE - 1), outline=color, width=5)
    draw.rounded_rectangle((x + 7, y + 7, x + 31, y + 31), radius=4, fill=(255, 255, 255))
    draw.text((x + 13, y + 9), str(rank), fill=BLACK, font=font(15, True))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    queries, targets = load_split("query"), load_split("target")
    query = load_descriptor("query")
    target = load_descriptor("target")
    query_labels = np.array([label for _, label in queries])
    target_labels = np.array([label for _, label in targets])

    frozen_order = np.argsort(-(query @ target.T), axis=1)
    refined_q, refined_t = refine(query, target, k=5, tau=0.2, lambda_min=0.0, lambda_max=0.7, rounds=2)
    pong_order = np.argsort(-(refined_q @ refined_t.T), axis=1)

    candidates = []
    for index, label in enumerate(query_labels):
        frozen_hits = int((target_labels[frozen_order[index, :5]] == label).sum())
        pong_hits = int((target_labels[pong_order[index, :5]] == label).sum())
        candidates.append((pong_hits - frozen_hits, pong_hits, -frozen_hits, index))

    selected = []
    used_labels = set()
    for _, _, _, index in sorted(candidates, reverse=True):
        if query_labels[index] in used_labels:
            continue
        if (target_labels[pong_order[index, :5]] == query_labels[index]).sum() <= (target_labels[frozen_order[index, :5]] == query_labels[index]).sum():
            continue
        selected.append(index)
        used_labels.add(query_labels[index])
        if len(selected) == 4:
            break
    if len(selected) != 4:
        raise RuntimeError("Could not identify four positive qualitative examples")

    x_query = MARGIN
    x_frozen = x_query + QUERY_TILE + GAP
    x_pong = x_frozen + TILE * 5 + GAP * 4 + GROUP_GAP
    row_height = max(QUERY_TILE + 18, TILE + 14)
    width = MARGIN * 2 + QUERY_TILE + GAP + TILE * 10 + GAP * 8 + GROUP_GAP
    height = HEADER + MARGIN + row_height * len(selected)
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((x_query, 8), "Query RGB", fill=BLACK, font=font(16, True))
    draw.text((x_frozen, 8), "Frozen retrieval", fill=BLACK, font=font(16, True))
    draw.text((x_pong, 8), "PONG retrieval", fill=BLACK, font=font(16, True))

    metadata = []
    for row, index in enumerate(selected):
        y = HEADER + MARGIN + row * row_height
        query_id, label = queries[index]
        query_image = fit(image_for("query", query_id, prefer_rgb=True), QUERY_TILE)
        canvas.paste(query_image, (x_query, y))
        draw.rectangle((x_query, y, x_query + QUERY_TILE - 1, y + QUERY_TILE - 1), outline=BLUE, width=4)
        draw.text((x_query, y + QUERY_TILE + 4), label, fill=GRAY, font=font(13, True))

        details = {"query_id": query_id, "label": label, "frozen": [], "pong": []}
        for name, start_x, ranking in (("frozen", x_frozen, frozen_order[index]), ("pong", x_pong, pong_order[index])):
            for rank, target_index in enumerate(ranking[:5], start=1):
                target_id, target_label = targets[target_index]
                correct = target_label == label
                draw_tile(canvas, (start_x + (rank - 1) * (TILE + GAP), y), image_for("target", target_id, prefer_rgb=False), rank, correct)
                details[name].append({"target_id": target_id, "target_label": target_label, "correct": bool(correct)})
        metadata.append(details)

    canvas.save(OUT, dpi=(300, 300))
    canvas.save(OUT_PDF, "PDF", resolution=300.0)
    META.write_text(json.dumps(metadata, indent=2))
    print(OUT)
    print(OUT_PDF)
    print(META)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
