#!/usr/bin/env python3
"""
prepare_subset.py
Build a fixed, reproducible subset from CelebA aligned images + identity mapping.
Output is ImageFolder style:

out_dir/
  images/<identity>/*.jpg
  split.json  # fixed train/test/all relative paths (relative to out_dir/images)

This allows fair accuracy/latency comparisons across commits.

Usage:
python scripts/prepare_subset.py \
  --celeba_images img_align_celeba/img_align_celeba \
  --celeba_identity identity_CelebA.txt \
  --out_dir data/celebA_subset \
  --num_ids 50 \
  --imgs_per_id 20 \
  --seed 1337
"""
from __future__ import annotations
import argparse, json, os, random, shutil
from pathlib import Path
from collections import defaultdict

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--celeba_images", required=True)
    ap.add_argument("--celeba_identity", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--num_ids", type=int, default=50)
    ap.add_argument("--imgs_per_id", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--test_ratio", type=float, default=0.2)
    args = ap.parse_args()

    random.seed(args.seed)

    img_dir = Path(args.celeba_images)
    id_file = Path(args.celeba_identity)
    out_dir = Path(args.out_dir)
    out_img = out_dir / "images"

    if not img_dir.exists():
        raise FileNotFoundError(img_dir)
    if not id_file.exists():
        raise FileNotFoundError(id_file)

    # Load identity map
    by_id = defaultdict(list)
    for line in id_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        fn, ident = line.strip().split()
        p = img_dir / fn
        if p.exists():
            by_id[ident].append(fn)

    # Filter IDs with enough images
    candidates = [i for i, fns in by_id.items() if len(fns) >= args.imgs_per_id]
    if len(candidates) < args.num_ids:
        raise RuntimeError(f"Not enough IDs with >= imgs_per_id. Have {len(candidates)}, need {args.num_ids}.")

    selected_ids = sorted(random.sample(candidates, args.num_ids))

    # Reset output
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_img.mkdir(parents=True, exist_ok=True)

    all_rel = []
    for ident in selected_ids:
        fns = sorted(by_id[ident])
        chosen = random.sample(fns, args.imgs_per_id)
        cls_dir = out_img / ident
        cls_dir.mkdir(parents=True, exist_ok=True)
        for fn in chosen:
            src = img_dir / fn
            dst = cls_dir / fn
            shutil.copy2(src, dst)
            rel = str(Path(ident) / fn).replace("\\", "/")
            all_rel.append(rel)

    # Fixed split
    random.shuffle(all_rel)
    n_test = int(len(all_rel) * args.test_ratio)
    test = all_rel[:n_test]
    train = all_rel[n_test:]

    split = {
        "seed": args.seed,
        "num_ids": args.num_ids,
        "imgs_per_id": args.imgs_per_id,
        "test_ratio": args.test_ratio,
        "train": train,
        "test": test,
        "all": all_rel,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "split.json").write_text(json.dumps(split, indent=2, ensure_ascii=False), encoding="utf-8")

    print("[OK] Wrote subset to:", out_dir)
    print(" - images/: ImageFolder structure")
    print(" - split.json: fixed file lists")

if __name__ == "__main__":
    main()
