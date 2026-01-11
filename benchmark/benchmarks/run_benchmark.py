#!/usr/bin/env python3
"""
run_benchmark.py
- Detect task: measures MTCNN face detection latency + memory.
- Train_eval task: trains a small classifier (ResNet18) for 1 epoch (default) and reports eval top-1 accuracy + latency + memory.

Outputs a JSON file containing:
- timestamps
- git info (if inside a git repo)
- hardware/software info
- metrics (latency/memory/accuracy)
- full command-line args

This script is intended for reproducible evidence in a public fork.
"""
from __future__ import annotations
import argparse, json, os, platform, subprocess, sys, time, math, random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import psutil

def _run(cmd: List[str]) -> Tuple[int, str]:
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
        return p.returncode, p.stdout.strip()
    except Exception as e:
        return 1, f"ERROR: {e}"

def get_git_info() -> Dict[str, str]:
    # Best effort; if not in git repo, return empty fields.
    info = {"commit": "", "branch": "", "status": ""}
    rc, out = _run(["git", "rev-parse", "HEAD"])
    if rc == 0:
        info["commit"] = out
    rc, out = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    if rc == 0:
        info["branch"] = out
    rc, out = _run(["git", "status", "--porcelain"])
    if rc == 0:
        info["status"] = "clean" if out == "" else "dirty"
    return info

def get_hw_sw_info() -> Dict[str, object]:
    info: Dict[str, object] = {}
    info["python"] = sys.version.replace("\n", " ")
    info["platform"] = platform.platform()
    info["machine"] = platform.machine()
    info["cpu_count_logical"] = psutil.cpu_count(logical=True)
    info["cpu_count_physical"] = psutil.cpu_count(logical=False)
    vm = psutil.virtual_memory()
    info["ram_total_gb"] = round(vm.total / (1024**3), 3)

    # Torch/CUDA info (optional)
    try:
        import torch
        info["torch_version"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["cuda_version"] = torch.version.cuda
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_count"] = torch.cuda.device_count()
    except Exception as e:
        info["torch_version"] = ""
        info["cuda_available"] = False
        info["torch_error"] = str(e)
    return info

def set_repro(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception:
        pass

def rss_mb() -> float:
    p = psutil.Process(os.getpid())
    return p.memory_info().rss / (1024**2)

def cuda_max_mem_mb() -> Optional[float]:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024**2)
        return None
    except Exception:
        return None

def _maybe_sync_cuda() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass

def load_split(split_json: Optional[str]) -> Optional[Dict[str, List[str]]]:
    if not split_json:
        return None
    p = Path(split_json)
    if not p.exists():
        raise FileNotFoundError(f"split_json not found: {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    return data

def iter_images_from_split(data_root: str, split: Optional[Dict[str, List[str]]], num_images: int) -> List[str]:
    root = Path(data_root)
    if split and "all" in split:
        rels = split["all"][:num_images]
        return [str(root / r) for r in rels]
    # fallback: scan filesystem
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    paths = []
    for p in root.rglob("*"):
        if p.suffix.lower() in exts:
            paths.append(str(p))
    paths.sort()
    return paths[:num_images]

def bench_detect(args) -> Dict[str, object]:
    import torch
    from PIL import Image
    from facenet_pytorch import MTCNN

    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else ("cpu" if args.device=="auto" else args.device))
    mtcnn = MTCNN(image_size=args.mtcnn_image_size, margin=args.mtcnn_margin, device=device)

    split = load_split(args.split_json)
    img_paths = iter_images_from_split(args.data_root, split, args.num_images)

    # Warmup
    warm = min(args.warmup, len(img_paths))
    for i in range(warm):
        img = Image.open(img_paths[i]).convert("RGB")
        _ = mtcnn(img)
    _maybe_sync_cuda()

    latencies = []
    # Reset cuda max memory stats for clean measurement
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    rss_before = rss_mb()
    t0_all = time.perf_counter()

    for r in range(args.repeats):
        t0 = time.perf_counter()
        for pth in img_paths:
            img = Image.open(pth).convert("RGB")
            _ = mtcnn(img)
        _maybe_sync_cuda()
        t1 = time.perf_counter()
        latencies.append((t1 - t0) / max(1, len(img_paths)))

    t1_all = time.perf_counter()
    rss_after = rss_mb()
    cuda_peak = cuda_max_mem_mb()

    return {
        "task": "detect",
        "num_images": len(img_paths),
        "repeats": args.repeats,
        "latency_ms_per_image_mean": 1000.0 * sum(latencies) / len(latencies),
        "latency_ms_per_image_std": 1000.0 * (math.sqrt(sum((x - sum(latencies)/len(latencies))**2 for x in latencies)/len(latencies)) if len(latencies)>1 else 0.0),
        "wall_time_s_total": (t1_all - t0_all),
        "memory_rss_mb_before": rss_before,
        "memory_rss_mb_after": rss_after,
        "cuda_peak_mem_mb": cuda_peak,
    }

def bench_train_eval(args) -> Dict[str, object]:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Subset
    from torchvision import datasets, transforms, models

    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else ("cpu" if args.device=="auto" else args.device))

    split = load_split(args.split_json)
    if split and ("train" in split and "test" in split):
        train_files = set(split["train"])
        test_files = set(split["test"])
    else:
        train_files, test_files = set(), set()

    tfm = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
    ])

    ds = datasets.ImageFolder(args.data_root, transform=tfm)

    if split and (train_files or test_files):
        # Map relpath list to indices
        # datasets.ImageFolder stores samples as (path, class_idx)
        root = Path(args.data_root).resolve()
        train_idx, test_idx = [], []
        for i, (pth, _) in enumerate(ds.samples):
            rel = str(Path(pth).resolve().relative_to(root)).replace("\\", "/")
            if rel in train_files:
                train_idx.append(i)
            elif rel in test_files:
                test_idx.append(i)
        if not train_idx or not test_idx:
            raise RuntimeError("split_json provided but did not match any files. Check data_root and split paths.")
        train_ds = Subset(ds, train_idx)
        test_ds = Subset(ds, test_idx)
    else:
        # Fallback: random split
        n = len(ds)
        n_train = int(0.8 * n)
        n_test = n - n_train
        gen = torch.Generator().manual_seed(args.seed)
        train_ds, test_ds = torch.utils.data.random_split(ds, [n_train, n_test], generator=gen)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=(device.type=="cuda"))
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=(device.type=="cuda"))

    model = models.resnet18(weights="DEFAULT" if hasattr(models, "ResNet18_Weights") else None)
    model.fc = nn.Linear(model.fc.in_features, len(ds.classes))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    rss_before = rss_mb()
    t0_all = time.perf_counter()

    # Train
    model.train()
    train_lat = []
    for epoch in range(args.epochs):
        for images, labels in train_loader:
            t0 = time.perf_counter()
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            _maybe_sync_cuda()
            t1 = time.perf_counter()
            train_lat.append(t1 - t0)

    # Eval
    model.eval()
    correct, total = 0, 0
    eval_lat = []
    with torch.no_grad():
        for images, labels in test_loader:
            t0 = time.perf_counter()
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            _maybe_sync_cuda()
            t1 = time.perf_counter()
            eval_lat.append(t1 - t0)

    t1_all = time.perf_counter()
    rss_after = rss_mb()
    cuda_peak = cuda_max_mem_mb()

    acc = (100.0 * correct / total) if total > 0 else 0.0

    return {
        "task": "train_eval",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "num_train_samples": len(train_ds),
        "num_test_samples": len(test_ds),
        "train_latency_ms_per_batch_mean": 1000.0 * (sum(train_lat)/len(train_lat) if train_lat else 0.0),
        "eval_latency_ms_per_batch_mean": 1000.0 * (sum(eval_lat)/len(eval_lat) if eval_lat else 0.0),
        "accuracy_top1_percent": acc,
        "wall_time_s_total": (t1_all - t0_all),
        "memory_rss_mb_before": rss_before,
        "memory_rss_mb_after": rss_after,
        "cuda_peak_mem_mb": cuda_peak,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["detect", "train_eval"], required=True)
    ap.add_argument("--data_root", required=True, help="ImageFolder root (class subfolders). For CelebA subset, use data/celebA_subset/images")
    ap.add_argument("--split_json", default=None, help="Optional fixed split json (from prepare_subset.py).")
    ap.add_argument("--device", default="auto", help="auto|cpu|cuda|mps")
    ap.add_argument("--out", required=True, help="Output JSON path")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--deterministic", action="store_true")

    # detect params
    ap.add_argument("--num_images", type=int, default=500)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--mtcnn_image_size", type=int, default=160)
    ap.add_argument("--mtcnn_margin", type=int, default=20)

    # train/eval params
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--img_size", type=int, default=160)
    ap.add_argument("--lr", type=float, default=3e-4)

    args = ap.parse_args()
    set_repro(args.seed, args.deterministic)

    started = time.strftime("%Y-%m-%d %H:%M:%S")
    meta = {
        "started_at": started,
        "argv": sys.argv,
        "cwd": os.getcwd(),
        "git": get_git_info(),
        "hw_sw": get_hw_sw_info(),
        "args": vars(args),
    }

    if args.task == "detect":
        metrics = bench_detect(args)
    else:
        metrics = bench_train_eval(args)

    payload = {**meta, "metrics": metrics}

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] Wrote: {outp}")

if __name__ == "__main__":
    main()
