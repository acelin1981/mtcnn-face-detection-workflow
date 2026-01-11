#!/usr/bin/env python3
"""
compare_results.py
Compare two benchmark JSON files and output a Markdown table (memory / latency / accuracy).
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Dict, Any

from tabulate import tabulate

def load(p: str) -> Dict[str, Any]:
    return json.loads(Path(p).read_text(encoding="utf-8"))

def pct_change(before: float, after: float) -> str:
    if before == 0:
        return "n/a"
    return f"{(after-before)/before*100:+.2f}%"

def fmt(v):
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", required=True)
    ap.add_argument("--after", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    b = load(args.before)
    a = load(args.after)
    bm = b["metrics"]
    am = a["metrics"]

    rows = []

    def add(metric_key, label, unit=""):
        bv = bm.get(metric_key, None)
        av = am.get(metric_key, None)
        # change for numeric
        ch = "n/a"
        if isinstance(bv, (int, float)) and isinstance(av, (int, float)):
            ch = pct_change(float(bv), float(av))
        rows.append([label, f"{fmt(bv)} {unit}".strip(), f"{fmt(av)} {unit}".strip(), ch])

    # Common
    add("memory_rss_mb_after", "RSS memory (after)", "MB")
    add("cuda_peak_mem_mb", "CUDA peak memory", "MB")
    add("wall_time_s_total", "Total wall time", "s")

    # Detect
    if bm.get("task") == "detect" and am.get("task") == "detect":
        add("latency_ms_per_image_mean", "Latency per image (mean)", "ms")
        add("latency_ms_per_image_std", "Latency per image (std)", "ms")

    # Train/Eval
    if bm.get("task") == "train_eval" and am.get("task") == "train_eval":
        add("train_latency_ms_per_batch_mean", "Train latency per batch (mean)", "ms")
        add("eval_latency_ms_per_batch_mean", "Eval latency per batch (mean)", "ms")
        add("accuracy_top1_percent", "Top-1 accuracy", "%")

    headers = ["Metric", "Before", "After", "Δ%"]
    md = tabulate(rows, headers=headers, tablefmt="github")

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(md + "\n", encoding="utf-8")
    print(f"[OK] Wrote: {outp}")

if __name__ == "__main__":
    main()
