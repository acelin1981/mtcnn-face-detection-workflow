#!/usr/bin/env python3
"""
bench_git.py
Run the benchmark at two git refs (commits/tags/branches) and generate a compare markdown.

This is the easiest way to generate "before/after" evidence for a fork:
- baseline ref
- optimized ref
Then you commit the produced JSON + compare md into your repo.
"""
from __future__ import annotations
import argparse, os, shutil, subprocess, sys
from pathlib import Path

def run(cmd, cwd=None):
    print("[CMD]", " ".join(cmd))
    subprocess.check_call(cmd, cwd=cwd)

def capture(cmd, cwd=None) -> str:
    return subprocess.check_output(cmd, cwd=cwd, text=True).strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--before_ref", required=True, help="git ref for baseline (tag/commit/branch)")
    ap.add_argument("--after_ref", required=True, help="git ref for after (tag/commit/branch)")
    ap.add_argument("--workdir", required=True, help="temp working directory (will be created/overwritten)")
    ap.add_argument("--task", choices=["detect", "train_eval"], required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--split_json", default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--num_images", type=int, default=500)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    repo_root = capture(["git", "rev-parse", "--show-toplevel"])
    repo_root = Path(repo_root).resolve()

    workdir = Path(args.workdir).resolve()
    if workdir.exists():
        shutil.rmtree(workdir)
    shutil.copytree(repo_root, workdir)

    results_dir = repo_root / "results"
    results_dir.mkdir(exist_ok=True)

    def run_at(ref: str, out_json: Path):
        run(["git", "checkout", "-f", ref], cwd=workdir)
        cmd = [sys.executable, "benchmarks/run_benchmark.py",
               "--task", args.task,
               "--data_root", args.data_root,
               "--device", args.device,
               "--out", str(out_json)]
        if args.split_json:
            cmd += ["--split_json", args.split_json]
        if args.task == "train_eval":
            cmd += ["--epochs", str(args.epochs),
                    "--batch_size", str(args.batch_size),
                    "--num_workers", str(args.num_workers)]
        else:
            cmd += ["--num_images", str(args.num_images),
                    "--warmup", str(args.warmup),
                    "--repeats", str(args.repeats)]
        run(cmd, cwd=workdir)

    before_json = results_dir / f"before_{args.before_ref}_{args.task}.json"
    after_json  = results_dir / f"after_{args.after_ref}_{args.task}.json"
    compare_md  = results_dir / f"compare_{args.task}_{args.before_ref}_vs_{args.after_ref}.md"

    run_at(args.before_ref, before_json)
    run_at(args.after_ref, after_json)

    run([sys.executable, "benchmarks/compare_results.py",
         "--before", str(before_json),
         "--after", str(after_json),
         "--out", str(compare_md)], cwd=workdir)

    # Copy outputs back to repo_root/results
    # (benchmarks write directly to repo_root/results; already there)

    print("[OK] Generated:")
    print(" -", before_json)
    print(" -", after_json)
    print(" -", compare_md)
    print("Now commit these under results/ as evidence.")

if __name__ == "__main__":
    main()
