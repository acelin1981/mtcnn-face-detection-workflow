# Reproducible Benchmark Pack (memory / latency / accuracy)

This folder is designed to be dropped into your forked repo to produce **reproducible** benchmark artifacts:
- **memory** (CPU RSS and optional CUDA max memory)
- **latency** (per-image detect latency; and per-batch train/eval latency)
- **accuracy** (top-1 on a fixed evaluation split)
- **full reproducibility** (seed, environment export, deterministic flags)

It also supports **before/after** comparisons between two Git commits/tags.

> Notes
> - You must provide the dataset path locally (CelebA or your own face dataset).
> - This pack does **not** download datasets automatically; it only consumes local data.
> - The benchmark scripts are written to be clear and auditable for visa evidence.

---

## 0) Folder layout

```
benchmarks/
  run_benchmark.py          # single-run benchmark -> JSON
  compare_results.py        # compare two JSON files -> Markdown table
  bench_git.py              # run at two commits/tags and compare automatically
scripts/
  prepare_subset.py         # build a fixed subset split for reproducible accuracy/latency tests
  export_env.py             # export environment info (pip freeze, torch/cuda) to results/
docs/
  BENCHMARK_WRITEUP.md      # copy/paste evidence writeup template (for Global Talent)
results/
  (generated files)
```

---

## 1) Install

Create a clean venv, then install your repo deps + benchmark deps:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate

pip install -U pip
pip install -r requirements.txt  # your repo requirements
pip install psutil tabulate
```

If using CUDA, install the correct PyTorch build for your system.

---

## 2) Prepare a reproducible subset split

You need a stable dataset directory. Two options:

### Option A: CelebA aligned images (recommended)
Assume you already have:
- `img_align_celeba/img_align_celeba/` (images)
- `identity_CelebA.txt` (filename -> identity id)

Then build a fixed subset:

```bash
python scripts/prepare_subset.py   --celeba_images img_align_celeba/img_align_celeba   --celeba_identity identity_CelebA.txt   --out_dir data/celebA_subset   --num_ids 50   --imgs_per_id 20   --seed 1337
```

This produces:
- `data/celebA_subset/images/<id>/*.jpg`
- `data/celebA_subset/split.json` (train/test split + fixed file lists)

### Option B: Your own ImageFolder dataset
Provide an ImageFolder-like directory:
`<root>/<class_name>/*.jpg`
and skip prepare_subset.

---

## 3) Run a single benchmark (creates JSON)

### 3.1 Face detection latency (MTCNN)
Run detect-only latency and memory:

```bash
python benchmarks/run_benchmark.py   --task detect   --data_root data/celebA_subset/images   --split_json data/celebA_subset/split.json   --device auto   --num_images 500   --warmup 20   --repeats 3   --out results/baseline_detect.json
```

### 3.2 Train+eval (ResNet18 classifier)
```bash
python benchmarks/run_benchmark.py   --task train_eval   --data_root data/celebA_subset/images   --split_json data/celebA_subset/split.json   --device auto   --epochs 1   --batch_size 32   --num_workers 4   --out results/baseline_train_eval.json
```

---

## 4) Compare "before vs after" (Markdown table)

After you have two JSON results (baseline vs optimized):

```bash
python benchmarks/compare_results.py   --before results/baseline_train_eval.json   --after  results/after_train_eval.json   --out    results/compare_train_eval.md
```

The markdown output is ready to paste into GitHub README.

---

## 5) One-command: benchmark two commits/tags automatically

Run benchmarks at two refs and auto-compare:

```bash
python benchmarks/bench_git.py   --before_ref baseline_v1   --after_ref  opt_v2   --workdir    /tmp/benchwork   --task train_eval   --data_root  /ABS/PATH/data/celebA_subset/images   --split_json /ABS/PATH/data/celebA_subset/split.json   --device auto
```

It produces:
- `results/before_<ref>_*.json`
- `results/after_<ref>_*.json`
- `results/compare_<task>_<before>_vs_<after>.md`

---

## 6) Reproducibility checklist (what reviewers like)

- Fixed split file list (`split.json`)
- Fixed random seed (`--seed`)
- Export environment (`scripts/export_env.py`)
- Report hardware (`CPU/GPU/RAM`) and OS
- Include raw JSON results in repo under `results/`

---

## 7) Suggested commit structure (for evidence)

- Commit A: `baseline_v1` (original)
- Commit B: `opt_v2` (your improvement)
- Add `results/compare_*.md` and `results/*.json`
- Add a short summary paragraph in README

See `docs/BENCHMARK_WRITEUP.md` for a copy/paste template.
