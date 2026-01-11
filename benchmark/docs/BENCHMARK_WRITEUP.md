# Benchmark writeup template (paste into your fork README)

## What changed (Before vs After)
- **Before** (`<git ref>`): <describe baseline behavior / configuration>
- **After** (`<git ref>`): <describe your improvements>

## How to reproduce
1. Prepare fixed subset:
   - Command:
     ```bash
     python scripts/prepare_subset.py --celeba_images <...> --celeba_identity <...> --out_dir data/celebA_subset --num_ids 50 --imgs_per_id 20 --seed 1337
     ```
   - Output: `data/celebA_subset/split.json` (fixed train/test lists)

2. Export environment:
   ```bash
   python scripts/export_env.py --out_dir results
   ```

3. Run benchmark:
   ```bash
   python benchmarks/run_benchmark.py --task train_eval --data_root data/celebA_subset/images --split_json data/celebA_subset/split.json --device auto --epochs 1 --batch_size 32 --num_workers 4 --out results/<ref>_train_eval.json
   ```

4. Compare:
   ```bash
   python benchmarks/compare_results.py --before results/<before>.json --after results/<after>.json --out results/compare.md
   ```

## Results summary
Paste the generated markdown table from `results/compare.md` here.

## Notes / limitations
- Accuracy depends on chosen subset and epoch count; keep subset fixed for fair comparison.
- If running on different hardware, report specs and do not mix results.
