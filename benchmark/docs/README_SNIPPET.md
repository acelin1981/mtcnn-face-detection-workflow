## Benchmark (memory / latency / accuracy)

This fork includes a reproducible benchmark pack under `benchmarks/` to compare **before vs after** changes.

- Prepare fixed subset: `python scripts/prepare_subset.py ...`
- Run benchmark: `python benchmarks/run_benchmark.py ...`
- Compare: `python benchmarks/compare_results.py ...`

Benchmark results (raw JSON + generated tables) are stored under `results/`.
