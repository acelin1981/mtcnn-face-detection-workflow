MTCNN Face Detection Workflow
-----------------------------
This repository provides a **public reference implementation and engineering-oriented analysis**
of the MTCNN (Multi-task Cascaded Convolutional Networks) face detection and alignment workflow.

It includes:
- A PyTorch-based MTCNN pipeline reference
- Step-by-step documentation of the detection stages (P-Net / R-Net / O-Net)
- A **reproducible benchmarking framework** to evaluate performance impact of implementation changes

Technical article (design rationale and background):
- *Understanding MTCNN: A Practical Guide to Multi-Task Cascaded Convolutional Networks*  
  https://medium.com/@ace.lin0121/understanding-mtcnn-a-practical-guide-to-multi-task-cascaded-convolutional-networks-for-face-efc6cc2a433f

---

Purpose
-----------------------------
Beyond providing a functional MTCNN reference, this project focuses on **engineering practices**
that are often under-documented in academic examples, including:

- Performance measurement methodology
- Before/after comparison of implementation changes
- Reproducibility of experimental results
- Clear separation between algorithmic behavior and system-level cost

This makes the repository suitable not only for learning MTCNN,
but also for **practical system optimization and evaluation**.


MTCNN Pipeline Overview
-----------------------------
The workflow follows the standard three-stage cascade:

1. **P-Net (Proposal Network)**  
   Generates candidate face bounding boxes across image scales.

2. **R-Net (Refine Network)**  
   Filters false positives and refines bounding box locations.

3. **O-Net (Output Network)**  
   Produces final bounding boxes and facial landmarks.

Each stage progressively trades computation cost for accuracy,
which makes performance evaluation and batching behavior especially important in real systems.


Reproducible Benchmark Framework
-----------------------------

To objectively evaluate performance impact, this repository includes a lightweight
benchmarking framework under the `benchmark/` directory.

What is measured
----------------------------
- **Latency**
  - Mean evaluation latency per batch
- **Memory**
  - Process RSS memory before and after evaluation
- **Accuracy**
  - Top-1 classification accuracy on a fixed evaluation subset
- **Reproducibility**
  - Fixed dataset split
  - Exported environment metadata
  - JSON-based raw results

All benchmarks are designed to be **repeatable and auditable**.

Benchmark Methodology
-----------------------------
- Dataset is organized in ImageFolder format
- A fixed train/test split is generated once and reused
- Baseline and optimized implementations are evaluated on the **same input set**
- Results are saved as JSON and compared using an automated script

Relevant scripts:
- `benchmark/scripts/prepare_subset.py`
- `benchmark/benchmarks/run_benchmark.py`
- `benchmark/benchmarks/compare_results.py`


Benchmark Results (Before vs After)
-----------------------------
The following table summarizes the evaluation-only benchmark results
using the same dataset split and batch configuration.

| Metric | Before | After | Δ% |
|---|---:|---:|---:|
| RSS memory (after) [MB] | 855.53 | 855.53 | +0.00% |
| Evaluation latency per batch (ms) | *see JSON* | *see JSON* | improved |
| Top-1 accuracy (%) | 100.0 | 100.0 | +0.00% |

Raw result files:
- `benchmark/result/baseline_eval_only.json`
- `benchmark/result/after_eval_only.json`
- `benchmark/result/compare_eval_only.md`

These artifacts are intentionally kept in the repository
to allow independent verification of the results.

Notes on Interpretation
-----------------------------
- Accuracy is intentionally held constant to isolate performance effects
- Memory usage remains stable, indicating no hidden buffering or leaks
- Latency differences are attributable to implementation-level optimizations,
  not algorithmic changes

This approach reflects common engineering practice in production ML systems,
where **performance predictability and reproducibility** are as important as raw accuracy.

Summary
-----------------------------
This repository demonstrates how a well-known algorithm (MTCNN)
can be treated as a **system-level engineering problem** rather than a black-box model.

By combining:
- Clear pipeline documentation
- Reproducible benchmarking
- Transparent before/after analysis

the project aims to serve as a practical reference for engineers
working on real-world face detection and ML inference pipelines.




Technical Article:  
------------------
Understanding MTCNN: A Practical Guide to Multi-Task Cascaded Convolutional Networks  
https://medium.com/@ace.lin0121/understanding-mtcnn-a-practical-guide-to-multi-task-cascaded-convolutional-networks-for-face-efc6cc2a433f


MTCNN Face Detection Workflow
=============================

This repository provides a public reference implementation and documentation of the MTCNN
(Multi-task Cascaded Convolutional Networks) face detection and alignment workflow.

It includes a PyTorch version of the MTCNN pipeline and supporting notes explaining the
end-to-end detection stages.

Purpose
-------

The goal of this repository is to make the MTCNN workflow and the related technical notes
publicly accessible for reference and verification. All materials in this repository are
intended for learning, reproducibility, and documentation.

Repository Contents
-------------------

MTCNN.py  
A PyTorch-based implementation of the MTCNN detection pipeline, including:
- Proposal generation using P-Net
- Refinement stage using R-Net
- Final stage using O-Net for face classification and landmark prediction
- Basic bounding box handling and workflow structure

Theory and Explanation of MTCNN.docx  
A technical document summarizing:
- The overall architecture of MTCNN
- How P-Net, R-Net and O-Net work together
- Concepts such as bounding box regression and non-maximum suppression
- Landmark refinement and multi-stage inference flow

README.md  
This file.

Current Status
--------------

This repository currently provides:
- Documentation describing how the MTCNN stages operate
- A reference PyTorch script showing the high-level workflow

This repository does not include datasets or full training scripts.

Planned Additions
-----------------

Future updates may include:
- Additional notes or documentation
- Example input and output illustrations

Versioning
----------

v1.0.0 – Initial public release (documentation and reference code)

Contact
-------

For questions or clarifications, please open an Issue on GitHub.
