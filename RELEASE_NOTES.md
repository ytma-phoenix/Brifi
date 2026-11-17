# Public core release — 2026-11-17

This release packages the core BRIFi method for WSDM 2027 with explicit
command-line entry points, dependencies, citation metadata, and the MIT License.
It does not contain the complete original experimental artifact.

## Core implementation corrections

- The public `brifi` path now uses `brifi.BudgetedFilter`, which implements
  Algorithm 1: raw-score top-c rejection during initialization, one-sided
  median/MAD calibration, a hard rejection cap, and retained-only history.
  The earlier generic prefilter ranked zero standardized scores before history
  existed and clipped large scores before ranking; neither behavior is used by
  the core release.
- The scorer is frozen, including BatchNorm statistics, and auxiliary/downstream
  inputs share RGB, unit-range, tensor-bilinear preprocessing.
- Downstream BRIFi execution requires a checkpoint. Detector training accepts an
  explicit independent auxiliary training directory. The earlier same-task
  fallback and implicit auxiliary benchmark additions are not used.
- FL defaults now match the manuscript's stated client count, participation,
  local contamination rate, optimizer, and round count. Histories persist per
  client; only sampled clients upload. A zero contaminated-client ratio means
  zero contaminated clients, and poison rates are not globally renormalized.
- Filtering metrics count actual sample visits, including multiple local epochs.
  The effective poison fraction is computed per pass rather than subtracting
  repeated rejections from a fixed dataset across epochs.
- Evaluation covers the six named trigger families, preserves original test
  labels, and excludes target-class examples from ASR. Evaluation seeds/restores
  Python, NumPy, and torch CPU random states. The initial classifier seed is
  reset after scorer construction/diagnostics for consistent paired runs.

These corrections can change outputs relative to the supplied earlier source.
No claim is made that the revised driver regenerates the published numerical
values without the original weights, corpus, configuration records, and complete
experiment implementations.

## Validation performed

Environment: Python 3.12, torch 2.6.0, torchvision 0.21.0, NumPy 1.26.3 on CPU.

- All ten Python files parsed successfully and the project modules imported.
- The CLI help and configuration-only dry run completed.
- A BRIFi run without a checkpoint failed before any data access, as intended.
- The synthetic smoke check passed.
- Eleven functional tests passed: bootstrap ordering/cap, retained-only history,
  unclipped ranking, bounded/client-local history, one-sided rejection,
  frozen parameters/BatchNorm, shared canonicalization, checkpoint load/refusal,
  reproducible ASR draws with RNG restoration, paired classifier initialization,
  and partial participation/zero contamination/sample-visit accounting.

The FL accounting tests use stubbed training/evaluation on synthetic objects.
They do not execute optimizer steps. No benchmark dataset was downloaded, no
model was trained, and no full-scale or multi-seed experiment was run while
preparing this release.

## Reproduction boundaries

Not included: the original GTSRB/EuroSAT/procedural corpus and base-image splits,
the original detector checkpoint, seed-level outputs, the server-aggregation
baselines, full same-stage baseline implementations, and the complete LOFO,
ablation, advanced-attack, and adaptive-attack experiment scripts.

The supplied DBA family is a local multi-patch sampler, not a full implementation
of distributed trigger-component assignment. The LGA, SoDa, and NBA helpers are
prototypes and are not wired into the CLI's FL runner. The legacy frequency,
warp, unified-channel, and split-learning configuration fields are retained for
source compatibility; the public CLI exposes only BRIFi and undefended FedAvg.
