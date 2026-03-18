# MOUSEDataPipeline Upgrade Plan

## Goals

1. Replace direct `logbook2mouse` usage with `mouse_logbook`.
2. Move metadata writing in `src/processstep_metadata_update.py` to the newer writer path, including missing fields such as `/entry1/sample/sampleowner`.
3. Add a minimal but effective `pytest` test suite around the migration.
4. Improve package structure, environment reproducibility, and runtime robustness without rewriting the whole pipeline at once.

## Progress Update

The following migration steps are now implemented in this repository:

- `src/processstep_metadata_update.py` now uses the newer `mouse_logbook` writer path through the `mouse-logbook write-nexus-metadata ...` CLI.
- `src/directory_processor.py` no longer constructs a logbook reader eagerly at startup.
- Logbook-reader access is now isolated in `src/logbook_support.py`.
- Process-step modules no longer import `logbook2mouse` directly; they use a local `LogbookReaderLike | None` type instead.
- `src/directory_processor_multibatch_nostack.sh` now exits on the first failed batch instead of printing a false success message.
- `requirements.txt` now includes `mouse_logbook` and `pint`.
- `src/logbook_support.py` now uses `mouse_logbook` only.
- `logbook2mouse` has been removed from `requirements.txt`.
- `src/processstep_metadata_update.py` now also writes `/entry1/sample/sampleowner` as a compatibility alias from the CLI-written `/entry1/sample/owner`.
- `src/processstep_translator_step_2.py` continues to shell out to `python3 -m HDF5Translator`.
- `src/directory_processor.py` now emits lightweight per-step timing logs when profiling is enabled.
- `src/directory_processor.py` now also creates per-repetition log files alongside generated `MOUSE_*.nxs` outputs when `log_per_datafile` is enabled.
- `src/directory_processor.py` now exposes a clearer CLI with built-in step presets and discovery flags for steps and presets.
- `src/directory_processor.py` and `DefaultsCarrier` now support a `parallel_workers` override to tune repetition-level thread-pool size for disk-heavy batch runs.
- The shell wrappers in `src/` now call the installed `mouse-directory-processor` command instead of invoking `src/directory_processor.py` directly.
- `processstep_translator_step_1.py`, `processstep_translator_step_2.py`, and `processstep_metadata_update.py` now skip expensive reruns when their outputs are already up to date.
- `pytest.ini`, `requirements-dev.txt`, and a first `tests/` suite have been added.
- `pyproject.toml` now provides package metadata, dependencies, and a `mouse-directory-processor` console entry point.
- `requirements-dev.txt` now installs the project in editable mode through `-e .[dev]`.
- `MOUSE_settings.yaml` now documents the `profile_steps` toggle.
- `MOUSE_settings.yaml` and `README.md` now document `log_per_datafile`.
- `MOUSE_settings.yaml`, the CLI, and `README.md` now document `parallel_workers` for first-run performance tuning.
- `README.md` now documents the preset-based CLI workflow and the discovery commands for steps and presets.
- `README.md` now also makes the installed `mouse-directory-processor` console command the primary user-facing entry point.
- `tests/` now also covers realistic Excel fixtures from `mouse_logbook/tests/data`.
- `DirectoryProcessor` and `YMD_class` now use explicit exceptions for core path/argument validation instead of runtime `assert` statements.
- The `repetition=0` orchestration path now works correctly instead of being rejected by truthiness checks.
- `utilities.py`, `processstep_thickness_from_absorption.py`, and `processstep_make_beam_mask.py` now use explicit validation exceptions instead of runtime `assert` statements in their core guard rails.
- `post_translation_operation_hdf5_stacker.py` and `processstep_calc_beam_flux_and_transmissions.py` now also use explicit validation exceptions instead of runtime `assert` statements in active runtime paths.
- `processstep_determine_beam_center.py`, `processstep_thickness_from_absorption.py`, and `processstep_stacker.py` no longer write progress/debug information to stdout; they now use logger output instead.
- Active `skimage` deprecation warnings have been addressed by updating beam-feature cleanup and weighted-centroid access to the current API.
- Beam-analysis compatibility shims now support both older and newer `scikit-image` APIs for morphology cleanup and weighted-centroid access.
- Active step execution now uses per-repetition child loggers, and the standalone `post_translation_operation_hdf5_stacker.py` script now also uses an explicit module/logger path instead of direct root-logger calls.
- `ruff`, `pre-commit`, and a repo-level `.pre-commit-config.yaml` have been added for incremental linting and formatting on touched files.
- `pyproject.toml` now exposes the linting tools both as a `pip` extra and as a `uv` dependency group.
- `periodictable` and `xraydb` are now declared directly as runtime dependencies because the `mouse_logbook` metadata writer requires them during chemistry and X-ray validation.
- The removed obsolete modules are no longer referenced from `pyproject.toml`.
- Fresh-environment validation has now been exercised successfully on Python 3.14 in both the runtime and `.[dev]` environments.
- The current local test suite passes: 43 tests.

## Current State Observations

- `src/directory_processor.py` now passes `None` to process steps unless a step explicitly opts in to reader construction.
- Most process steps only accept the reader in their signature, and none of the active runtime steps currently dereference it directly.
- `src/processstep_metadata_update.py` remains a thin CLI wrapper around `mouse_logbook`.
- Expensive subprocess-heavy steps now avoid rerunning when their outputs are newer than their inputs and configuration sources.
- Batch-level parallelism can now be capped explicitly when storage contention makes the default thread count too aggressive.
- The entry points still default to `python`, but the batch script now allows overriding the interpreter through `PYTHON_BIN`.
- The repository now has `pytest` scaffolding, an editable-install path via `pyproject.toml`, updated README usage examples, and a small integration-oriented test suite.

## Findings About `mouse_logbook`

- The local `.venv` currently contains `mouse-logbook==0.1.2`.
- The CLI now exposes `write-nexus-metadata`, and that command works locally on the configured dataset.
- The installed package provides a legacy-compatible reader facade.
- The upstream reader now normalizes volume fractions, which removed one earlier compatibility blocker.
- With the current `MOUSE_settings.yaml` data, `mouse_logbook.Logbook2MouseReader` now loads successfully against the configured corpus.

## Migration Strategy

The repository has already completed the dependency migration. The remaining work is now hardening: better packaging, broader tests, and using the new profiling logs to identify actual bottlenecks before optimizing.

## Phase 1: Packaging And Environment

### Tasks

1. Add a `pyproject.toml` for reproducible installs.
2. Keep runtime and development dependencies split cleanly.
3. Standardize the execution environment used by scripts and docs.
   - Runtime and `.[dev]` installs have now been validated on Python 3.14
   - The package CLI is in place, while shell wrappers remain convenience entry points
4. Document `PYTHON_BIN` usage for the shell wrapper.
5. Document `profile_steps` in `MOUSE_settings.yaml` and user-facing docs.

### Exit Criteria

- A fresh environment can install the package and run the CLI entry point reproducibly.
- The same environment can run tests.

### Status

Mostly complete.

- `pyproject.toml`, `requirements-dev.txt`, and `pytest.ini` exist.
- Editable installation works with `./.venv/bin/python -m pip install -e '.[dev]' --no-deps`.
- `MOUSE_settings.yaml` now documents `profile_steps`.
- `README.md` now documents editable installation, the `mouse-directory-processor` entry point, and `PYTHON_BIN` for the shell wrapper.
- `README.md` now also documents `pre-commit` setup and the incremental lint/format workflow.
- `README.md` now documents how to refresh stale dev-tool environments for both `pip` and `uv`.
- `pytest` tests now cover:
  - metadata updates via the `mouse-logbook` CLI writer path
  - metadata CLI failure propagation
  - `DirectoryProcessor` startup without eager reader construction
  - explicit `DirectoryProcessor` validation errors for missing coordinates and missing paths
  - path parsing validation in `YMD_class.extract_metadata_from_path`
  - utility validation failures in `reduce_extra_image_dimensions` and `label_main_feature`
  - reader initialization against a small Excel fixture
  - reader initialization against realistic `mouse_logbook` example sheets
  - metadata export against realistic `mouse_logbook` example sheets and a synthetic `.nxs` file
  - translator step 2 subprocess dispatch
  - background-file metadata writing with a synthetic `.nxs` file
  - cleanup of intermediate step-1 output files
  - explicit validation failures in `processstep_thickness_from_absorption` and `processstep_make_beam_mask`
  - stacker configuration and input validation plus a synthetic stacker smoke test
  - explicit validation failure for a mismatched beam-coverage mask in `processstep_calc_beam_flux_and_transmissions`
  - beam-center smoke testing with a synthetic detector image
  - quiet execution for stacker, thickness, and beam-center steps without stray stdout output
  - per-repetition log-file creation and opt-out behavior in `DirectoryProcessor`
  - explicit logger-path usage in the standalone stacker script
  - CLI step-presets, step discovery, and preset discovery
  - rerun-skipping behavior for translator step 1, translator step 2, and metadata update
  - configurable parallel worker limits in both config and CLI
  - compatibility behavior across older and newer `scikit-image` APIs used in beam analysis
- The remaining packaging gap is mainly representative clean-room usage validation beyond installation itself.

### Linting Status

- `ruff` is configured in `pyproject.toml` with an intentionally small initial rule set:
  - syntax/correctness-focused linting
  - unused imports/names
  - import sorting
- `ruff format` is configured as the initial style enforcer.
- The current lint configuration is meant for incremental adoption on changed files rather than an immediate full-repo cleanup.

## Phase 2: Expand Test Coverage

The first `pytest` scaffolding is in place. The next step is to cover the main failure paths and operational guarantees.

### Priority tests

1. Metadata writer failure propagation.
   - Simulate CLI failure and assert that the step raises a useful error.
2. Parallel execution behavior.
   - Assert that step exceptions from the thread pool propagate to the batch runner.
3. Profiling behavior.
   - Assert that `PROFILE ...` messages are emitted when enabled and absent when disabled.
4. Representative step smoke tests.
   - Add one or two small-file tests for high-value steps that do not need the full production dataset.
5. Reader-fixture coverage.
   - Keep the Excel fixtures small, explicit, and isolated from the real corpus.
   - Example logbook/project sheets from `mouse_logbook` tests are now folded into regression coverage.

### Exit Criteria

- `pytest` covers the current migration path and at least one important failure path.
- The core orchestration behavior can be exercised without the full production dataset.

### Status

- In progress.
- Current tests cover the happy path for metadata writing, metadata CLI failure propagation, lazy reader construction, profiling enable/disable behavior, parallel-error propagation, reader initialization on both synthetic and realistic Excel fixtures, metadata export using realistic Excel fixtures plus a synthetic `.nxs` file, translator step 2 subprocess dispatch, and small-file smoke/validation paths in the stacker, cleanup, background-file, beam-center, beam-mask, beam-flux, and thickness-related steps.
- Current tests also cover rerun-skipping behavior for translator step 1, translator step 2, and metadata update.
- The biggest gaps are broader orchestration coverage and smoke tests for more numerically heavy processing steps.

## Phase 3: Runtime Robustness Cleanup

These are maintainability improvements that should now be done against the migrated codebase rather than against the old dependency boundary.

### Recommended cleanup items

1. Continue replacing runtime `assert` statements with explicit exceptions and user-facing validation errors in any remaining active runtime modules.
2. Centralize logging setup instead of relying on implicit logger reuse.
3. Remove any remaining `print(...)` debugging from active runtime paths and keep command/progress output in the logger only.
4. Consider a small step registry instead of raw `importlib.import_module(...)` strings once the pipeline behavior is better covered by tests.
5. Keep subprocess-based step wrappers simple and explicit unless a direct-library path clearly improves both performance and maintainability.
6. Gradually widen the `ruff` rule set once the existing touched-file workflow is stable.

## Phase 4: Performance Investigation

The pipeline has now been profiled enough to identify the first hot spots. The in-process replacements for metadata export and translator step 2 were tried and backed out because they did not improve real performance enough to justify the added complexity.

### Tasks

1. Run a representative batch with `profile_steps: true`.
2. Collect step-level timings and identify the slowest stages.
   - Current measurements point to translator step 2 at roughly 10 s per repetition and metadata update at roughly 1.5 s per repetition.
   - Cheap rerun-avoidance is now in place, so the remaining hot path is mostly first-run work rather than repeated development reruns.
   - `parallel_workers` is now available as a low-effort tuning knob when first-run throughput is limited by disk contention rather than CPU.
3. Separate likely costs:
   - HDF5 read/write I/O
   - translator template-copy overhead and HDF5 I/O
   - metadata CLI startup and workbook parsing cost
   - thread-pool contention on shared disk access
   - expensive numerical steps such as beam-center detection
4. Only then decide whether to optimize concurrency, reduce file opens, cache metadata, or restructure steps.

### Exit Criteria

- Profiling data exists for a representative run.
- At least one concrete performance hypothesis is confirmed with measurements.

## Definition Of Done For This Migration

The migration can be considered complete when all of the following are true:

- the runtime no longer imports `logbook2mouse`
- the configured project/logbook corpus is readable through `mouse_logbook`
- metadata updates are written through the new writer path or a stable wrapper around it
- `/entry1/sample/sampleowner` is present in updated measurement files
- `pytest` covers the reader adapter and metadata update behavior
- the documented CLI path works from a fresh environment

## Immediate Next Step

The highest-value next implementation step is:

1. expand `pytest` coverage to selected real processing steps with small-file smoke tests
2. keep using `pre-commit` on touched files and gradually widen `ruff` coverage once the touched-file workflow stays stable
3. collect and review real profiling output from representative first-run batches
4. consider one or two small usability improvements for the CLI output, such as a `--show-config` view or friendlier preset descriptions
5. if first-run performance still matters after tuning `parallel_workers`, move the deeper optimization work into `HDF5Translator` or more targeted I/O reductions

The dependency swap is complete in code. The remaining work is now testing, packaging, and cleanup.
