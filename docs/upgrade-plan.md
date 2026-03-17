# MOUSEDataPipeline Upgrade Plan

## Goals

1. Replace direct `logbook2mouse` usage with `mouse_logbook`.
2. Move metadata writing in `src/processstep_metadata_update.py` to the newer writer path, including missing fields such as `/entry1/sample/sampleowner`.
3. Add a minimal but effective `pytest` test suite around the migration.
4. Improve package structure, environment reproducibility, and runtime robustness without rewriting the whole pipeline at once.

## Progress Update

The following migration steps are now implemented in this repository:

- `src/processstep_metadata_update.py` now delegates metadata updates to `mouse-logbook write-nexus-metadata ...`.
- `src/directory_processor.py` no longer constructs a logbook reader eagerly at startup.
- Logbook-reader access is now isolated in `src/logbook_support.py`.
- Process-step modules no longer import `logbook2mouse` directly; they use a local `LogbookReaderLike | None` type instead.
- `src/directory_processor_multibatch_nostack.sh` now exits on the first failed batch instead of printing a false success message.
- `requirements.txt` now includes `mouse_logbook` and `pint`.
- `src/processstep_metadata_update.py` has been simplified to a pure `mouse-logbook write-nexus-metadata ...` wrapper.
- `src/logbook_support.py` now uses `mouse_logbook` only.
- `logbook2mouse` has been removed from `requirements.txt`.
- `src/processstep_metadata_update.py` now also writes `/entry1/sample/sampleowner` as a compatibility alias from the CLI-written `/entry1/sample/owner`.
- `src/directory_processor.py` now emits lightweight per-step timing logs when profiling is enabled.
- `pytest.ini`, `requirements-dev.txt`, and a first `tests/` suite have been added.
- The current local test suite passes: 4 tests.

What is still transitional:

- No `pyproject.toml` or package metadata has been added yet.

## Current State Observations

- `src/directory_processor.py` now passes `None` to process steps unless a step explicitly opts in to reader construction.
- Most process steps only accept the reader in their signature, and none of the active runtime steps currently dereference it directly.
- `src/processstep_metadata_update.py` is now a thin CLI wrapper around `mouse_logbook`.
- The entry points still default to `python`, but the batch script now allows overriding the interpreter through `PYTHON_BIN`.
- The repository now has `pytest` scaffolding and a small integration-oriented test suite, but still has no `pyproject.toml`.

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
   - Short term: call `./.venv/bin/python`
   - Long term: expose a package CLI or console script
4. Document `PYTHON_BIN` usage for the shell wrapper.
5. Document `profile_steps` in `MOUSE_settings.yaml` and user-facing docs.

### Exit Criteria

- A fresh environment can install the package and run the CLI entry point reproducibly.
- The same environment can run tests.

### Status

Partially complete.

- `requirements-dev.txt` and `pytest.ini` exist.
- `pytest` tests now cover:
  - metadata updates via the CLI writer
  - `DirectoryProcessor` startup without eager reader construction
  - reader initialization against a small Excel fixture
- Packaging metadata and a cleaner developer install path still need to be added.

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

### Exit Criteria

- `pytest` covers the current migration path and at least one important failure path.
- The core orchestration behavior can be exercised without the full production dataset.

### Status

- Started.
- Current tests cover the happy path for metadata writing, lazy reader construction, and reader initialization on a small fixture.
- The biggest gaps are failure paths, more orchestration coverage, and step-specific smoke tests.

## Phase 3: Runtime Robustness Cleanup

These are maintainability improvements that should now be done against the migrated codebase rather than against the old dependency boundary.

### Recommended cleanup items

1. Replace runtime `assert` statements with explicit exceptions and user-facing validation errors.
2. Centralize logging setup instead of relying on implicit logger reuse.
3. Remove any remaining `print(...)` debugging from runtime paths.
4. Consider a small step registry instead of raw `importlib.import_module(...)` strings once the pipeline behavior is better covered by tests.
5. Decide whether metadata-step subprocess execution should remain CLI-based long-term or move to a direct library call later.

## Phase 4: Performance Investigation

The pipeline appears slower than expected, but the likely bottleneck is still unverified. The new profiling output should be used before any optimization work.

### Tasks

1. Run a representative batch with `profile_steps: true`.
2. Collect step-level timings and identify the slowest stages.
3. Separate likely costs:
   - HDF5 read/write I/O
   - subprocess startup
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

1. add a `pyproject.toml` and split runtime vs development dependencies
2. expand `pytest` coverage to failure paths, parallel execution, and selected processing steps
3. tighten CLI/runtime documentation around `.venv` and `PYTHON_BIN`
4. collect and review real profiling output from representative batches

The dependency swap is complete in code. The remaining work is now testing, packaging, and cleanup.
