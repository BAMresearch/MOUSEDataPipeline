# MOUSEDataPipeline Upgrade Plan

## Goals

1. Replace direct `logbook2mouse` usage with `mouse_logbook`.
2. Move metadata writing in `src/processstep_metadata_update.py` to the newer writer path, including missing fields such as `/entry1/sample/sampleowner`.
3. Add a minimal but effective `pytest` test suite around the migration.
4. Improve package structure, environment reproducibility, and runtime robustness without rewriting the whole pipeline at once.

## Progress Update

The following migration steps are now implemented in this repository:

- `src/processstep_metadata_update.py` now prefers `mouse-logbook write-nexus-metadata ...` and only falls back to the legacy in-process metadata updater if the CLI path fails.
- `src/directory_processor.py` no longer constructs a logbook reader eagerly at startup.
- Logbook-reader access is now isolated in `src/logbook_support.py`.
- Process-step modules no longer import `logbook2mouse` directly; they use a local `LogbookReaderLike | None` type instead.
- `src/directory_processor_multibatch_nostack.sh` now exits on the first failed batch instead of printing a false success message.
- `requirements.txt` now includes `mouse_logbook` and `pint`.
- `src/processstep_metadata_update.py` has been simplified to a pure `mouse-logbook write-nexus-metadata ...` wrapper.
- `src/logbook_support.py` now uses `mouse_logbook` only.
- `logbook2mouse` has been removed from `requirements.txt`.

What is still transitional:

- No `pytest` suite or package metadata has been added yet.

## Current State Observations

- `src/directory_processor.py` now passes `None` to process steps unless a step explicitly opts in to reader construction.
- Most process steps only accept the reader in their signature. The only current consumer of reader internals in this repository remains `src/processstep_metadata_update.py`, and only for legacy fallback.
- `src/processstep_metadata_update.py` is now a thin CLI wrapper around `mouse_logbook`.
- The entry points still default to `python`, but the batch script now allows overriding the interpreter through `PYTHON_BIN`.
- The repository currently has no `pyproject.toml`, no `pytest` setup, and no tests.

## Findings About `mouse_logbook`

- The local `.venv` currently contains `mouse-logbook==0.1.2`.
- The CLI now exposes `write-nexus-metadata`, and that command works locally on the configured dataset.
- The installed package provides a legacy-compatible reader facade.
- The upstream reader now normalizes volume fractions, which removed one earlier compatibility blocker.
- With the current `MOUSE_settings.yaml` data, `mouse_logbook.Logbook2MouseReader` now loads successfully against the configured corpus.

## Migration Strategy

The safest route is a staged migration with a local compatibility boundary. That avoids touching every process step at once and lets us land tests before changing behavior.

## Phase 0: Lock Migration Decisions

Before implementation starts, confirm three decisions:

1. Which `mouse_logbook` revision should be the target.
   - The installed `0.1.0` package does not expose `nexuswriter`, so either a newer Git revision is required or the writer work needs a local temporary abstraction.
2. How strict the new project parser should be.
   - Either fix proposal sheets to satisfy strict validation, or provide a custom/relaxed parser that matches current operational data.
3. What should populate `/entry1/sample/sampleowner`.
   - Most likely candidates are project responsible name, user, or a dedicated sample owner field if the newer package exposes one.

## Phase 1: Stabilize the Runtime Environment

### Tasks

1. Add a `pyproject.toml` for reproducible installs.
2. Split runtime and development dependencies.
3. Add `pytest` as a development dependency.
4. Standardize the execution environment used by scripts and docs.
   - Short term: call `./.venv/bin/python`
   - Long term: expose a package CLI or console script
5. Update README usage examples to reflect the actual supported interpreter path.

### Exit Criteria

- A fresh environment can install the package and run the CLI entry point reproducibly.
- The same environment can run tests.

## Phase 2: Introduce a Local Logbook Adapter

Create one repository-owned abstraction layer between the pipeline and any third-party logbook package.

### Proposed module

- `src/logbook_adapter.py`

### Responsibilities

1. Construct the configured reader.
2. Provide a stable lookup API by `(ymd, batch)`.
3. Normalize field access needed by the pipeline.
4. Hide differences between:
   - `logbook2mouse`
   - `mouse_logbook` legacy facade
   - any future writer/parser shape

### Suggested normalized output

Use repository-owned dataclasses or plain typed objects for:

- proposal metadata
- sample metadata
- sample components
- sample position metadata
- derived metadata needed for writing NeXus fields

### Why this matters

After this step, `directory_processor.py` and process steps no longer need to know which third-party package is behind the reader.

### Exit Criteria

- Only the adapter imports `logbook2mouse` or `mouse_logbook`.
- `directory_processor.py` depends on the adapter instead of a third-party reader class.

## Phase 3: Make `mouse_logbook` Compatible With Current Data

This phase should happen before switching the live pipeline over to `mouse_logbook`.

### Tasks

1. Reproduce the current parser failure in an automated test fixture.
2. Decide how to handle proposal sheets where fractions do not sum to `~1.0`.
3. Choose one of these approaches:
   - fix the source proposal files and keep strict validation
   - inject a custom `project_parser` into `mouse_logbook.Logbook2MouseReader`
   - add a local normalization layer that mirrors current `logbook2mouse` behavior
4. Verify that all project files referenced by the configured logbook can be read successfully.

### Recommendation

Prefer repository-side adaptation first, not a big-bang data cleanup. The pipeline should be able to read the current corpus before the dependency swap is finalized.

### Exit Criteria

- `mouse_logbook` can load the current configured logbook/project set without breaking active processing.

## Phase 4: Refactor `processstep_metadata_update.py`

This is the main functional change requested in this round.

### Refactor shape

Split the current module into small units:

1. `find_entry(...)`
2. `read_measurement_energy(...)`
3. `build_metadata_payload(...)`
4. `write_metadata(...)`

### Writer strategy

The initial integration has now been reduced to a very small wrapper:

- call `mouse-logbook write-nexus-metadata <logbook.xlsx> <projects_dir> <output.nxs> --ymd <YYYYMMDD> --batch-num <batch>`

The old in-process metadata updater has been removed from the runtime path.

### Required metadata scope

At minimum, the rewritten step should handle:

- `/entry1/collection_identifier`
- background identifiers
- experiment and proposal metadata
- sample ID, name, composition
- `/entry1/sample/sampleowner`
- sample thickness
- matrix fraction
- sample position fields
- sample components subtree

### Robustness improvements

- make writes idempotent
- handle missing optional metadata cleanly
- replace broad subprocess-only exception handling with actual HDF5/logbook errors
- keep the metadata update path independent from third-party in-memory object shape

### Exit Criteria

- The metadata step updates a representative `.nxs` file correctly through the CLI writer.
- `/entry1/sample/sampleowner` is written.
- The step no longer depends on direct access to third-party object graphs.

## Phase 5: Add `pytest` Coverage

Start with focused tests that pin the migration behavior.

### Unit tests

1. `YMD` parsing and directory metadata extraction
2. adapter lookup by `(ymd, batch)`
3. metadata payload mapping from normalized entry data
4. energy extraction from HDF5
5. optional-field handling for missing backgrounds or incomplete sample metadata

### Integration tests

1. `processstep_metadata_update.run(...)` writing into a temp NeXus file
2. `directory_processor.py` using the adapter and selected steps
3. reader initialization against representative proposal fixtures

### Fixture strategy

- Use small synthetic HDF5 fixtures for NeXus writes.
- Prefer stubbed normalized logbook entries for most tests.
- Keep Excel-based integration fixtures small and explicit so tests do not depend on the full real logbook corpus.

### Exit Criteria

- `pytest` covers the adapter and metadata migration path.
- Regression tests exist for the known parser/data compatibility issue.

## Phase 6: Structural Cleanup

These are follow-on maintainability tasks that become easier once the adapter and tests exist.

### Recommended cleanup items

1. Move from ad hoc `src/*.py` scripts toward a package layout.
2. Replace runtime `assert` statements with explicit exceptions and user-facing validation errors.
3. Centralize logging setup instead of reusing the `DefaultsCarrier` logger implicitly.
4. Remove `print(...)` debugging from runtime paths.
5. Make shell scripts thin wrappers only, or replace them with documented CLI commands.
6. Consider a small step registry instead of raw `importlib.import_module(...)` strings once the pipeline is stable.

## Suggested PR Sequence

1. Documentation plus environment setup
   - add `pyproject.toml`
   - add `pytest`
   - document the interpreter/venv expectation
2. Introduce the adapter without changing behavior
   - keep `logbook2mouse` behind the adapter first
3. Verify `mouse_logbook` against the current data corpus
4. Rewrite `processstep_metadata_update.py` around the CLI writer
5. Remove the `logbook2mouse` dependency
6. Add tests around the stabilized path

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

1. add `pytest` coverage for:
   - CLI-first metadata updates
   - `DirectoryProcessor` startup without eager reader construction
   - reader initialization against small representative fixtures
2. add a `pyproject.toml` and split runtime vs development dependencies
3. tighten CLI/runtime documentation around `.venv` and `PYTHON_BIN`

The dependency swap is complete in code. The remaining work is now testing, packaging, and cleanup.
