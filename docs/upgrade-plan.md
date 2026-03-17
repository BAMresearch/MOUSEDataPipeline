# MOUSEDataPipeline Upgrade Plan

## Goals

1. Replace direct `logbook2mouse` usage with `mouse_logbook`.
2. Move metadata writing in `src/processstep_metadata_update.py` to the newer writer path, including missing fields such as `/entry1/sample/sampleowner`.
3. Add a minimal but effective `pytest` test suite around the migration.
4. Improve package structure, environment reproducibility, and runtime robustness without rewriting the whole pipeline at once.

## Current State Observations

- `src/directory_processor.py` constructs `logbook2mouse.logbook_reader.Logbook2MouseReader` directly and passes it into every process step.
- Most process steps only accept the reader in their signature. The only current consumer of reader internals in this repository is `src/processstep_metadata_update.py`.
- `src/processstep_metadata_update.py` performs direct HDF5 writes itself through `HDF5Translator` elements. It is not isolated behind a writer abstraction.
- The current metadata updater depends on the old object model:
  - `entry.sampleposition`
  - `entry.sample.density`
  - `entry.sample.calculate_overall_properties(...)`
  - component fields such as `volume_fraction`, `mass_fraction`, and `name`
- `requirements.txt` still installs `logbook2mouse` from Git.
- The entry points `src/directory_processor.py` and `src/directory_processor_multibatch_nostack.sh` call `python`, but the current shell environment does not expose the same interpreter as the project `.venv`.
- The repository currently has no `pyproject.toml`, no `pytest` setup, and no tests.

## Findings About `mouse_logbook`

- The local `.venv` contains `mouse-logbook==0.1.0`.
- In that installed version, `Logbook2MouseReader` is available via `from mouse_logbook import Logbook2MouseReader`.
- The installed package does not currently expose a `mouse_logbook.nexuswriter` module.
- The installed package provides a legacy-compatible reader facade, but not a full legacy-compatible nested object model.
- Important compatibility differences observed from the installed package:
  - new entry object uses `entry.positions`, not `entry.sampleposition`
  - parsed sample components use `vol_frac`, `mass_frac`, and `component_name`
  - the parsed sample model in the installed package does not provide `density` or `calculate_overall_properties(...)`
- With the current `MOUSE_settings.yaml` data, the old `logbook2mouse` reader loads successfully, but the installed `mouse_logbook` reader fails on at least one real project sheet due to stricter validation:
  - `2025012_Andrea_BAM3p6.xlsx: invalid Sample_Info: sampleId=1: volFrac sums to 0.300000, expected ~1.0`

This means the migration is not just an import replacement. There are two separate blockers:

1. parser/data compatibility with current proposal sheets
2. metadata-updater assumptions about the old nested object model

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

1. If the target `mouse_logbook` revision provides the desired NeXus writer, use it behind a thin local wrapper.
2. If not, add a repository-owned writer abstraction now and swap the backend later.

That keeps the processing step stable even if the upstream writer API changes.

If the new CLI remains stable, the initial integration can be extremely small:

- call `mouse-logbook write-nexus-metadata <logbook.xlsx> <projects_dir> <output.nxs> --ymd <YYYYMMDD> --batch-num <batch>`
- keep the current in-process writer as a temporary fallback during migration

That would remove almost all direct logbook-object handling from `processstep_metadata_update.py` immediately.

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

### Derived values

The current code also writes:

- sample density
- overall attenuation (`overall_mu`)

Those should not depend on third-party sample-object methods anymore. Move the calculations into repository-owned functions, or make them optional with explicit warnings when the necessary inputs are unavailable.

### Robustness improvements

- make writes idempotent
- handle missing optional metadata cleanly
- replace broad subprocess-only exception handling with actual HDF5/logbook errors
- remove hidden assumptions about attribute names from third-party objects

### Exit Criteria

- The metadata step updates a representative `.nxs` fixture correctly.
- `/entry1/sample/sampleowner` is written.
- The step works from normalized adapter data, not directly from the third-party object graph.

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
3. regression test for the strict parser failure case

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
3. Add failing tests for `mouse_logbook` compatibility gaps
4. Make `mouse_logbook` read the current data corpus
5. Rewrite `processstep_metadata_update.py` around normalized data and a writer abstraction
6. Switch the adapter backend from `logbook2mouse` to `mouse_logbook`
7. Remove the `logbook2mouse` dependency

## Definition Of Done For This Migration

The migration can be considered complete when all of the following are true:

- the runtime no longer imports `logbook2mouse`
- the configured project/logbook corpus is readable through `mouse_logbook`
- metadata updates are written through the new writer path or a stable wrapper around it
- `/entry1/sample/sampleowner` is present in updated measurement files
- `pytest` covers the reader adapter and metadata update behavior
- the documented CLI path works from a fresh environment

## Immediate Next Step

The highest-value first implementation step is:

1. add the environment/test scaffolding
2. introduce the local adapter while still using `logbook2mouse`
3. write a regression test that captures the current `mouse_logbook` parser failure

That gives a safe base for the actual dependency swap and the metadata-writer rewrite.
