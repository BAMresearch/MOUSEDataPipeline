from __future__ import annotations

import subprocess
from pathlib import Path

import h5py
import pytest

import processstep_metadata_update


def _read_scalar_string(dataset) -> str:
    value = dataset[()]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def test_metadata_update_cli_writer_updates_nexus_file(mini_dataset):
    processstep_metadata_update.run(
        mini_dataset.repetition_dir,
        mini_dataset.defaults,
        logbook_reader=None,
        logger=mini_dataset.defaults.logger,
    )

    with h5py.File(mini_dataset.output_file, "r") as h5f:
        assert "/entry1/sample/sampleowner" in h5f
        assert _read_scalar_string(h5f["/entry1/sample/sampleowner"]) == "Project Owner"
        assert "/entry1/proposal/proposalid" in h5f
        assert _read_scalar_string(h5f["/entry1/proposal/proposalid"]) == "2026001"
        assert "/entry1/processing_required_metadata/procpipeline" in h5f
        assert _read_scalar_string(h5f["/entry1/processing_required_metadata/procpipeline"]) == "test-pipeline"


def test_metadata_update_cli_writer_updates_nexus_file_from_example_mouse_logbook(mouse_logbook_example_dataset):
    processstep_metadata_update.run(
        mouse_logbook_example_dataset.repetition_dir,
        mouse_logbook_example_dataset.defaults,
        logbook_reader=None,
        logger=mouse_logbook_example_dataset.defaults.logger,
    )

    with h5py.File(mouse_logbook_example_dataset.output_file, "r") as h5f:
        assert "/entry1/sample/sampleowner" in h5f
        assert _read_scalar_string(h5f["/entry1/sample/sampleowner"]) == "Test user"
        assert "/entry1/proposal/proposalid" in h5f
        assert _read_scalar_string(h5f["/entry1/proposal/proposalid"]) == "2025002"
        assert "/entry1/sample/name" in h5f
        assert _read_scalar_string(h5f["/entry1/sample/name"]) == "Vacuum"
        assert "/entry1/processing_required_metadata/procpipeline" in h5f
        assert (
            _read_scalar_string(h5f["/entry1/processing_required_metadata/procpipeline"])
            == "20251010_standard_logq.nxs"
        )


def test_metadata_update_cli_writer_failure_propagates(mini_dataset, monkeypatch):
    monkeypatch.setattr(
        processstep_metadata_update,
        "_resolve_mouse_logbook_cli",
        lambda: Path("/tmp/mouse-logbook"),
    )

    def fake_run(*args, **kwargs):
        raise subprocess.CalledProcessError(
            returncode=2,
            cmd=kwargs.get("args", args[0] if args else []),
            stderr="simulated failure",
        )

    monkeypatch.setattr(processstep_metadata_update.subprocess, "run", fake_run)

    with pytest.raises(subprocess.CalledProcessError, match="returned non-zero exit status 2"):
        processstep_metadata_update.run(
            mini_dataset.repetition_dir,
            mini_dataset.defaults,
            logbook_reader=None,
            logger=mini_dataset.defaults.logger,
        )
