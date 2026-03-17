from __future__ import annotations

import h5py
import pytest

import processstep_thickness_from_absorption


def test_thickness_from_absorption_raises_for_missing_input_file(mini_dataset):
    mini_dataset.output_file.unlink()

    with pytest.raises(FileNotFoundError, match="Input file .* does not exist"):
        processstep_thickness_from_absorption.run(
            mini_dataset.repetition_dir,
            mini_dataset.defaults,
            logbook_reader=None,
            logger=mini_dataset.defaults.logger,
        )


def test_thickness_from_absorption_writes_metadata_without_stdout(mini_dataset, capsys):
    with h5py.File(mini_dataset.output_file, "a") as h5f:
        h5f.require_dataset("/entry1/sample/overall_mu", shape=(), dtype="f4")[...] = 1000.0
        h5f.require_dataset("/entry1/sample/transmission", shape=(), dtype="f4")[...] = 0.8
        h5f.require_dataset("/entry1/sample/samplethickness", shape=(), dtype="f4")[...] = -1.0
        h5f.require_dataset(
            "/entry1/processing_required_metadata/background_file",
            shape=(),
            dtype=h5py.string_dtype(encoding="utf-8"),
        )[...] = ""

    processstep_thickness_from_absorption.run(
        mini_dataset.repetition_dir,
        mini_dataset.defaults,
        logbook_reader=None,
        logger=mini_dataset.defaults.logger,
    )

    captured = capsys.readouterr()
    assert captured.out == ""

    with h5py.File(mini_dataset.output_file, "r") as h5f:
        assert h5f["/entry1/sample/absorptionDerivedThickness"][()] > 0
        assert h5f["/entry1/sample/thickness"][()] > 0
