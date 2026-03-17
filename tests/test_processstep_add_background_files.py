from __future__ import annotations

import h5py

import processstep_add_background_files


def _read_scalar_string(dataset) -> str:
    value = dataset[()]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def test_add_background_files_writes_relative_paths(mini_dataset):
    with h5py.File(mini_dataset.output_file, "a") as h5f:
        h5f.require_dataset("/entry1/instrument/configuration", shape=(), dtype="i4")[...] = 7
        h5f.require_dataset(
            "/entry1/processing_required_metadata/background_identifier",
            shape=(),
            dtype=h5py.string_dtype(encoding="utf-8"),
        )[...] = "20260311_4"
        h5f.require_dataset(
            "/entry1/processing_required_metadata/dispersant_background_identifier",
            shape=(),
            dtype=h5py.string_dtype(encoding="utf-8"),
        )[...] = "None"

    processstep_add_background_files.run(
        mini_dataset.repetition_dir,
        mini_dataset.defaults,
        logbook_reader=None,
        logger=mini_dataset.defaults.logger,
    )

    expected_background = (
        mini_dataset.defaults.data_dir
        / "2026"
        / "20260311"
        / "MOUSE_20260311_4_7_stacked.nxs"
    ).relative_to(mini_dataset.repetition_dir, walk_up=True)

    with h5py.File(mini_dataset.output_file, "r") as h5f:
        assert _read_scalar_string(h5f["/entry1/processing_required_metadata/background_file"]) == str(expected_background)
        assert _read_scalar_string(h5f["/entry1/processing_required_metadata/dispersed_background_file"]) == ""
