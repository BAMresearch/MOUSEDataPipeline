from __future__ import annotations

import logging
from pathlib import Path

import h5py
import numpy as np
import pytest

import post_translation_operation_hdf5_stacker


def _write_stackable_input(filename: Path, data: np.ndarray, mask_file: Path):
    filename.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(filename, "w") as h5f:
        h5f.require_dataset(
            "/entry1/instrument/detector00/data",
            shape=data.shape,
            dtype="f4",
        )[...] = data
        h5f.require_dataset("/entry1/sample/beam/flux", shape=(), dtype="f4")[...] = 123.0
        h5f.require_dataset("/entry1/sample/beam/incident_wavelength", shape=(), dtype="f4")[...] = 0.1
        h5f.require_dataset("/entry1/sample/transmission", shape=(), dtype="f4")[...] = float(np.mean(data))
        h5f.require_dataset(
            "/entry1/processing/direct_beam_profile/beam_analysis/centerOfMass",
            shape=(2,),
            dtype="f4",
        )[...] = [1.0, 2.0]
        h5f.require_dataset(
            "/entry1/processing_required_metadata/mask_file",
            shape=(),
            dtype=h5py.string_dtype(encoding="utf-8"),
        )[...] = "../Masks/mask.nxs"

    mask_file.parent.mkdir(parents=True, exist_ok=True)
    mask_file.write_text("mask", encoding="utf-8")


def _write_dataset(filename: Path, path: str, data) -> None:
    with h5py.File(filename, "a") as h5f:
        array = np.asarray(data)
        h5f.require_dataset(path, shape=array.shape, dtype=array.dtype)[...] = array


def _read_scalar_string(dataset) -> str:
    value = dataset[()]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def test_stacker_main_stacks_and_adjusts_paths(tmp_path: Path):
    input_a = tmp_path / "inputs" / "a.nxs"
    input_b = tmp_path / "inputs" / "b.nxs"
    mask_file = tmp_path / "Masks" / "mask.nxs"
    output_file = tmp_path / "stacked.nxs"
    config_file = tmp_path / "stacker.yaml"

    _write_stackable_input(input_a, np.full((2, 2), 1.0, dtype=np.float32), mask_file)
    _write_stackable_input(input_b, np.full((2, 2), 3.0, dtype=np.float32), mask_file)
    config_file.write_text(
        "\n".join(
            [
                "stack_datasets:",
                "  - entry1/instrument/detector00/data",
                "  - entry1/sample/transmission",
                "calculate_average:",
                "  - entry1/sample/transmission",
                "adjust_relative_path_oneup:",
                "  - entry1/processing_required_metadata/mask_file",
            ]
        ),
        encoding="utf-8",
    )

    post_translation_operation_hdf5_stacker.main(
        output=output_file,
        auxiliary_files=[input_a, input_b],
        config=config_file,
    )

    with h5py.File(output_file, "r") as h5f:
        np.testing.assert_allclose(
            h5f["/entry1/instrument/detector00/data"][()],
            np.array(
                [
                    np.full((2, 2), 1.0, dtype=np.float32),
                    np.full((2, 2), 3.0, dtype=np.float32),
                ]
            ),
        )
        np.testing.assert_allclose(h5f["/entry1/sample/transmission"][()], np.array([1.0, 3.0], dtype=np.float32))
        assert h5f["/entry1/sample/transmission_averaged/mean"][()] == pytest.approx(2.0)
        assert _read_scalar_string(h5f["/entry1/processing_required_metadata/mask_file"]) == "Masks/mask.nxs"


def test_stacker_main_uses_provided_logger(tmp_path: Path, caplog):
    input_a = tmp_path / "inputs" / "a.nxs"
    mask_file = tmp_path / "Masks" / "mask.nxs"
    output_file = tmp_path / "stacked.nxs"
    config_file = tmp_path / "stacker.yaml"

    _write_stackable_input(input_a, np.full((2, 2), 1.0, dtype=np.float32), mask_file)
    config_file.write_text("stack_datasets:\n  - entry1/sample/transmission\n", encoding="utf-8")

    logger = logging.getLogger("test_stacker")
    caplog.set_level(logging.INFO, logger="test_stacker")

    post_translation_operation_hdf5_stacker.main(
        output=output_file,
        auxiliary_files=[input_a],
        config=config_file,
        logger=logger,
    )

    assert "Post-translation processing complete." in caplog.text


def test_stacker_main_can_match_metadata_rank_to_detector_data(tmp_path: Path):
    input_a = tmp_path / "inputs" / "a.nxs"
    input_b = tmp_path / "inputs" / "b.nxs"
    mask_file = tmp_path / "Masks" / "mask.nxs"
    output_file = tmp_path / "stacked.nxs"
    config_file = tmp_path / "stacker.yaml"

    _write_stackable_input(input_a, np.full((1, 2, 3), 1.0, dtype=np.float32), mask_file)
    _write_stackable_input(input_b, np.full((1, 2, 3), 3.0, dtype=np.float32), mask_file)
    _write_dataset(input_a, "/entry1/sample/custom_metadata", np.array([7.0], dtype=np.float32))
    _write_dataset(input_b, "/entry1/sample/custom_metadata", np.array([9.0], dtype=np.float32))
    config_file.write_text(
        "\n".join(
            [
                "stack_datasets:",
                "  - entry1/instrument/detector00/data",
                "  - entry1/sample/custom_metadata",
            ]
        ),
        encoding="utf-8",
    )

    post_translation_operation_hdf5_stacker.main(
        output=output_file,
        auxiliary_files=[input_a, input_b],
        config=config_file,
        match_detector_data_rank=True,
    )

    with h5py.File(output_file, "r") as h5f:
        assert h5f["/entry1/instrument/detector00/data"].shape == (2, 1, 2, 3)
        assert h5f["/entry1/sample/custom_metadata"].shape == (2, 1, 1, 1)
        np.testing.assert_allclose(
            h5f["/entry1/sample/custom_metadata"][()],
            np.array([[[[7.0]]], [[[9.0]]]], dtype=np.float32),
        )


def test_stacker_main_requires_auxiliary_files(tmp_path: Path):
    config_file = tmp_path / "stacker.yaml"
    config_file.write_text("stack_datasets: []\n", encoding="utf-8")

    with pytest.raises(ValueError, match="At least one file is required for stacking"):
        post_translation_operation_hdf5_stacker.main(
            output=tmp_path / "stacked.nxs",
            auxiliary_files=[],
            config=config_file,
        )


def test_stacker_main_requires_stack_datasets_section(tmp_path: Path):
    config_file = tmp_path / "stacker.yaml"
    config_file.write_text("calculate_average: []\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must contain a 'stack_datasets' section"):
        post_translation_operation_hdf5_stacker.main(
            output=tmp_path / "stacked.nxs",
            auxiliary_files=[tmp_path / "input.nxs"],
            config=config_file,
        )


def test_new_concat_raises_for_missing_input_file(tmp_path: Path):
    missing_file = tmp_path / "missing.nxs"

    with pytest.raises(FileNotFoundError, match="does not exist in the list of files to stack"):
        post_translation_operation_hdf5_stacker.newNewConcat(
            outputFile=tmp_path / "stacked.nxs",
            filenames=[missing_file],
            stackItems=["entry1/instrument/detector00/data"],
        )
