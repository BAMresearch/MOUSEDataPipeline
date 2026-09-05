from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

import stacked_mouse_to_modacor


def _write_stacked_file(filename: Path) -> None:
    with h5py.File(filename, "w") as h5f:
        h5f.create_dataset(
            "/entry1/instrument/detector00/data",
            data=np.ones((3, 1, 4, 5), dtype=np.float32),
            chunks=(1, 1, 4, 5),
            compression="gzip",
        )
        scalar = h5f.create_dataset(
            "/entry1/sample/transmission",
            data=np.array([[0.8], [0.9], [1.0]], dtype=np.float32),
            chunks=(1, 1),
            compression="gzip",
        )
        scalar.attrs["units"] = "dimensionless"
        h5f.create_dataset("/entry1/experiment/stage_temperature", data=np.array([20.0, 21.0, 22.0]))
        h5f.create_dataset("/entry1/sample/not_for_modacor", data=np.array([[1.0], [2.0], [3.0]]))
        h5f.create_dataset(
            "/entry1/processing/direct_beam_profile/beam_analysis/centerOfMass",
            data=np.array([[1.0, 2.0], [1.1, 2.1], [1.2, 2.2]], dtype=np.float32),
            chunks=(1, 2),
            compression="gzip",
        )
        h5f.create_dataset("/entry1/sample/name", data=np.bytes_("sample"))


def test_convert_stacked_file_to_modacor_pads_only_scalar_repetition_datasets(tmp_path: Path):
    input_file = tmp_path / "stacked.nxs"
    output_file = tmp_path / "stacked_modacor.nxs"
    _write_stacked_file(input_file)

    changed = stacked_mouse_to_modacor.convert_stacked_file_to_modacor(input_file, output_file)

    assert changed == [
        ("entry1/experiment/stage_temperature", (3,), (3, 1, 1, 1)),
        ("entry1/sample/transmission", (3, 1), (3, 1, 1, 1)),
    ]
    with h5py.File(output_file, "r") as h5f:
        transmission = h5f["/entry1/sample/transmission"]
        assert transmission.shape == (3, 1, 1, 1)
        assert transmission.chunks == (1, 1, 1, 1)
        assert transmission.compression == "gzip"
        assert transmission.attrs["units"] == "dimensionless"
        np.testing.assert_allclose(transmission[()].squeeze(), np.array([0.8, 0.9, 1.0], dtype=np.float32))

        assert h5f["/entry1/experiment/stage_temperature"].shape == (3, 1, 1, 1)
        assert h5f["/entry1/instrument/detector00/data"].shape == (3, 1, 4, 5)
        assert h5f["/entry1/processing/direct_beam_profile/beam_analysis/centerOfMass"].shape == (3, 2)
        assert h5f["/entry1/sample/not_for_modacor"].shape == (3, 1)
        assert h5f["/entry1/sample/name"].shape == ()


def test_find_datasets_to_pad_uses_hardcoded_mouse_paths(tmp_path: Path):
    input_file = tmp_path / "stacked.nxs"
    _write_stacked_file(input_file)

    changed = stacked_mouse_to_modacor.find_datasets_to_pad(input_file)

    assert changed == [
        ("entry1/experiment/stage_temperature", (3,), (3, 1, 1, 1)),
        ("entry1/sample/transmission", (3, 1), (3, 1, 1, 1)),
    ]
