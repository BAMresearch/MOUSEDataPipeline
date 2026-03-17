from __future__ import annotations

import h5py
import numpy as np
import pytest

import processstep_determine_beam_center


def test_determine_beam_center_writes_center_without_stdout(mini_dataset, capsys):
    image = np.zeros((30, 30), dtype=np.float32)
    image[10:20, 12:22] = 5.0

    with h5py.File(mini_dataset.output_file, "a") as h5f:
        h5f.require_dataset(
            "/entry1/processing/direct_beam_profile/data",
            shape=image.shape,
            dtype="f4",
        )[...] = image

    processstep_determine_beam_center.run(
        mini_dataset.repetition_dir,
        mini_dataset.defaults,
        logbook_reader=None,
        logger=mini_dataset.defaults.logger,
    )

    captured = capsys.readouterr()
    assert captured.out == ""

    with h5py.File(mini_dataset.output_file, "r") as h5f:
        center = h5f["/entry1/processing/direct_beam_profile/beam_analysis/centerOfMass"][()]
        assert center[0] == pytest.approx(14.5, abs=0.2)
        assert center[1] == pytest.approx(16.5, abs=0.2)
