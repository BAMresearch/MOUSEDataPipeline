from __future__ import annotations

import h5py
import pytest

import processstep_make_beam_mask


def test_make_beam_mask_raises_for_invalid_sample_detector_distance(mini_dataset):
    with h5py.File(mini_dataset.output_file, "a") as h5f:
        h5f.require_dataset(
            "/entry1/processing/direct_beam_profile/beam_analysis/centerOfMass",
            shape=(2,),
            dtype="f4",
        )[...] = [5.0, 5.0]
        h5f.require_dataset(
            "/entry1/instrument/detector00/data",
            shape=(10, 10),
            dtype="f4",
        )[...] = 0.0
        det_x = h5f.require_dataset(
            "/entry1/instrument/detector00/transformations/det_x",
            shape=(),
            dtype="f4",
        )
        det_x[...] = 0.0
        det_x.attrs["units"] = "m"
        sample_x = h5f.require_dataset(
            "/entry1/sample/transformations/sample_x",
            shape=(),
            dtype="f4",
        )
        sample_x[...] = 1.0
        sample_x.attrs["units"] = "m"

    with pytest.raises(ValueError, match="invalid sample-detector distance"):
        processstep_make_beam_mask.run(
            mini_dataset.repetition_dir,
            mini_dataset.defaults,
            logbook_reader=None,
            logger=mini_dataset.defaults.logger,
        )
