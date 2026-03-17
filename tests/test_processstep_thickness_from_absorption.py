from __future__ import annotations

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
