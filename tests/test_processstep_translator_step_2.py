from __future__ import annotations

import os
from types import SimpleNamespace

import processstep_translator_step_2


def test_translator_step_2_shells_out_to_hdf5translator(mini_dataset, monkeypatch):
    calls: list[list[str]] = []
    step_1_file = (
        mini_dataset.repetition_dir
        / f"MOUSE_{mini_dataset.ymd}_{mini_dataset.batch_num}_{mini_dataset.repetition}_step_1.nxs"
    )
    input_file = mini_dataset.repetition_dir / "eiger_1_master.h5"
    step_1_file.write_bytes(b"step1")
    input_file.write_bytes(b"master")

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return SimpleNamespace(stdout="")

    monkeypatch.setattr(processstep_translator_step_2.subprocess, "run", fake_run)

    processstep_translator_step_2.run(
        mini_dataset.repetition_dir,
        mini_dataset.defaults,
        logbook_reader=None,
        logger=mini_dataset.defaults.logger,
    )

    assert calls == [
        [
            "python3",
            "-m",
            "HDF5Translator",
            "-C",
            str(mini_dataset.defaults.translator_template_dir / "BAM_new_MOUSE_dectris_adder_configuration.yaml"),
            "-T",
            str(step_1_file),
            "-I",
            str(input_file),
            "-O",
            str(mini_dataset.output_file),
            "-d",
        ]
    ]


def test_translator_step_2_can_run_skips_when_output_is_up_to_date(mini_dataset):
    step_1_file = (
        mini_dataset.repetition_dir
        / f"MOUSE_{mini_dataset.ymd}_{mini_dataset.batch_num}_{mini_dataset.repetition}_step_1.nxs"
    )
    input_file = mini_dataset.repetition_dir / "eiger_1_master.h5"
    config_file = mini_dataset.defaults.translator_template_dir / "BAM_new_MOUSE_dectris_adder_configuration.yaml"

    step_1_file.write_bytes(b"step1")
    input_file.write_bytes(b"master")
    mini_dataset.output_file.write_bytes(b"output")
    config_file.write_bytes(b"config")

    os.utime(step_1_file, (100, 100))
    os.utime(input_file, (100, 100))
    os.utime(config_file, (100, 100))
    os.utime(mini_dataset.output_file, (200, 200))

    assert (
        processstep_translator_step_2.can_run(
            mini_dataset.repetition_dir,
            mini_dataset.defaults,
            logbook_reader=None,
            logger=mini_dataset.defaults.logger,
        )
        is False
    )
