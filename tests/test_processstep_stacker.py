from __future__ import annotations

import logging
from types import SimpleNamespace

import processstep_stacker


def test_stacker_logs_command_without_printing(mini_dataset, monkeypatch, caplog, capsys):
    processed_file = mini_dataset.repetition_dir / f"MOUSE_{mini_dataset.ymd}_{mini_dataset.batch_num}_0.nxs"
    processed_file.write_bytes(b"processed")
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return SimpleNamespace(stdout="stacked")

    monkeypatch.setattr(processstep_stacker, "get_processed_files", lambda dir_path: [processed_file])
    monkeypatch.setattr(
        processstep_stacker,
        "sort_processed_files_by_instrument_configuration",
        lambda processed_files, logger: {"7": processed_files},
    )
    monkeypatch.setattr(processstep_stacker.subprocess, "run", fake_run)

    mini_dataset.defaults.logger.setLevel(logging.DEBUG)
    caplog.set_level(logging.DEBUG, logger="DefaultsCarrier")

    processstep_stacker.run(
        mini_dataset.repetition_dir,
        mini_dataset.defaults,
        logbook_reader=None,
        logger=mini_dataset.defaults.logger,
    )

    captured = capsys.readouterr()
    assert captured.out == ""
    assert len(calls) == 1
    assert calls[0][:2] == [
        "python3",
        str(mini_dataset.defaults.post_translation_dir / "post_translation_operation_hdf5_stacker.py"),
    ]
    assert "Running stacker command:" in caplog.text
