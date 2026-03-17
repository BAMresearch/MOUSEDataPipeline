from __future__ import annotations

import logging
from types import SimpleNamespace

import directory_processor
from YMD_class import YMD
from directory_processor import DirectoryProcessor


def test_directory_processor_startup_is_lazy(mini_dataset):
    processor = DirectoryProcessor(defaults=mini_dataset.defaults, steps=[])

    assert processor.logbook_reader is None


def test_directory_processor_runs_non_reader_step_without_building_reader(mini_dataset, monkeypatch, caplog):
    processor = DirectoryProcessor(defaults=mini_dataset.defaults, steps=["fake_step"])
    calls: list[str] = []

    fake_module = SimpleNamespace(
        requires_logbook_reader=False,
        can_run=lambda dir_path, defaults, logbook_reader, logger: True,
        run=lambda dir_path, defaults, logbook_reader, logger: calls.append("run"),
    )

    monkeypatch.setattr(directory_processor.importlib, "import_module", lambda name: fake_module)
    monkeypatch.setattr(
        directory_processor,
        "build_logbook_reader",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("reader should not be built")),
    )

    caplog.set_level(logging.INFO, logger="DefaultsCarrier")
    processor._run_processing_step(
        "fake_step",
        mini_dataset.repetition_dir,
        YMD(mini_dataset.ymd),
        mini_dataset.batch_num,
        mini_dataset.repetition,
    )

    assert calls == ["run"]
    assert processor.logbook_reader is None
    assert "PROFILE step=fake_step" in caplog.text
