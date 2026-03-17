from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

import directory_processor
from directory_processor import DirectoryProcessor
from YMD_class import YMD, extract_metadata_from_path


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

    monkeypatch.setattr(
        directory_processor,
        "importlib",
        SimpleNamespace(import_module=lambda name: fake_module),
    )
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


def test_directory_processor_writes_per_datafile_log(mini_dataset, monkeypatch):
    processor = DirectoryProcessor(defaults=mini_dataset.defaults, steps=["fake_step"])

    fake_module = SimpleNamespace(
        requires_logbook_reader=False,
        can_run=lambda dir_path, defaults, logbook_reader, logger: True,
        run=lambda dir_path, defaults, logbook_reader, logger: logger.info("fake step body"),
    )

    monkeypatch.setattr(
        directory_processor,
        "importlib",
        SimpleNamespace(import_module=lambda name: fake_module),
    )

    processor.process_directory(single_dir=mini_dataset.repetition_dir)

    log_file = (
        mini_dataset.repetition_dir
        / f"MOUSE_{mini_dataset.ymd}_{mini_dataset.batch_num}_{mini_dataset.repetition}.processing.log"
    )
    assert log_file.is_file()
    log_text = log_file.read_text(encoding="utf-8")
    assert "Starting processing for directory" in log_text
    assert "Running step: fake_step" in log_text
    assert "fake step body" in log_text


def test_directory_processor_can_disable_per_datafile_logs(mini_dataset, monkeypatch):
    mini_dataset.defaults.log_per_datafile = False
    processor = DirectoryProcessor(defaults=mini_dataset.defaults, steps=["fake_step"])

    fake_module = SimpleNamespace(
        requires_logbook_reader=False,
        can_run=lambda dir_path, defaults, logbook_reader, logger: True,
        run=lambda dir_path, defaults, logbook_reader, logger: logger.info("fake step body"),
    )

    monkeypatch.setattr(
        directory_processor,
        "importlib",
        SimpleNamespace(import_module=lambda name: fake_module),
    )

    processor.process_directory(single_dir=mini_dataset.repetition_dir)

    log_file = (
        mini_dataset.repetition_dir
        / f"MOUSE_{mini_dataset.ymd}_{mini_dataset.batch_num}_{mini_dataset.repetition}.processing.log"
    )
    assert not log_file.exists()


def test_directory_processor_omits_profile_logs_when_disabled(mini_dataset, monkeypatch, caplog):
    mini_dataset.defaults.profile_steps = False
    processor = DirectoryProcessor(defaults=mini_dataset.defaults, steps=["fake_step"])

    fake_module = SimpleNamespace(
        requires_logbook_reader=False,
        can_run=lambda dir_path, defaults, logbook_reader, logger: True,
        run=lambda dir_path, defaults, logbook_reader, logger: None,
    )

    monkeypatch.setattr(
        directory_processor,
        "importlib",
        SimpleNamespace(import_module=lambda name: fake_module),
    )

    caplog.set_level(logging.INFO, logger="DefaultsCarrier")
    processor._run_processing_step(
        "fake_step",
        mini_dataset.repetition_dir,
        YMD(mini_dataset.ymd),
        mini_dataset.batch_num,
        mini_dataset.repetition,
    )

    assert "PROFILE step=fake_step" not in caplog.text


def test_directory_processor_propagates_parallel_step_errors(mini_dataset, monkeypatch):
    processor = DirectoryProcessor(defaults=mini_dataset.defaults, steps=["fake_step"])
    second_dir = mini_dataset.repetition_dir.parent / f"{mini_dataset.ymd}_{mini_dataset.batch_num}_1"
    second_dir.mkdir(parents=True, exist_ok=True)

    fake_module = SimpleNamespace(
        requires_logbook_reader=False,
        can_process_repetitions_in_parallel=True,
        can_run=lambda dir_path, defaults, logbook_reader, logger: True,
        run=lambda dir_path, defaults, logbook_reader, logger: (
            (_ for _ in ()).throw(RuntimeError("parallel step failed")) if dir_path == second_dir else None
        ),
    )

    monkeypatch.setattr(
        directory_processor,
        "importlib",
        SimpleNamespace(import_module=lambda name: fake_module),
    )
    monkeypatch.setattr(
        DirectoryProcessor,
        "_get_all_repetitions_directories",
        lambda self, ymd, batch: [mini_dataset.repetition_dir, second_dir],
    )

    with pytest.raises(RuntimeError, match="parallel step failed"):
        processor.process_batch(mini_dataset.ymd, mini_dataset.batch_num, parallel=True)


def test_directory_processor_resolve_directory_accepts_repetition_zero(mini_dataset):
    processor = DirectoryProcessor(defaults=mini_dataset.defaults, steps=[])

    resolved_dir, ymd, batch, repetition = processor._resolve_directory(
        single_dir=None,
        ymd=mini_dataset.ymd,
        batch=mini_dataset.batch_num,
        repetition=0,
    )

    assert resolved_dir == mini_dataset.repetition_dir
    assert ymd.YMD == mini_dataset.ymd
    assert batch == mini_dataset.batch_num
    assert repetition == 0


def test_directory_processor_resolve_directory_requires_complete_coordinates(mini_dataset):
    processor = DirectoryProcessor(defaults=mini_dataset.defaults, steps=[])

    with pytest.raises(ValueError, match="Either single_dir or ymd, batch, and repetition must be provided."):
        processor._resolve_directory(
            single_dir=None,
            ymd=mini_dataset.ymd,
            batch=mini_dataset.batch_num,
            repetition=None,
        )


def test_directory_processor_resolve_directory_raises_for_missing_path(mini_dataset):
    processor = DirectoryProcessor(defaults=mini_dataset.defaults, steps=[])

    with pytest.raises(FileNotFoundError, match="Provided path is not an existing directory"):
        processor._resolve_directory(
            single_dir=mini_dataset.repetition_dir / "does_not_exist",
            ymd=None,
            batch=None,
            repetition=None,
        )


def test_extract_metadata_from_path_raises_value_error_for_invalid_format():
    with pytest.raises(ValueError, match="Invalid directory format"):
        extract_metadata_from_path(Path("invalid-directory-name"))
