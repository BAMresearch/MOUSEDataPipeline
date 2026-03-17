from __future__ import annotations

from logbook_support import build_logbook_reader


def test_build_logbook_reader_loads_small_fixture(mini_dataset):
    reader = build_logbook_reader(
        mini_dataset.defaults.logbook_file,
        mini_dataset.defaults.projects_dir,
        logger=mini_dataset.defaults.logger,
    )

    assert type(reader).__module__ == "mouse_logbook.legacy"
    assert len(reader.entries) == 1
    assert reader.entries[0].ymd == mini_dataset.ymd
    assert reader.entries[0].batchnum == mini_dataset.batch_num


def test_build_logbook_reader_loads_example_mouse_logbook_fixture(mouse_logbook_example_dataset):
    reader = build_logbook_reader(
        mouse_logbook_example_dataset.logbook_file,
        mouse_logbook_example_dataset.projects_dir,
        logger=mouse_logbook_example_dataset.defaults.logger,
    )

    assert type(reader).__module__ == "mouse_logbook.legacy"
    assert len(reader.entries) == 28
    selected = [entry for entry in reader.entries if entry.ymd == "20251220" and entry.batchnum == 2]
    assert len(selected) == 1
    assert selected[0].proposal == "2025002"
    assert selected[0].sampleid == 1
    assert selected[0].sampos == "Cu B1 20251219"
