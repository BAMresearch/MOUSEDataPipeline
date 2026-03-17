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
