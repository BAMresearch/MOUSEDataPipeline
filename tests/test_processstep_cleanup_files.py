from __future__ import annotations

import processstep_cleanup_files


def test_cleanup_files_removes_step_1_output(mini_dataset):
    step_1_file = mini_dataset.repetition_dir / f"MOUSE_{mini_dataset.ymd}_{mini_dataset.batch_num}_{mini_dataset.repetition}_step_1.nxs"
    step_1_file.write_text("temporary", encoding="utf-8")

    processstep_cleanup_files.run(
        mini_dataset.repetition_dir,
        mini_dataset.defaults,
        logbook_reader=None,
        logger=mini_dataset.defaults.logger,
    )

    assert not step_1_file.exists()
