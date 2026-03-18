from __future__ import annotations

import os

import processstep_translator_step_1


def test_translator_step_1_can_run_skips_when_output_is_up_to_date(mini_dataset):
    direct_beam_dir = mini_dataset.repetition_dir / "beam_profile"
    sample_beam_dir = mini_dataset.repetition_dir / "beam_profile_through_sample"
    direct_beam_dir.mkdir(parents=True, exist_ok=True)
    sample_beam_dir.mkdir(parents=True, exist_ok=True)

    root_im_craw = mini_dataset.repetition_dir / "im_craw.nxs"
    root_master = mini_dataset.repetition_dir / "eiger_1_master.h5"
    direct_im_craw = direct_beam_dir / "im_craw.nxs"
    direct_master = direct_beam_dir / "eiger_2_master.h5"
    sample_im_craw = sample_beam_dir / "im_craw.nxs"
    sample_master = sample_beam_dir / "eiger_3_master.h5"
    config_file = mini_dataset.defaults.translator_template_dir / "BAM_new_MOUSE_xenocs_translator_configuration.yaml"
    output_file = (
        mini_dataset.repetition_dir
        / f"MOUSE_{mini_dataset.ymd}_{mini_dataset.batch_num}_{mini_dataset.repetition}_step_1.nxs"
    )

    for path in (
        root_im_craw,
        root_master,
        direct_im_craw,
        direct_master,
        sample_im_craw,
        sample_master,
        config_file,
        output_file,
    ):
        path.write_bytes(b"x")

    os.utime(root_im_craw, (100, 100))
    os.utime(root_master, (100, 100))
    os.utime(direct_im_craw, (100, 100))
    os.utime(direct_master, (100, 100))
    os.utime(sample_im_craw, (100, 100))
    os.utime(sample_master, (100, 100))
    os.utime(config_file, (100, 100))
    os.utime(output_file, (200, 200))

    assert (
        processstep_translator_step_1.can_run(
            mini_dataset.repetition_dir,
            mini_dataset.defaults,
            logbook_reader=None,
            logger=mini_dataset.defaults.logger,
        )
        is False
    )
