from __future__ import annotations

import h5py

import processstep_metadata_update


def _read_scalar_string(dataset) -> str:
    value = dataset[()]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def test_metadata_update_cli_writer_updates_nexus_file(mini_dataset):
    processstep_metadata_update.run(
        mini_dataset.repetition_dir,
        mini_dataset.defaults,
        logbook_reader=None,
        logger=mini_dataset.defaults.logger,
    )

    with h5py.File(mini_dataset.output_file, "r") as h5f:
        assert "/entry1/sample/sampleowner" in h5f
        assert _read_scalar_string(h5f["/entry1/sample/sampleowner"]) == "Project Owner"
        assert "/entry1/proposal/proposalid" in h5f
        assert _read_scalar_string(h5f["/entry1/proposal/proposalid"]) == "2026001"
        assert "/entry1/processing_required_metadata/procpipeline" in h5f
        assert _read_scalar_string(h5f["/entry1/processing_required_metadata/procpipeline"]) == "test-pipeline"
