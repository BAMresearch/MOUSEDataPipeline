from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from defaults_carrier import DefaultsCarrier


def _write_logbook_xlsx(path: Path, ymd: str, batch_num: int, proposal_id: str, sample_id: int, sampos: str):
    logbook_df = pd.DataFrame(
        [
            {
                "converttoscript": 1,
                "date": pd.Timestamp(ymd),
                "Proposal": proposal_id,
                "sampleid": sample_id,
                "User": "Test User",
                "batchnum": batch_num,
                "matrixfraction": 0.9,
                "samplethickness": 0.001,
                "sampos": sampos,
                "protocol": "test-protocol",
                "procpipeline": "test-pipeline",
                "notes": "test-note",
                "bgdate": pd.Timestamp(ymd),
                "bgnumber": batch_num,
                "dbgdate": pd.NaT,
                "dbgnumber": pd.NA,
            }
        ]
    )
    environments_df = pd.DataFrame(
        [
            {
                "sampos": sampos,
                "xsam": 1.5,
                "ysam": 0.0,
            }
        ]
    )

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        logbook_df.to_excel(writer, sheet_name="Sheet1", index=False, startrow=2)
        environments_df.to_excel(writer, sheet_name="Sample Environments", index=False, startrow=2)


def _write_project_xlsx(path: Path, proposal_id: str, sample_id: int):
    project_info_df = pd.DataFrame(
        {
            "Field": ["proposal", "name", "organisation", "email", "title", "what"],
            "Value": [
                proposal_id,
                "Project Owner",
                "BAM",
                "owner@example.com",
                "Fixture Project",
                "Fixture project description",
            ],
        }
    )
    sample_info_df = pd.DataFrame(
        [
            {
                "sampleId": sample_id,
                "sampleName": "Fixture Sample",
                "componentId": "matrix",
                "componentName": "Matrix",
                "composition": "H2O",
                "density": 1.0,
                "volFrac": 1.0,
                "massFrac": 1.0,
                "componentConnection": "",
                "componentConnectedTo": "",
            }
        ]
    )

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        project_info_df.to_excel(writer, sheet_name="Project_Info", index=False)
        sample_info_df.to_excel(writer, sheet_name="Sample_Info", index=False, startrow=2)


@pytest.fixture
def mini_dataset(tmp_path: Path):
    ymd = "20260311"
    batch_num = 2
    repetition = 0
    proposal_id = "2026001"
    sample_id = 1
    sampos = "Cu S1"

    vsi_root = tmp_path / "vsi"
    saxs_dir = vsi_root / "Measurements" / "SAXS002"
    post_translation_dir = tmp_path / "post_translation"
    translator_template_dir = tmp_path / "translator_templates"
    data_dir = tmp_path / "data"
    masks_dir = data_dir / "Masks"
    stacker_config_dir = data_dir / "StackerConfigurations"
    stacker_config_file = stacker_config_dir / "stacker_config.yaml"
    projects_dir = tmp_path / "projects"
    logbooks_dir = tmp_path / "logbooks"

    for path in (
        vsi_root,
        saxs_dir,
        post_translation_dir,
        translator_template_dir,
        data_dir,
        masks_dir,
        stacker_config_dir,
        projects_dir,
        logbooks_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)

    year_dir = projects_dir / proposal_id[:4]
    year_dir.mkdir(parents=True, exist_ok=True)
    project_file = year_dir / f"{proposal_id}_fixture.xlsx"
    logbook_file = logbooks_dir / "fixture_logbook.xlsx"

    _write_project_xlsx(project_file, proposal_id, sample_id)
    _write_logbook_xlsx(logbook_file, ymd, batch_num, proposal_id, sample_id, sampos)
    stacker_config_file.write_text("{}", encoding="utf-8")

    repetition_dir = data_dir / ymd[:4] / ymd / f"{ymd}_{batch_num}_{repetition}"
    repetition_dir.mkdir(parents=True, exist_ok=True)
    output_file = repetition_dir / f"MOUSE_{ymd}_{batch_num}_{repetition}.nxs"
    with h5py.File(output_file, "w"):
        pass

    defaults = DefaultsCarrier(
        vsi_root=vsi_root,
        saxs_dir=saxs_dir,
        post_translation_dir=post_translation_dir,
        translator_template_dir=translator_template_dir,
        data_dir=data_dir,
        masks_dir=masks_dir,
        projects_dir=projects_dir,
        logbook_file=logbook_file,
        stacker_config_file=stacker_config_file,
        logging_level="INFO",
        profile_steps=True,
    )

    return SimpleNamespace(
        defaults=defaults,
        ymd=ymd,
        batch_num=batch_num,
        repetition=repetition,
        repetition_dir=repetition_dir,
        output_file=output_file,
        project_file=project_file,
        logbook_file=logbook_file,
    )
