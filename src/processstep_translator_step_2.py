from pathlib import Path
import subprocess
from YMD_class import extract_metadata_from_path
from checkers import len_files_in_path, processing_possible
from defaults_carrier import DefaultsCarrier
from logbook2mouse.logbook_reader import Logbook2MouseReader
import logging


def can_run(dir_path: Path, defaults: DefaultsCarrier, logbook_reader: Logbook2MouseReader, logger: logging.Logger) -> bool:
    """
    Checks if the translator step should run. Besides the base files, we don't need anything...
    """
    ymd, batch, repetition = extract_metadata_from_path(dir_path)
    step_1_file = dir_path / f'mouse_{ymd}_step_1.nxs'
    if not step_1_file.is_file():
        logger.info(f"Step 2 translation not possible for {dir_path}, step 1 result file missing at: {step_1_file}")
        return False

    return True

def run(dir_path: Path, defaults: DefaultsCarrier, logbook_reader: Logbook2MouseReader, logger: logging.Logger):
    """
    Executes the first translator processing step.
    """
    ymd, batch, repetition = extract_metadata_from_path(dir_path)
    try:

        # encode: python3 -m HDF5Translator -C BAM_new_MOUSE_dectris_adder_configuration.yaml -I ./20250101_17_0/eiger_3_master.h5 -T ./20250101_17_0/testBAM.nxs -O ./20250101_17_0/testBAM_Dadd.nxs -d

        input_file = next(dir_path.glob('eiger_*_master.h5'), None)
        template_file = dir_path / f'mouse_{ymd}_step_1.nxs'
        output_file = dir_path / f'mouse_{ymd}_step_2.nxs'
        cmd = [
            'python3', '-m', 'HDF5Translator',
            '-C', str(defaults.translator_template_dir / 'BAM_new_MOUSE_dectris_adder_configuration.yaml'),
            '-T', str(template_file),
            '-I', str(input_file),
            '-O', str(output_file), 
            '-d'
        ]
        logger.info(f"Starting translator step 2 for {input_file}")
        subprocess.run(cmd, check=True)
        logger.info(f"Completed translator step 2 for {input_file}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error during translator subprocess: {e}")
        raise
