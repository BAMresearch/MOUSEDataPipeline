from pathlib import Path
import subprocess
from YMD_class import extract_metadata_from_path
from checkers import processing_possible
from defaults_carrier import DefaultsCarrier
from logbook2mouse.logbook_reader import Logbook2MouseReader
import logging


# Flag indicating whether this process step can be executed in parallel on multiple repetitions
can_process_repetitions_in_parallel = True

def can_run(dir_path: Path, defaults: DefaultsCarrier, logbook_reader: Logbook2MouseReader, logger: logging.Logger) -> bool:
    """
    Checks if the translator step should run. Besides the base files, we don't need anything...
    """
    if not processing_possible(dir_path):
        logger.info(f"Step 1 translation not possible for {dir_path}, required files missing... check DEBUG for details")
        logger.debug(f"Required files missing for step 1 translation not possible for {dir_path}. The following were not found: {processing_possible(dir_path, return_list=True)}")
        return False

    return True

def determine_calibration_configuration(ymd: str):
    """
    Selects the translator configuration / calibration suitable for the given ymd.

    The limits of the ymd range are estimated in some cases.
    Known calibrations and stage position changes:
    - 2026-02-25: forward one position on the kinematic feet
    """
    if int(ymd) >= 20260209:
        return 'BAM_new_MOUSE_xenocs_translator_configuration-gi.yaml'
    elif 20251211 <= int(ymd) < 20260209:
        return 'outdated_20260209/BAM_new_MOUSE_xenocs_translator_configuration-gi.yaml'
    else:
        return 'outdated_20251112/BAM_new_MOUSE_xenocs_translator_configuration-gi.yaml'


def run(dir_path: Path, defaults: DefaultsCarrier, logbook_reader: Logbook2MouseReader, logger: logging.Logger):
    """
    Executes the first translator processing step.
    """
    ymd, batch, repetition = extract_metadata_from_path(dir_path)
    print(dir_path)
    try:

        # encode: python3 -m HDF5Translator -C BAM_new_MOUSE_xenocs_translator_configuration.yaml -I ./20250101_17_0/im_craw.nxs -O ./20250101_17_0/testBAM.nxs -d

        input_file = dir_path / 'im_craw.nxs'
        output_file = dir_path / f'MOUSE_{ymd}_{batch}_{repetition}_step_1.nxs'
        translator_configuration = determine_calibration_configuration(ymd)
        cmd = [
            'python3', '-m', 'HDF5Translator',
            '-C', str(defaults.translator_template_dir / translator_configuration),
            '-I', str(input_file),
            '-O', str(output_file),
            '-d',
            '-v'
        ]
        logger.info(f"Starting translator step 1 for {input_file}")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.debug(result.stdout)
        logger.info(f"Completed translator step 1 for {input_file}")
    except subprocess.CalledProcessError as e:
        # Print the standard output and standard error
        logger.info("Subprocess failed with stderr:")
        logger.info(e.stderr)
        # Optionally, also print the standard output
        logger.info("Subprocess output was:")
        logger.info(e.stdout)
        logger.error(f"Error during translator subprocess: {e}")
        raise
