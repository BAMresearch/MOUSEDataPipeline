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
    step_2_file = dir_path / f'mouse_{ymd}_step_2.nxs'
    if not step_2_file.is_file():
        logger.info(f"Beamanalysis not possible for {dir_path}, step 2 result file missing at: {step_2_file}")
        return False

    return True

def run(dir_path: Path, defaults: DefaultsCarrier, logbook_reader: Logbook2MouseReader, logger: logging.Logger):
    """
    Executes the first translator processing step.
    """
    ymd, batch, repetition = extract_metadata_from_path(dir_path)
    try:

        # encode: 
        # python3 ../../src/tools/post_translation_operation_MOUSE_beamanalysis.py -f 20250101_17_0/testBAM_Dadd.nxs -v -k roi_size=25 image_type="sample_beam"
        # python3 ../../src/tools/post_translation_operation_MOUSE_beamanalysis.py -f 20250101_17_0/testBAM_Dadd.nxs -v -k roi_size=25 image_type="direct_beam"


        input_file = dir_path / f'mouse_{ymd}_step_2.nxs'
        cmd1 = [
            'python3', 'post_translation_operation_MOUSE_beamanalysis.py',
            '-f', str(input_file),
            '-v', 
            '-k', 'roi_size=25', 'image_type="sample_beam"',
        ]
        cmd2 = [
            'python3', 'post_translation_operation_MOUSE_beamanalysis.py',
            '-f', str(input_file),
            '-v', 
            '-k', 'roi_size=25', 'image_type="direct_beam"',
        ]


        logger.info(f"Starting beam analysis for {input_file}")
        result = subprocess.run(cmd1, check=True, capture_output=True, text=True)
        print(result.stdout)
        result = subprocess.run(cmd2, check=True, capture_output=True, text=True)
        print(result.stdout)
        logger.info(f"Completed beam analysis for {input_file}")
    except subprocess.CalledProcessError as e:
        # Print the standard output and standard error
        print("Subprocess failed with stderr:")
        print(e.stderr)
        # Optionally, also print the standard output
        print("Subprocess output was:")
        print(e.stdout)
        logger.error(f"Error during translator subprocess: {e}")
        raise
