import logging
import subprocess
from pathlib import Path
from typing import Optional, Tuple
import attrs
from defaults_carrier import DefaultsCarrier, load_config_from_yaml
from checkers import processing_possible
from YMD_class import YMD
from logbook2mouse.logbook_reader import Logbook2MouseReader


@attrs.define
class DirectoryProcessor:
    """
    A class to manage and execute directory processing tasks.
    """
    defaults: DefaultsCarrier = attrs.field(validator=attrs.validators.instance_of(DefaultsCarrier))
    logbook_reader: Logbook2MouseReader = attrs.field(init=False, default=None)
    logger: logging.Logger = attrs.field(init=False, default=None)
    
    def __attrs_post_init__(self):
        """
        Post-initialization setup.
        """
        self.logger = self.defaults.logger
        self.logger.debug(f"Initializing {self.__class__.__name__}...")
        # Initialize the logbook reader
        self.logbook_reader = Logbook2MouseReader(
            self.defaults.logbook_file, 
            project_base_path=self.defaults.projects_dir
        )

    def process_directory(
        self, 
        single_dir: Optional[Path] = None, 
        ymd: Optional[str] = None, 
        batch: Optional[int] = None, 
        repetition: Optional[int] = None
    ):
        """
        Processes a single repetition directory through a sequence of steps.
        """
        # Parse input arguments and get the directory path
        single_dir, ymd, batch, repetition = self._resolve_directory(
            single_dir=single_dir, ymd=ymd, batch=batch, repetition=repetition
        )

        # Validate processing preconditions
        if not processing_possible(single_dir):
            print(f"Processing not possible for {single_dir}")
            return

        print(f"Processing directory: {single_dir}")

        # Step 1: Locate Eiger file
        eiger_file = self._locate_eiger_file(single_dir)
        if not eiger_file:
            print(f"No Eiger file found in {single_dir}")
            return

        # Step 2: Define paths and run subprocess
        self._run_processing_pipeline(single_dir, ymd, batch, repetition, eiger_file)

    def _resolve_directory(
        self, 
        single_dir: Optional[Path], 
        ymd: Optional[str], 
        batch: Optional[int], 
        repetition: Optional[int]
    ) -> Tuple[Path, YMD, int, int]:
        """
        Resolves and validates the input arguments to determine the directory path and metadata.
        """
        if single_dir:
            # Generate YMD, batch, and repetition from single_dir
            assert single_dir.is_dir(), f"Provided path is not an existing directory: {single_dir}"
            ymd, batch, repetition = self._extract_metadata_from_path(single_dir)
        else:
            # Ensure all required metadata is provided
            assert ymd and batch and repetition, "Either single_dir or ymd, batch, and repetition must be provided."
            ymd = YMD(ymd)
            batch = int(batch)
            repetition = int(repetition)
            single_dir = self._get_directory_path(ymd, batch, repetition)
            assert single_dir.is_dir(), f"Directory (generated from supplied ymd, batch and repetition) does not exist: {single_dir}"

        return single_dir, ymd, batch, repetition

    def _extract_metadata_from_path(self, dir_path: Path) -> Tuple[YMD, int, int]:
        """
        Extracts YMD, batch, and repetition metadata from a directory path.
        """
        last_path = dir_path.parts[-1]
        parts = last_path.split('_')
        assert len(parts) == 3, f"Provided path does not match the expected format: {dir_path}"
        ymd, batch, repetition = parts
        return YMD(ymd), int(batch), int(repetition)

    def _get_directory_path(self, ymd: YMD, batch: int, repetition: int) -> Path:
        """
        Constructs the directory path from YMD, batch, and repetition.
        """
        return self.defaults.data_dir / ymd.get_year() / f"{ymd}_{batch}_{repetition}"

    def _run_subprocess(self, input_file: Path, output_file: Path):
        """
        Executes the translator script as a subprocess.
        """
        cmd = [
            'python3', '-m', self.translator_script,
            '-C', str(self.translator_config),
            '-I', str(input_file),
            '-O', str(output_file)
        ]
        try:
            subprocess.run(cmd, check=True)
            print(f"Subprocess completed successfully for {input_file}")
        except subprocess.CalledProcessError as e:
            print(f"Error during subprocess execution: {e}")
            raise


def main():
    """
    Main entry point for the processing script.
    """
    # Parse command-line arguments (or adjust as needed)
    import argparse

    parser = argparse.ArgumentParser(description="Process directories using DirectoryProcessor.")
    parser.add_argument('--config', type=str, required=True, help="Path to the configuration file.")
    parser.add_argument('--single_dir', type=str, help="Path to a single repetition directory.")
    parser.add_argument('--ymd', type=str, help="YMD string (if not using single_dir).")
    parser.add_argument('--batch', type=int, help="Batch number (if not using single_dir).")
    parser.add_argument('--repetition', type=int, help="Repetition number (if not using single_dir).")

    args = parser.parse_args()

    # Load DefaultsCarrier
    defaults = DefaultsCarrier(**load_config_from_yaml(args.config))

    # Instantiate DirectoryProcessor
    processor = DirectoryProcessor(
        defaults=defaults,
    )

    # Process a single directory
    processor.process_directory(
        single_dir=Path(args.single_dir) if args.single_dir else None,
        ymd=args.ymd,
        batch=args.batch,
        repetition=args.repetition
    )


if __name__ == '__main__':
    main()
