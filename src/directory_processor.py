import importlib
import logging
from pathlib import Path
from typing import Optional, Tuple, List
import attrs
from defaults_carrier import DefaultsCarrier, load_config_from_yaml
from checkers import processing_possible
from YMD_class import YMD, extract_metadata_from_path
from logbook2mouse.logbook_reader import Logbook2MouseReader


@attrs.define
class DirectoryProcessor:
    """
    A class to manage and execute directory processing tasks using modular steps.
    """
    defaults: DefaultsCarrier = attrs.field(validator=attrs.validators.instance_of(DefaultsCarrier))
    logbook_reader: Logbook2MouseReader = attrs.field(init=False, default=None)
    logger: logging.Logger = attrs.field(init=False, default=None)
    steps: List[str] = attrs.field(factory=list)  # List of processing step module names

    def __attrs_post_init__(self):
        """
        Post-initialization setup.
        """
        self.logger = self.defaults.logger
        self.logger.debug(f"Initializing {self.__class__.__name__}...")
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
        Processes a single repetition directory through a sequence of modular steps.
        """
        try:
            single_dir, ymd, batch, repetition = self._resolve_directory(
                single_dir=single_dir, ymd=ymd, batch=batch, repetition=repetition
            )

            self.logger.info(f"Starting processing for directory: {single_dir}")

            for step_name in self.steps:
                self._run_processing_step(step_name, single_dir, ymd, batch, repetition)

            self.logger.info(f"Completed processing for directory: {single_dir}, with steps: {self.steps}")

        except Exception as e:
            self.logger.error(f"Error processing directory: {single_dir}. Exception: {e}")
            raise

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
            assert single_dir.is_dir(), f"Provided path is not an existing directory: {single_dir}"
            ymd, batch, repetition = extract_metadata_from_path(single_dir)
        else:
            assert ymd and batch and repetition, "Either single_dir or ymd, batch, and repetition must be provided."
            ymd = YMD(ymd)
            batch = int(batch)
            repetition = int(repetition)
            single_dir = self._get_directory_path(ymd, batch, repetition)
            assert single_dir.is_dir(), f"Directory does not exist: {single_dir}"

        return single_dir, ymd, batch, repetition

    def _get_directory_path(self, ymd: YMD, batch: int, repetition: int) -> Path:
        """
        Constructs the directory path from YMD, batch, and repetition.
        """
        return self.defaults.data_dir / ymd.get_year() / str(ymd) / f"{ymd}_{batch}_{repetition}"

    def _run_processing_step(self, step_name: str, dir_path: Path, ymd: YMD, batch: int, repetition: int):
        """
        Dynamically loads and runs a processing step module with logging.
        """
        try:
            module = importlib.import_module(step_name)
            if hasattr(module, "can_run") and hasattr(module, "run"):
                if module.can_run(dir_path, self.defaults, self.logbook_reader, self.logger):
                    self.logger.info(f"Running step: {step_name}")
                    module.run(dir_path, self.defaults, self.logbook_reader, self.logger)
                else:
                    self.logger.info(f"Step skipped: {step_name}")
            else:
                self.logger.error(f"Module {step_name} must define 'can_run' and 'run' functions.")
        except Exception as e:
            self.logger.error(f"Error in step {step_name}: {e}")
            raise


def main():
    """
    Main entry point for the processing script.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Process directories using DirectoryProcessor.")
    parser.add_argument('--config', type=str, required=True, help="Path to the configuration yaml file (contains paths).")
    parser.add_argument('--single_dir', type=str, help="Path to a single repetition directory to process.")
    parser.add_argument('--ymd', type=str, help="YMD string (if not using single_dir).")
    parser.add_argument('--batch', type=int, help="Batch number (if not using single_dir).")
    parser.add_argument('--repetition', type=int, help="Repetition number (if not using single_dir).")
    parser.add_argument('--steps', type=str, nargs='+', help="List of processing step module names.", required=True)

    args = parser.parse_args()

    defaults = DefaultsCarrier(**load_config_from_yaml(args.config))
    processor = DirectoryProcessor(
        defaults=defaults,
        steps=args.steps
    )

    processor.process_directory(
        single_dir=Path(args.single_dir) if args.single_dir else None,
        ymd=args.ymd,
        batch=args.batch,
        repetition=args.repetition
    )


if __name__ == '__main__':
    main()
