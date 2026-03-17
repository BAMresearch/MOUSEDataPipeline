import concurrent.futures
import importlib
import logging
from pathlib import Path
from time import perf_counter
from typing import List, Optional, Tuple

import attrs

from defaults_carrier import DefaultsCarrier, load_config_from_yaml
from logbook_support import LogbookReaderLike, build_logbook_reader
from YMD_class import YMD, extract_metadata_from_path


@attrs.define
class DirectoryProcessor:
    """
    A class to manage and execute directory processing tasks using modular steps.
    """

    defaults: DefaultsCarrier = attrs.field(validator=attrs.validators.instance_of(DefaultsCarrier))
    logbook_reader: LogbookReaderLike | None = attrs.field(init=False, default=None)
    logger: logging.Logger = attrs.field(init=False, default=None)
    steps: List[str] = attrs.field(factory=list)  # List of processing step module names

    def __attrs_post_init__(self):
        """
        Post-initialization setup.
        """
        self.logger = self.defaults.logger
        self.logger.setLevel(self.defaults.logging_level.upper())
        self.logger.debug(f"Initializing {self.__class__.__name__}...")

    def _log_profile(self, message: str, *args):
        if self.defaults.profile_steps:
            self.logger.info("PROFILE " + message, *args)

    def _get_logbook_reader(self) -> LogbookReaderLike:
        if self.logbook_reader is None:
            started_at = perf_counter()
            self.logbook_reader = build_logbook_reader(
                self.defaults.logbook_file,
                self.defaults.projects_dir,
                logger=self.logger,
            )
            self._log_profile("logbook_reader_init elapsed=%.3fs", perf_counter() - started_at)
        return self.logbook_reader

    def process_directory(
        self,
        single_dir: Optional[Path] = None,
        ymd: Optional[str] = None,
        batch: Optional[int] = None,
        repetition: Optional[int] = None,
    ):
        """
        Processes a single repetition directory through a sequence of modular steps.
        """
        process_started_at = perf_counter()
        try:
            single_dir, ymd, batch, repetition = self._resolve_directory(
                single_dir=single_dir, ymd=ymd, batch=batch, repetition=repetition
            )

            self.logger.info(f"Starting processing for directory: {single_dir}")

            for step_name in self.steps:
                self._run_processing_step(step_name, single_dir, ymd, batch, repetition)

            self.logger.info(f"Completed processing for directory: {single_dir}, with steps: {self.steps}")
            self._log_profile(
                "process_directory dir=%s elapsed=%.3fs",
                single_dir,
                perf_counter() - process_started_at,
            )

        except Exception as e:
            self.logger.error(f"Error processing directory: {single_dir}. Exception: {e}")
            raise

    def process_batch(self, ymd: str, batch: int, parallel: bool = False):
        batch_started_at = perf_counter()
        ymd = YMD(ymd)
        directories = self._get_all_repetitions_directories(ymd, batch)

        for step_name in self.steps:
            step_started_at = perf_counter()
            step_module = importlib.import_module(step_name)
            if not getattr(step_module, "can_process_repetitions_in_parallel", False):
                logging.info(f"{step_module} cannot process repetitions in parallel.")
                # Run this step sequentially
                for directory in directories:
                    self._run_processing_step(step_name, directory, ymd, batch, None)
            elif parallel:
                logging.info(f"using {step_module} to process repetitions in parallel.")
                # Run this step in parallel
                self._run_steps_in_parallel(step_name, directories, ymd, batch)
            else:
                logging.info(f"{step_module} can process repetitions in parallel, but not requested.")
                for directory in directories:
                    self._run_processing_step(step_name, directory, ymd, batch, None)
            self._log_profile(
                "batch_step step=%s dirs=%d parallel=%s elapsed=%.3fs",
                step_name,
                len(directories),
                parallel and getattr(step_module, "can_process_repetitions_in_parallel", False),
                perf_counter() - step_started_at,
            )
        self._log_profile(
            "process_batch ymd=%s batch=%s dirs=%d elapsed=%.3fs",
            ymd,
            batch,
            len(directories),
            perf_counter() - batch_started_at,
        )

    def _run_steps_in_parallel(self, step_name: str, directories: List[Path], ymd: YMD, batch: int):
        parallel_started_at = perf_counter()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(self._run_processing_step, step_name, directory, ymd, batch, None)
                for directory in directories
            ]
            for future in concurrent.futures.as_completed(futures):
                future.result()
        self._log_profile(
            "parallel_step step=%s dirs=%d elapsed=%.3fs",
            step_name,
            len(directories),
            perf_counter() - parallel_started_at,
        )

    def _get_all_repetitions_directories(self, ymd: YMD, batch: int) -> List[Path]:
        """
        Returns the list of Path objects for all repetition directories in a batch.
        """
        base_dir = self.defaults.data_dir / ymd.get_year() / str(ymd)
        return list(base_dir.glob(f"{ymd}_{batch}_*/"))

    def _resolve_directory(
        self, single_dir: Optional[Path], ymd: Optional[str], batch: Optional[int], repetition: Optional[int]
    ) -> Tuple[Path, YMD, int, int]:
        """
        Resolves and validates the input arguments to determine the directory path and metadata.
        """
        if single_dir:
            if not single_dir.is_dir():
                raise FileNotFoundError(f"Provided path is not an existing directory: {single_dir}")
            ymd, batch, repetition = extract_metadata_from_path(single_dir)
        else:
            if ymd is None or batch is None or repetition is None:
                raise ValueError("Either single_dir or ymd, batch, and repetition must be provided.")
            ymd = YMD(ymd)
            batch = int(batch)
            repetition = int(repetition)
            single_dir = self._get_directory_path(ymd, batch, repetition)
            if not single_dir.is_dir():
                raise FileNotFoundError(f"Directory does not exist: {single_dir}")

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
        step_started_at = perf_counter()
        try:
            module = importlib.import_module(step_name)
            logbook_reader = self._get_logbook_reader() if getattr(module, "requires_logbook_reader", False) else None
            if hasattr(module, "can_run") and hasattr(module, "run"):
                can_run_started_at = perf_counter()
                should_run = module.can_run(dir_path, self.defaults, logbook_reader, self.logger)
                can_run_elapsed = perf_counter() - can_run_started_at
                if should_run:
                    self.logger.info(f"Running step: {step_name}")
                    run_started_at = perf_counter()
                    module.run(dir_path, self.defaults, logbook_reader, self.logger)
                    run_elapsed = perf_counter() - run_started_at
                    self._log_profile(
                        "step=%s dir=%s can_run=%.3fs run=%.3fs total=%.3fs",
                        step_name,
                        dir_path,
                        can_run_elapsed,
                        run_elapsed,
                        perf_counter() - step_started_at,
                    )
                else:
                    self.logger.info(f"Step skipped: {step_name}")
                    self._log_profile(
                        "step=%s dir=%s skipped can_run=%.3fs total=%.3fs",
                        step_name,
                        dir_path,
                        can_run_elapsed,
                        perf_counter() - step_started_at,
                    )
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
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the configuration yaml file (contains paths)."
    )
    parser.add_argument("--single_dir", type=str, help="Path to a single repetition directory to process.")
    parser.add_argument("--ymd", type=str, help="YMD string (if not using single_dir).")
    parser.add_argument("--batch", type=int, help="Batch number (if not using single_dir).")
    parser.add_argument("--repetition", type=int, help="Repetition number (if not using single_dir).")
    parser.add_argument("--steps", type=str, nargs="+", help="List of processing step module names.", required=True)
    parser.add_argument("--parallel", action="store_true", help="Enable parallel processing of repetitions.")

    args = parser.parse_args()

    defaults = DefaultsCarrier(**load_config_from_yaml(args.config))
    processor = DirectoryProcessor(defaults=defaults, steps=args.steps)

    if args.single_dir is not None or args.repetition is not None:
        processor.process_directory(
            single_dir=Path(args.single_dir) if args.single_dir else None,
            ymd=args.ymd,
            batch=args.batch,
            repetition=args.repetition,
        )
    else:
        if args.ymd is None or args.batch is None:
            parser.error("Processing all repetitions requires YMD and batch.")
        processor.process_batch(ymd=args.ymd, batch=args.batch, parallel=args.parallel)


if __name__ == "__main__":
    main()
