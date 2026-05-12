import concurrent.futures
import contextvars
import hashlib
import importlib
import logging
import threading
from pathlib import Path
from time import perf_counter
from typing import List, Optional, Tuple

import attrs

from defaults_carrier import DefaultsCarrier, load_config_from_yaml
from logbook_support import LogbookReaderLike, build_logbook_reader
from utilities import processed_file_scope
from YMD_class import YMD, extract_metadata_from_path

STEP_PRESETS: dict[str, list[str]] = {
    "preprocess": [
        "processstep_translator_step_1",
        "processstep_translator_step_2",
        "processstep_average_to_counts",
        "processstep_cleanup_files",
        "processstep_add_mask_file",
        "processstep_metadata_update",
        "processstep_determine_beam_center",
        "processstep_make_beam_mask",
        "processstep_calc_beam_flux_and_transmissions",
        "processstep_calc_beam_shape_info",
        "processstep_add_background_files",
        "processstep_transmission_correction_factor_propagator",
        "processstep_apply_transmission_correction_factor",
        "processstep_thickness_from_absorption",
        "processstep_transmission_thickness_flux_table",
    ],
    "stackonly": [
        "processstep_stacker",
    ],
}


def discover_process_steps() -> list[str]:
    processstep_dir = Path(__file__).resolve().parent
    return sorted(path.stem for path in processstep_dir.glob("processstep_*.py") if path.stem != "processstep_template")


def resolve_requested_steps(steps: list[str] | None, step_preset: str | None) -> list[str]:
    if step_preset and steps:
        raise ValueError("Use either --steps or --step-preset, not both.")
    if step_preset:
        return STEP_PRESETS[step_preset]
    if steps:
        return steps
    raise ValueError("Specify processing steps with --steps or choose a --step-preset.")


@attrs.define
class DirectoryProcessor:
    """
    A class to manage and execute directory processing tasks using modular steps.
    """

    defaults: DefaultsCarrier = attrs.field(validator=attrs.validators.instance_of(DefaultsCarrier))
    logbook_reader: LogbookReaderLike | None = attrs.field(init=False, default=None)
    logger: logging.Logger = attrs.field(init=False, default=None)
    steps: List[str] = attrs.field(factory=list)  # List of processing step module names
    require_complete: bool = attrs.field(default=False)
    complete_marker: str = attrs.field(default="COMPLETE", converter=str)
    _directory_loggers: dict[Path, logging.Logger] = attrs.field(init=False, factory=dict, repr=False)
    _logger_lock: threading.Lock = attrs.field(init=False, factory=threading.Lock, repr=False)

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

    def _get_parallel_workers(self) -> int | None:
        return self.defaults.parallel_workers

    def _get_directory_logger(self, dir_path: Path) -> logging.Logger:
        if not self.defaults.log_per_datafile:
            return self.logger

        resolved_dir = dir_path.resolve()
        with self._logger_lock:
            existing_logger = self._directory_loggers.get(resolved_dir)
            if existing_logger is not None:
                return existing_logger

            ymd, batch, repetition = extract_metadata_from_path(resolved_dir)
            log_file = resolved_dir / f"MOUSE_{ymd}_{batch}_{repetition}.processing.log"
            logger_name = f"{self.logger.name}.dir.{hashlib.sha1(str(resolved_dir).encode('utf-8')).hexdigest()[:12]}"
            directory_logger = logging.getLogger(logger_name)
            directory_logger.setLevel(logging.DEBUG)
            directory_logger.propagate = True

            resolved_log_file = log_file.resolve()
            if not any(
                isinstance(handler, logging.FileHandler) and Path(handler.baseFilename) == resolved_log_file
                for handler in directory_logger.handlers
            ):
                directory_logger.addHandler(self.defaults.build_file_handler(resolved_log_file, level=logging.DEBUG))

            self._directory_loggers[resolved_dir] = directory_logger
            return directory_logger

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
            directory_logger = self._get_directory_logger(single_dir)

            directory_logger.info(f"Starting processing for directory: {single_dir}")

            for step_name in self.steps:
                self._run_processing_step(step_name, single_dir, ymd, batch, repetition)

            directory_logger.info(f"Completed processing for directory: {single_dir}, with steps: {self.steps}")
            self._log_profile(
                "process_directory dir=%s elapsed=%.3fs",
                single_dir,
                perf_counter() - process_started_at,
            )

        except Exception as e:
            error_logger = (
                self._get_directory_logger(single_dir)
                if single_dir is not None and single_dir.is_dir()
                else self.logger
            )
            error_logger.error(f"Error processing directory: {single_dir}. Exception: {e}")
            raise

    def process_batch(
        self,
        ymd: str,
        batch: int,
        parallel: bool = False,
        require_complete: bool | None = None,
        complete_marker: str | None = None,
    ):
        batch_started_at = perf_counter()
        ymd = YMD(ymd)
        directories = self._get_all_repetitions_directories(ymd, batch)
        require_complete = self.require_complete if require_complete is None else require_complete
        complete_marker = self.complete_marker if complete_marker is None else str(complete_marker)
        if require_complete:
            self._validate_complete_marker(complete_marker)
            original_directory_count = len(directories)
            directories, skipped_directories = self._split_complete_repetition_directories(
                directories,
                complete_marker,
            )
            self.logger.info(
                "Batch snapshot for %s batch %s requires marker %s: selected %d of %d repetition directories.",
                ymd,
                batch,
                complete_marker,
                len(directories),
                original_directory_count,
            )
            if skipped_directories:
                self.logger.info(
                    "Skipping repetition directories without %s at batch startup: %s",
                    complete_marker,
                    ", ".join(str(directory) for directory in skipped_directories),
                )
        else:
            self.logger.info(
                "Batch snapshot for %s batch %s selected %d repetition directories.",
                ymd,
                batch,
                len(directories),
            )

        with processed_file_scope(directories if require_complete else None):
            for step_name in self.steps:
                step_started_at = perf_counter()
                step_module = importlib.import_module(step_name)
                if not getattr(step_module, "can_process_repetitions_in_parallel", False):
                    self.logger.info(f"{step_module} cannot process repetitions in parallel.")
                    # Run this step sequentially
                    for directory in directories:
                        self._run_processing_step(step_name, directory, ymd, batch, None)
                elif parallel:
                    self.logger.info(
                        "%s will process repetitions in parallel with workers=%s.",
                        step_module,
                        self._get_parallel_workers() if self._get_parallel_workers() is not None else "default",
                    )
                    # Run this step in parallel
                    self._run_steps_in_parallel(step_name, directories, ymd, batch)
                else:
                    self.logger.info(f"{step_module} can process repetitions in parallel, but not requested.")
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
        parallel_workers = self._get_parallel_workers()
        with concurrent.futures.ThreadPoolExecutor(max_workers=parallel_workers) as executor:
            futures = [
                executor.submit(
                    contextvars.copy_context().run,
                    self._run_processing_step,
                    step_name,
                    directory,
                    ymd,
                    batch,
                    None,
                )
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
        return sorted(
            base_dir.glob(f"{ymd}_{batch}_*/"),
            key=lambda directory: extract_metadata_from_path(directory)[2],
        )

    def _validate_complete_marker(self, complete_marker: str):
        marker_path = Path(complete_marker)
        if not complete_marker or marker_path.is_absolute():
            raise ValueError("--complete-marker must be a relative file name.")

    def _split_complete_repetition_directories(
        self,
        directories: List[Path],
        complete_marker: str,
    ) -> tuple[List[Path], List[Path]]:
        complete_directories = []
        incomplete_directories = []
        for directory in directories:
            if (directory / complete_marker).is_file():
                complete_directories.append(directory)
            else:
                incomplete_directories.append(directory)
        return complete_directories, incomplete_directories

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
        step_logger = self._get_directory_logger(dir_path)
        try:
            module = importlib.import_module(step_name)
            logbook_reader = self._get_logbook_reader() if getattr(module, "requires_logbook_reader", False) else None
            if hasattr(module, "can_run") and hasattr(module, "run"):
                can_run_started_at = perf_counter()
                should_run = module.can_run(dir_path, self.defaults, logbook_reader, step_logger)
                can_run_elapsed = perf_counter() - can_run_started_at
                if should_run:
                    step_logger.info(f"Running step: {step_name}")
                    run_started_at = perf_counter()
                    module.run(dir_path, self.defaults, logbook_reader, step_logger)
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
                    step_logger.info(f"Step skipped: {step_name}")
                    self._log_profile(
                        "step=%s dir=%s skipped can_run=%.3fs total=%.3fs",
                        step_name,
                        dir_path,
                        can_run_elapsed,
                        perf_counter() - step_started_at,
                    )
            else:
                step_logger.error(f"Module {step_name} must define 'can_run' and 'run' functions.")
        except Exception as e:
            step_logger.error(f"Error in step {step_name}: {e}")
            raise


def build_arg_parser():
    """
    Build the command line parser for the directory processor.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Process MOUSE repetition directories or full batches using modular processing steps.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Examples:\n"
            "  mouse-directory-processor --single_dir /path/to/20260311_1_0 --step-preset preprocess\n"
            "  mouse-directory-processor --ymd 20260311 --batch 1 --parallel --step-preset preprocess\n"
            "  mouse-directory-processor --list-steps\n"
            "  mouse-directory-processor --list-step-presets"
        ),
    )
    parser.add_argument(
        "--config",
        type=str,
        default="MOUSE_settings.yaml",
        help="Path to the configuration yaml file (contains paths).",
    )
    parser.add_argument("--single_dir", type=str, help="Path to a single repetition directory to process.")
    parser.add_argument("--ymd", type=str, help="YMD string (if not using single_dir).")
    parser.add_argument("--batch", type=int, help="Batch number (if not using single_dir).")
    parser.add_argument("--repetition", type=int, help="Repetition number (if not using single_dir).")
    parser.add_argument("--steps", type=str, nargs="+", help="Explicit list of processing step module names.")
    parser.add_argument(
        "--step-preset",
        choices=sorted(STEP_PRESETS),
        help="Named processing-step preset for common workflows.",
    )
    parser.add_argument("--list-steps", action="store_true", help="List discovered processstep modules and exit.")
    parser.add_argument(
        "--list-step-presets",
        action="store_true",
        help="List built-in step presets and exit.",
    )
    parser.add_argument("--parallel", action="store_true", help="Enable parallel processing of repetitions.")
    parser.add_argument(
        "--parallel-workers",
        type=int,
        help="Maximum number of worker threads for parallel repetition processing. Defaults to ThreadPoolExecutor behavior.",
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Only process repetition directories that contain the completion marker at batch startup.",
    )
    parser.add_argument(
        "--complete-marker",
        default="COMPLETE",
        help="Completion marker file name to require inside each repetition directory when --require-complete is used.",
    )
    return parser


def main(argv: list[str] | None = None):
    """
    Main entry point for the processing script.
    """
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.list_steps:
        for step_name in discover_process_steps():
            print(step_name)
        return

    if args.list_step_presets:
        for preset_name, steps in STEP_PRESETS.items():
            print(f"{preset_name}:")
            for step_name in steps:
                print(f"  {step_name}")
        return

    try:
        requested_steps = resolve_requested_steps(args.steps, args.step_preset)
    except ValueError as exc:
        parser.error(str(exc))

    defaults = DefaultsCarrier(**load_config_from_yaml(args.config))
    if args.parallel_workers is not None:
        if args.parallel_workers <= 0:
            parser.error("--parallel-workers must be a positive integer.")
        defaults.parallel_workers = args.parallel_workers
    processor = DirectoryProcessor(defaults=defaults, steps=requested_steps)

    if args.single_dir is not None or args.repetition is not None:
        processor.process_directory(
            single_dir=Path(args.single_dir) if args.single_dir else None,
            ymd=args.ymd,
            batch=args.batch,
            repetition=args.repetition,
        )
        return

    if args.ymd is None or args.batch is None:
        parser.error("Batch processing requires --ymd and --batch.")
    if args.repetition is not None:
        parser.error("--repetition only applies to single-directory processing.")

    processor.process_batch(
        ymd=args.ymd,
        batch=args.batch,
        parallel=args.parallel,
        require_complete=args.require_complete,
        complete_marker=args.complete_marker,
    )


if __name__ == "__main__":
    main()
