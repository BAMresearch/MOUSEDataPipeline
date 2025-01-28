import os
import subprocess
import argparse
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileSystemEvent
import logging
import time

from YMD_class import extract_metadata_from_path
from directory_processor import DirectoryProcessor
from defaults_carrier import DefaultsCarrier, load_config_from_yaml
from checkers import processing_possible, already_processed

logging.basicConfig(level=logging.INFO)


class WatcherFileSystemEventHandler(FileSystemEventHandler):
    def __init__(self, processor: DirectoryProcessor, logger: logging.Logger):
        self.processor = processor
        self.logger = logger

    def on_created(self, event: FileSystemEvent):
        if event.is_directory:
            self.logger.info(f"New directory detected: {event.src_path}")
            dir_path = Path(event.src_path)
            
            if already_processed(dir_path):
                self.logger.info(f"Directory already processed: {dir_path}")
                return

            # Wait for files to stabilize
            timeout = 600  # 10 minutes
            stability_check_interval = 10

            start_time = time.time()
            while time.time() - start_time < timeout:
                if processing_possible(dir_path):
                    self.logger.info(f"Processing possible for {dir_path}")
                    ymd, batch, repetition = extract_metadata_from_path(dir_path)
                    self.processor.process_directory(
                        single_dir=dir_path,
                        ymd=ymd,
                        batch=batch,
                        repetition=repetition
                    )
                    return
                else:
                    self.logger.info(f"Waiting for directory to stabilize: {dir_path}")
                    time.sleep(stability_check_interval)

            self.logger.warning(f"Timed out waiting for stabilization of directory: {dir_path}")


def main():
    parser = argparse.ArgumentParser(description="Watch a directory tree for changes and process new directories.")
    parser.add_argument(
        "input_path",
        help="The top-level directory path to monitor."
    )
    parser.add_argument('--config', type=str, required=True, help="Path to the configuration yaml file.")

    args = parser.parse_args()

    config = load_config_from_yaml(args.config)
    defaults = DefaultsCarrier(**config)
    logger = logging.getLogger(__name__)

    steps = [
        'processstep_translator_step_1',
        'processstep_translator_step_2',
        'processstep_beamanalysis',
        'processstep_cleanup_files',
        'processstep_add_mask_file',
        'processstep_metadata_update',
        'processstep_add_background_files',
        'processstep_thickness_from_absorption',
        'processstep_transmission_thickness_flux_table',
        'processstep_stacker'
    ]

    processor = DirectoryProcessor(defaults=defaults, steps=steps)

    event_handler = WatcherFileSystemEventHandler(processor=processor, logger=logger)
    observer = Observer()
    observer.schedule(event_handler, path=args.input_path, recursive=True)

    try:
        observer.start()
        logger.info(f"Started watching directory: {args.input_path}")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    main()