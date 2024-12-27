import os
import subprocess
import argparse
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileSystemEvent
import attrs
import logging

logging.basicConfig(level=logging.INFO)

@attrs.define
class DirectoryProcessor(FileSystemEventHandler):
    """
    File Watcher and Processor for HDF5 Translation

    This script uses the `watchdog` library to monitor a directory tree for new
    directories and processes files within those directories. It specifically checks
    for the presence of the `im_craw.nxs` file and ensures that a translation
    process is executed if the `translated.nxs` file does not yet exist. It is designed
    to handle multi-step processing pipelines for HDF5 translation.

    Features:
    - Watches for new directories in a specified directory tree.
    - Triggers processing for directories containing all necessary source files.
    - Utilizes `subprocess` to execute translation commands.

    Prerequisites:
    - `watchdog` library: Install using `pip install watchdog`.

    Functions:
    - process_directory(dir_path): Checks if translations are required and executes them.
    - on_created(event): Responds to new directory creation and starts processing as needed.

    Usage:
    - Define the `processing_dirs` path with the top-level directory to be monitored.
    - Extend `process_directory` to add further processing steps after initial translation.
    - Adjust `subprocess.run` commands for specific translation tool invocation.

    Notes:
    - The script runs indefinitely until interrupted (e.g., by a keyboard interruption).
    - Handles basic setup for watching directory events and processing them.
    - Expand the translation logic and path handling as per your data and requirements.

    """
    processing_directories: Path = attrs.field(converter=Path, validator=[attrs.validators.instance_of(Path)])


    def __attrs_post_init__(self):
        assert self.processing_directories.is_dir(), "The specified directory to watch does not exist."


    def process_directory(self, dir_path:Path):
        if not self.translation_needed(dir_path):
            return

        print(f"Processing directory: {dir_path}")

        eiger_file = next(dir_path.glob('eiger_*_master.h5'), None)
        if not eiger_file:
            print(f"No eiger file found for {dir_path}")
            return

        craw_path = dir_path / 'im_craw.nxs'
        translated_path = dir_path / 'translated.nxs'

        subprocess.run([
            'python3', '-m', 'HDF5Translator',
            '-C', 'data/TranslatorConfigurations/BAM_new_MOUSE_xenocs_translator_configuration.yaml',
            '-I', str(craw_path), '-O', str(translated_path)
        ])

    def on_created(self, event: FileSystemEvent):
        if event.is_directory:
            self.process_directory(Path(event.src_path))

def main():
    parser = argparse.ArgumentParser(description="Watch a directory tree for changes and process new directories.")
    parser.add_argument(
        "input_path",
        help="The top-level directory path to monitor."
    )
    args = parser.parse_args()
    event_handler = DirectoryProcessor(args.input_path)
    observer = Observer()
    observer.schedule(event_handler, path=args.input_path, recursive=True)

    try:
        observer.start()
        while True:
            pass
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    main()