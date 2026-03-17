from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Protocol, Sequence

from mouse_logbook import Logbook2MouseReader


class LogbookReaderLike(Protocol):
    entries: Sequence[Any]


def build_logbook_reader(
    logbook_file: Path,
    project_base_path: Path,
    logger: logging.Logger | None = None,
) -> LogbookReaderLike:
    reader = Logbook2MouseReader(
        logbook_file,
        project_base_path=project_base_path,
    )
    if logger:
        logger.info("Using mouse_logbook for logbook access.")
    return reader
