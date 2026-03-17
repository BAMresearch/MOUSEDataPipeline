from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Protocol, Sequence


class LogbookReaderLike(Protocol):
    entries: Sequence[Any]


def build_logbook_reader(
    logbook_file: Path,
    project_base_path: Path,
    logger: logging.Logger | None = None,
) -> LogbookReaderLike:
    errors: list[str] = []

    for backend_name in ("mouse_logbook", "logbook2mouse"):
        try:
            if backend_name == "mouse_logbook":
                from mouse_logbook import Logbook2MouseReader
            else:
                from logbook2mouse.logbook_reader import Logbook2MouseReader

            reader = Logbook2MouseReader(
                logbook_file,
                project_base_path=project_base_path,
            )
            if logger:
                logger.info("Using %s for logbook access.", backend_name)
            return reader
        except Exception as exc:
            errors.append(f"{backend_name}: {exc}")
            if logger:
                logger.warning("Failed to initialize %s logbook reader: %s", backend_name, exc)

    raise RuntimeError(
        "Unable to initialize any supported logbook reader. "
        + " | ".join(errors)
    )
