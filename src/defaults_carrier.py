import logging
from pathlib import Path
from typing import Optional

import attrs
import yaml


# Validators and Converters
def if_not_none_is_path_and_exists(instance, attribute, value):
    """
    Validator to ensure the value is a Path object and the path exists.
    """
    if value is None:
        return
    if not isinstance(value, Path):
        raise ValueError(f"{attribute.name} must be a Path object.")
    if not value.exists():
        raise ValueError(f"{attribute.name} path '{value}' does not exist.")


def convert_to_path_or_none(value):
    """
    Convert value to Path if not None, otherwise return None.
    """
    return Path(value) if value else None


def convert_to_int_or_none(value):
    """
    Convert value to int if not None, otherwise return None.
    """
    return int(value) if value is not None else None


def if_not_none_is_positive_int(instance, attribute, value):
    """
    Validator to ensure the value is a positive integer when provided.
    """
    if value is None:
        return
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{attribute.name} must be a positive integer or None.")


# Configuration Loader
def load_config_from_yaml(file_path: str) -> dict:
    """
    Load configuration from a YAML file.
    """
    try:
        with open(file_path, "r") as yaml_file:
            return yaml.safe_load(yaml_file) or {}
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file '{file_path}' not found.")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML file: {e}")


# DefaultsCarrier Class
@attrs.define
class DefaultsCarrier:
    """
    A class to manage default paths and settings.
    """

    vsi_root: Path = attrs.field(converter=Path, validator=[if_not_none_is_path_and_exists])
    post_translation_dir: Path = attrs.field(converter=Path, validator=[if_not_none_is_path_and_exists])
    translator_template_dir: Path = attrs.field(converter=Path, validator=[if_not_none_is_path_and_exists])

    saxs_dir: Optional[Path] = attrs.field(
        default=None, converter=convert_to_path_or_none, validator=[if_not_none_is_path_and_exists]
    )
    data_dir: Optional[Path] = attrs.field(
        default=None, converter=convert_to_path_or_none, validator=[if_not_none_is_path_and_exists]
    )
    masks_dir: Optional[Path] = attrs.field(
        default=None, converter=convert_to_path_or_none, validator=[if_not_none_is_path_and_exists]
    )
    projects_dir: Optional[Path] = attrs.field(
        default=None, converter=convert_to_path_or_none, validator=[if_not_none_is_path_and_exists]
    )

    logbook_file: Optional[Path] = attrs.field(
        default=None, converter=convert_to_path_or_none, validator=[if_not_none_is_path_and_exists]
    )
    stacker_config_file: Optional[Path] = attrs.field(
        default=None, converter=convert_to_path_or_none, validator=[if_not_none_is_path_and_exists]
    )

    logging_level: str = attrs.field(default="INFO", converter=str)
    profile_steps: bool = attrs.field(default=True)
    log_per_datafile: bool = attrs.field(default=True)
    parallel_workers: Optional[int] = attrs.field(
        default=None,
        converter=convert_to_int_or_none,
        validator=[if_not_none_is_positive_int],
    )
    stacker_match_detector_data_rank: bool = attrs.field(default=False)
    log_to_file: bool = attrs.field(default=False)
    log_file: Optional[Path] = attrs.field(default=None, converter=convert_to_path_or_none)
    logger: logging.Logger = attrs.field(init=False)
    _log_formatter: logging.Formatter = attrs.field(init=False, repr=False)

    def __attrs_post_init__(self):
        """
        Post-initialization setup for logging and default paths.
        """
        self._setup_logger()

        # Set defaults for optional paths
        self.saxs_dir = self.saxs_dir or self.vsi_root / "Measurements" / "SAXS002"
        self.data_dir = self.data_dir or self.saxs_dir / "data"
        self.masks_dir = self.masks_dir or self.data_dir / "Masks"
        self.logbook_file = self.logbook_file or self.saxs_dir / "logbooks" / "logbook_MOUSE.xlsx"
        self.stacker_config_file = (
            self.stacker_config_file or self.data_dir / "StackerConfigurations" / "stacker_config.yaml"
        )
        self.projects_dir = self.projects_dir or self.vsi_root / "Proposals" / "SAXS002"

        self.logger.info("DefaultsCarrier initialized with provided or default paths.")

    def _setup_logger(self):
        """
        Configure logging for the carrier.
        """
        self._log_formatter = logging.Formatter(
            "%(asctime)s %(levelname)s [%(threadName)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        self.logger = logging.getLogger("DefaultsCarrier")
        self.logger.handlers.clear()
        self.logger.setLevel(logging.DEBUG)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(self.logging_level.upper())
        console_handler.setFormatter(self._log_formatter)
        self.logger.addHandler(console_handler)

        if self.log_to_file:
            if not self.log_file:
                raise ValueError("Log file path must be provided when log_to_file is enabled.")
            self.logger.addHandler(self.build_file_handler(self.log_file, level=self.logging_level.upper()))

    def build_file_handler(self, log_file: Path, level: int | str = logging.DEBUG) -> logging.FileHandler:
        """
        Create a consistently formatted file handler for pipeline logs.
        """
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(self._log_formatter)
        return file_handler


# Factory Function
def create_defaults_carrier_from_config(config_file: Optional[str] = None) -> DefaultsCarrier:
    """
    Factory function to create a DefaultsCarrier instance from a configuration file.
    """
    config = load_config_from_yaml(config_file) if config_file else {}

    required_keys = ["vsi_root", "post_translation_dir", "translator_template_dir"]
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise ValueError(f"Missing required configuration keys: {', '.join(missing_keys)}")

    return DefaultsCarrier(
        vsi_root=config.get("vsi_root"),
        post_translation_dir=config.get("post_translation_dir"),
        translator_template_dir=config.get("translator_template_dir"),
        saxs_dir=config.get("saxs_dir", None),
        data_dir=config.get("data_dir", None),
        masks_dir=config.get("masks_dir", None),
        logbook_file=config.get("logbook_file", None),
        stacker_config_file=config.get("stacker_config_file", None),
        projects_dir=config.get("projects_dir", None),
        logging_level=config.get("logging_level", "INFO"),
        profile_steps=config.get("profile_steps", True),
        log_per_datafile=config.get("log_per_datafile", True),
        parallel_workers=config.get("parallel_workers", None),
        stacker_match_detector_data_rank=config.get("stacker_match_detector_data_rank", False),
        log_to_file=config.get("log_to_file", False),
        log_file=config.get("log_file", None),
    )


if __name__ == "__main__":
    # Example: Loading from a YAML configuration
    defaults = create_defaults_carrier_from_config("MOUSE_settings.yaml")
    # print(defaults.data_dir)
