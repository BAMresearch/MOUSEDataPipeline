import yaml
from typing import Optional
import attrs
from pathlib import Path
import logging


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


# Configuration Loader
def load_config_from_yaml(file_path: str) -> dict:
    """
    Load configuration from a YAML file.
    """
    try:
        with open(file_path, 'r') as yaml_file:
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

    saxs_dir: Optional[Path] = attrs.field(default=None, converter=convert_to_path_or_none, validator=[if_not_none_is_path_and_exists])
    data_dir: Optional[Path] = attrs.field(default=None, converter=convert_to_path_or_none, validator=[if_not_none_is_path_and_exists])
    masks_dir: Optional[Path] = attrs.field(default=None, converter=convert_to_path_or_none, validator=[if_not_none_is_path_and_exists])
    projects_dir: Optional[Path] = attrs.field(default=None, converter=convert_to_path_or_none, validator=[if_not_none_is_path_and_exists])

    logbook_file: Optional[Path] = attrs.field(default=None, converter=convert_to_path_or_none, validator=[if_not_none_is_path_and_exists])
    logging_level: str = attrs.field(default='INFO', converter=str)
    log_to_file: bool = attrs.field(default=False)
    log_file: Optional[Path] = attrs.field(default=None, converter=convert_to_path_or_none)
    logger: logging.Logger = attrs.field(init=False)

    def __attrs_post_init__(self):
        """
        Post-initialization setup for logging and default paths.
        """
        self._setup_logger()

        # Set defaults for optional paths
        self.saxs_dir = self.saxs_dir or self.vsi_root / 'Measurements' / 'SAXS002'
        self.data_dir = self.data_dir or self.saxs_dir / 'data'
        self.masks_dir = self.masks_dir or self.data_dir / 'Masks'
        self.logbooks = self.logbooks or self.saxs_dir / 'logbooks'
        self.projects_dir = self.projects_dir or self.vsi_root / 'Proposals' / 'SAXS002'

        self.logger.info("DefaultsCarrier initialized with provided or default paths.")

    def _setup_logger(self):
        """
        Configure logging for the carrier.
        """
        self.logger = logging.getLogger('DefaultsCarrier')
        self.logger.setLevel(self.logging_level.upper())

        if self.log_to_file:
            if not self.log_file:
                raise ValueError("Log file path must be provided when log_to_file is enabled.")
            # Ensure the directory for the log file exists
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(self.log_file)
            self.logger.addHandler(file_handler)
        else:
            console_handler = logging.StreamHandler()
            self.logger.addHandler(console_handler)


# Factory Function
def create_defaults_carrier_from_config(config_file: Optional[str] = None) -> DefaultsCarrier:
    """
    Factory function to create a DefaultsCarrier instance from a configuration file.
    """
    config = load_config_from_yaml(config_file) if config_file else {}

    required_keys = ['vsi_root', 'post_translation_dir', 'translator_template_dir']
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise ValueError(f"Missing required configuration keys: {', '.join(missing_keys)}")

    return DefaultsCarrier(
        vsi_root=config.get('vsi_root'),
        post_translation_dir=config.get('post_translation_dir'),
        translator_template_dir=config.get('translator_template_dir'),
        saxs_dir=config.get('saxs_dir', None),
        data_dir=config.get('data_dir', None),
        masks_dir=config.get('masks_dir', None),
        logbooks=config.get('logbooks', None),
        projects_dir=config.get('projects_dir', None),
        logging_level=config.get('logging_level', 'INFO'),
        log_to_file=config.get('log_to_file', False),
        log_file=config.get('log_file', None)
    )

if __name__ == '__main__':
    # Example: Loading from a YAML configuration
    defaults = create_defaults_carrier_from_config("MOUSE_settings.yaml")
    print(defaults.data_dir)
