from pathlib import Path
from typing import Tuple
import attrs
from datetime import datetime

def validate_ymd(instance, attribute, value):
    # Check if the string is of length 8 and matches YYYYMMDD format
    if len(value) != 8:
        raise ValueError(f"{attribute.name} must be a string of length 8")
    
    try:
        # Attempt to parse the string as a date
        datetime.strptime(value, "%Y%m%d")
    except ValueError:
        raise ValueError(f"{attribute.name} must be in the format YYYYMMDD")

@attrs.define
class YMD:
    YMD: str = attrs.field(converter=str, validator=[attrs.validators.instance_of(str), validate_ymd])

    def __repr__(self):
        # return the YMD string
        return f"{self.YMD}"
    
    def as_int(self) -> int:
        # Convert and return the YMD string as an integer
        return int(self.YMD)

    def get_year(self) -> str:
        # Extract and return the year from the YMD string
        return self.YMD[:4]

def extract_metadata_from_path(dir_path: Path) -> Tuple[YMD, int, int]:
    """
    Extracts YMD, batch, and repetition metadata from a directory path.
    """
    # check if this is not actually a file path, and if so, get the parent directory
    if dir_path.is_file():
        dir_path = dir_path.parent
    last_path = dir_path.parts[-1]
    parts = last_path.split('_')
    assert len(parts) == 3, f"Invalid directory format: {dir_path}"
    ymd, batch, repetition = parts
    return YMD(ymd), int(batch), int(repetition)
