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

    def get_year(self) -> str:
        # Extract and return the year from the YMD string
        return self.YMD[:4]
