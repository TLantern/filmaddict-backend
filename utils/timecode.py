import re
from typing import Union


def seconds_to_timecode(seconds: float) -> str:
    """
    Convert seconds to HH:MM:SS format.

    Args:
        seconds: Time in seconds (can be negative)

    Returns:
        Timecode string in HH:MM:SS format
    """
    if seconds < 0:
        sign = "-"
        seconds = abs(seconds)
    else:
        sign = ""

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    return f"{sign}{hours:02d}:{minutes:02d}:{secs:02d}"


def timecode_to_seconds(timecode: str) -> float:
    """
    Parse HH:MM:SS or HH:MM:SS.mmm format to seconds.

    Args:
        timecode: Timecode string (e.g., "01:23:45" or "01:23:45.67")

    Returns:
        Time in seconds

    Raises:
        ValueError: If timecode format is invalid
    """
    if not timecode:
        raise ValueError("Timecode cannot be empty")

    negative = timecode.startswith("-")
    if negative:
        timecode = timecode[1:]

    pattern = r"^(\d{1,2}):(\d{2}):(\d{2})(?:\.(\d{1,3}))?$"
    match = re.match(pattern, timecode)

    if not match:
        raise ValueError(f"Invalid timecode format: {timecode}. Expected HH:MM:SS or HH:MM:SS.mmm")

    hours = int(match.group(1))
    minutes = int(match.group(2))
    seconds = int(match.group(3))
    milliseconds = int(match.group(4) or 0)

    if minutes >= 60:
        raise ValueError(f"Invalid minutes value: {minutes} (must be < 60)")
    if seconds >= 60:
        raise ValueError(f"Invalid seconds value: {seconds} (must be < 60)")

    total_seconds = hours * 3600 + minutes * 60 + seconds + milliseconds / 1000.0

    return -total_seconds if negative else total_seconds

