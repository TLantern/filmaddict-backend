import logging
import os
import re
import subprocess
from typing import List

from models import SilenceSegment, SpeechSegment, TimelineItem, TranscriptSegment

logger = logging.getLogger(__name__)


def _detect_silence_with_ffmpeg(audio_path: str) -> List[SilenceSegment]:
    """
    Detect silence segments using ffmpeg's silencedetect filter.
    This is a fallback when pydub is not available.
    
    Args:
        audio_path: Full path to the audio file
        
    Returns:
        List of SilenceSegment objects with start, end, and duration
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    # MIN_SILENCE_DURATION = 2.0  # seconds
    # min_silence_len = 400  # milliseconds (0.4 seconds)
    # silence_thresh = -40  # dBFS
    
    # ffmpeg silencedetect parameters:
    # - silence_thresh: silence threshold in dB (negative value)
    # - duration: minimum silence duration to detect (in seconds)
    silence_thresh_db = -40
    min_silence_duration = 0.4  # 400ms minimum for detection
    
    # Run ffmpeg silencedetect filter
    # Output format: silence_start: X.XXX | silence_end: Y.YYY
    cmd = [
        'ffmpeg',
        '-i', audio_path,
        '-af', f'silencedetect=noise={silence_thresh_db}dB:d={min_silence_duration}',
        '-f', 'null',
        '-'
    ]
    
    try:
        result = subprocess.run(
            cmd,
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            check=False  # ffmpeg returns non-zero for silencedetect, that's ok
        )
        
        # Parse ffmpeg output to find silence_start and silence_end
        # Format: silence_start: 10.5
        #         silence_end: 12.3 | silence_duration: 1.8
        stderr_output = result.stderr
        
        silence_starts = []
        silence_ends = []
        
        # Find all silence_start and silence_end markers
        for line in stderr_output.split('\n'):
            # Match: silence_start: 10.5
            start_match = re.search(r'silence_start:\s*([\d.]+)', line)
            if start_match:
                silence_starts.append(float(start_match.group(1)))
            
            # Match: silence_end: 12.3
            end_match = re.search(r'silence_end:\s*([\d.]+)', line)
            if end_match:
                silence_ends.append(float(end_match.group(1)))
        
        # Pair up silence start/end times
        silence_segments = []
        MIN_SILENCE_DURATION = 2.0  # Only include silences >= 2 seconds
        
        # Ensure we have matching pairs
        min_len = min(len(silence_starts), len(silence_ends))
        for i in range(min_len):
            start = silence_starts[i]
            end = silence_ends[i]
            duration = end - start
            
            if duration >= MIN_SILENCE_DURATION:
                silence_segment = SilenceSegment(
                    start=start,
                    end=end,
                    duration=duration
                )
                silence_segments.append(silence_segment)
        
        return silence_segments
        
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg error detecting silence: {e.stderr if e.stderr else str(e)}")
        raise Exception(f"FFmpeg silence detection failed: {str(e)}") from e
    except Exception as e:
        logger.error(f"Error in FFmpeg silence detection: {str(e)}", exc_info=True)
        raise Exception(f"Failed to detect silence with FFmpeg: {str(e)}") from e


def detect_silence_segments(audio_path: str) -> List[SilenceSegment]:
    """
    Detect silence segments in audio file.
    Tries pydub first, falls back to ffmpeg if pydub is not available.
    
    Args:
        audio_path: Full path to the audio file
        
    Returns:
        List of SilenceSegment objects with start, end, and duration
        
    Raises:
        FileNotFoundError: If audio file doesn't exist
        Exception: If silence detection fails
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    # Try pydub first
    try:
        from pydub import AudioSegment
        from pydub.silence import detect_silence
        
        logger.info(f"🔇 Detecting silence in audio file using pydub: {audio_path}")
        
        # Load audio file
        audio = AudioSegment.from_file(audio_path)
        
        # Detect silence: min_silence_len=400ms, silence_thresh=-40dBFS
        min_silence_len = 400  # milliseconds
        silence_thresh = -40  # dBFS
        
        silence_ranges = detect_silence(
            audio,
            min_silence_len=min_silence_len,
            silence_thresh=silence_thresh
        )
        
        # Convert from milliseconds to seconds and create SilenceSegment objects
        # Only include silences >= 2 seconds
        MIN_SILENCE_DURATION = 2.0  # seconds
        silence_segments = []
        for start_ms, end_ms in silence_ranges:
            start_sec = start_ms / 1000.0
            end_sec = end_ms / 1000.0
            duration = end_sec - start_sec
            
            # Filter: only include silences >= 2 seconds
            if duration >= MIN_SILENCE_DURATION:
                silence_segment = SilenceSegment(
                    start=start_sec,
                    end=end_sec,
                    duration=duration
                )
                silence_segments.append(silence_segment)
        
        logger.info(f"✅ Detected {len(silence_segments)} silence segments (>= {MIN_SILENCE_DURATION}s)")
        if silence_segments:
            total_silence = sum(s.duration for s in silence_segments)
            logger.debug(f"📊 Total silence duration: {total_silence:.2f}s")
        
        return silence_segments
        
    except ImportError:
        # pydub not available, use ffmpeg fallback
        logger.info(f"🔇 pydub not available, using FFmpeg for silence detection: {audio_path}")
        try:
            silence_segments = _detect_silence_with_ffmpeg(audio_path)
            MIN_SILENCE_DURATION = 2.0
            logger.info(f"✅ Detected {len(silence_segments)} silence segments (>= {MIN_SILENCE_DURATION}s) using FFmpeg")
            if silence_segments:
                total_silence = sum(s.duration for s in silence_segments)
                logger.debug(f"📊 Total silence duration: {total_silence:.2f}s")
            return silence_segments
        except Exception as e:
            logger.error(f"FFmpeg silence detection failed: {str(e)}", exc_info=True)
            raise Exception(f"Failed to detect silence: {str(e)}") from e
    except Exception as e:
        logger.error(f"Error detecting silence with pydub: {str(e)}", exc_info=True)
        # Try ffmpeg as fallback
        logger.info(f"🔇 pydub failed, trying FFmpeg fallback for silence detection: {audio_path}")
        try:
            silence_segments = _detect_silence_with_ffmpeg(audio_path)
            MIN_SILENCE_DURATION = 2.0
            logger.info(f"✅ Detected {len(silence_segments)} silence segments (>= {MIN_SILENCE_DURATION}s) using FFmpeg")
            return silence_segments
        except Exception as fallback_error:
            logger.error(f"Both pydub and FFmpeg silence detection failed: {str(fallback_error)}", exc_info=True)
            raise Exception(f"Failed to detect silence: {str(e)} (pydub) and {str(fallback_error)} (ffmpeg)") from e


def merge_transcript_with_silence(
    speech_segments: List[TranscriptSegment],
    silence_segments: List[SilenceSegment]
) -> List[TimelineItem]:
    """
    Merge speech and silence segments into a unified timeline sorted by start time.
    
    Args:
        speech_segments: List of speech transcript segments
        silence_segments: List of detected silence segments
        
    Returns:
        List of TimelineItem objects (union of SpeechSegment and SilenceSegment) sorted by start time
    """
    timeline: List[TimelineItem] = []
    
    # Convert speech segments to SpeechSegment objects
    for segment in speech_segments:
        speech_segment = SpeechSegment(
            start=segment.start,
            end=segment.end,
            text=segment.text
        )
        timeline.append(speech_segment)
    
    # Add silence segments
    timeline.extend(silence_segments)
    
    # Sort by start time
    timeline.sort(key=lambda item: item.start)
    
    logger.info(
        f"✅ Merged timeline: {len(speech_segments)} speech segments, "
        f"{len(silence_segments)} silence segments, "
        f"{len(timeline)} total items"
    )
    
    return timeline


def format_timeline_as_transcript(timeline: List[TimelineItem]) -> str:
    """
    Format timeline items as a transcript string with silence markers.
    
    Speech segments output their text, silence segments output [SILENCE X.XXs].
    Items are joined with spaces.
    
    Args:
        timeline: List of TimelineItem objects (sorted by start time)
        
    Returns:
        Formatted transcript string with silence markers
    """
    formatted_parts = []
    
    for item in timeline:
        if item.type == "speech":
            if isinstance(item, SpeechSegment):
                formatted_parts.append(item.text)
        elif item.type == "silence":
            if isinstance(item, SilenceSegment):
                formatted_parts.append(f"[SILENCE {item.duration:.2f}s]")
    
    result = " ".join(formatted_parts)
    logger.debug(f"📝 Formatted transcript: {len(formatted_parts)} parts, {len(result)} characters")
    return result

