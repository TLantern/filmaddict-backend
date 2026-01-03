import logging
import os
import subprocess
import tempfile
from typing import List, Tuple, Optional
from uuid import UUID

import ffmpeg
from sqlalchemy.ext.asyncio import AsyncSession

from db import crud
from utils.storage import get_storage_instance, S3Storage, get_video_path

logger = logging.getLogger(__name__)

EXPORT_FORMATS = {
    "mp4": {
        "extension": "mp4",
        "mime_type": "video/mp4",
        "description": "MP4 (H.264) - Default delivery",
    },
    "mov_prores422": {
        "extension": "mov",
        "mime_type": "video/quicktime",
        "description": "MOV (ProRes 422) - For real editors & post houses",
    },
    "mov_prores4444": {
        "extension": "mov",
        "mime_type": "video/quicktime",
        "description": "MOV (ProRes 4444) - For real editors & post houses",
    },
    "webm": {
        "extension": "webm",
        "mime_type": "video/webm",
        "description": "WebM (VP9) - YouTube optimization / modern web",
    },
    "xml": {
        "extension": "xml",
        "mime_type": "application/xml",
        "description": "XML (Final Cut Pro)",
    },
    "edl": {
        "extension": "edl",
        "mime_type": "text/plain",
        "description": "EDL (Premiere / Resolve)",
    },
    "aaf": {
        "extension": "aaf",
        "mime_type": "application/octet-stream",
        "description": "AAF (Avid / Pro pipelines)",
    },
}


def get_keep_segments(
    video_path: str, segments_to_remove: List[Tuple[float, float]]
) -> List[Tuple[float, float]]:
    """Calculate segments to keep from segments to remove."""
    if not segments_to_remove:
        try:
            probe = ffmpeg.probe(video_path)
            video_duration = float(probe['format'].get('duration', 0))
            return [(0.0, video_duration)] if video_duration > 0 else []
        except Exception:
            return [(0.0, 1000.0)]
    
    sorted_segments = sorted(segments_to_remove, key=lambda x: x[0])
    
    try:
        probe = ffmpeg.probe(video_path)
        video_duration = float(probe['format'].get('duration', sorted_segments[-1][1] + 10))
    except Exception:
        video_duration = sorted_segments[-1][1] + 10
    
    keep_segments = []
    current_time = 0.0
    
    for start, end in sorted_segments:
        if current_time < start:
            keep_segments.append((current_time, start))
        current_time = max(current_time, end)
    
    if current_time < video_duration:
        keep_segments.append((current_time, video_duration))
    
    return keep_segments


def _get_ffmpeg_input_kwargs() -> dict:
    """Get common FFmpeg input kwargs for error tolerance."""
    return {
        "err_detect": "ignore_err",
        "fflags": "+genpts+igndts+discardcorrupt",
    }


def export_mp4(
    video_path: str,
    keep_segments: List[Tuple[float, float]],
    output_path: str,
) -> None:
    """Export video as MP4 (H.264)."""
    # Filter out invalid segments (too small or invalid times)
    valid_segments = [(s, e) for s, e in keep_segments if e > s and (e - s) >= 0.1]
    if not valid_segments:
        raise ValueError("No valid segments to export")
    
    input_kwargs = _get_ffmpeg_input_kwargs()
    
    if len(valid_segments) == 1:
        start, end = valid_segments[0]
        duration = end - start
        stream = ffmpeg.input(video_path, ss=start, t=duration, **input_kwargs)
        stream = ffmpeg.output(
            stream,
            output_path,
            vcodec="libx264",
            acodec="aac",
            preset="medium",
            crf=23,
            **{"movflags": "faststart", "avoid_negative_ts": "make_zero", "max_muxing_queue_size": "1024"},
        )
        ffmpeg.run(stream, overwrite_output=True, quiet=True, capture_stderr=True)
    else:
        temp_files = []
        segment_paths = []
        try:
            for i, (start, end) in enumerate(valid_segments):
                duration = end - start
                if duration < 0.1:
                    logger.warning(f"Skipping segment {i}: duration {duration}s too small")
                    continue
                    
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f"_seg_{i}.mp4")
                temp_files.append(temp_file.name)
                temp_file.close()
                
                try:
                    segment_stream = ffmpeg.input(video_path, ss=start, t=duration, **input_kwargs)
                    segment_stream = ffmpeg.output(
                        segment_stream,
                        temp_files[-1],
                        vcodec="libx264",
                        acodec="aac",
                        preset="medium",
                        crf=23,
                        **{"movflags": "faststart", "avoid_negative_ts": "make_zero", "max_muxing_queue_size": "1024"},
                    )
                    # Use subprocess with timeout to prevent hangs (30 seconds per segment max)
                    segment_timeout = max(30, int(duration * 2))
                    try:
                        cmd = ffmpeg.compile(segment_stream, overwrite_output=True)
                        subprocess.run(
                            cmd,
                            capture_output=True,
                            timeout=segment_timeout,
                            check=True
                        )
                    except subprocess.TimeoutExpired:
                        logger.warning(f"Segment {i} extraction timed out after {segment_timeout}s, skipping")
                        continue
                    except subprocess.CalledProcessError as e:
                        logger.error(f"Segment {i} extraction failed: {e.stderr.decode() if e.stderr else str(e)}")
                        continue
                    
                    if os.path.exists(temp_files[-1]) and os.path.getsize(temp_files[-1]) > 1000:
                        segment_paths.append(temp_files[-1])
                    else:
                        logger.warning(f"Segment {i} extraction failed or produced empty file")
                except Exception as e:
                    logger.error(f"Error extracting segment {i} ({start}-{end}): {str(e)[:200]}")
                    continue
            
            if not segment_paths:
                raise ValueError("No valid segments were extracted")
            
            if len(segment_paths) == 1:
                # If only one segment, just copy it
                import shutil
                shutil.copy2(segment_paths[0], output_path)
            else:
                concat_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt')
                for segment_path in segment_paths:
                    concat_file.write(f"file '{segment_path}'\n")
                concat_file.close()
                
                # Use stream copy for concatenation (much faster, no re-encoding)
                # Segments are already encoded, so we can just copy streams
                concat_stream = ffmpeg.input(concat_file.name, format='concat', safe=0)
                concat_stream = ffmpeg.output(
                    concat_stream,
                    output_path,
                    c="copy",  # Stream copy - no re-encoding
                    movflags="faststart",
                )
                # Use subprocess with timeout for concat (120 seconds max for large files)
                try:
                    cmd = ffmpeg.compile(concat_stream, overwrite_output=True)
                    subprocess.run(cmd, capture_output=True, timeout=120, check=True)
                except subprocess.TimeoutExpired:
                    logger.error("Concat operation timed out after 120s")
                    raise Exception("Video concatenation timed out after 120 seconds")
                except subprocess.CalledProcessError as e:
                    error_msg = e.stderr.decode() if e.stderr else str(e)
                    logger.error(f"Concat operation failed: {error_msg}")
                    raise Exception(f"Video concatenation failed: {error_msg}")
                
                os.remove(concat_file.name)
        finally:
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    try:
                        os.remove(temp_file)
                    except Exception:
                        pass


def export_mov_prores(
    video_path: str,
    keep_segments: List[Tuple[float, float]],
    output_path: str,
    prores_profile: str = "prores_ks",
    prores_preset: str = "422",
) -> None:
    """Export video as MOV with ProRes codec."""
    # Filter out invalid segments (too small or invalid times)
    valid_segments = [(s, e) for s, e in keep_segments if e > s and (e - s) >= 0.1]
    if not valid_segments:
        raise ValueError("No valid segments to export")
    
    input_kwargs = _get_ffmpeg_input_kwargs()
    
    if len(valid_segments) == 1:
        start, end = valid_segments[0]
        duration = end - start
        stream = ffmpeg.input(video_path, ss=start, t=duration, **input_kwargs)
        output_kwargs = {
            "vcodec": prores_profile,
            "acodec": "pcm_s24le",
            "movflags": "faststart",
            "avoid_negative_ts": "make_zero",
            "max_muxing_queue_size": "1024",
        }
        if prores_profile == "prores_ks":
            output_kwargs["profile"] = prores_preset
        else:
            output_kwargs[f"{prores_profile}_profile"] = prores_preset
        stream = ffmpeg.output(stream, output_path, **output_kwargs)
        ffmpeg.run(stream, overwrite_output=True, quiet=True, capture_stderr=True)
    else:
        temp_files = []
        segment_paths = []
        try:
            for i, (start, end) in enumerate(valid_segments):
                duration = end - start
                if duration < 0.1:
                    logger.warning(f"Skipping segment {i}: duration {duration}s too small")
                    continue
                    
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f"_seg_{i}.mov")
                temp_files.append(temp_file.name)
                temp_file.close()
                
                try:
                    segment_stream = ffmpeg.input(video_path, ss=start, t=duration, **input_kwargs)
                    output_kwargs = {
                        "vcodec": prores_profile,
                        "acodec": "pcm_s24le",
                        "movflags": "faststart",
                        "avoid_negative_ts": "make_zero",
                        "max_muxing_queue_size": "1024",
                    }
                    if prores_profile == "prores_ks":
                        output_kwargs["profile"] = prores_preset
                    else:
                        output_kwargs[f"{prores_profile}_profile"] = prores_preset
                    segment_stream = ffmpeg.output(segment_stream, temp_files[-1], **output_kwargs)
                    ffmpeg.run(segment_stream, overwrite_output=True, quiet=True, capture_stderr=True)
                    
                    if os.path.exists(temp_files[-1]) and os.path.getsize(temp_files[-1]) > 1000:
                        segment_paths.append(temp_files[-1])
                    else:
                        logger.warning(f"Segment {i} extraction failed or produced empty file")
                except Exception as e:
                    logger.error(f"Error extracting segment {i} ({start}-{end}): {str(e)[:200]}")
                    continue
            
            if not segment_paths:
                raise ValueError("No valid segments were extracted")
            
            if len(segment_paths) == 1:
                # If only one segment, just copy it
                import shutil
                shutil.copy2(segment_paths[0], output_path)
            else:
                concat_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt')
                for segment_path in segment_paths:
                    concat_file.write(f"file '{segment_path}'\n")
                concat_file.close()
                
                concat_stream = ffmpeg.input(concat_file.name, format='concat', safe=0)
                output_kwargs = {
                    "vcodec": prores_profile,
                    "acodec": "pcm_s24le",
                    "movflags": "faststart",
                    "avoid_negative_ts": "make_zero",
                    "max_muxing_queue_size": "1024",
                }
                if prores_profile == "prores_ks":
                    output_kwargs["profile"] = prores_preset
                else:
                    output_kwargs[f"{prores_profile}_profile"] = prores_preset
                concat_stream = ffmpeg.output(concat_stream, output_path, **output_kwargs)
                ffmpeg.run(concat_stream, overwrite_output=True, quiet=True, capture_stderr=True)
                
                os.remove(concat_file.name)
        finally:
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    try:
                        os.remove(temp_file)
                    except Exception:
                        pass


def export_webm(
    video_path: str,
    keep_segments: List[Tuple[float, float]],
    output_path: str,
) -> None:
    """Export video as WebM (VP9)."""
    # Filter out invalid segments (too small or invalid times)
    valid_segments = [(s, e) for s, e in keep_segments if e > s and (e - s) >= 0.1]
    if not valid_segments:
        raise ValueError("No valid segments to export")
    
    input_kwargs = _get_ffmpeg_input_kwargs()
    
    if len(valid_segments) == 1:
        start, end = valid_segments[0]
        duration = end - start
        stream = ffmpeg.input(video_path, ss=start, t=duration, **input_kwargs)
        stream = ffmpeg.output(
            stream,
            output_path,
            vcodec="libvpx-vp9",
            acodec="libopus",
            crf=30,
            **{"b:v": 0, "avoid_negative_ts": "make_zero"},
        )
        ffmpeg.run(stream, overwrite_output=True, quiet=True, capture_stderr=True)
    else:
        temp_files = []
        segment_paths = []
        try:
            for i, (start, end) in enumerate(valid_segments):
                duration = end - start
                if duration < 0.1:
                    logger.warning(f"Skipping segment {i}: duration {duration}s too small")
                    continue
                    
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f"_seg_{i}.webm")
                temp_files.append(temp_file.name)
                temp_file.close()
                
                try:
                    segment_stream = ffmpeg.input(video_path, ss=start, t=duration, **input_kwargs)
                    segment_stream = ffmpeg.output(
                        segment_stream,
                        temp_files[-1],
                        vcodec="libvpx-vp9",
                        acodec="libopus",
                        crf=30,
                        **{"b:v": 0, "avoid_negative_ts": "make_zero"},
                    )
                    ffmpeg.run(segment_stream, overwrite_output=True, quiet=True, capture_stderr=True)
                    
                    if os.path.exists(temp_files[-1]) and os.path.getsize(temp_files[-1]) > 1000:
                        segment_paths.append(temp_files[-1])
                    else:
                        logger.warning(f"Segment {i} extraction failed or produced empty file")
                except Exception as e:
                    logger.error(f"Error extracting segment {i} ({start}-{end}): {str(e)[:200]}")
                    continue
            
            if not segment_paths:
                raise ValueError("No valid segments were extracted")
            
            if len(segment_paths) == 1:
                # If only one segment, just copy it
                import shutil
                shutil.copy2(segment_paths[0], output_path)
            else:
                concat_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt')
                for segment_path in segment_paths:
                    concat_file.write(f"file '{segment_path}'\n")
                concat_file.close()
                
                concat_stream = ffmpeg.input(concat_file.name, format='concat', safe=0)
                concat_stream = ffmpeg.output(
                    concat_stream,
                    output_path,
                    vcodec="libvpx-vp9",
                    acodec="libopus",
                    crf=30,
                    **{"b:v": 0, "avoid_negative_ts": "make_zero"},
                )
                ffmpeg.run(concat_stream, overwrite_output=True, quiet=True, capture_stderr=True)
                
                os.remove(concat_file.name)
        finally:
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    try:
                        os.remove(temp_file)
                    except Exception:
                        pass


def export_xml_fcp(
    video_path: str,
    keep_segments: List[Tuple[float, float]],
    output_path: str,
) -> None:
    """Export Final Cut Pro XML."""
    try:
        probe = ffmpeg.probe(video_path)
        video_info = next(s for s in probe['streams'] if s.get('codec_type') == 'video')
        width = int(video_info.get('width', 1920))
        height = int(video_info.get('height', 1080))
        frame_rate = eval(video_info.get('r_frame_rate', '30/1'))
    except Exception:
        width, height, frame_rate = 1920, 1080, 30.0
    
    xml_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE xmeml>
<xmeml version="5">
  <project>
    <name>FilmAddict Export</name>
    <children>
      <sequence>
        <name>Sequence 1</name>
        <rate>
          <timebase>{int(frame_rate)}</timebase>
          <ntsc>FALSE</ntsc>
        </rate>
        <media>
          <video>
            <format>
              <samplecharacteristics>
                <rate>
                  <timebase>{int(frame_rate)}</timebase>
                  <ntsc>FALSE</ntsc>
                </rate>
                <width>{width}</width>
                <height>{height}</height>
              </samplecharacteristics>
            </format>
            <track>
'''
    
    current_frame = 0
    for start, end in keep_segments:
        start_frame = int(start * frame_rate)
        end_frame = int(end * frame_rate)
        duration_frames = end_frame - start_frame
        
        xml_content += f'''              <clipitem id="clip_{current_frame}">
                <name>Video Clip</name>
                <start>{current_frame}</start>
                <end>{current_frame + duration_frames}</end>
                <in>{start_frame}</in>
                <out>{end_frame}</out>
                <file>
                  <pathurl>{video_path}</pathurl>
                </file>
              </clipitem>
'''
        current_frame += duration_frames
    
    xml_content += '''            </track>
          </video>
        </media>
      </sequence>
    </children>
  </project>
</xmeml>'''
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(xml_content)


def export_edl(
    video_path: str,
    keep_segments: List[Tuple[float, float]],
    output_path: str,
) -> None:
    """Export EDL (Edit Decision List) for Premiere/Resolve."""
    edl_content = "TITLE: FilmAddict Export\n"
    edl_content += "FCM: NON-DROP FRAME\n\n"
    
    reel_number = 1
    event_number = 1
    
    for start, end in keep_segments:
        start_timecode = format_timecode(start)
        end_timecode = format_timecode(end)
        duration = end - start
        duration_timecode = format_timecode(duration)
        
        edl_content += f"{event_number:03d}  {reel_number:03d}     V     C        {start_timecode} {end_timecode} {start_timecode} {end_timecode}\n"
        edl_content += f"* FROM CLIP NAME: {os.path.basename(video_path)}\n"
        edl_content += f"* SOURCE FILE: {video_path}\n\n"
        
        event_number += 1
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(edl_content)


def format_timecode(seconds: float) -> str:
    """Format seconds to timecode (HH:MM:SS:FF)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    frames = int((seconds % 1) * 30)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}:{frames:02d}"


def export_aaf(
    video_path: str,
    keep_segments: List[Tuple[float, float]],
    output_path: str,
) -> None:
    """Export AAF (Advanced Authoring Format) for Avid."""
    aaf_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<AAF xmlns="http://www.aafassociation.org/2008/AVC">
  <Header>
    <Version>1.2</Version>
  </Header>
  <Content>
    <Composition>
      <Name>FilmAddict Export</Name>
      <Segments>
"""
    
    for i, (start, end) in enumerate(keep_segments):
        aaf_content += f"""        <Segment>
          <ID>{i + 1}</ID>
          <StartTime>{start}</StartTime>
          <Duration>{end - start}</Duration>
          <SourceFile>{video_path}</SourceFile>
        </Segment>
"""
    
    aaf_content += """      </Segments>
    </Composition>
  </Content>
</AAF>"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(aaf_content)


async def export_video(
    video_id: UUID,
    export_format: str,
    segments_to_remove: Optional[List[Tuple[float, float]]] = None,
    db: AsyncSession = None,
) -> Tuple[str, str]:
    """
    Export video in specified format with optional cuts applied.
    
    Args:
        video_id: UUID of the video to export
        export_format: Format key (mp4, mov_prores422, mov_prores4444, webm, xml, edl, aaf)
        segments_to_remove: Optional list of (start, end) tuples to remove
        db: Database session (required)
        
    Returns:
        Tuple of (output_file_path, mime_type)
    """
    if export_format not in EXPORT_FORMATS:
        raise ValueError(f"Unsupported export format: {export_format}")
    
    format_info = EXPORT_FORMATS[export_format]
    
    if db is None:
        raise ValueError("Database session is required")
    
    video = await crud.get_video_by_id(db, video_id)
    if not video:
        raise ValueError(f"Video not found: {video_id}")
    
    video_path = await get_video_path(video_id, db, download_local=True)
    if not video_path:
        raise ValueError(f"Video file not found: {video_id}")
    
    if isinstance(get_storage_instance(), S3Storage) and video_path.startswith("http"):
        import urllib.request
        temp_video_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_video_path = temp_video_file.name
        temp_video_file.close()
        urllib.request.urlretrieve(video_path, temp_video_path)
        actual_video_path = temp_video_path
    else:
        actual_video_path = video_path
        temp_video_path = None
    
    try:
        if segments_to_remove is None:
            segments_to_remove = []
            if video.pending_cuts:
                segments_to_remove = [
                    (cut["start_time"], cut["end_time"])
                    for cut in video.pending_cuts
                ]
        
        keep_segments = get_keep_segments(actual_video_path, segments_to_remove)
        
        output_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=f".{format_info['extension']}"
        )
        output_path = output_file.name
        output_file.close()
        
        # Run export in executor to avoid blocking
        import asyncio
        loop = asyncio.get_event_loop()
        
        # Calculate timeout based on video duration and segments (allow 10 seconds per segment, minimum 60s)
        try:
            probe = ffmpeg.probe(actual_video_path)
            video_duration = float(probe['format'].get('duration', 0))
            timeout_seconds = max(60, int(len(keep_segments) * 10) + int(video_duration * 0.1))
        except:
            timeout_seconds = max(60, len(keep_segments) * 10)
        if export_format == "mp4":
            try:
                await asyncio.wait_for(
                    loop.run_in_executor(None, export_mp4, actual_video_path, keep_segments, output_path),
                    timeout=timeout_seconds
                )
            except asyncio.TimeoutError:
                logger.error(f"FFmpeg export timed out after {timeout_seconds}s")
                if os.path.exists(output_path):
                    try:
                        os.remove(output_path)
                    except:
                        pass
                raise Exception(f"Video export timed out after {timeout_seconds} seconds")
        elif export_format == "mov_prores422":
            try:
                await asyncio.wait_for(
                    loop.run_in_executor(None, export_mov_prores, actual_video_path, keep_segments, output_path, "prores_ks", "422"),
                    timeout=timeout_seconds
                )
            except asyncio.TimeoutError:
                logger.error(f"FFmpeg export timed out after {timeout_seconds}s")
                if os.path.exists(output_path):
                    try:
                        os.remove(output_path)
                    except:
                        pass
                raise Exception(f"Video export timed out after {timeout_seconds} seconds")
        elif export_format == "mov_prores4444":
            try:
                await asyncio.wait_for(
                    loop.run_in_executor(None, export_mov_prores, actual_video_path, keep_segments, output_path, "prores_ks", "4444"),
                    timeout=timeout_seconds
                )
            except asyncio.TimeoutError:
                logger.error(f"FFmpeg export timed out after {timeout_seconds}s")
                if os.path.exists(output_path):
                    try:
                        os.remove(output_path)
                    except:
                        pass
                raise Exception(f"Video export timed out after {timeout_seconds} seconds")
        elif export_format == "webm":
            try:
                await asyncio.wait_for(
                    loop.run_in_executor(None, export_webm, actual_video_path, keep_segments, output_path),
                    timeout=timeout_seconds
                )
            except asyncio.TimeoutError:
                logger.error(f"FFmpeg export timed out after {timeout_seconds}s")
                if os.path.exists(output_path):
                    try:
                        os.remove(output_path)
                    except:
                        pass
                raise Exception(f"Video export timed out after {timeout_seconds} seconds")
        elif export_format == "xml":
            await loop.run_in_executor(None, export_xml_fcp, actual_video_path, keep_segments, output_path)
        elif export_format == "edl":
            await loop.run_in_executor(None, export_edl, actual_video_path, keep_segments, output_path)
        elif export_format == "aaf":
            await loop.run_in_executor(None, export_aaf, actual_video_path, keep_segments, output_path)
        else:
            raise ValueError(f"Export format {export_format} not implemented")
        
        return output_path, format_info['mime_type']
    
    finally:
        if temp_video_path and os.path.exists(temp_video_path):
            try:
                os.remove(temp_video_path)
            except Exception:
                pass

