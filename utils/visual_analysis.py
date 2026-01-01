import asyncio
import logging
import os
import tempfile
from typing import List, Optional

import ffmpeg

from models import SemanticSegment
from utils.model_loader import get_vjepa2_model

# Optional dependencies - only needed for local visual analysis (not used when Colab is enabled)
try:
    import cv2
    import numpy as np
    _cv2_numpy_available = True
except ImportError:
    _cv2_numpy_available = False

# colab_vjepa2 is only imported when needed (not in main Colab flow)
logger = logging.getLogger(__name__)


def extract_frames(
    video_path: str,
    segment: SemanticSegment,
    interval: float = 2.0,
) -> List:
    """
    Extract frames from video segment at regular intervals.
    
    Args:
        video_path: Path to video file or HTTP URL
        segment: SemanticSegment with start_time and end_time
        interval: Time interval between frames in seconds
        
    Returns:
        List of frame arrays (numpy arrays)
    """
    if not _cv2_numpy_available:
        raise ImportError(
            "cv2 and numpy are required for frame extraction. Install with:\n"
            "pip install opencv-python numpy\n"
            "Note: Frame extraction runs locally even when using Colab for visual analysis."
        )
    
    import cv2
    import numpy as np
    
    frames = []
    is_http_url = video_path.startswith("http://") or video_path.startswith("https://")
    
    if is_http_url:
        # Download to temp file first
        import urllib.request
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_path = temp_file.name
        temp_file.close()
        try:
            urllib.request.urlretrieve(video_path, temp_path)
            actual_path = temp_path
        except Exception as e:
            logger.error(f"Failed to download video from URL: {str(e)}")
            return []
    else:
        actual_path = video_path
        temp_path = None
    
    try:
        # Extract frames using FFmpeg
        current_time = segment.start_time
        
        while current_time < segment.end_time:
            try:
                out, _ = (
                    ffmpeg
                    .input(actual_path, ss=current_time)
                    .output('pipe:', vframes=1, format='rawvideo', pix_fmt='rgb24', s='256x256')
                    .run(capture_stdout=True, quiet=True)
                )
                
                frame = np.frombuffer(out, np.uint8).reshape([256, 256, 3])
                frames.append(frame)
                
                current_time += interval
            except Exception as e:
                logger.warning(f"Failed to extract frame at {current_time}s: {str(e)}")
                current_time += interval
                continue
        
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        
        return frames
        
    except Exception as e:
        logger.error(f"Error extracting frames: {str(e)}", exc_info=True)
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass
        return []


def compute_visual_change_scores_local(
    frames: List,
) -> float:
    """
    Compute visual change score using local V-JEPA2 model.
    
    Args:
        frames: List of frame arrays
        
    Returns:
        Visual change score (0.0-1.0)
    """
    if len(frames) < 2:
        return 0.5  # Neutral score
    
    if not _cv2_numpy_available:
        logger.warning("numpy/cv2 not available, using fallback visual analysis")
        return compute_visual_change_scores_fallback(frames)
    
    import numpy as np
    
    try:
        model = get_vjepa2_model()
        if model is None:
            logger.warning("V-JEPA2 model not available locally, using fallback")
            return compute_visual_change_scores_fallback(frames)
        
        # Check if torch is available
        try:
            import torch
        except ImportError:
            logger.warning("PyTorch not available, using fallback visual analysis")
            return compute_visual_change_scores_fallback(frames)
        
        device = next(model.parameters()).device
        
        # Preprocess frames
        frame_tensors = []
        for frame in frames:
            # Normalize and convert to tensor
            frame_normalized = frame.astype(np.float32) / 255.0
            frame_tensor = torch.from_numpy(frame_normalized).permute(2, 0, 1).unsqueeze(0).to(device)
            frame_tensors.append(frame_tensor)
        
        # Get embeddings
        embeddings = []
        with torch.no_grad():
            for frame_tensor in frame_tensors:
                embedding = model(frame_tensor)
                embeddings.append(embedding.cpu().numpy().flatten())
        
        # Compute embedding deltas
        deltas = []
        for i in range(1, len(embeddings)):
            delta = np.linalg.norm(embeddings[i] - embeddings[i-1])
            deltas.append(delta)
        
        # Normalize to 0-1 range
        if deltas:
            max_delta = max(deltas) if max(deltas) > 0 else 1.0
            visual_change_score = min(1.0, np.mean(deltas) / max_delta)
        else:
            visual_change_score = 0.5
        
        return float(visual_change_score)
        
    except Exception as e:
        logger.error(f"Error computing visual change scores locally: {str(e)}", exc_info=True)
        return compute_visual_change_scores_fallback(frames)


def compute_visual_change_scores_fallback(
    frames: List,
) -> float:
    """
    Fallback: Simple frame difference using histogram comparison.
    
    Args:
        frames: List of frame arrays
        
    Returns:
        Visual change score (0.0-1.0)
    """
    if len(frames) < 2:
        return 0.5
    
    try:
        deltas = []
        for i in range(1, len(frames)):
            # Compute histogram difference
            hist1 = cv2.calcHist([frames[i-1]], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            hist2 = cv2.calcHist([frames[i]], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            
            # Compare histograms
            diff = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
            deltas.append(1.0 - diff)  # Convert similarity to difference
        
        visual_change_score = float(np.mean(deltas)) if deltas else 0.5
        return visual_change_score
        
    except Exception as e:
        logger.warning(f"Error in fallback visual analysis: {str(e)}")
        return 0.5  # Default neutral score


async def compute_visual_change_scores(
    video_path: str,
    segments: List[SemanticSegment],
) -> List[float]:
    """
    Compute visual change scores for segments using V-JEPA2.
    
    Supports three modes:
    - local: Load V-JEPA2 model directly (requires GPU)
    - colab: Send frames to Colab API endpoint
    - disabled: Use simple frame difference fallback
    
    Args:
        video_path: Path to video file or HTTP URL
        segments: List of SemanticSegment objects
        
    Returns:
        List of visual change scores (0.0-1.0) for each segment
    """
    if not segments:
        return []
    
    vjepa2_mode = os.getenv("VJEPA2_MODE", "colab").lower()
    logger.info(f"Computing visual change scores using mode: {vjepa2_mode}")
    
    visual_scores = []
    
    for segment_idx, segment in enumerate(segments):
        try:
            # Extract frames from segment
            frames = extract_frames(video_path, segment, interval=2.0)
            
            if not frames:
                logger.warning(f"No frames extracted for segment {segment.segment_id}, using default score")
                visual_scores.append(0.5)
                continue
            
            if vjepa2_mode == "local":
                # Use local V-JEPA2 model
                score = compute_visual_change_scores_local(frames)
                visual_scores.append(score)
                
            elif vjepa2_mode == "colab":
                # Send to Colab API
                colab_url = os.getenv("COLAB_API_URL")
                if not colab_url:
                    logger.warning("COLAB_API_URL not set, falling back to simple frame difference")
                    score = compute_visual_change_scores_fallback(frames)
                else:
                    try:
                        from utils.colab_vjepa2 import send_frames_to_colab_async
                        scores = await send_frames_to_colab_async(frames, colab_url)
                        import numpy as np
                        score = float(np.mean(scores)) if scores else 0.5
                    except Exception as e:
                        logger.warning(f"Colab API failed: {str(e)}, using fallback")
                        score = compute_visual_change_scores_fallback(frames)
                visual_scores.append(score)
                
            else:  # disabled or unknown
                # Use simple fallback
                score = compute_visual_change_scores_fallback(frames)
                visual_scores.append(score)
                
        except Exception as e:
            logger.error(f"Error computing visual score for segment {segment.segment_id}: {str(e)}")
            visual_scores.append(0.5)  # Default neutral score
    
    logger.info(f"Computed {len(visual_scores)} visual change scores")
    return visual_scores
