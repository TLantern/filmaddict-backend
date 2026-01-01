import base64
import logging
from typing import List, Any

import httpx
from PIL import Image

# numpy is optional - only needed when actually calling this function
logger = logging.getLogger(__name__)


def send_frames_to_colab(
    frames: List[Any],
    colab_url: str,
    timeout: float = 60.0,
) -> List[float]:
    """
    Send frame batches to Colab API endpoint for V-JEPA2 processing.
    
    Args:
        frames: List of numpy arrays (frames as images)
        colab_url: URL of the Colab API endpoint
        timeout: Request timeout in seconds
        
    Returns:
        List of visual change scores (0.0-1.0) for each frame pair
    """
    if not frames:
        return []
    
    try:
        import numpy as np
    except ImportError:
        raise ImportError("numpy is required for send_frames_to_colab. Install with: pip install numpy")
    
    logger.info(f"Sending {len(frames)} frames to Colab API: {colab_url}")
    
    try:
        # Convert frames to base64 encoded images
        frame_data = []
        for frame in frames:
            # Convert numpy array to PIL Image
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8)
            
            if len(frame.shape) == 3:
                img = Image.fromarray(frame)
            else:
                # Grayscale
                img = Image.fromarray(frame, mode='L').convert('RGB')
            
            # Convert to base64
            import io
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG')
            img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            frame_data.append(img_base64)
        
        # Send to Colab API
        with httpx.Client(timeout=timeout) as client:
            response = client.post(
                f"{colab_url}/process_frames",
                json={"frames": frame_data},
            )
            response.raise_for_status()
            result = response.json()
            
            visual_scores = result.get("visual_change_scores", [])
            logger.info(f"Received {len(visual_scores)} visual change scores from Colab")
            return visual_scores
            
    except httpx.TimeoutException:
        logger.error(f"Timeout sending frames to Colab API")
        raise Exception("Colab API timeout")
    except httpx.HTTPError as e:
        logger.error(f"HTTP error sending frames to Colab: {str(e)}")
        raise Exception(f"Colab API error: {str(e)}")
    except Exception as e:
        logger.error(f"Error sending frames to Colab: {str(e)}", exc_info=True)
        raise Exception(f"Failed to send frames to Colab: {str(e)}")


async def send_frames_to_colab_async(
    frames: List[Any],
    colab_url: str,
    timeout: float = 60.0,
) -> List[float]:
    """Async wrapper for send_frames_to_colab."""
    import asyncio
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: send_frames_to_colab(frames, colab_url, timeout),
    )
