import asyncio
import logging
import os
from typing import List, Dict, Tuple, Optional

import httpx

from models import Word, Sentence, SemanticSegment

logger = logging.getLogger(__name__)


async def process_video_on_colab(video_url: str, enable_visual: bool = True) -> Tuple[List[Word], List[Sentence], List[SemanticSegment], Dict[int, float]]:
    """
    Send video processing request to Colab API.
    
    Returns all processing results in one call - 10x faster than local processing.
    
    Args:
        video_url: URL to video file (S3 presigned URL or direct URL)
        enable_visual: Whether to enable visual analysis (default: True)
        
    Returns:
        Tuple of (words, sentences, segments, visual_scores)
    """
    colab_url = os.getenv("COLAB_API_URL")
    if not colab_url:
        raise ValueError("COLAB_API_URL environment variable not set")
    
    # Strip trailing slash to avoid double slashes in URLs
    colab_url = colab_url.rstrip('/')
    
    logger.info(f"[ColabProcessor] 🚀 Sending video processing request to Colab: {colab_url}")
    logger.info(f"[ColabProcessor] 📹 Video URL: {video_url[:80]}...")
    logger.info(f"[ColabProcessor] 🎬 Visual analysis: {enable_visual}")
    
    timeout = httpx.Timeout(600.0)  # 10 minute timeout for long videos
    logger.info(f"[ColabProcessor] ⏱️  Timeout set to {timeout.read} seconds")

    try:
        logger.info(f"[ColabProcessor] 📡 Sending POST request to {colab_url}/process_video...")

        # Configure SSL settings to be more tolerant of ngrok certificates
        # Disable SSL verification for ngrok URLs (they use self-signed certs)
        verify_ssl = False if "ngrok" in colab_url else True

        # Retry logic for SSL errors
        max_retries = 3
        retry_delay = 2  # seconds

        for attempt in range(max_retries):
            try:
                logger.info(f"[ColabProcessor] 📡 Attempt {attempt + 1}/{max_retries} - Sending POST request to {colab_url}/process_video...")

                async with httpx.AsyncClient(
                    timeout=timeout,
                    verify=verify_ssl,
                    follow_redirects=True
                ) as client:
                    # Add ngrok bypass header if using ngrok
                    headers = {}
                    if "ngrok" in colab_url:
                        headers["ngrok-skip-browser-warning"] = "true"

                    response = await client.post(
                        f"{colab_url}/process_video",
                        json={
                            "video_url": video_url,
                            "enable_visual": enable_visual
                        },
                        headers=headers,
                    )
                    break  # Success, exit retry loop

            except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout) as e:
                if "SSL" in str(e) or "decryption failed" in str(e).lower():
                    if attempt < max_retries - 1:
                        logger.warning(f"[ColabProcessor] ⚠️ SSL error on attempt {attempt + 1}, retrying in {retry_delay}s: {str(e)}")
                        await asyncio.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                        continue
                    else:
                        logger.error(f"[ColabProcessor] ❌ SSL error after {max_retries} attempts: {str(e)}")
                        raise Exception(f"SSL connection failed after {max_retries} attempts: {str(e)}")
                else:
                    # Non-SSL connection error, don't retry
                    raise e
            except Exception as e:
                # For other errors, don't retry
                if attempt == 0:  # Only log on first attempt to avoid spam
                    logger.error(f"[ColabProcessor] ❌ Error on attempt {attempt + 1}: {str(e)}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                    continue
                raise e
            logger.info(f"[ColabProcessor] ✅ Received response: {response.status_code}")
            response.raise_for_status()
            logger.info(f"[ColabProcessor] 📊 Parsing response JSON...")
            result = response.json()
            logger.info(f"[ColabProcessor] ✅ Response parsed successfully")
            
            # Convert response to model objects
            logger.info(f"[ColabProcessor] 🔄 Converting response to model objects...")
            words_data = result.get("words", [])
            logger.info(f"[ColabProcessor] 📝 Converting {len(words_data)} words...")
            words = [Word(**w) for w in words_data]
            
            sentences_data = result.get("sentences", [])
            logger.info(f"[ColabProcessor] 📝 Converting {len(sentences_data)} sentences...")
            sentences = []
            for i, s_data in enumerate(sentences_data):
                # Convert word dicts to Word objects
                word_objs = [Word(**w) for w in s_data.get("words", []) if isinstance(w, dict)]
                sentence = Sentence(
                    start=s_data["start"],
                    end=s_data["end"],
                    text=s_data.get("text", ""),
                    words=word_objs
                )
                sentences.append(sentence)
            
            segments_data = result.get("segments", [])
            logger.info(f"[ColabProcessor] 📝 Converting {len(segments_data)} segments...")
            segments = []
            for seg_data in segments_data:
                segment = SemanticSegment(
                    segment_id=seg_data["segment_id"],
                    start_time=seg_data["start_time"],
                    end_time=seg_data["end_time"],
                    text=seg_data["text"],
                    embedding=seg_data.get("embedding")
                )
                segments.append(segment)
            
            visual_scores = result.get("visual_scores", {})
            # Convert string keys to int
            visual_scores = {int(k): float(v) for k, v in visual_scores.items()}
            logger.info(f"[ColabProcessor] 📝 Processed {len(visual_scores)} visual scores")
            
            logger.info(
                f"[ColabProcessor] ✅ Colab processing complete: {len(words)} words, {len(sentences)} sentences, "
                f"{len(segments)} segments, {len(visual_scores)} visual scores"
            )
            
            return words, sentences, segments, visual_scores
            
    except httpx.TimeoutException:
        logger.error(f"[ColabProcessor] ❌ Colab API timeout after {timeout.read} seconds")
        raise Exception("Colab processing timeout - video may be too long")
    except httpx.HTTPStatusError as e:
        error_text = e.response.text[:500] if e.response.text else "No error text"
        logger.error(
            f"[ColabProcessor] ❌ Colab API HTTP error: {e.response.status_code}\n"
            f"Response: {error_text}\n"
            f"URL: {colab_url}/process_video"
        )
        raise Exception(f"Colab API HTTP {e.response.status_code}: {error_text}")
    except Exception as e:
        logger.error(f"[ColabProcessor] ❌ Error sending to Colab: {str(e)}", exc_info=True)
        raise Exception(f"Failed to process video on Colab: {str(e)}")

