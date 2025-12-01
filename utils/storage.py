import logging
import os
import tempfile
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional
from uuid import UUID

import boto3
from botocore.exceptions import ClientError
from fastapi import UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./uploads")
VIDEOS_DIR = os.path.join(UPLOAD_DIR, "videos")


class StorageInterface(ABC):
    """Abstract interface for video storage backends."""

    @abstractmethod
    async def store_video(self, file_content: bytes, filename: str) -> str:
        """
        Store video file content.
        
        Args:
            file_content: Video file content as bytes
            filename: Original filename (for extension detection)
            
        Returns:
            Relative storage path (e.g., "videos/uuid-filename.mp4")
        """
        pass

    @abstractmethod
    def get_video_path(self, relative_path: str) -> str:
        """
        Get full path for reading a video file.
        
        Args:
            relative_path: Relative storage path returned by store_video()
            
        Returns:
            Full path to the video file
        """
        pass

    @abstractmethod
    def delete_video(self, relative_path: str) -> bool:
        """
        Delete a video file.
        
        Args:
            relative_path: Relative storage path
            
        Returns:
            True if deleted successfully, False otherwise
        """
        pass


class LocalStorage(StorageInterface):
    """Local disk storage implementation."""

    def __init__(self, upload_dir: str = None):
        self.upload_dir = upload_dir or os.getenv("UPLOAD_DIR", "./uploads")
        self.videos_dir = os.path.join(self.upload_dir, "videos")
        os.makedirs(self.videos_dir, exist_ok=True)

    async def store_video(self, file_content: bytes, filename: str) -> str:
        """Store video file to local disk."""
        file_extension = Path(filename).suffix if filename else ".mp4"
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        storage_path = os.path.join(self.videos_dir, unique_filename)
        
        with open(storage_path, "wb") as f:
            f.write(file_content)
        
        relative_path = os.path.join("videos", unique_filename)
        logger.info(f"Stored video file: {relative_path}")
        return relative_path

    def get_video_path(self, relative_path: str) -> str:
        """Get full path for reading a video file."""
        return os.path.join(self.upload_dir, relative_path)

    def delete_video(self, relative_path: str) -> bool:
        """Delete a video file from local disk."""
        try:
            full_path = self.get_video_path(relative_path)
            if os.path.exists(full_path):
                os.remove(full_path)
                logger.info(f"Deleted video file: {relative_path}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error deleting video {relative_path}: {str(e)}")
            return False


class S3Storage(StorageInterface):
    """AWS S3 storage implementation."""

    def __init__(
        self,
        bucket_name: str = None,
        access_key_id: str = None,
        secret_access_key: str = None,
        region: str = None,
        endpoint_url: str = None,
    ):
        self.bucket_name = bucket_name or os.getenv("S3_BUCKET_NAME")
        if not self.bucket_name:
            raise ValueError("S3_BUCKET_NAME environment variable is required")
        
        self.access_key_id = access_key_id or os.getenv("AWS_ACCESS_KEY_ID")
        self.secret_access_key = secret_access_key or os.getenv("AWS_SECRET_ACCESS_KEY")
        self.region = region or os.getenv("AWS_REGION", "us-east-1")
        
        endpoint_url_raw = endpoint_url or os.getenv("S3_ENDPOINT_URL")
        if endpoint_url_raw:
            endpoint_url_raw = endpoint_url_raw.strip()
            if endpoint_url_raw.startswith("#") or not endpoint_url_raw:
                self.endpoint_url = None
            elif endpoint_url_raw.startswith(("http://", "https://")):
                self.endpoint_url = endpoint_url_raw
            else:
                logger.warning(f"Invalid S3_ENDPOINT_URL format, ignoring: {endpoint_url_raw}")
                self.endpoint_url = None
        else:
            self.endpoint_url = None
        
        if not self.access_key_id or not self.secret_access_key:
            raise ValueError("AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables are required")
        
        s3_config = {
            "aws_access_key_id": self.access_key_id,
            "aws_secret_access_key": self.secret_access_key,
            "region_name": self.region,
        }
        
        if self.endpoint_url:
            s3_config["endpoint_url"] = self.endpoint_url
        
        self.s3_client = boto3.client("s3", **s3_config)
        self.presigned_url_expiration = int(os.getenv("S3_PRESIGNED_URL_EXPIRATION", "3600"))
        
        logger.info(f"Initialized S3Storage with bucket: {self.bucket_name}, region: {self.region}")

    async def store_video(self, file_content: bytes, filename: str) -> str:
        """Store video file to S3."""
        file_extension = Path(filename).suffix if filename else ".mp4"
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        s3_key = f"videos/{unique_filename}"
        
        try:
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=file_content,
                ContentType="video/mp4",
            )
            logger.info(f"Stored video file to S3: {s3_key}")
            return s3_key
        except ClientError as e:
            logger.error(f"Error storing video to S3: {str(e)}")
            raise Exception(f"Failed to store video to S3: {str(e)}")

    def store_video_from_file(self, file_path: str, filename: str) -> str:
        """
        Stream upload video file from local path directly to S3.
        This avoids loading the entire file into memory.
        
        Args:
            file_path: Path to local file to upload
            filename: Original filename (for extension detection)
            
        Returns:
            Relative storage path (e.g., "videos/uuid-filename.mp4")
        """
        file_extension = Path(filename).suffix if filename else ".mp4"
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        s3_key = f"videos/{unique_filename}"
        
        try:
            with open(file_path, "rb") as f:
                self.s3_client.upload_fileobj(
                    f,
                    self.bucket_name,
                    s3_key,
                    ExtraArgs={"ContentType": "video/mp4"},
                )
            logger.info(f"Streamed video file to S3: {s3_key}")
            return s3_key
        except ClientError as e:
            logger.error(f"Error streaming video to S3: {str(e)}")
            raise Exception(f"Failed to stream video to S3: {str(e)}")
        except Exception as e:
            logger.error(f"Error reading file for S3 upload: {str(e)}")
            raise Exception(f"Failed to read file for upload: {str(e)}")

    async def store_video_from_url(self, url: str, filename: str) -> str:
        """
        Stream video directly from URL to S3 without saving to local disk.
        
        Args:
            url: Direct URL to video stream
            filename: Original filename (for extension detection)
            
        Returns:
            Relative storage path (e.g., "videos/uuid-filename.mp4")
        """
        import urllib.request
        import asyncio
        
        file_extension = Path(filename).suffix if filename else ".mp4"
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        s3_key = f"videos/{unique_filename}"
        
        def _stream_upload():
            """Synchronous function to stream from URL to S3."""
            try:
                # Stream from URL using urllib
                with urllib.request.urlopen(url, timeout=300) as response:
                    # Upload directly to S3 using the response stream
                    self.s3_client.upload_fileobj(
                        response,
                        self.bucket_name,
                        s3_key,
                        ExtraArgs={"ContentType": "video/mp4"},
                    )
                
                logger.info(f"Streamed video from URL to S3: {s3_key}")
                return s3_key
            except urllib.error.URLError as e:
                logger.error(f"Error streaming video from URL: {str(e)}")
                raise Exception(f"Failed to stream video from URL: {str(e)}")
            except ClientError as e:
                logger.error(f"Error streaming video to S3: {str(e)}")
                raise Exception(f"Failed to stream video to S3: {str(e)}")
        
        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, _stream_upload)
        except Exception as e:
            logger.error(f"Error in async stream upload: {str(e)}")
            raise

    def get_video_path(self, relative_path: str) -> str:
        """
        Get presigned URL for reading a video file from S3.
        
        For local file processing, use download_to_local() instead.
        """
        try:
            url = self.s3_client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.bucket_name, "Key": relative_path},
                ExpiresIn=self.presigned_url_expiration,
            )
            return url
        except ClientError as e:
            logger.error(f"Error generating presigned URL for {relative_path}: {str(e)}")
            raise Exception(f"Failed to generate presigned URL: {str(e)}")

    def download_to_local(self, relative_path: str, local_path: str = None) -> str:
        """
        Download video from S3 to a local file.
        
        Args:
            relative_path: S3 key (relative path)
            local_path: Optional local file path. If not provided, creates a temp file.
            
        Returns:
            Path to the local file
        """
        if local_path is None:
            file_extension = Path(relative_path).suffix
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_extension)
            local_path = temp_file.name
            temp_file.close()
        
        try:
            self.s3_client.download_file(self.bucket_name, relative_path, local_path)
            logger.info(f"Downloaded {relative_path} from S3 to {local_path}")
            return local_path
        except ClientError as e:
            logger.error(f"Error downloading {relative_path} from S3: {str(e)}")
            raise Exception(f"Failed to download video from S3: {str(e)}")

    def store_clip_from_file(self, file_path: str, filename: str) -> str:
        """
        Stream upload clip file from local path directly to S3.
        
        Args:
            file_path: Path to local clip file to upload
            filename: Original filename (for extension detection)
            
        Returns:
            Relative storage path (e.g., "clips/uuid.mp4")
        """
        file_extension = Path(filename).suffix if filename else ".mp4"
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        s3_key = f"clips/{unique_filename}"
        
        try:
            with open(file_path, "rb") as f:
                self.s3_client.upload_fileobj(
                    f,
                    self.bucket_name,
                    s3_key,
                    ExtraArgs={"ContentType": "video/mp4"},
                )
            logger.info(f"Streamed clip file to S3: {s3_key}")
            return s3_key
        except ClientError as e:
            logger.error(f"Error streaming clip to S3: {str(e)}")
            raise Exception(f"Failed to stream clip to S3: {str(e)}")
        except Exception as e:
            logger.error(f"Error reading clip file for S3 upload: {str(e)}")
            raise Exception(f"Failed to read clip file for upload: {str(e)}")

    def store_thumbnail_from_file(self, file_path: str, filename: str) -> str:
        """
        Stream upload thumbnail file from local path directly to S3.
        
        Args:
            file_path: Path to local thumbnail file to upload
            filename: Original filename (for extension detection)
            
        Returns:
            Relative storage path (e.g., "thumbnails/uuid.jpg")
        """
        file_extension = Path(filename).suffix if filename else ".jpg"
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        s3_key = f"thumbnails/{unique_filename}"
        
        try:
            with open(file_path, "rb") as f:
                self.s3_client.upload_fileobj(
                    f,
                    self.bucket_name,
                    s3_key,
                    ExtraArgs={"ContentType": "image/jpeg"},
                )
            logger.info(f"Streamed thumbnail file to S3: {s3_key}")
            return s3_key
        except ClientError as e:
            logger.error(f"Error streaming thumbnail to S3: {str(e)}")
            raise Exception(f"Failed to stream thumbnail to S3: {str(e)}")
        except Exception as e:
            logger.error(f"Error reading thumbnail file for S3 upload: {str(e)}")
            raise Exception(f"Failed to read thumbnail file for upload: {str(e)}")

    def delete_video(self, relative_path: str) -> bool:
        """Delete a video file from S3."""
        try:
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=relative_path)
            logger.info(f"Deleted video file from S3: {relative_path}")
            return True
        except ClientError as e:
            logger.error(f"Error deleting video {relative_path} from S3: {str(e)}")
            return False


def get_storage() -> StorageInterface:
    """
    Factory function to get the appropriate storage instance.
    
    Returns:
        StorageInterface implementation based on configuration
    """
    storage_type = os.getenv("STORAGE_TYPE", "s3").lower()
    
    if storage_type == "s3":
        return S3Storage()
    elif storage_type == "local":
        return LocalStorage()
    # Future: Add R2 implementation here
    # elif storage_type == "r2":
    #     return R2Storage(...)
    else:
        logger.warning(f"Unknown storage type '{storage_type}', defaulting to S3")
        return S3Storage()


# Global storage instance
_storage_instance: Optional[StorageInterface] = None


def get_storage_instance() -> StorageInterface:
    """Get or create the global storage instance."""
    global _storage_instance
    if _storage_instance is None:
        _storage_instance = get_storage()
    return _storage_instance


async def store_video(file: UploadFile) -> str:
    """
    Store uploaded video file using the configured storage backend.
    
    Args:
        file: FastAPI UploadFile object
        
    Returns:
        Storage path relative to uploads directory (e.g., "videos/uuid-filename.mp4")
    """
    storage = get_storage_instance()
    content = await file.read()
    filename = file.filename or "video.mp4"
    return await storage.store_video(content, filename)


async def get_video_path(video_id: UUID, db: AsyncSession, download_local: bool = True) -> Optional[str]:
    """
    Get full storage path for a video by ID.
    
    For S3 storage, downloads the file to a temporary local file if download_local=True
    (needed for ffmpeg processing). Otherwise returns a presigned URL.
    
    Args:
        video_id: UUID of the video
        db: Database session
        download_local: If True and storage is S3, download to temp file. Default True.
        
    Returns:
        Full storage path (local file path or presigned URL) or None if video not found
    """
    from db.crud import get_video_by_id
    
    video = await get_video_by_id(db, video_id)
    if not video:
        return None
    
    storage = get_storage_instance()
    
    if isinstance(storage, S3Storage) and download_local:
        return storage.download_to_local(video.storage_path)
    
    return storage.get_video_path(video.storage_path)

