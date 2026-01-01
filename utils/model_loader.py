import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# ML libraries are optional - only needed if Colab is unavailable
_torch_available = False
_transformers_available = False
_sentence_transformers_available = False

try:
    import torch
    _torch_available = True
except ImportError:
    pass

try:
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    _transformers_available = True
except ImportError:
    pass

try:
    from sentence_transformers import SentenceTransformer
    _sentence_transformers_available = True
except ImportError:
    pass

# Disable tokenizers parallelism to avoid deadlocks when forking
if _transformers_available:
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

_whisper_model = None
_whisper_processor = None
_embedding_model = None
_vjepa2_model = None


def get_device() -> str:
    """Get the best available device (CUDA if available, else CPU)."""
    if not _torch_available:
        raise ImportError(
            "PyTorch is not installed. Install it with: pip install torch\n"
            "Or use Colab processing by setting COLAB_API_URL environment variable."
        )
    import torch
    use_gpu = os.getenv("USE_GPU", "true").lower() == "true"
    if use_gpu and torch.cuda.is_available():
        device = "cuda"
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        logger.info("Using CPU")
    return device


def get_whisper_model():
    """Load Whisper large-v3 model (singleton pattern)."""
    if not _torch_available or not _transformers_available:
        raise ImportError(
            "PyTorch and Transformers are not installed. Install them with:\n"
            "pip install torch transformers\n"
            "Or use Colab processing by setting COLAB_API_URL environment variable."
        )
    
    import torch
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    
    global _whisper_model, _whisper_processor
    
    if _whisper_model is None:
        model_name = os.getenv("WHISPER_MODEL", "openai/whisper-large-v3")
        logger.info(f"Loading Whisper model: {model_name}")
        
        device = get_device()
        _whisper_processor = WhisperProcessor.from_pretrained(model_name)
        _whisper_model = WhisperForConditionalGeneration.from_pretrained(
            model_name,
            dtype=torch.float16 if device == "cuda" else torch.float32,
        ).to(device)
        _whisper_model.eval()
        
        logger.info(f"Whisper model loaded on {device}")
    
    return _whisper_model, _whisper_processor


def get_embedding_model():
    """Load BGE embedding model (singleton pattern)."""
    if not _torch_available or not _sentence_transformers_available:
        raise ImportError(
            "PyTorch and SentenceTransformers are not installed. Install them with:\n"
            "pip install torch sentence-transformers\n"
            "Or use Colab processing by setting COLAB_API_URL environment variable."
        )
    
    from sentence_transformers import SentenceTransformer
    
    global _embedding_model
    
    if _embedding_model is None:
        model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")
        logger.info(f"Loading embedding model: {model_name}")
        
        device = get_device()
        _embedding_model = SentenceTransformer(model_name, device=device)
        logger.info(f"Embedding model loaded on {device}")
    
    return _embedding_model


def get_vjepa2_model():
    """Load V-JEPA2 model (singleton pattern, only if local mode)."""
    global _vjepa2_model
    
    if _vjepa2_model is None:
        vjepa2_mode = os.getenv("VJEPA2_MODE", "colab").lower()
        
        if vjepa2_mode == "local":
            if not _torch_available or not _transformers_available:
                logger.warning(
                    "PyTorch/Transformers not available for V-JEPA2. "
                    "Install with: pip install torch transformers\n"
                    "Or use Colab mode (default) by setting VJEPA2_MODE=colab"
                )
                _vjepa2_model = None
                return None
            
            model_name = os.getenv("VJEPA2_MODEL", "facebook/vjepa2-vitl-fpc64-256")
            logger.info(f"Loading V-JEPA2 model: {model_name}")
            
            try:
                from transformers import AutoModel
                device = get_device()
                _vjepa2_model = AutoModel.from_pretrained(model_name).to(device)
                _vjepa2_model.eval()
                logger.info(f"V-JEPA2 model loaded on {device}")
            except Exception as e:
                logger.warning(f"Failed to load V-JEPA2 model locally: {str(e)}")
                _vjepa2_model = None
        else:
            logger.info("V-JEPA2 mode is not 'local', skipping model loading")
            _vjepa2_model = None
    
    return _vjepa2_model
