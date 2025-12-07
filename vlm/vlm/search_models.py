"""
Search Model Abstraction Module

Supports multiple retrieval models:
- VisRAG
- RemoteCLIP
- DGTRS-CLIP
- RS5M-CLIP
"""

import torch
import torch.nn.functional as F
import numpy as np
from abc import ABC, abstractmethod
from PIL import Image
import warnings
import traceback


class BaseSearchModel(ABC):
    """Abstract base class for search models."""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.preprocess = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._loaded = False

    @abstractmethod
    def load(self):
        """Load the model. Should set self._loaded = True on success."""
        raise NotImplementedError

    @abstractmethod
    def encode_text(self, texts: list) -> np.ndarray:
        """Encode a list of text strings into embeddings."""
        raise NotImplementedError

    @abstractmethod
    def encode_images(self, images: list) -> np.ndarray:
        """Encode a list of PIL Images into embeddings."""
        raise NotImplementedError

    def encode(self, text_or_image_list: list) -> np.ndarray:
        """
        Encode a list of texts or images.
        Automatically detects the input type.
        """
        if len(text_or_image_list) == 0:
            return np.array([])

        if isinstance(text_or_image_list[0], str):
            return self.encode_text(text_or_image_list)
        else:
            return self.encode_images(text_or_image_list)

    @property
    def is_loaded(self):
        return self._loaded


class VisRAGSearchModel(BaseSearchModel):
    """VisRAG retrieval model (openbmb/VisRAG-Ret)."""

    def __init__(self, model_path: str = 'openbmb/VisRAG-Ret'):
        super().__init__()
        self.model_path = model_path

    def load(self):
        if self._loaded:
            return True

        try:
            from transformers import AutoModel, AutoTokenizer

            print(f"[VisRAG] Loading model from {self.model_path}...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, trust_remote_code=True
            )
            self.model = AutoModel.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True
            ).to(self.device)
            self.model.eval()
            self._loaded = True
            print("[VisRAG] Model loaded successfully.")
            return True
        except Exception as e:
            warnings.warn(f"[VisRAG] Failed to load model from {self.model_path}: {e}")
            traceback.print_exc()
            return False

    def _unweighted_mean_pooling(self, hidden, attention_mask):
        reps = torch.mean(hidden, dim=1).float()
        return reps

    @torch.no_grad()
    def encode_text(self, texts: list) -> np.ndarray:
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        max_batch_size = 64
        embedding_list = []

        for i in range(0, len(texts), max_batch_size):
            batch_texts = texts[i:i + max_batch_size]
            inputs = {
                "text": batch_texts,
                'image': [None] * len(batch_texts),
                'tokenizer': self.tokenizer
            }
            outputs = self.model(**inputs)
            hidden = outputs.last_hidden_state

            reps = self._unweighted_mean_pooling(hidden, outputs.attention_mask)
            embeddings = F.normalize(reps, p=2, dim=1).detach().cpu().numpy()
            embedding_list.append(embeddings)

        return np.concatenate(embedding_list)

    @torch.no_grad()
    def encode_images(self, images: list) -> np.ndarray:
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        max_batch_size = 64
        embedding_list = []

        for i in range(0, len(images), max_batch_size):
            batch_images = images[i:i + max_batch_size]
            inputs = {
                "text": [''] * len(batch_images),
                'image': batch_images,
                'tokenizer': self.tokenizer
            }
            outputs = self.model(**inputs)
            hidden = outputs.last_hidden_state

            reps = self._unweighted_mean_pooling(hidden, outputs.attention_mask)
            embeddings = F.normalize(reps, p=2, dim=1).detach().cpu().numpy()
            embedding_list.append(embeddings)

        return np.concatenate(embedding_list)


class RemoteCLIPSearchModel(BaseSearchModel):
    """RemoteCLIP retrieval model for remote sensing images."""

    SUPPORTED_MODELS = ['RN50', 'ViT-B-32', 'ViT-L-14']

    def __init__(self, model_name: str = 'ViT-L-14', ckpt_path: str = None):
        super().__init__()

        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model: {model_name}. Choose from {self.SUPPORTED_MODELS}")

        self.model_name = model_name
        self.ckpt_path = ckpt_path

    def load(self):
        if self._loaded:
            return True

        try:
            import open_clip

            print(f"[RemoteCLIP] Loading model {self.model_name}...")

            self.model, _, self.preprocess = open_clip.create_model_and_transforms(self.model_name)
            self.tokenizer = open_clip.get_tokenizer(self.model_name)

            if self.ckpt_path:
                print(f"[RemoteCLIP] Loading checkpoint from {self.ckpt_path}...")
                ckpt = torch.load(self.ckpt_path, map_location="cpu")
                message = self.model.load_state_dict(ckpt)
                print(f"[RemoteCLIP] Checkpoint loaded: {message}")

            self.model = self.model.to(self.device).eval()
            self._loaded = True
            print("[RemoteCLIP] Model loaded successfully.")
            return True
        except Exception as e:
            warnings.warn(f"[RemoteCLIP] Failed to load model: {e}")
            traceback.print_exc()
            return False

    @torch.no_grad()
    def encode_text(self, texts: list) -> np.ndarray:
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        max_batch_size = 64
        embedding_list = []

        for i in range(0, len(texts), max_batch_size):
            batch_texts = texts[i:i + max_batch_size]
            text_tokens = self.tokenizer(batch_texts).to(self.device)
            text_features = self.model.encode_text(text_tokens)
            text_features = F.normalize(text_features, p=2, dim=1)
            embedding_list.append(text_features.detach().cpu().numpy())

        return np.concatenate(embedding_list)

    @torch.no_grad()
    def encode_images(self, images: list) -> np.ndarray:
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        max_batch_size = 64
        embedding_list = []

        for i in range(0, len(images), max_batch_size):
            batch_images = images[i:i + max_batch_size]
            image_tensors = torch.stack([self.preprocess(img) for img in batch_images]).to(self.device)
            image_features = self.model.encode_image(image_tensors)
            image_features = F.normalize(image_features, p=2, dim=1)
            embedding_list.append(image_features.detach().cpu().numpy())

        return np.concatenate(embedding_list)


class DGTRSCLIPSearchModel(BaseSearchModel):
    """DGTRS-CLIP retrieval model for remote sensing images (Long-CLIP based)."""

    def __init__(self, ckpt_path: str = None):
        super().__init__()
        self.ckpt_path = ckpt_path
        self.longclip = None  

    def load(self):
        if self._loaded:
            return True

        try:
            from .model import longclip
            self.longclip = longclip

            print(f"[DGTRS-CLIP] Loading model from {self.ckpt_path}...")

            self.model, self.preprocess = longclip.load(self.ckpt_path, device=self.device)
            self.model.eval()

            self._loaded = True
            print("[DGTRS-CLIP] Model loaded successfully.")
            return True
        except Exception as e:
            warnings.warn(f"[DGTRS-CLIP] Failed to load model: {e}")
            traceback.print_exc()
            return False

    @torch.no_grad()
    def encode_text(self, texts: list) -> np.ndarray:
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        max_batch_size = 64
        embedding_list = []

        for i in range(0, len(texts), max_batch_size):
            batch_texts = texts[i:i + max_batch_size]
            text_tokens = self.longclip.tokenize(batch_texts).to(self.device)
            text_features = self.model.encode_text(text_tokens)
            text_features = F.normalize(text_features, p=2, dim=1)
            embedding_list.append(text_features.float().detach().cpu().numpy())

        return np.concatenate(embedding_list)

    @torch.no_grad()
    def encode_images(self, images: list) -> np.ndarray:
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        max_batch_size = 64
        embedding_list = []

        for i in range(0, len(images), max_batch_size):
            batch_images = images[i:i + max_batch_size]
            image_tensors = torch.stack([self.preprocess(img) for img in batch_images]).to(self.device)
            image_features = self.model.encode_image(image_tensors)
            image_features = F.normalize(image_features, p=2, dim=1)
            embedding_list.append(image_features.float().detach().cpu().numpy())

        return np.concatenate(embedding_list)


class RS5MCLIPSearchModel(BaseSearchModel):
    """RS5M-CLIP retrieval model for remote sensing images (ViT-B/32 based)."""

    def __init__(self, ckpt_path: str = None):
        super().__init__()
        self.ckpt_path = ckpt_path

    def load(self):
        if self._loaded:
            return True

        try:
            import open_clip
            from .inference.inference_tool import get_preprocess

            print(f"[RS5M-CLIP] Loading model from {self.ckpt_path}...")
            self.model, _, _ = open_clip.create_model_and_transforms("ViT-B/32", pretrained="openai")
            self.tokenizer = open_clip.get_tokenizer("ViT-B-32")

            if self.ckpt_path:
                print(f"[RS5M-CLIP] Loading checkpoint from {self.ckpt_path}...")
                checkpoint = torch.load(self.ckpt_path, map_location="cpu")
                msg = self.model.load_state_dict(checkpoint, strict=False)
                print(f"[RS5M-CLIP] Checkpoint loaded: {msg}")

            self.model = self.model.to(self.device).eval()

            self.preprocess = get_preprocess(image_resolution=224)

            self._loaded = True
            print("[RS5M-CLIP] Model loaded successfully.")
            return True
        except Exception as e:
            warnings.warn(f"[RS5M-CLIP] Failed to load model: {e}")
            traceback.print_exc()
            return False

    @torch.no_grad()
    def encode_text(self, texts: list) -> np.ndarray:
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        max_batch_size = 64
        embedding_list = []

        for i in range(0, len(texts), max_batch_size):
            batch_texts = texts[i:i + max_batch_size]
            text_tokens = self.tokenizer(batch_texts).to(self.device)
            text_features = self.model.encode_text(text_tokens)
            text_features = F.normalize(text_features, p=2, dim=1)
            embedding_list.append(text_features.float().detach().cpu().numpy())

        return np.concatenate(embedding_list)

    @torch.no_grad()
    def encode_images(self, images: list) -> np.ndarray:
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        max_batch_size = 64
        embedding_list = []

        for i in range(0, len(images), max_batch_size):
            batch_images = images[i:i + max_batch_size]
            image_tensors = torch.stack([self.preprocess(img) for img in batch_images]).to(self.device)
            image_features = self.model.encode_image(image_tensors)
            image_features = F.normalize(image_features, p=2, dim=1)
            embedding_list.append(image_features.float().detach().cpu().numpy())

        return np.concatenate(embedding_list)


def create_search_model(model_type: str, **kwargs) -> BaseSearchModel:
    """
    Factory function to create search models.

    Args:
        model_type: 'visrag', 'remoteclip', 'dgtrs', or 'rs5m'
        **kwargs: Model-specific parameters
    """
    model_type = model_type.lower()

    if model_type == 'visrag':
        model_path = kwargs.get('model_path', 'openbmb/VisRAG-Ret')
        return VisRAGSearchModel(model_path=model_path)

    elif model_type == 'remoteclip':
        model_name = kwargs.get('model_name', 'ViT-L-14')
        ckpt_path = kwargs.get('ckpt_path', None)
        return RemoteCLIPSearchModel(model_name=model_name, ckpt_path=ckpt_path)

    elif model_type in ['dgtrs', 'dgtrs-clip', 'dgtrsclip']:
        ckpt_path = kwargs.get('ckpt_path', None)
        return DGTRSCLIPSearchModel(ckpt_path=ckpt_path)

    elif model_type in ['rs5m', 'rs5m-clip', 'rs5mclip']:
        ckpt_path = kwargs.get('ckpt_path', None)
        return RS5MCLIPSearchModel(ckpt_path=ckpt_path)

    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose from ['visrag', 'remoteclip', 'dgtrs', 'rs5m']")