import torch

torch.set_grad_enabled(False)
torch.manual_seed(1234)
from .base import BaseModel
from .llava import LLaVA
from .qwen2_vl import Qwen2VLChat
