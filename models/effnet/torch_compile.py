import os
import torch
import numpy as np
from class_model import Model
from transformers import EfficientNetForImageClassification
from .common import get_efficientnet_version, get_image_size, is_fp16

class Model(Model):
  """EfficientNet inference with using default Torch.Compile"""
  def __init__(self):
    super().__init__()
    self.model = None
    self.device = 'cuda'
    if not torch.cuda.is_available():
      raise Exception('CUDA is not available')
    self.version = get_efficientnet_version()
    self.half = is_fp16()
    self.imgsz = get_image_size(self.version)
    self.model_id = f'google/efficientnet-{self.version}'
  def read(self):
    model = EfficientNetForImageClassification.from_pretrained(self.model_id)
    model.eval()
    model.to(self.device)
    if self.half:
      model = model.half()
    # Compile the model for improved performance
    self.model = torch.compile(model, mode='default')
  def prepare(self):
    # Create random input tensor (B, C, H, W)
    dtype = torch.float16 if self.half else torch.float32
    self.input_data = torch.randn(
        self.batch_size, 3, self.imgsz, self.imgsz,
        dtype=dtype,
        device=self.device
    )
    min_val = self.input_data.min()
    max_val = self.input_data.max()
    self.input_data = (self.input_data - min_val) / (max_val - min_val)
  def inference(self):
    with torch.no_grad():
      return self.model(pixel_values=self.input_data).logits
  def shutdown(self):
    if self.model is not None:
      del self.model
      self.model = None
      torch.cuda.empty_cache() if torch.cuda.is_available() else None
