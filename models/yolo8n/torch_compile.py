import os
import torch
import numpy as np
from class_model import Model
from ultralytics import YOLO
from .common import (
    WEIGHTS_BASE_URL,
    get_image_size,
    get_model_name,
    get_yolo_task,
    is_fp16,
    weights_name,
)

class Model(Model):
  """YOLOv8n inference with using default Torch.Compile"""
  def __init__(self):
    super().__init__()
    self.model = None
    self.device = 'cuda'
    if not torch.cuda.is_available():
      raise Exception('CUDA is not available')
    self.model_name = get_model_name()
    self.task = get_yolo_task()
    self.half = is_fp16()
    self.imgsz = get_image_size(self.task)
    self.model_path = weights_name(self.model_name, self.task)
  def prepare_batch(self, batch_size):
    self.details[f'Model File (batch {batch_size})'] = f"{self.model_path} {self.imgsz} {'fp16' if self.half else 'fp32'}"
  def read(self):
    if not os.path.exists(self.model_path):
      try:
        import urllib.request
        urllib.request.urlretrieve(WEIGHTS_BASE_URL + self.model_path, self.model_path)
      except Exception as e:
        raise Exception(f'Failed to download YOLO model {e}')
    if not os.path.exists(self.model_path):
      raise Exception(f'Model file {self.model_path} not found')
    self.model = YOLO(self.model_path)
    self.model.to(self.device)
    if self.half:
      self.model.model.half()
    # Compile the model for improved performance
    self.model.model = torch.compile(self.model.model, mode='default')
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
      return self.model(self.input_data, verbose=False)
  def shutdown(self):
    if self.model is not None:
      del self.model
      torch.cuda.empty_cache() if torch.cuda.is_available() else None
