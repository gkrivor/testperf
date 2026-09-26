import os
from class_model import Model
import numpy as np
import openvino as ov
from .common import (
    get_image_size,
    get_model_name,
    get_yolo_task,
    is_fp16,
    onnx_name,
    try_export_model,
)

class Model(Model):
  """YOLOv8n inference with using OpenVINO"""
  def __init__(self):
    super().__init__()
    self.core = ov.Core()
    self.ov_model = None
    self.compiled_model = None
    self.model_name = get_model_name()
    self.task = get_yolo_task()
    self.half = is_fp16()
    self.imgsz = get_image_size(self.task)
  def prepare_batch(self, batch_size):
    file_path = self.get_file_path(onnx_name(self.model_name, self.task, batch_size, self.half))
    try_export_model(file_path, batch_size, self.half, model_name=self.model_name, task=self.task, imgsz=self.imgsz)
    self.details[f'Model File (batch {batch_size})'] = file_path
  def read(self):
    file_path = self.get_file_path(onnx_name(self.model_name, self.task, self.batch_size, self.half))
    self.ov_model = self.core.read_model(file_path)
    self.compiled_model = self.core.compile_model(self.ov_model, 'CPU')
  def prepare(self):
    self.input_data = {
      'images': np.random.randn(self.batch_size, 3, self.imgsz, self.imgsz).astype(np.float32),
    }
  def inference(self):
    return self.compiled_model(self.input_data)
  def shutdown(self):
    pass
