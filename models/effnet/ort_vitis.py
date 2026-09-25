import os
from class_model import Model
import numpy as np
import onnxruntime as ort
from .common import (
    get_efficientnet_version,
    get_image_size,
    get_input_np_dtype,
    is_fp16,
    onnx_name,
    try_export_model,
)

class Model(Model):
  """EfficientNet inference with using Vitis AI Execution Provider"""
  def __init__(self):
    super().__init__()
    self.sess = None
    self.sess_data = {'providers': ['VitisAIExecutionProvider']}
    self.version = get_efficientnet_version()
    self.half = is_fp16()
    self.imgsz = get_image_size(self.version)
    if not self.sess_data['providers'][0] in ort.get_available_providers():
      raise Exception(f'Vitis AI Execution Provider is not available')
  def prepare_batch(self, batch_size):
    file_path = self.get_file_path(onnx_name(self.version, batch_size, self.half))
    try_export_model(file_path, batch_size, self.version, self.half)
  def read(self):
    file_path = self.get_file_path(onnx_name(self.version, self.batch_size, self.half))
    self.sess = ort.InferenceSession(file_path, **self.sess_data)
  def prepare(self):
    self.input_data = {
      'pixel_values': np.random.randn(self.batch_size, 3, self.imgsz, self.imgsz).astype(get_input_np_dtype(self.half)),
    }
  def inference(self):
    return self.sess.run([], input_feed=self.input_data)
  def shutdown(self):
    pass
