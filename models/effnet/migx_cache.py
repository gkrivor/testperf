import os
from class_model import Model
import numpy as np
import migraphx
from .common import (
    get_efficientnet_version,
    get_image_size,
    get_input_np_dtype,
    is_fp16,
    onnx_name,
    try_export_model,
)

class Model(Model):
  """EfficientNet inference using direct MIGraphX with cache"""
  def __init__(self):
    super().__init__()
    self.model = None
    self.version = get_efficientnet_version()
    self.half = is_fp16()
    self.imgsz = get_image_size(self.version)
  def prepare_batch(self, batch_size):
    file_path = self.get_file_path(onnx_name(self.version, batch_size, self.half))
    try_export_model(file_path, batch_size, self.version, self.half)
    # Compile model to MIGraphX binary cache
    cache_path = file_path[:-4] + 'mxr'
    if not os.path.exists(cache_path):
      try:
        model = migraphx.parse_onnx(file_path)
        model.compile(migraphx.get_target("gpu"))
        migraphx.save(model, cache_path)
        del model
      except Exception as e:
        raise Exception(f'Failed to compile and save MIGraphX model: {e}')
  def read(self):
    file_path = self.get_file_path(onnx_name(self.version, self.batch_size, self.half))
    cache_path = file_path[:-4] + 'mxr'
    self.model = migraphx.load(cache_path)
  def prepare(self):
    self.input_data = np.random.randn(self.batch_size, 3, self.imgsz, self.imgsz).astype(get_input_np_dtype(self.half))
  def inference(self):
    return self.model.run({'pixel_values': self.input_data})
  def shutdown(self):
    if self.model:
      del self.model
      self.model = None
