import os
import sys
from class_model import Model
import numpy as np
import torch
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
  """EfficientNet inference with using MIGraphX Execution Provider with cache (GPU io-binding)"""
  def __init__(self):
    super().__init__()
    self.sess = None
    self.sess_data = {'providers': ['MIGraphXExecutionProvider']}
    self.device = 'cuda'
    if not torch.cuda.is_available():
      raise Exception('CUDA is not available')
    self.version = get_efficientnet_version()
    self.half = is_fp16()
    self.imgsz = get_image_size(self.version)
    self._input_tensor = None
    if not self.sess_data['providers'][0] in ort.get_available_providers():
      raise Exception(f'MIGraphX Execution Provider is not available')
  def prepare_batch(self, batch_size):
    file_path = self.get_file_path(onnx_name(self.version, batch_size, self.half))
    try_export_model(file_path, batch_size, self.version, self.half)
    cache_path = file_path[:-4] + 'migx'
    if not os.path.exists(cache_path):
      try:
        #os.environ['ORT_MIGRAPHX_CACHE_PATH'] = self.get_file_path('')
        os.environ['ORT_MIGRAPHX_MODEL_CACHE_PATH'] = cache_path
        os.makedirs(cache_path, exist_ok=True)
        self.sess = ort.InferenceSession(file_path, **self.sess_data)
        del self.sess
        del os.environ['ORT_MIGRAPHX_MODEL_CACHE_PATH']
      except Exception as e:
        raise Exception(f'Failed to save compiled model {e}')
  def read(self):
    #os.environ['ORT_MIGRAPHX_CACHE_PATH'] = self.get_file_path('')
    file_path = self.get_file_path(onnx_name(self.version, self.batch_size, self.half))
    os.environ['ORT_MIGRAPHX_MODEL_CACHE_PATH'] = file_path[:-4] + 'migx'
    self.sess = ort.InferenceSession(file_path, **self.sess_data)
  def prepare(self):
    self.input_data = self.sess.io_binding()
    np_dtype = get_input_np_dtype(self.half)
    torch_dtype = torch.float16 if self.half else torch.float32
    images_shape = [self.batch_size, 3, self.imgsz, self.imgsz]
    self._input_tensor = torch.rand(images_shape, dtype=torch_dtype, device=self.device)
    self.input_data.bind_input('pixel_values', 'cuda', 0, np_dtype, images_shape, self._input_tensor.data_ptr())
    self.input_data.bind_output('logits', 'cuda')
  def inference(self):
    self.sess.run_with_iobinding(self.input_data)
  def shutdown(self):
    try:
      del os.environ['ORT_MIGRAPHX_CACHE_PATH']
      del os.environ['ORT_MIGRAPHX_MODEL_CACHE_PATH']
    except Exception as e:
      pass
