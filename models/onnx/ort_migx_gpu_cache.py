import os
import numpy as np
import torch
import onnxruntime as ort
from class_model import Model
from . import common


class Model(Model):
  """ONNX model inference using MIGraphX Execution Provider with cache and IO binding"""
  def __init__(self):
    super().__init__()
    self.sess = None
    self.sess_data = {'providers': ['MIGraphXExecutionProvider']}
    self.device = 'cuda'
    self.model_file = None
    self._gpu_tensors = []
    if not torch.cuda.is_available():
      raise Exception('CUDA is not available')
    if not self.sess_data['providers'][0] in ort.get_available_providers():
      raise Exception('MIGraphX Execution Provider is not available')
  def _cache_path(self):
    base = os.path.splitext(os.path.basename(self.model_file))[0]
    return self.get_file_path(base + '.migx')
  def prepare_batch(self, batch_size):
    self.model_file = common.get_model_path()
    cache_path = self._cache_path()
    if not os.path.exists(cache_path):
      try:
        os.environ['ORT_MIGRAPHX_MODEL_CACHE_PATH'] = cache_path
        os.makedirs(cache_path, exist_ok=True)
        self.sess = ort.InferenceSession(self.model_file, **self.sess_data)
        del self.sess
        del os.environ['ORT_MIGRAPHX_MODEL_CACHE_PATH']
      except Exception as e:
        raise Exception(f'Failed to save compiled model {e}')
  def read(self):
    if self.model_file is None:
      self.model_file = common.get_model_path()
    os.environ['ORT_MIGRAPHX_MODEL_CACHE_PATH'] = self._cache_path()
    self.sess = ort.InferenceSession(self.model_file, **self.sess_data)
  def prepare(self):
    self.input_data = self.sess.io_binding()
    self._gpu_tensors = []
    for node in common.get_model_inputs(self.sess):
      shape = list(common.resolve_shape(node['shape'], self.batch_size))
      np_dtype = common.get_ort_input_np_dtype(node['type'])
      tensor = torch.rand(shape, device=self.device).to(
        dtype=getattr(torch, np.dtype(np_dtype).name, torch.float32))
      # Keep a reference so the GPU memory is not freed before inference
      self._gpu_tensors.append(tensor)
      self.input_data.bind_input(node['name'], 'cuda', 0, np_dtype, shape, tensor.data_ptr())
    for node in common.get_model_outputs(self.sess):
      self.input_data.bind_output(node['name'], 'cuda')
  def inference(self):
    self.sess.run_with_iobinding(self.input_data)
  def shutdown(self):
    self._gpu_tensors = []
    try:
      del os.environ['ORT_MIGRAPHX_CACHE_PATH']
      del os.environ['ORT_MIGRAPHX_MODEL_CACHE_PATH']
    except Exception:
      pass
