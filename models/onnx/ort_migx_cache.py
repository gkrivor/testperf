import os
import onnxruntime as ort
from class_model import Model
from . import common


class Model(Model):
  """ONNX model inference using MIGraphX Execution Provider with cache"""
  def __init__(self):
    super().__init__()
    self.sess = None
    self.sess_data = {'providers': ['MIGraphXExecutionProvider']}
    self.model_file = None
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
    self.input_data = common.random_input_feed(self.sess, self.batch_size)
  def inference(self):
    return self.sess.run([], input_feed=self.input_data)
  def shutdown(self):
    try:
      del os.environ['ORT_MIGRAPHX_CACHE_PATH']
      del os.environ['ORT_MIGRAPHX_MODEL_CACHE_PATH']
    except Exception:
      pass
