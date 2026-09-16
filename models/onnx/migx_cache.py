import os
import onnxruntime as ort
import migraphx
from class_model import Model
from . import common


class Model(Model):
  """ONNX model inference using direct MIGraphX with cache"""
  def __init__(self):
    super().__init__()
    self.model = None
    self.sess = None  # ONNX Runtime session used only to introspect inputs
    self.model_file = None
  def _cache_path(self):
    base = os.path.splitext(os.path.basename(self.model_file))[0]
    return self.get_file_path(base + '.mxr')
  def prepare_batch(self, batch_size):
    self.model_file = common.get_model_path()
    cache_path = self._cache_path()
    if not os.path.exists(cache_path):
      try:
        model = migraphx.parse_onnx(self.model_file)
        model.compile(migraphx.get_target('gpu'))
        migraphx.save(model, cache_path)
        del model
      except Exception as e:
        raise Exception(f'Failed to compile and save MIGraphX model: {e}')
  def read(self):
    if self.model_file is None:
      self.model_file = common.get_model_path()
    self.model = migraphx.load(self._cache_path())
    self.sess = ort.InferenceSession(self.model_file)
  def prepare(self):
    self.input_data = common.random_input_feed(self.sess, self.batch_size)
  def inference(self):
    return self.model.run(self.input_data)
  def shutdown(self):
    if self.model:
      del self.model
      self.model = None
