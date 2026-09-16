import onnxruntime as ort
import openvino as ov
from class_model import Model
from . import common


class Model(Model):
  """ONNX model inference using OpenVINO"""
  def __init__(self):
    super().__init__()
    self.core = ov.Core()
    self.ov_model = None
    self.compiled_model = None
    self.sess = None  # ONNX Runtime session used only to introspect inputs
    self.model_file = None
  def prepare_batch(self, batch_size):
    self.model_file = common.get_model_path()
  def read(self):
    if self.model_file is None:
      self.model_file = common.get_model_path()
    self.ov_model = self.core.read_model(self.model_file)
    self.compiled_model = self.core.compile_model(self.ov_model, 'CPU')
    # Lightweight CPU session to list model inputs for random data generation
    self.sess = ort.InferenceSession(self.model_file)
  def prepare(self):
    self.input_data = common.random_input_feed(self.sess, self.batch_size)
  def inference(self):
    return self.compiled_model(self.input_data)
  def shutdown(self):
    pass
