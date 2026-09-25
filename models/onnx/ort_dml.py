import onnxruntime as ort
from class_model import Model
from . import common


class Model(Model):
  """ONNX model inference using DirectML Execution Provider"""
  def __init__(self):
    super().__init__()
    self.sess = None
    self.sess_data = {'providers': ['DmlExecutionProvider']}
    self.model_file = None
    if not self.sess_data['providers'][0] in ort.get_available_providers():
      raise Exception('DirectML Execution Provider is not available')
  def prepare_batch(self, batch_size):
    self.model_file = common.get_model_path()
    self.details[f'Model File (batch {batch_size})'] = self.model_file
  def read(self):
    if self.model_file is None:
      self.model_file = common.get_model_path()
    self.sess = ort.InferenceSession(self.model_file, **self.sess_data)
  def prepare(self):
    self.input_data = common.random_input_feed(self.sess, self.batch_size)
  def inference(self):
    return self.sess.run([], input_feed=self.input_data)
  def shutdown(self):
    pass
