import os
from class_model import Model
import numpy as np
import onnxruntime as ort
from .common import (
    get_image_size,
    get_input_np_dtype,
    get_model_name,
    get_yolo_task,
    is_fp16,
    onnx_name,
    try_export_model,
)

class Model(Model):
  """YOLOv8n inference with using MIGraphX Execution Provider"""
  def __init__(self):
    super().__init__()
    self.sess = None
    self.sess_data = {'providers': ['MIGraphXExecutionProvider']}
    self.model_name = get_model_name()
    self.task = get_yolo_task()
    self.half = is_fp16()
    self.imgsz = get_image_size(self.task)
    if not self.sess_data['providers'][0] in ort.get_available_providers():
      raise Exception(f'MIGraphX Execution Provider is not available')
  def prepare_batch(self, batch_size):
    file_path = self.get_file_path(onnx_name(self.model_name, self.task, batch_size, self.half))
    try_export_model(file_path, batch_size, self.half, model_name=self.model_name, task=self.task, imgsz=self.imgsz)
    self.details[f'Model File (batch {batch_size})'] = file_path
  def read(self):
    file_path = self.get_file_path(onnx_name(self.model_name, self.task, self.batch_size, self.half))
    self.sess = ort.InferenceSession(file_path, **self.sess_data)
  def prepare(self):
    self.input_data = {
      'images': np.random.randn(self.batch_size, 3, self.imgsz, self.imgsz).astype(get_input_np_dtype(self.half)),
    }
  def inference(self):
    #return self.sess.run(['output0'], input_feed=self.input_data)
    return self.sess.run([], input_feed=self.input_data)
  def shutdown(self):
    pass
