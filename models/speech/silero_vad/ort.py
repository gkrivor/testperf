from .common import SileroOrt


class Model(SileroOrt):
  """Silero VAD streaming inference using ONNX Runtime CPU Execution Provider"""
  providers = ['CPUExecutionProvider']
