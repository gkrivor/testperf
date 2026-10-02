from .common import SileroOrt


class Model(SileroOrt):
  """Silero VAD streaming inference using ONNX Runtime MIGraphX Execution Provider"""
  providers = ['MIGraphXExecutionProvider', 'CPUExecutionProvider']
