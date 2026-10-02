from .common import SileroOrt


class Model(SileroOrt):
  """Silero VAD streaming inference using ONNX Runtime CUDA Execution Provider"""
  providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
