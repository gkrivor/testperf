from .torch_common import SileroRebuiltStream


class Model(SileroRebuiltStream):
  """Silero VAD streaming inference using the rebuilt nn.Module in eager mode"""
