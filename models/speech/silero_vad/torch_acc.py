import torch

from ..common import SAMPLE_RATE
from .torch_common import SileroTorchStream


class Model(SileroTorchStream):
  """Silero VAD streaming inference using the shipped TorchScript model"""
  def read(self):
    self.model = torch.jit.load(self.model_file, map_location=self.device).eval()

  def reset_state(self):
    self.model.reset_states()

  def forward_chunk(self, chunk):
    return self.model(chunk, SAMPLE_RATE)
