import numpy as np
import torch

from ..common import AudioModel
from .common import WINDOW, get_silero_file


class Model(AudioModel):
  """Silero VAD streaming GPU performance test (shipped TorchScript).

  Pure performance test: read the model and run inference over sliding windows
  of the provided source_buffer. GPU only.
  """
  def __init__(self):
    super().__init__()
    self.device = torch.device('cuda')
    self.model_sliding_window = WINDOW / self.model_sample_rate  # 512-sample chunk
    self.model = None

  def read(self):
    self.model = torch.jit.load(get_silero_file('silero_vad.jit'), map_location=self.device).eval()

  def prepare(self):
    super().prepare()
    pos = int(self.source_position)
    window = self.source_buffer[pos:pos + WINDOW]
    if len(window) < WINDOW:
      window = np.pad(window, (0, WINDOW - len(window)))
    self.batch = torch.from_numpy(np.ascontiguousarray(window)).reshape(1, -1).repeat(self.batch_size, 1).to(self.device)

  def inference(self):
    with torch.inference_mode():
      prob = self.model(self.batch, self.model_sample_rate)
      torch.cuda.synchronize(self.device)
    return prob

  def shutdown(self):
    if self.model is not None:
      del self.model
      self.model = None
      torch.cuda.empty_cache()
