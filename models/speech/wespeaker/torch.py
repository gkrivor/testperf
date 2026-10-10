import numpy as np
import torch

from ..common import AudioModel


class Model(AudioModel):
  """WeSpeaker ResNet34 speaker embedding GPU performance test (pyannote.audio).

  Pure performance test: read the model and run inference over sliding windows
  of the provided source_buffer. GPU only.
  """
  def __init__(self):
    super().__init__()
    self.device = torch.device('cuda')
    self.model_sliding_window = 5  # seconds
    self.model = None

  def read(self):
    from pyannote.audio import Model as PyannoteModel
    self.model = PyannoteModel.from_pretrained('pyannote/wespeaker-voxceleb-resnet34-LM').to(self.device).eval()

  def prepare(self):
    super().prepare()
    samples = self.model_sample_rate * self.model_sliding_window
    window = self.source_buffer[self.source_position:self.source_position + samples]
    if len(window) < samples:
      window = np.pad(window, (0, samples - len(window)))
    self.batch = torch.from_numpy(np.ascontiguousarray(window)).reshape(1, 1, -1).repeat(self.batch_size, 1, 1).to(self.device)

  def inference(self):
    with torch.inference_mode():
      embeddings = self.model(self.batch)
      torch.cuda.synchronize(self.device)
    return embeddings

  def shutdown(self):
    if self.model is not None:
      del self.model
      self.model = None
      torch.cuda.empty_cache()
