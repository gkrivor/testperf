import numpy as np
import torch

from ..common import AudioModel


class Model(AudioModel):
  """MMS spoken language identification (Wav2Vec2) GPU performance test.

  Pure performance test: read the model and run inference over sliding windows
  of the provided source_buffer. GPU only.
  """
  def __init__(self):
    super().__init__()
    self.device = torch.device('cuda')
    self.dtype = torch.bfloat16
    self.model_sliding_window = 5  # seconds
    self.model = None
    self.extractor = None

  def read(self):
    from transformers import AutoFeatureExtractor, Wav2Vec2ForSequenceClassification
    self.extractor = AutoFeatureExtractor.from_pretrained('facebook/mms-lid-126')
    self.model = Wav2Vec2ForSequenceClassification.from_pretrained('facebook/mms-lid-126').to(self.device, dtype=self.dtype).eval()

  def prepare(self):
    super().prepare()
    samples = self.model_sample_rate * self.model_sliding_window
    window = self.source_buffer[self.source_position:self.source_position + samples]
    if len(window) < samples:
      window = np.pad(window, (0, samples - len(window)))
    values = self.extractor(window, sampling_rate=self.model_sample_rate, return_tensors='pt')['input_values']
    self.batch = values.repeat(self.batch_size, 1).to(self.device, dtype=self.dtype)

  def inference(self):
    with torch.inference_mode():
      logits = self.model(self.batch).logits
      torch.cuda.synchronize(self.device)
    return logits

  def shutdown(self):
    if self.model is not None:
      del self.model
      self.model = None
      torch.cuda.empty_cache()
