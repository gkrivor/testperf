import numpy as np
import torch

from ..common import SAMPLE_RATE, SpeechModel, get_arg, has_flag, torch_device, torch_dtype, torch_sync


class Model(SpeechModel):
  """MMS spoken language identification (Wav2Vec2) inference with using PyTorch.

  Batch N runs N files of similar length per forward pass (padded with an attention mask).
    --model ID            Hugging Face model (default facebook/mms-lid-126)
    --dtype bf16|fp16|fp32  (default bf16)
    --crop-seconds S      crop/tile every file to S seconds (fixed-length latency runs)
    --score               top-1 accuracy against --expected-label (default eng), outside timing
  """
  def __init__(self):
    super().__init__()
    self.device = torch_device()
    self.dtype = torch_dtype('bf16')
    self.model_id = get_arg('--model', 'facebook/mms-lid-126')
    self.crop_seconds = get_arg('--crop-seconds', None, float)
    self.expected_label = get_arg('--expected-label', 'eng')
    self.model = None
    self.extractor = None
    self.inputs = {}

  def prepare_batch(self, batch_size):
    super().prepare_batch(batch_size)
    self.details['Model'] = self.model_id
    self.details['Device / DType'] = f'{self.device} / {self.dtype}'
    if self.crop_seconds:
      self.details['Crop Seconds'] = self.crop_seconds

  def read(self):
    from transformers import AutoFeatureExtractor, Wav2Vec2ForSequenceClassification
    self.extractor = AutoFeatureExtractor.from_pretrained(self.model_id)
    self.model = Wav2Vec2ForSequenceClassification.from_pretrained(self.model_id).to(self.device, dtype=self.dtype).eval()

  def waveform(self, index):
    audio = self.corpus.audio(index)
    if not self.crop_seconds:
      return audio
    samples = int(self.crop_seconds * SAMPLE_RATE)
    return np.resize(audio, samples) if len(audio) < samples else audio[:samples]

  def unit_audio_seconds(self, unit):
    return self.crop_seconds * len(unit) if self.crop_seconds else super().unit_audio_seconds(unit)

  def prepare(self):
    super().prepare()
    if self.batch_size in self.inputs:
      return
    batches = []
    for unit in self.get_units():
      features = self.extractor([self.waveform(i) for i in unit], sampling_rate=SAMPLE_RATE,
                                return_tensors='pt', padding=True, return_attention_mask=True)
      values, mask = features['input_values'], features.get('attention_mask')
      if self.device.type == 'cuda':
        values = values.pin_memory()
        mask = mask.pin_memory() if mask is not None else None
      batches.append((values, mask))
    self.inputs[self.batch_size] = batches
    if has_flag('--score'):
      self.score()

  def forward(self, values, mask):
    values = values.to(self.device, dtype=self.dtype, non_blocking=True)
    mask = mask.to(self.device, non_blocking=True) if mask is not None else None
    return self.model(values, attention_mask=mask).logits

  def score(self):
    labels = self.model.config.id2label
    correct = 0
    total = 0
    with torch.inference_mode():
      for values, mask in self.inputs[self.batch_size]:
        for label in self.forward(values, mask).argmax(dim=-1).tolist():
          correct += labels[label] == self.expected_label
          total += 1
    self.details[f'Top-1 "{self.expected_label}" (batch {self.batch_size})'] = f'{correct}/{total}'

  def inference(self):
    values, mask = self.inputs[self.batch_size][self.current_unit_index()]
    with torch.inference_mode():
      logits = self.forward(values, mask)
      torch_sync(self.device)
    return logits

  def shutdown(self):
    if self.model is not None:
      del self.model
      self.model = None
      if self.device.type == 'cuda':
        torch.cuda.empty_cache()
