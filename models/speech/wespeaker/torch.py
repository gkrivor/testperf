import os

import torch

from ..common import SAMPLE_RATE, SpeechModel, get_arg, torch_device, torch_sync


class Model(SpeechModel):
  """WeSpeaker ResNet34 speaker embedding inference with using PyTorch (pyannote.audio).

  Batch N embeds N files of similar length per forward pass.
    --model ID                   (default pyannote/wespeaker-voxceleb-resnet34-LM, HF_TOKEN if gated)
    --window whole|sliding       whole utterance (zero-padded within a batch) or sliding windows
    --window-seconds S           sliding window length (default 5.0, the training chunk length)
    --step-seconds S             sliding window step (default 2.5)
  MIOpen/cuDNN tune kernels on first use; the warm-up run absorbs that before timing.
  """
  def __init__(self):
    super().__init__()
    self.device = torch_device()
    self.model_id = get_arg('--model', 'pyannote/wespeaker-voxceleb-resnet34-LM')
    self.window = get_arg('--window', 'whole')
    if self.window not in ('whole', 'sliding'):
      raise Exception(f'Unsupported --window {self.window}, use whole or sliding')
    self.window_samples = int(get_arg('--window-seconds', 5.0, float) * SAMPLE_RATE)
    self.step_samples = int(get_arg('--step-seconds', 2.5, float) * SAMPLE_RATE)
    self.model = None
    self.inputs = {}

  def prepare_batch(self, batch_size):
    super().prepare_batch(batch_size)
    self.details['Model'] = self.model_id
    self.details['Device'] = str(self.device)
    self.details['Window'] = self.window if self.window == 'whole' else \
      f'sliding {self.window_samples / SAMPLE_RATE} s, step {self.step_samples / SAMPLE_RATE} s'

  def read(self):
    from pyannote.audio import Model as PyannoteModel
    token = os.environ.get('HF_TOKEN')
    model = PyannoteModel.from_pretrained(self.model_id, use_auth_token=token) if token else PyannoteModel.from_pretrained(self.model_id)
    self.model = model.to(self.device).eval()

  def make_units(self, batch_size):
    if self.window == 'whole':
      return super().make_units(batch_size)
    long_enough = [i for i in range(len(self.corpus)) if len(self.corpus.audio(i)) >= self.window_samples]
    if not long_enough:
      raise Exception(f'No file is at least {self.window_samples / SAMPLE_RATE} s long')
    return self.corpus.groups(batch_size, long_enough)

  def unit_tensor(self, unit):
    if self.window == 'whole':
      waves = [torch.from_numpy(self.corpus.audio(i)) for i in unit]
      batch = torch.zeros((len(waves), 1, max(len(w) for w in waves)))
      for row, wave in enumerate(waves):
        batch[row, 0, :len(wave)] = wave
      return batch
    windows = []
    for i in unit:
      wave = torch.from_numpy(self.corpus.audio(i))
      windows.append(wave.unfold(0, self.window_samples, self.step_samples))
    return torch.cat(windows).unsqueeze(1)

  def prepare(self):
    super().prepare()
    if self.batch_size in self.inputs:
      return
    tensors = [self.unit_tensor(unit) for unit in self.get_units()]
    if self.device.type == 'cuda':
      tensors = [t.pin_memory() for t in tensors]
    self.inputs[self.batch_size] = tensors
    if self.window == 'sliding':
      self.details[f'Windows (batch {self.batch_size})'] = sum(t.shape[0] for t in tensors)

  def inference(self):
    with torch.inference_mode():
      embeddings = self.model(self.inputs[self.batch_size][self.current_unit_index()].to(self.device, non_blocking=True))
      torch_sync(self.device)
    return embeddings

  def shutdown(self):
    if self.model is not None:
      del self.model
      self.model = None
      if self.device.type == 'cuda':
        torch.cuda.empty_cache()
