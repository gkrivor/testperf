"""PyTorch Silero VAD: the shipped TorchScript, and its 16 kHz branch rebuilt as an nn.Module.

torch.compile cannot trace the shipped TorchScript module, so the 16 kHz branch is
rebuilt with the same parameter names and loaded from the JIT weights with strict=True.

  --device cuda|cpu   (default cuda)
  --verify            compare every real chunk with the shipped TorchScript (outside timing)
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common import SAMPLE_RATE, has_flag, torch_device, torch_sync
from .common import CONTEXT, STATE_SIZE, WINDOW, SileroStream


class Block(nn.Module):
  def __init__(self, cin, cout, stride):
    super().__init__()
    self.reparam_conv = nn.Conv1d(cin, cout, 3, stride=stride, padding=1)

  def forward(self, x):
    return F.relu(self.reparam_conv(x))


class STFT(nn.Module):
  def __init__(self):
    super().__init__()
    self.register_buffer('forward_basis_buffer', torch.zeros(258, 1, 256))

  def forward(self, x):
    x = F.pad(x, (0, 64), mode='reflect').unsqueeze(1)
    t = F.conv1d(x, self.forward_basis_buffer, stride=128)
    real, imag = t[:, :129], t[:, 129:]
    return torch.sqrt(real * real + imag * imag)


class Decoder(nn.Module):
  def __init__(self):
    super().__init__()
    self.rnn = nn.LSTMCell(128, 128)
    self.decoder = nn.Sequential(nn.Dropout(0.1), nn.ReLU(), nn.Conv1d(128, 1, 1), nn.Sigmoid())

  def forward(self, x, state):
    h, c = self.rnn(x.squeeze(-1), (state[0], state[1]))
    return self.decoder(h.unsqueeze(-1)), torch.stack([h, c])


class SileroVAD16k(nn.Module):
  def __init__(self):
    super().__init__()
    self.stft = STFT()
    self.encoder = nn.Sequential(Block(129, 128, 1), Block(128, 64, 2), Block(64, 64, 2), Block(64, 128, 1))
    self.decoder = Decoder()

  def forward(self, x, state):
    y, state = self.decoder(self.encoder(self.stft(x)), state)
    return y.squeeze(1).mean(dim=1, keepdim=True), state


def load_rebuilt(jit_path, device):
  jit = torch.jit.load(jit_path, map_location=device).eval()
  model = SileroVAD16k().to(device).eval()
  model.load_state_dict(jit._model.state_dict(), strict=True)
  return model


def step(model, chunk, context, state):
  x = torch.cat([context, chunk], dim=1)
  prob, state = model(x, state)
  return prob, x[:, -CONTEXT:], state


class SileroTorchStream(SileroStream):
  """Silero VAD PyTorch base: each chunk is copied to the device and its probability back, then synchronized"""
  def __init__(self):
    super().__init__('silero_vad.jit')
    self.device = torch_device()
    self.model = None
    self.host_chunks = {}
    self.verified = set()

  def prepare_batch(self, batch_size):
    super().prepare_batch(batch_size)
    self.details['Device'] = str(self.device)
    self.details['Torch Version'] = torch.__version__

  def prepare(self):
    super().prepare()
    if self.batch_size not in self.host_chunks:
      host = torch.from_numpy(self.stream_chunks[self.batch_size])
      self.host_chunks[self.batch_size] = host.pin_memory() if self.device.type == 'cuda' else host
    self.chunk_dev = torch.empty((self.batch_size, WINDOW), device=self.device)
    self.prob_host = torch.empty(self.batch_size)
    if self.device.type == 'cuda':
      self.prob_host = self.prob_host.pin_memory()
    if has_flag('--verify') and self.batch_size not in self.verified:
      self.verified.add(self.batch_size)
      self.verify()

  def inference(self):
    with torch.inference_mode():
      step_index = self.current_unit_index()
      if step_index == 0:
        self.reset_state()
      self.chunk_dev.copy_(self.host_chunks[self.batch_size][step_index], non_blocking=True)
      prob = self.forward_chunk(self.chunk_dev)
      self.prob_host.copy_(prob[:, 0], non_blocking=True)
      torch_sync(self.device)
      return self.prob_host

  def all_probabilities(self, reset_state, forward_chunk):
    host = self.host_chunks[self.batch_size]
    probs = np.zeros(self.stream_chunks[self.batch_size].shape[:2], dtype=np.float32)
    with torch.inference_mode():
      reset_state()
      for index in range(host.shape[0]):
        probs[index] = forward_chunk(host[index].to(self.device))[:, 0].float().cpu().numpy()
    return probs

  def verify(self):
    """Compare all real chunks of this batch with the shipped TorchScript model."""
    reference_model = torch.jit.load(self.model_file, map_location=self.device).eval()
    reference = self.all_probabilities(
      reference_model.reset_states,
      lambda chunk: reference_model(chunk, SAMPLE_RATE))
    if self.model is None:
      self.read()
    probs = self.all_probabilities(self.reset_state, self.forward_chunk)
    mask = self.stream_mask[self.batch_size]
    diff = float(np.abs(probs[mask] - reference[mask]).max())
    flips = int(((probs[mask] >= 0.5) != (reference[mask] >= 0.5)).sum())
    self.details[f'Verify vs TorchScript (batch {self.batch_size})'] = f'max |dp| {diff:.2e}, decision flips at 0.5: {flips} of {int(mask.sum())} chunks'

  def reset_state(self):
    raise NotImplementedError

  def forward_chunk(self, chunk):
    raise NotImplementedError

  def shutdown(self):
    if self.model is not None:
      del self.model
      self.model = None
      if self.device.type == 'cuda':
        torch.cuda.empty_cache()


class SileroRebuiltStream(SileroTorchStream):
  """Rebuilt nn.Module, optionally wrapped (e.g. torch.compile) by subclasses"""
  def make_step(self, model):
    return lambda chunk, context, state: step(model, chunk, context, state)

  def read(self):
    self.model = load_rebuilt(self.model_file, self.device)
    self.step_fn = self.make_step(self.model)

  def reset_state(self):
    self.context = torch.zeros((self.batch_size, CONTEXT), device=self.device)
    self.state = torch.zeros((2, self.batch_size, STATE_SIZE), device=self.device)

  def forward_chunk(self, chunk):
    prob, self.context, self.state = self.step_fn(chunk, self.context, self.state)
    return prob
