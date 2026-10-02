"""Silero VAD streaming loop shared by every runtime.

One timed inference run is one 512-sample (32 ms at 16 kHz) chunk for every stream in
the batch: batch size N runs N files in lockstep, the way a server batches N live
streams. Each stream carries 64 samples of left context and the LSTM state between
chunks. Shorter streams are zero-padded; only real audio counts towards RTFx.

  --model PATH    silero_vad.jit / silero_vad.onnx (default: from the silero-vad 6.2.0 wheel)
  --runs N        chunks per batch (default 1000, i.e. 32 s of audio per stream)
"""
import hashlib
import json
import os
import urllib.request
import zipfile

import numpy as np

from ..common import SAMPLE_RATE, SpeechModel, get_arg, has_flag, ort_kernel_placement, ort_session, temp_path

WINDOW = 512
CONTEXT = 64
STATE_SIZE = 128
SILERO_VERSION = '6.2.0'
SILERO_SHA256 = {
  'silero_vad.jit': 'e1122837f4154c511485fe0b9c64455f7b929c96fbb8d79fbdb336383ebd3720',
  'silero_vad.onnx': '1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3',
}


def file_sha256(path):
  digest = hashlib.sha256()
  with open(path, 'rb') as f:
    for block in iter(lambda: f.read(1 << 20), b''):
      digest.update(block)
  return digest.hexdigest()


def get_silero_file(file_name):
  """Return --model, or ``file_name`` from the silero-vad wheel on PyPI (cached in temp/)."""
  path = get_arg('--model')
  if path:
    if not os.path.exists(path):
      raise Exception(f'Model file not found: {path}')
    return path
  dest = temp_path(f'silero_vad_{SILERO_VERSION}', file_name)
  if not dest.exists():
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(f'https://pypi.org/pypi/silero-vad/{SILERO_VERSION}/json') as response:
      release = json.load(response)
    wheel = next(item for item in release['urls'] if item['filename'].endswith('.whl'))
    wheel_path = dest.parent / wheel['filename']
    if not wheel_path.exists():
      urllib.request.urlretrieve(wheel['url'], wheel_path.as_posix())
    with zipfile.ZipFile(wheel_path) as archive:
      dest.write_bytes(archive.read(f'silero_vad/data/{file_name}'))
  return dest.as_posix()


def chunk_audio(audio):
  """Split into zero-padded 512-sample chunks. Returns (chunks [n, 512], real samples per chunk)."""
  count = -(-len(audio) // WINDOW)
  chunks = np.zeros((count, WINDOW), dtype=np.float32)
  chunks.reshape(-1)[:len(audio)] = audio
  real = np.full(count, WINDOW)
  real[-1] = len(audio) - WINDOW * (count - 1)
  return chunks, real


class SileroStream(SpeechModel):
  """Silero VAD streaming base: subclasses implement reset_state() and run_chunk()"""
  def __init__(self, file_name):
    super().__init__()
    self.total_inference_runs = 1000
    self.model_file = get_silero_file(file_name)
    self.stream_chunks = {}  # batch size -> [steps, batch, 512] float32
    self.stream_mask = {}    # batch size -> [steps, batch] bool, True for real chunks

  def prepare_batch(self, batch_size):
    super().prepare_batch(batch_size)
    if 'Model File' not in self.details:
      self.details['Model File'] = self.model_file
      sha = file_sha256(self.model_file)
      known = sha == SILERO_SHA256.get(self.model_file.replace('\\', '/').rsplit('/', 1)[-1])
      self.details['Model SHA256'] = sha + (f' (silero-vad {SILERO_VERSION})' if known else '')

  def make_units(self, batch_size):
    streams = [i % len(self.corpus) for i in range(batch_size)]
    chunked = [chunk_audio(self.corpus.audio(i)) for i in streams]
    steps = max(len(chunks) for chunks, _ in chunked)
    host = np.zeros((steps, batch_size, WINDOW), dtype=np.float32)
    mask = np.zeros((steps, batch_size), dtype=bool)
    real_samples = np.zeros(steps)
    for stream, (chunks, real) in enumerate(chunked):
      host[:len(chunks), stream] = chunks
      mask[:len(chunks), stream] = True
      real_samples[:len(chunks)] += real
    self.stream_chunks[batch_size] = host
    self.stream_mask[batch_size] = mask
    return [samples / SAMPLE_RATE for samples in real_samples]

  def unit_audio_seconds(self, unit):
    return unit

  def prepare(self):
    super().prepare()
    self.details[f'Streams x Chunks (batch {self.batch_size})'] = f'{self.batch_size} x {len(self.get_units())}'
    self.reset_state()

  def inference(self):
    step = self.current_unit_index()
    if step == 0:
      self.reset_state()
    return self.run_chunk(step)

  def reset_state(self):
    raise NotImplementedError

  def run_chunk(self, step):
    raise NotImplementedError


class SileroNumpyStream(SileroStream):
  """Silero VAD base for runtimes fed with numpy arrays (ONNX Runtime, MIGraphX)"""
  def reset_state(self):
    self.state = np.zeros((2, self.batch_size, STATE_SIZE), dtype=np.float32)
    self.context = np.zeros((self.batch_size, CONTEXT), dtype=np.float32)

  def window(self, step):
    return np.concatenate([self.context, self.stream_chunks[self.batch_size][step]], axis=1)


class SileroOrt(SileroNumpyStream):
  """Silero VAD base for ONNX Runtime execution providers.

  --no-cpu-fallback   fail instead of silently running unsupported nodes on CPU
  Reports record which execution provider ran each kernel, so a CPU fallback is visible.
  """
  providers = ['CPUExecutionProvider']

  def __init__(self):
    super().__init__('silero_vad.onnx')
    self.disable_cpu_fallback = has_flag('--no-cpu-fallback')
    self.sess = None
    self.sr = np.array(SAMPLE_RATE, dtype=np.int64)

  def feed(self, window, state):
    return {'input': window, 'state': state, 'sr': self.sr}

  def prepare_batch(self, batch_size):
    super().prepare_batch(batch_size)
    self.details['Requested Providers'] = ', '.join(self.providers)
    self.details['CPU Fallback'] = 'disabled' if self.disable_cpu_fallback else 'enabled'
    window = np.zeros((batch_size, CONTEXT + WINDOW), dtype=np.float32)
    state = np.zeros((2, batch_size, STATE_SIZE), dtype=np.float32)
    counts, active = ort_kernel_placement(self.model_file, self.providers, self.feed(window, state), self.disable_cpu_fallback)
    self.details['Session Providers'] = ', '.join(active)
    self.details[f'Kernels Per Provider (batch {batch_size})'] = ', '.join(f'{k}: {v}' for k, v in sorted(counts.items()))
    on_cpu = counts.get('CPUExecutionProvider', 0)
    if self.providers[0] != 'CPUExecutionProvider' and on_cpu:
      print(f'{{ "Warning": "{on_cpu} of {sum(counts.values())} kernels ran on CPUExecutionProvider instead of {self.providers[0]}" }},', flush=True)

  def read(self):
    self.sess = ort_session(self.model_file, self.providers, self.disable_cpu_fallback)

  def run_chunk(self, step):
    window = self.window(step)
    out, self.state = self.sess.run(None, self.feed(window, self.state))
    self.context = window[:, -CONTEXT:]
    return out
