"""Shared helpers for speech models: audio corpus, work units, WER and runtime utilities.

Corpus selection (any speech test):
  --audio-dir DIR     directory with 16 kHz .wav/.flac files (or SPEECH_AUDIO_DIR)
  --download          download LibriSpeech test-clean into temp/ and use it
  --synthetic N       generate N synthetic clips (performance only, no references)
  --subset FILE       text file with one file name per line, relative to the audio dir
  --max-files N       use only the first N files
  --refs FILE         tab-separated "utterance-id<TAB>transcript" (default: refs.tsv in the audio dir)
"""
import io
import json
import os
import re
import sys
import tarfile
import urllib.request
import wave
from pathlib import Path

import numpy as np

import settings
from class_model import Model

SAMPLE_RATE = 16000
AUDIO_EXTENSIONS = ('.wav', '.flac')
LIBRISPEECH_TEST_CLEAN_URL = 'https://www.openslr.org/resources/12/test-clean.tar.gz'


def get_arg(name, default=None, cast=str):
  """Return the value following ``name`` in sys.argv, or ``default`` when absent."""
  if name not in sys.argv:
    return default
  index = sys.argv.index(name)
  if index + 1 >= len(sys.argv) or sys.argv[index + 1].startswith('--'):
    raise Exception(f'Missing value for {name}')
  return cast(sys.argv[index + 1])


def get_list_arg(name, default, cast=str):
  """Comma-separated list argument, e.g. --concurrency 1,8,32."""
  value = get_arg(name)
  if value is None:
    return list(default)
  return [cast(item.strip()) for item in value.split(',') if item.strip()]


def has_flag(name):
  return name in sys.argv


def temp_path(*parts):
  return settings.APP_PATH.joinpath('temp', *parts)


def use_sweep_as_batches(count):
  """Run sweep points 0..count-1 as test_perf batches when --batch-size is not given."""
  if '--batch-size' in sys.argv:
    return
  for module_name in ('__main__', 'test_perf'):
    module = sys.modules.get(module_name)
    if module is not None and hasattr(module, 'batches') and hasattr(module, 'model_name'):
      module.batches = list(range(count))


# Audio IO

def read_audio(path):
  """Return (mono float32 samples, sample rate). Uses soundfile when installed, else 16-bit PCM WAV only."""
  try:
    import soundfile as sf
    audio, sample_rate = sf.read(str(path), dtype='float32', always_2d=False)
  except ImportError:
    if Path(path).suffix.lower() != '.wav':
      raise Exception(f'soundfile is required to read {path}')
    with wave.open(str(path), 'rb') as wav:
      if wav.getsampwidth() != 2:
        raise Exception(f'Only 16-bit PCM WAV can be read without soundfile: {path}')
      sample_rate = wav.getframerate()
      channels = wav.getnchannels()
      audio = np.frombuffer(wav.readframes(wav.getnframes()), dtype='<i2').astype(np.float32) / 32768.0
      if channels > 1:
        audio = audio.reshape(-1, channels)
  if audio.ndim > 1:
    audio = audio.mean(axis=1)
  return np.ascontiguousarray(audio, dtype=np.float32), sample_rate


def audio_duration(path):
  try:
    import soundfile as sf
    info = sf.info(str(path))
    return info.frames / info.samplerate
  except ImportError:
    with wave.open(str(path), 'rb') as wav:
      return wav.getnframes() / wav.getframerate()


def write_wav(path, audio, sample_rate=SAMPLE_RATE):
  pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype('<i2')
  with wave.open(str(path), 'wb') as wav:
    wav.setnchannels(1)
    wav.setsampwidth(2)
    wav.setframerate(sample_rate)
    wav.writeframes(pcm.tobytes())


# Corpora

def prepare_librispeech(dest=None):
  """Download LibriSpeech test-clean (~350 MB) and convert it to 16 kHz PCM16 WAV plus refs.tsv."""
  dest = Path(dest) if dest else temp_path('librispeech', 'test-clean-wav')
  if (dest / 'refs.tsv').exists() and any(dest.glob('*.wav')):
    return dest
  import soundfile as sf  # FLAC decoding
  archive = temp_path('librispeech', 'test-clean.tar.gz')
  archive.parent.mkdir(parents=True, exist_ok=True)
  if not archive.exists():
    print(f'{{ "Downloading": "{LIBRISPEECH_TEST_CLEAN_URL}" }},', flush=True)
    partial = archive.with_suffix('.part')
    urllib.request.urlretrieve(LIBRISPEECH_TEST_CLEAN_URL, partial.as_posix())
    os.replace(partial, archive)
  dest.mkdir(parents=True, exist_ok=True)
  refs = []
  with tarfile.open(archive, 'r:gz') as tar:
    for member in tar:
      if not member.isfile():
        continue
      if member.name.endswith('.flac'):
        # Decode from memory: seeking inside a gzip tar member re-reads the archive from the start
        audio, sample_rate = sf.read(io.BytesIO(tar.extractfile(member).read()), dtype='float32')
        write_wav(dest / (Path(member.name).stem + '.wav'), audio, sample_rate)
      elif member.name.endswith('.trans.txt'):
        for line in tar.extractfile(member).read().decode('utf-8').splitlines():
          key, _, text = line.partition(' ')
          if key:
            refs.append(f'{key}\t{text}')
  (dest / 'refs.tsv').write_text('\n'.join(sorted(refs)) + '\n', encoding='utf-8')
  return dest


def prepare_synthetic(count, dest=None, seed=0):
  """Generate ``count`` deterministic 2-15 s clips (gated tone plus noise). For performance runs only."""
  dest = Path(dest) if dest else temp_path('speech_synthetic', str(count))
  if dest.exists() and len(list(dest.glob('*.wav'))) == count:
    return dest
  dest.mkdir(parents=True, exist_ok=True)
  rng = np.random.default_rng(seed)
  for index in range(count):
    seconds = float(rng.uniform(2.0, 15.0))
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    gate = np.sin(2 * np.pi * 0.5 * t) > 0
    audio = 0.3 * np.sin(2 * np.pi * rng.uniform(100, 400) * t) * gate + 0.02 * rng.standard_normal(t.shape)
    write_wav(dest / f'synthetic-{index:04d}.wav', audio.astype(np.float32))
  return dest


def load_refs(path):
  refs = {}
  for line in Path(path).read_text(encoding='utf-8').splitlines():
    key, _, text = line.partition('\t')
    if key and text:
      refs[key] = text
  return refs


class Corpus:
  """An ordered list of audio files with durations, cached samples and optional reference transcripts."""
  def __init__(self, directory, names=None, max_files=None, refs_file=None):
    self.directory = Path(directory)
    if names is None:
      files = sorted(p for p in self.directory.iterdir() if p.suffix.lower() in AUDIO_EXTENSIONS)
    else:
      files = [self.directory / name for name in names]
      missing = [p for p in files if not p.is_file()]
      if missing:
        raise Exception(f'{len(missing)} subset files are missing, first: {missing[0]}')
    if max_files is not None:
      files = files[:max_files]
    if not files:
      raise Exception(f'No audio files found in {self.directory}')
    self.files = files
    self.is_subset = names is not None or max_files is not None
    self.durations = [audio_duration(p) for p in files]
    refs_file = Path(refs_file) if refs_file else self.directory / 'refs.tsv'
    self.refs = load_refs(refs_file) if refs_file.is_file() else {}
    self._audio = {}

  def __len__(self):
    return len(self.files)

  @property
  def total_seconds(self):
    return sum(self.durations)

  def audio(self, index):
    if index not in self._audio:
      audio, sample_rate = read_audio(self.files[index])
      if sample_rate != SAMPLE_RATE:
        raise Exception(f'{self.files[index]} is {sample_rate} Hz, only {SAMPLE_RATE} Hz is supported')
      self._audio[index] = audio
    return self._audio[index]

  def reference(self, index):
    return self.refs.get(self.files[index].stem)

  def by_length(self):
    return sorted(range(len(self.files)), key=lambda i: self.durations[i])

  def groups(self, batch_size, indices=None):
    """Groups of ``batch_size`` files of similar length, so padding inside a batch stays small."""
    order = indices if indices is not None else self.by_length()
    order = sorted(order, key=lambda i: self.durations[i])
    return [order[i:i + batch_size] for i in range(0, len(order), batch_size)]

  def materialize(self):
    """Directory that contains exactly this corpus, for tools that take a directory."""
    if not self.is_subset:
      return self.directory
    dest = temp_path('speech_subset', f'{self.directory.name}_{len(self.files)}')
    if dest.exists() and len(list(dest.iterdir())) == len(self.files):
      return dest
    dest.mkdir(parents=True, exist_ok=True)
    for path in self.files:
      link = dest / path.name
      if link.exists():
        continue
      try:
        os.symlink(path.resolve(), link)
      except OSError:
        import shutil
        shutil.copyfile(path, link)
    return dest


_corpus = None

def get_corpus():
  global _corpus
  if _corpus is not None:
    return _corpus
  synthetic = get_arg('--synthetic', None, int)
  if synthetic:
    directory = prepare_synthetic(synthetic)
  elif has_flag('--download'):
    directory = prepare_librispeech(get_arg('--audio-dir'))
  else:
    directory = get_arg('--audio-dir') or os.environ.get('SPEECH_AUDIO_DIR') or temp_path('librispeech', 'test-clean-wav')
  if not Path(directory).is_dir():
    raise Exception(f'Audio directory not found: {Path(directory).as_posix()}. '
                    'Use --audio-dir DIR, --download (LibriSpeech test-clean) or --synthetic N')
  names = None
  subset = get_arg('--subset')
  if subset:
    names = [line.strip() for line in Path(subset).read_text().splitlines() if line.strip()]
  _corpus = Corpus(directory, names, get_arg('--max-files', None, int), get_arg('--refs'))
  return _corpus


# Accuracy

def normalize_words(text):
  """Lowercase, keep [a-z0-9'] and split on whitespace."""
  return re.sub(r"[^a-z0-9' ]+", ' ', text.lower()).split()


def word_errors(reference, hypothesis):
  """Return (substitutions, deletions, insertions, reference word count) for one utterance."""
  ref = normalize_words(reference)
  hyp = normalize_words(hypothesis)
  # Each cell is (edits, substitutions, deletions, insertions)
  previous = [(j, 0, 0, j) for j in range(len(hyp) + 1)]
  for i in range(1, len(ref) + 1):
    current = [(i, 0, i, 0)]
    for j in range(1, len(hyp) + 1):
      if ref[i - 1] == hyp[j - 1]:
        current.append(previous[j - 1])
        continue
      sub, dele, ins = previous[j - 1], previous[j], current[j - 1]
      current.append(min(
        (sub[0] + 1, sub[1] + 1, sub[2], sub[3]),
        (dele[0] + 1, dele[1], dele[2] + 1, dele[3]),
        (ins[0] + 1, ins[1], ins[2], ins[3] + 1),
      ))
    previous = current
  _, substitutions, deletions, insertions = previous[-1]
  return substitutions, deletions, insertions, len(ref)


def word_error_rate(pairs):
  """WER over (reference, hypothesis) pairs. Returns a dict with the percentage and the error counts."""
  totals = [0, 0, 0, 0]
  for reference, hypothesis in pairs:
    for index, value in enumerate(word_errors(reference, hypothesis)):
      totals[index] += value
  substitutions, deletions, insertions, words = totals
  return {
    'wer_percent': 100.0 * (substitutions + deletions + insertions) / words if words else None,
    'substitutions': substitutions,
    'deletions': deletions,
    'insertions': insertions,
    'reference_words': words,
  }


# Base class

class SpeechModel(Model):
  """Base class for speech models measured on an audio corpus.

  Each timed inference run processes one work unit. By default a unit is a group of
  ``batch_size`` files of similar length; subclasses may redefine units (e.g. one
  streaming chunk). The audio covered by all timed runs goes to ``audio_seconds`` so
  reports show RTFx next to latency.
  """
  def __init__(self):
    super().__init__()
    self.corpus = None
    self.units = {}

  def prepare_batch(self, batch_size):
    if self.corpus is None:
      self.corpus = get_corpus()
      self.details['Audio Directory'] = self.corpus.directory.as_posix()
      self.details['Audio Files'] = len(self.corpus)
      self.details['Audio Seconds (Corpus)'] = round(self.corpus.total_seconds, 3)

  def make_units(self, batch_size):
    return self.corpus.groups(batch_size)

  def unit_audio_seconds(self, unit):
    return sum(self.corpus.durations[i] for i in unit)

  def get_units(self):
    if self.batch_size not in self.units:
      self.units[self.batch_size] = self.make_units(self.batch_size)
      if not self.units[self.batch_size]:
        raise Exception(f'No work units for batch {self.batch_size}')
    return self.units[self.batch_size]

  def prepare(self):
    units = self.get_units()
    self.audio_seconds[self.batch_size] = sum(
      self.unit_audio_seconds(units[i % len(units)]) for i in range(self.total_inference_runs))

  def current_unit_index(self):
    # Timed run N (1-based, see test_perf) processes unit N-1, wrapping around the corpus
    return (max(self.current_inference_run, 1) - 1) % len(self.get_units())

  def current_unit(self):
    return self.get_units()[self.current_unit_index()]


# PyTorch helpers

def torch_device():
  """--device (default cuda, which is also the ROCm device name in PyTorch)."""
  import torch
  device = torch.device(get_arg('--device', 'cuda'))
  if device.type == 'cuda' and not torch.cuda.is_available():
    raise Exception('CUDA/ROCm device is not available, use --device cpu to run on CPU')
  return device


def torch_dtype(default='fp32'):
  import torch
  name = get_arg('--dtype', default)
  dtypes = {'fp32': torch.float32, 'fp16': torch.float16, 'bf16': torch.bfloat16}
  if name not in dtypes:
    raise Exception(f'Unsupported --dtype {name}, use one of {", ".join(dtypes)}')
  return dtypes[name]


def torch_sync(device):
  import torch
  if device.type == 'cuda':
    torch.cuda.synchronize(device)


# ONNX Runtime helpers

_ort_plugins_registered = False

def register_ort_plugin_providers():
  """Register execution providers shipped as plugin packages (onnxruntime-ep-migraphx)."""
  global _ort_plugins_registered
  if _ort_plugins_registered:
    return
  _ort_plugins_registered = True
  try:
    import migraphx  # noqa: F401  must be loaded before onnxruntime in plugin builds
  except ImportError:
    pass
  import onnxruntime as ort
  try:
    import onnxruntime_ep_migraphx as ep
  except ImportError:
    return
  for name, path in zip(ep.get_ep_names(), ep.get_library_paths()):
    try:
      ort.register_execution_provider_library(name, path)
    except Exception as e:
      print(f'{{ "Warning": {json.dumps(f"Failed to register {name}: {e}")} }},', flush=True)


def ort_session(model_path, providers, disable_cpu_fallback=False, profile_prefix=None):
  register_ort_plugin_providers()
  import onnxruntime as ort
  missing = [p for p in providers if p != 'CPUExecutionProvider' and p not in ort.get_available_providers()]
  if missing:
    raise Exception(f'{", ".join(missing)} is not available, available: {", ".join(ort.get_available_providers())}')
  options = ort.SessionOptions()
  if disable_cpu_fallback:
    options.add_session_config_entry('session.disable_cpu_ep_fallback', '1')
  if profile_prefix:
    options.enable_profiling = True
    options.profile_file_prefix = profile_prefix
  return ort.InferenceSession(model_path, sess_options=options, providers=providers)


def ort_kernel_placement(model_path, providers, feed, disable_cpu_fallback=False):
  """Run ``feed`` once with profiling and count executed kernels per execution provider.

  A subgraph compiled by an EP (e.g. MIGraphX) counts as one kernel.
  """
  sess = ort_session(model_path, providers, disable_cpu_fallback, temp_path('ort_placement').as_posix())
  sess.run(None, feed)
  profile = sess.end_profiling()
  with open(profile, 'r') as f:
    events = json.load(f)
  os.remove(profile)
  counts = {}
  seen = set()
  for event in events:
    name = event.get('name', '')
    provider = event.get('args', {}).get('provider')
    if event.get('cat') != 'Node' or not provider or not name.endswith('_kernel_time') or name in seen:
      continue
    seen.add(name)
    counts[provider] = counts.get(provider, 0) + 1
  return counts, sess.get_providers()
