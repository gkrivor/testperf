"""Runners for the NeMo-Speech.cpp command line tool (GGUF ASR and diarization models).

The tool does its own concurrency, streaming and timing, so each test_perf batch index is
one sweep point (e.g. lookahead x concurrency) and one inference run is one tool call.
Without --batch-size every sweep point runs; --batch-size 0,2 picks points by index.

  --model PATH.gguf       model file (required)
  --nemo-speech CMD       tool command line (default NEMO_SPEECH_CMD or "nemo-speech")
  --nemo-gpu N            value for the tool's --gpu (default 0)
  NEMO_SPEECH_EXTRA_ARGS  extra arguments appended to every tool call
"""
import json
import os
import subprocess
from time import perf_counter

from class_model import Model
from .common import get_arg, get_corpus, use_sweep_as_batches


class NemoSpeechCLI(Model):
  """Base class for NeMo-Speech.cpp command line runners"""
  result_columns = []

  def __init__(self):
    super().__init__()
    self.total_inference_runs = 1
    self.empty_runs = 0
    self.command = (get_arg('--nemo-speech') or os.environ.get('NEMO_SPEECH_CMD', 'nemo-speech')).split()
    self.extra_args = os.environ.get('NEMO_SPEECH_EXTRA_ARGS', '').split()
    self.model_file = get_arg('--model')
    if not self.model_file:
      raise Exception('Missing --model "path/to/model.gguf"')
    self.gpu = get_arg('--nemo-gpu', '0')
    self.corpus = None
    self.points = self.sweep_points()
    self.command_line = []
    self.details = [
      ['Model File', self.model_file],
      ['Tool', ' '.join(self.command + self.extra_args)],
      ['Results', 'Point', *self.result_columns],
    ]
    use_sweep_as_batches(len(self.points))

  def sweep_points(self):
    """List of dicts with at least 'name' and 'args' (tool arguments for that point)."""
    raise NotImplementedError

  def build_command(self, point):
    raise NotImplementedError

  def parse_result(self, point, stdout, wall_seconds):
    """Return a dict with the result_columns keys."""
    raise NotImplementedError

  def prepare_batch(self, batch_size):
    if batch_size < 0 or batch_size >= len(self.points):
      raise Exception(f'Sweep point {batch_size} does not exist, valid points: 0..{len(self.points) - 1}')
    if self.corpus is None:
      self.corpus = get_corpus()
      self.details.insert(1, ['Audio Directory', self.corpus.directory.as_posix()])
      self.details.insert(2, ['Audio Files', len(self.corpus), 'Audio Seconds', round(self.corpus.total_seconds, 3)])
    print(f'{{ "Sweep Point {batch_size}": "{self.points[batch_size]["name"]}" }},', flush=True)

  def warm_up(self):
    pass

  def prepare(self):
    self.command_line = [str(item) for item in self.build_command(self.points[self.batch_size])]

  def inference(self):
    point = self.points[self.batch_size]
    print(f'{{ "Command Line": {json.dumps(" ".join(self.command_line))} }},', flush=True)
    start = perf_counter()
    result = subprocess.run(self.command_line, capture_output=True, text=True)
    wall_seconds = perf_counter() - start
    if result.returncode != 0:
      print(f'{{ "Tool Output": {json.dumps(result.stdout[-4000:] + result.stderr[-4000:])} }},', flush=True)
      raise RuntimeError(f'{self.command[0]} failed with exit code {result.returncode} for "{point["name"]}"')
    row = self.parse_result(point, result.stdout, wall_seconds)
    print(f'{{ "Speech Result": {json.dumps({"point": point["name"], **row})} }},', flush=True)
    self.details.append(['Result', point['name'], *[row.get(column) for column in self.result_columns]])
    return row


def parse_json_output(stdout):
  """The tool prints a single JSON object on stdout (log lines go to stderr)."""
  start, end = stdout.find('{'), stdout.rfind('}')
  if start < 0 or end < start:
    raise RuntimeError(f'No JSON found in tool output: {stdout[-1000:]}')
  return json.loads(stdout[start:end + 1])


class NemoSpeechASR(NemoSpeechCLI):
  """Base for "nemo-speech bench asr" runners.

    --concurrency LIST     concurrent utterances (one point per value)
    --repetitions N        passes over the corpus per point (-n, default 1)
    --warmup N             tool warm-up utterances (default 1)
    --language TAG         (default en-US)
  """
  mode = None

  def build_command(self, point):
    return [
      *self.command, 'bench', 'asr', self.corpus.materialize().as_posix(),
      '--model', self.model_file, '--gpu', self.gpu, '--mode', self.mode,
      *point['args'],
      '-n', get_arg('--repetitions', '1'), '--warmup', get_arg('--warmup', '1'),
      '--language', get_arg('--language', 'en-US'), '--json',
      *self.extra_args,
    ]

  def parse_result(self, point, stdout, wall_seconds):
    data = parse_json_output(stdout)
    run = data['runs'][0]
    row = {key: run.get(key) for key in self.result_columns if key in run}
    row['load_ms'] = data.get('load_ms')
    row.update(self.point_columns(point, run))
    return row

  def point_columns(self, point, run):
    return {}
