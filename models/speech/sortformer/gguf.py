import itertools
import shutil

from ..common import get_list_arg, temp_path
from ..nemo_speech import NemoSpeechCLI
from .common import compute_ms_per_chunk, designed_wait_ms, selected_geometries

# Runtime presets of the tool, with their published designed waits
PRESETS = {'streaming': 1600, 'offline': 8000}


class Model(NemoSpeechCLI):
  """Streaming Sortformer diarization with using NeMo-Speech.cpp (nemo-speech diarize, GGUF).

    --geometry LIST      explicit --diar-* geometries (default budget_480ms,card_1.04s,card_30.4s)
    --preset LIST        tool presets instead of / in addition to geometries (streaming, offline)
    --concurrency LIST   files in flight on one model (default 1,8)
  Wall time is the whole tool process, so RTFx includes startup and model load (~0.2 s).
  compute_ms_per_chunk = chunk ms / RTFx is the mean device time per chunk step; added to the
  designed wait it estimates end-to-end latency (capture, transport and queueing excluded).
  Do not combine --preset with a v2.1 GGUF: its presets select the v2 1.6 s / 8 s geometries.
  """
  result_columns = ['concurrency', 'designed_wait_ms', 'rtfx', 'compute_ms_per_chunk', 'rttm_files',
                    'segments', 'wall_seconds']

  def sweep_points(self):
    configurations = []
    presets = get_list_arg('--preset', [])
    unknown = [p for p in presets if p not in PRESETS]
    if unknown:
      raise Exception(f'Unknown --preset {", ".join(unknown)}, use one of {", ".join(PRESETS)}')
    for preset in presets:
      configurations.append({'name': f'preset {preset}', 'designed_wait_ms': PRESETS[preset],
                             'frames': None, 'args': ['--preset', preset]})
    default_geometries = [] if presets else ['budget_480ms', 'card_1.04s', 'card_30.4s']
    for name, frames in selected_geometries(default_geometries):
      args = list(itertools.chain.from_iterable((f'--diar-{key}', value) for key, value in frames.items()))
      configurations.append({'name': name, 'designed_wait_ms': designed_wait_ms(frames), 'frames': frames, 'args': args})
    points = []
    for configuration, concurrency in itertools.product(configurations, get_list_arg('--concurrency', [1, 8], int)):
      points.append({**configuration, 'name': f'{configuration["name"]}, concurrency {concurrency}',
                     'concurrency': concurrency, 'args': ['-c', concurrency, *configuration['args']]})
    return points

  def output_dir(self):
    return temp_path('sortformer_rttm', f'point_{self.batch_size}')

  def build_command(self, point):
    output_dir = self.output_dir()
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True)
    return [
      *self.command, 'diarize', self.corpus.materialize().as_posix(),
      '--model', self.model_file, '--gpu', self.gpu, *point['args'],
      '--format', 'rttm', '--output-dir', output_dir.as_posix(), '--force',
      *self.extra_args,
    ]

  def parse_result(self, point, stdout, wall_seconds):
    rttm_files = sorted(self.output_dir().glob('*.rttm'))
    segments = sum(1 for path in rttm_files for line in path.read_text().splitlines() if line.startswith('SPEAKER'))
    rtfx = self.corpus.total_seconds / wall_seconds
    return {
      'concurrency': point['concurrency'],
      'designed_wait_ms': point['designed_wait_ms'],
      'rtfx': rtfx,
      'compute_ms_per_chunk': compute_ms_per_chunk(point['frames'], rtfx) if point['frames'] else None,
      'rttm_files': len(rttm_files),
      'segments': segments,
      'wall_seconds': wall_seconds,
    }
