import itertools

from ..common import get_arg, get_list_arg
from ..nemo_speech import NemoSpeechASR

FRAME_MS = 80


class Model(NemoSpeechASR):
  """Nemotron ASR streaming latency and throughput with using NeMo-Speech.cpp (nemo-speech bench asr --mode stream).

    --right-context LIST   RNNT right context in 80 ms frames (default 0,3,13 = 80, 320, 1120 ms lookahead)
    --concurrency LIST     concurrent streams (default 1,4,8,16)
    --budget-ms MS         latency budget for the within_budget column (default 500)
  First partial is wall time from the first audio push to the first non-empty hypothesis.
  within_budget requires both the lookahead and the first-partial p95 to be inside the budget.
  """
  mode = 'stream'
  result_columns = ['lookahead_ms', 'concurrency', 'rtfx', 'first_partial_ms_p50', 'first_partial_ms_p95',
                    'within_budget', 'transcript_mismatches', 'utterances_per_second', 'wall_seconds', 'load_ms']

  def sweep_points(self):
    points = []
    for right_context, concurrency in itertools.product(
        get_list_arg('--right-context', [0, 3, 13], int), get_list_arg('--concurrency', [1, 4, 8, 16], int)):
      lookahead_ms = (right_context + 1) * FRAME_MS
      points.append({
        'name': f'lookahead {lookahead_ms} ms, concurrency {concurrency}',
        'lookahead_ms': lookahead_ms,
        'args': ['-c', concurrency, '--asr.streaming.rnnt_right_context', right_context],
      })
    return points

  def point_columns(self, point, run):
    budget_ms = get_arg('--budget-ms', 500.0, float)
    p95 = run.get('first_partial_ms_p95')
    return {
      'lookahead_ms': point['lookahead_ms'],
      'within_budget': p95 is not None and point['lookahead_ms'] <= budget_ms and p95 <= budget_ms,
    }
