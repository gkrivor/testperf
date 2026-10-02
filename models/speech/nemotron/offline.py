from ..common import get_arg, get_list_arg
from ..nemo_speech import NemoSpeechASR


class Model(NemoSpeechASR):
  """Nemotron ASR offline throughput with using NeMo-Speech.cpp (nemo-speech bench asr --mode offline).

    --concurrency LIST   (default 1,8,32,64,256)
    --bucket-ms MS       --asr.batching.offline_bucket_ms for every point
  transcript_mismatches counts utterances whose text differs from the tool's concurrency-1 pass.
  """
  mode = 'offline'
  result_columns = ['rtfx', 'utterances_per_second', 'transcript_mismatches', 'utterances',
                    'audio_seconds', 'wall_seconds', 'load_ms']

  def sweep_points(self):
    bucket_ms = get_arg('--bucket-ms')
    points = []
    for concurrency in get_list_arg('--concurrency', [1, 8, 32, 64, 256], int):
      args = ['-c', concurrency]
      name = f'concurrency {concurrency}'
      if bucket_ms:
        args += ['--asr.batching.offline_bucket_ms', bucket_ms]
        name += f', bucket {bucket_ms} ms'
      points.append({'name': name, 'args': args})
    return points
