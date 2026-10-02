import shutil

from ..common import get_arg, get_list_arg, temp_path, word_error_rate
from ..nemo_speech import NemoSpeechCLI


class Model(NemoSpeechCLI):
  """Nemotron ASR word error rate with using NeMo-Speech.cpp (nemo-speech transcribe).

    --concurrency LIST   (default 8)
    --language TAG       (default en-US)
  Needs reference transcripts (refs.tsv, see --refs). Punctuation is disabled and text is
  lowercased keeping [a-z0-9'] before scoring. RTFx includes process start and model load.
  """
  result_columns = ['wer_percent', 'substitutions', 'deletions', 'insertions', 'reference_words',
                    'scored_files', 'missing_hypotheses', 'rtfx_including_load', 'wall_seconds']

  def sweep_points(self):
    return [{'name': f'concurrency {c}', 'concurrency': c, 'args': ['-c', c]}
            for c in get_list_arg('--concurrency', [8], int)]

  def output_dir(self, point):
    return temp_path('nemotron_hypotheses', f'c{point["concurrency"]}')

  def build_command(self, point):
    if not self.corpus.refs:
      raise Exception('No reference transcripts: pass --refs FILE or use --download')
    output_dir = self.output_dir(point)
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True)
    return [
      *self.command, 'transcribe', self.corpus.materialize().as_posix(),
      '--model', self.model_file, '--gpu', self.gpu, '--language', get_arg('--language', 'en-US'),
      *point['args'], '--no-punctuation', '--output-dir', output_dir.as_posix(),
      *self.extra_args,
    ]

  def parse_result(self, point, stdout, wall_seconds):
    output_dir = self.output_dir(point)
    pairs = []
    missing = 0
    for index, path in enumerate(self.corpus.files):
      reference = self.corpus.reference(index)
      if reference is None:
        continue
      hypothesis = output_dir / f'{path.stem}.txt'
      if not hypothesis.exists():
        missing += 1
        continue
      pairs.append((reference, hypothesis.read_text(encoding='utf-8')))
    if not pairs:
      raise RuntimeError(f'No hypotheses named <utterance>.txt found in {output_dir.as_posix()}')
    row = word_error_rate(pairs)
    row.update({
      'scored_files': len(pairs),
      'missing_hypotheses': missing,
      'rtfx_including_load': self.corpus.total_seconds / wall_seconds,
      'wall_seconds': wall_seconds,
    })
    return row
