"""Stand-in for the NeMo-Speech.cpp CLI: answers "bench asr", "transcribe" and "diarize" like the real tool."""
import json
import sys
from pathlib import Path


def arg(name, default=None):
  return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


command = sys.argv[1:3] if sys.argv[1] == 'bench' else sys.argv[1:2]
audio_dir = Path(sys.argv[3] if command == ['bench', 'asr'] else sys.argv[2])
files = sorted(audio_dir.glob('*.wav'))
concurrency = int(arg('-c', '1'))

if command == ['bench', 'asr']:
  run = {'audio_seconds': 100.0, 'concurrency': concurrency, 'rtfx': 10.0 * concurrency,
         'transcript_mismatches': 0, 'utterances': len(files), 'utterances_per_second': 1.5, 'wall_seconds': 10.0}
  if arg('--mode') == 'stream':
    right_context = int(arg('--asr.streaming.rnnt_right_context', '0'))
    run.update({'first_partial_ms_p50': 90.0 + right_context, 'first_partial_ms_p95': 120.0 * concurrency})
  print(json.dumps({'command': 'bench asr', 'files': len(files), 'load_ms': 1.5, 'mode': arg('--mode'), 'runs': [run]}))
elif command == ['transcribe']:
  output_dir = Path(arg('--output-dir'))
  for path in files:
    (output_dir / f'{path.stem}.txt').write_text('hello world')
elif command == ['diarize']:
  output_dir = Path(arg('--output-dir'))
  for path in files:
    (output_dir / f'{path.stem}.rttm').write_text(f'SPEAKER {path.stem} 1 0.00 1.00 <NA> <NA> speaker_0 <NA> <NA>\n')
else:
  sys.exit(f'unsupported command {sys.argv[1:]}')
