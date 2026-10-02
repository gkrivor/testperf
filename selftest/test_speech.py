import json
import os
import subprocess
import sys

import numpy as np
import openpyxl
import pytest

from common import get_combined_output

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PARENT_DIR)
from models.speech.common import Corpus, prepare_synthetic, word_error_rate, word_errors, write_wav

FAKE_TOOL = os.path.join(PARENT_DIR, 'selftest', 'models', 'fake_nemo_speech.py')


def run_test_perf(args, env=None):
  result = subprocess.run([sys.executable, os.path.join(PARENT_DIR, 'test_perf.py')] + args,
                          cwd=PARENT_DIR, capture_output=True, text=True, env=env)
  out = get_combined_output(result.stdout, result.stderr)
  assert result.returncode == 0, out
  return json.loads(out)['Steps']


def steps_with(steps, key):
  return [step[key] for step in steps if isinstance(step, dict) and key in step]


def fake_tool_env():
  env = os.environ.copy()
  env['NEMO_SPEECH_CMD'] = f'{sys.executable} {FAKE_TOOL}'
  return env


class TestWordErrors:
  def test_identical(self):
    assert word_errors('Hello, World!', 'hello world') == (0, 0, 0, 2)

  def test_substitution_deletion_insertion(self):
    assert word_errors('a b c d', 'a x c') == (1, 1, 0, 4)
    assert word_errors('a b', 'a b c') == (0, 0, 1, 2)

  def test_keeps_apostrophes(self):
    assert word_errors("don't stop", 'dont stop') == (1, 0, 0, 2)

  def test_rate(self):
    result = word_error_rate([('a b c d', 'a b c d'), ('a b c d', 'a b')])
    assert result['wer_percent'] == pytest.approx(25.0)
    assert result['deletions'] == 2


class TestCorpus:
  def make_corpus(self, tmp_path, seconds):
    for index, value in enumerate(seconds):
      write_wav(tmp_path / f'utt-{index}.wav', np.zeros(int(value * 16000), dtype=np.float32))
    (tmp_path / 'refs.tsv').write_text('utt-0\thello world\n')
    return Corpus(tmp_path)

  def test_durations_and_refs(self, tmp_path):
    corpus = self.make_corpus(tmp_path, [1.0, 3.0, 2.0])
    assert corpus.durations == pytest.approx([1.0, 3.0, 2.0])
    assert corpus.reference(0) == 'hello world'
    assert corpus.reference(1) is None

  def test_groups_by_length(self, tmp_path):
    corpus = self.make_corpus(tmp_path, [1.0, 3.0, 2.0, 0.5])
    assert corpus.groups(2) == [[3, 0], [2, 1]]

  def test_subset_and_max_files(self, tmp_path):
    self.make_corpus(tmp_path, [1.0, 3.0, 2.0])
    corpus = Corpus(tmp_path, names=['utt-2.wav', 'utt-0.wav'], max_files=1)
    assert [p.name for p in corpus.files] == ['utt-2.wav']
    assert corpus.is_subset


class TestSpeechModelRun:
  def test_rtfx_in_summary_and_report(self):
    steps = run_test_perf(['selftest.models.speech_sleep_model', '--synthetic', '3', '--batch-size', '1,2', '--runs', '4'])
    summaries = {s['Batch Size']: s for s in steps_with(steps, 'Inference Summary')}
    times = steps_with(steps, 'Inference Times')
    assert set(summaries) == {1, 2}
    for batch, run_times in zip([1, 2], times):
      audio = float(summaries[batch]['Audio Seconds'])
      total = sum(float(t['Time']) for t in run_times)
      assert audio > 0
      assert float(summaries[batch]['RTFx']) == pytest.approx(audio / total)
    workbook = steps_with(steps, 'Workbook')[0]
    sheet = openpyxl.load_workbook(workbook)['Inference']
    labels = [row[0] for row in sheet.iter_rows(values_only=True)]
    assert 'RTFx' in labels and 'Audio Seconds (Timed Runs)' in labels
    os.remove(workbook)

  def test_audio_seconds_cover_timed_runs(self):
    # 3 clips at batch 1 -> 3 units; 4 runs wrap around to the shortest clip again
    steps = run_test_perf(['selftest.models.speech_sleep_model', '--synthetic', '3', '--runs', '4'])
    corpus = Corpus(prepare_synthetic(3))
    durations = sorted(corpus.durations)
    summary = steps_with(steps, 'Inference Summary')[0]
    assert float(summary['Audio Seconds']) == pytest.approx(sum(durations) + durations[0])
    os.remove(steps_with(steps, 'Workbook')[0])


class TestNemoSpeechRunners:
  def test_stream_sweep_runs_every_point(self):
    steps = run_test_perf(['models.speech.nemotron.stream', '--model', 'fake.gguf', '--synthetic', '3',
                           '--right-context', '0,3', '--concurrency', '1,4', '--budget-ms', '300'], fake_tool_env())
    results = steps_with(steps, 'Speech Result')
    assert [r['point'] for r in results] == [
      'lookahead 80 ms, concurrency 1', 'lookahead 80 ms, concurrency 4',
      'lookahead 320 ms, concurrency 1', 'lookahead 320 ms, concurrency 4']
    assert [r['within_budget'] for r in results] == [True, False, False, False]
    assert results[1]['rtfx'] == 40.0
    os.remove(steps_with(steps, 'Workbook')[0])

  def test_batch_size_selects_points(self):
    steps = run_test_perf(['models.speech.nemotron.offline', '--model', 'fake.gguf', '--synthetic', '3',
                           '--concurrency', '1,8,32', '--batch-size', '2'], fake_tool_env())
    assert [r['point'] for r in steps_with(steps, 'Speech Result')] == ['concurrency 32']
    os.remove(steps_with(steps, 'Workbook')[0])

  def test_sortformer_gguf_geometry(self):
    steps = run_test_perf(['models.speech.sortformer.gguf', '--model', 'fake.gguf', '--synthetic', '3',
                           '--geometry', 'budget_480ms', '--concurrency', '1'], fake_tool_env())
    result = steps_with(steps, 'Speech Result')[0]
    assert result['designed_wait_ms'] == 480
    assert result['rttm_files'] == 3 and result['segments'] == 3
    assert result['compute_ms_per_chunk'] == pytest.approx(480 / result['rtfx'])
    os.remove(steps_with(steps, 'Workbook')[0])

  def test_transcribe_wer(self, tmp_path):
    refs = tmp_path / 'refs.tsv'
    refs.write_text('synthetic-0000\thello world\nsynthetic-0001\thello there\n')
    steps = run_test_perf(['models.speech.nemotron.transcribe', '--model', 'fake.gguf', '--synthetic', '3',
                           '--refs', str(refs)], fake_tool_env())
    result = steps_with(steps, 'Speech Result')[0]
    assert result['scored_files'] == 2
    assert result['wer_percent'] == pytest.approx(25.0)
    os.remove(steps_with(steps, 'Workbook')[0])
