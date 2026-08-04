import os
import sys
import subprocess
from .vllm_cli import Model as BaseModel
import json
from pathlib import Path

instance = BaseModel()

bench_config = {}

# Option 1: Copy original settings from base configuration
bench_config.update(instance.bench_config)

# Option 2: Set/override fully custom configuration
"""
bench_config = {
    'backend': 'vllm',
    'model': instance.vllm_config['model'],
    'base-url': 'http://' + instance.vllm_config['host'] + ':' + str(instance.vllm_config['port']),
    'dataset-name': 'random',
    'random-input-len': 1024,
    'random-output-len': 128,
    'num-prompts': 500,
    'request-rate': 'inf'
}
"""

# Step 0: Start server
instance._start_server()

# Setting batch sizes from command line
batches = [1]
if '--batch-size' in sys.argv:
  try:
    batches = [int(x) for x in sys.argv[sys.argv.index('--batch-size') + 1].split(',')]
    # @TODO: Use batch size as an argument for vllm bench serve
  except Exception as e:
    print(f'{{ "Error": "Failed to set batch size {e}, using default [{", ".join(map(str, batches))}]" }},')

if not os.path.exists('./temp'):
  os.makedirs('./temp')

result_filepath = Path('.') / 'temp' / 'vllm_bench_result.json'
all_results = {}

for batch in batches:
  print(f'{{ "Processing Batch": {batch} }},')

  if result_filepath.exists():
    result_filepath.unlink()

  bench_command_line = ['vllm', 'bench', 'serve', '--save-result', '--result-filename', result_filepath.as_posix()]
  for key, value in bench_config.items():
    bench_command_line.append(f'--{key}')
    if value is None or value == '': continue
    bench_command_line.append(str(value))
  bench_environment = instance.get_environment_variables()

  # Step 1: Run perf command
  print(f'{{ "Running Performance Test": "{" ".join(bench_command_line)}" }},')
  try:
    all_results[batch] = None

    result = subprocess.run(bench_command_line, capture_output=True, text=True, env=bench_environment)
    
    if result.returncode != 0:
      print(f'{{ "VLLM Bench Environment Variables": "{bench_environment}" }},', flush=True)
      print(f'{{ "VLLM Bench Command Line": "{' '.join(bench_command_line)}" }},', flush=True)
      print(f'{{ "VLLM Bench Result": "{result.returncode}" }},', flush=True)
      print(f'{{ "VLLM Bench Output": "{result.stdout}" }},', flush=True)
      print(f'{{ "VLLM Bench Error": "{result.stderr}" }},', flush=True)
      print(f'{{ "Error": "Failed to run vllm bench" }},')
      continue
    
    # Parse output
    with open(result_filepath.as_posix(), 'r') as f:
        all_results[batch] = json.load(f)
    
  except Exception as e:
    print(f'{{ "Error": "Failed to run performance test {e}" }},')
    continue

# Step 2: Build report
try:
    import reports
    try:
        reports.vllm_bench_report(instance, instance.vllm_config['model'] + '_custom', batches, all_results)
    except Exception as e:
        print(f'{{ "Error": "Failed to generate VLLM Bench report: {e}" }},')
        pass
    try:
        reports.vllm_bench_report_html(instance, instance.vllm_config['model'] + '_custom', batches, all_results)
    except Exception as e:
        print(f'{{ "Error": "Failed to generate VLLM Bench report HTML: {e}" }},')
        pass
except Exception as e:
    print(f'{{ "Error": "Failed to generate VLLM Bench report: {e}" }},')
    pass

print('{ "Status": "Done" }')
del instance

if __name__ != "__main__":
  exit(0)
