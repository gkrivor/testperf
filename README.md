# Performance Benchmarking framework

## Overview

This framework is used to run a different workloads (originally focused on model inference), measure time of each run, and provide results in machine and human readable formats.

## Installation

### Requirements
 - Python
 - [Optional] OpenPyXL - for generating reports in Excel format
 - [Optional] Docker - for running tasks in predefined environment
 - [Optional] Excel - for running aggregation tool

## Running a single test

Using 'test_perf.py' script you can automate your performance testing by using a common tasks for performance calculations (prepares environment, reading tests, warming up before tests, measuring time of each run).

Most important feature is automatic generation of performance analysis with charts.

### Arguments
- --batch-size - list of batch sizes has to be verified
- --only-prepare - Only prepare the batch will be run, no inference will be run

### Examples

Simple run of YOLO11 Large model benchmarking using ONNXRuntime with default settings, batch size is 1.

```bash
python test_perf.py models.yolo11l.ort
```

Simple run of YOLO11 Large model benchmarking using ONNXRuntime with default settings, custom set of batch size: 1, 2, 4, 8, 16.

```bash
python test_perf.py models.yolo11l.ort --batch-size 1,2,4,8,16
```

## Speech models

Speech tests live in `models/speech` and run on a real audio corpus instead of random tensors. A model that sets `audio_seconds` gets RTFx (audio seconds / inference seconds, higher is better) in the inference summary and the report.

Audio corpus (16 kHz):
- `--download` - download LibriSpeech test-clean (~350 MB) into `temp/` with reference transcripts
- `--audio-dir DIR` - use your own `.wav`/`.flac` directory (or `SPEECH_AUDIO_DIR`); optional `refs.tsv` with `utterance-id<TAB>text`
- `--subset FILE`, `--max-files N` - limit the corpus
- `--synthetic N` - generated clips for a quick performance-only run

| Test | Runtime | One timed run |
|---|---|---|
| `models.speech.silero_vad.{ort,ort_cuda,ort_migx,migx_cache,torch,torch_eager,torch_compile}` | ONNX Runtime, MIGraphX, PyTorch | one 32 ms chunk for each of N lockstep streams |
| `models.speech.mms_lid.torch` | PyTorch (transformers) | N files of similar length |
| `models.speech.wespeaker.torch` | PyTorch (pyannote.audio) | N files, whole utterance or sliding windows |
| `models.speech.sortformer.nemo_torch` | NeMo PyTorch | `diarize()` of N files |
| `models.speech.unity2_aligner.fairseq2` | fairseq2 0.2 | N files aligned to their transcripts |
| `models.speech.nemotron.{offline,stream,transcribe}` | NeMo-Speech.cpp CLI | one tool call per sweep point |
| `models.speech.sortformer.gguf` | NeMo-Speech.cpp CLI | one tool call per sweep point |

Batch size is the number of streams or files per run. NeMo-Speech.cpp runners time themselves (concurrency, streaming first-partial latency), so there each batch index is a sweep point; without `--batch-size` all points run. Every file documents its options in the class docstring.

The ONNX Silero runners report how many kernels each execution provider ran, so a silent CPU fallback is visible; `--no-cpu-fallback` makes it an error. The stock `silero_vad.onnx` does not compile with MIGraphX; `migx_cache` needs an If-free export.

```bash
python test_perf.py models.speech.silero_vad.torch_compile --download --max-files 300 --verify --runs 2000 --batch-size 1,8,32
python test_perf.py models.speech.mms_lid.torch --download --score --batch-size 1,8,16
python test_perf.py models.speech.nemotron.stream --download --model temp/models/nemotron-3.5-asr-streaming-0.6b.q8_0.gguf --right-context 3 --concurrency 1,16
```

Docker images and an example `docker_runner` config are in `dockers/speech`.

## Running batch tasks using docker images

The provided `docker_runner.py` script allows you to automate the running of multiple benchmarking tasks across different Docker container configurations. It supports running batched tests, managing container lifecycle, and customizing Docker execution.

### Usage

```bash
python docker_runner.py [OPTIONS]
```

### Key Options

- `--config <file>`
  Loads configuration from a provided JSON-file

- `--show-config`
  Display current docker configurations and exit.

- `--only-prepare`
  Only prepare the batch will be run, no inference will be run, applied to all configs

- `--case <test_name>`
  Run a specific test case (e.g., `models.yolo8n.ort`).

- `--batch-size <sizes>`
  Set batch sizes as a comma-separated list (default: 1).
  Example: `--batch-size 1,4,8`

- `--continue <index>`
  Start execution from a specific configuration index.

- `--count <value>`
  Limit the number of configurations to run.

- `--single <index>`
  Run only the configuration at the specified index.
  Supports negative indexing (e.g., `-1` for last config).

#### Container Management

- `--dont-remove`
  Keep Docker image after execution (the default is to remove them).

- `--shell`
  Open an interactive shell in the container (must be used with `--single`).

#### Execution Control

- `--fake`
  Dry-run mode: show command lines to be executed without actually running them.

### Examples

Show current configuration:
```bash
python docker_runner.py --config configs/my_config.json
```

```bash
python docker_runner.py --show-config
```

Run all configurations:
```bash
python docker_runner.py
```

Run specific configuration with custom batch sizes:
```bash
python docker_runner.py --single 0 --batch-size 1,4,8,16
```

Run a specific test case on configuration 2:
```bash
python docker_runner.py --single 2 --case models.yolo11l.ort
```

Continue from configuration 3 onward:
```bash
python docker_runner.py --continue 3
```

Open a shell in the first configuration:
```bash
python docker_runner.py --single 0 --shell
```

Perform a dry-run to see the commands that would be executed:
```bash
python docker_runner.py --fake
```

### Configuration File

The script looks for a file named `docker_runner.json` in the current directory.
This file should be a JSON list of configuration objects, each containing:

- `docker_image`: (Optional) Name of the Docker image to use
- `docker_file`: Path to Dockerfile (used to build the image if it doesn't exist)
- `dont_remove`: (Optional) Whether to keep the container after execution
- `only_prepare`: (Optional) Only prepare the batch will be run, no inference will be run, applied only to selected run
- `docker_custom_run`: (Optional) Custom docker run command
- `docker_hostname`: (Optional) Hostname to set in the container
- `tests`: List of test cases to run

#### Example `docker_runner.json`

```json
[
  {
    "docker_file": "./dockers/com.org_name.docker_image_name",
    "dont_remove": false,
    "docker_custom_run": "",
    "only_prepare": true,
    "docker_hostname": "test_machine_A",
    "tests": ["models.yolo8n.ort", "models.yolo8n.ort_dml"]
  },
  {
    "docker_image": "my_image_2",
    "docker_file": "",
    "dont_remove": true,
    "docker_custom_run": "docker run -it --gpus all",
    "docker_hostname": "test_machine_B",
    "tests": ["models.yolo11l.ort", "models.yolo11l.ort_ov"]
  }
]
```

For more details, use `python docker_runner.py --help`.