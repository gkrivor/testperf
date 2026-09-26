import os
import sys

# Default YOLO model family name. The task-specific weights/ONNX file names are
# derived from this (e.g. 'yolov8n' + '-cls' -> 'yolov8n-cls').
DEFAULT_MODEL_NAME = 'yolov8n'

# Default YOLO task when --task is not provided.
DEFAULT_TASK = 'classify'

# Supported YOLO tasks -> (Ultralytics weight suffix, square input image size).
# Ultralytics ships a dedicated checkpoint per task, e.g. yolov8n-cls.pt for
# classification (224x224) and yolov8n-seg/pose/obb.pt for the rest (640x640).
YOLO_TASKS = {
    'classify': ('-cls', 224),
    'segment': ('-seg', 640),
    'pose': ('-pose', 640),
    'obb': ('-obb', 640),
}

# Base URL for the Ultralytics released weights.
WEIGHTS_BASE_URL = 'https://github.com/ultralytics/assets/releases/download/v8.4.0/'


def _valid_tasks_message():
  tasks = ', '.join(f'{t} ({size})' for t, (_, size) in YOLO_TASKS.items())
  return f'Valid YOLO tasks: {tasks}'


def get_model_name(argv=None):
  """Resolve the YOLO model name (default 'yolov8n')."""
  argv = sys.argv if argv is None else argv
  if '--model' in argv:
    try:
      name = argv[argv.index('--model') + 1]
    except IndexError:
      raise Exception('Missing value after --model.')
    name = str(name).strip().lower()
    if not name:
      raise Exception('Empty --model value.')
    return name
  return DEFAULT_MODEL_NAME


def get_yolo_task(argv=None):
  """Resolve the YOLO task from the --task argument (default 'classify')."""
  argv = sys.argv if argv is None else argv
  if '--task' in argv:
    try:
      task = argv[argv.index('--task') + 1]
    except IndexError:
      raise Exception(f'Missing value after --task. {_valid_tasks_message()}')
    task = str(task).strip().lower()
    if task not in YOLO_TASKS:
      raise Exception(f'Unknown YOLO task "{task}". {_valid_tasks_message()}')
    return task
  return DEFAULT_TASK


def get_image_size(task):
  """Return the square input image size for a YOLO task (e.g. 'classify' -> 224)."""
  task = str(task).strip().lower()
  if task not in YOLO_TASKS:
    raise Exception(f'Unknown YOLO task "{task}". {_valid_tasks_message()}')
  return YOLO_TASKS[task][1]


def _task_suffix(task):
  task = str(task).strip().lower()
  if task not in YOLO_TASKS:
    raise Exception(f'Unknown YOLO task "{task}". {_valid_tasks_message()}')
  return YOLO_TASKS[task][0]


def is_fp16(argv=None):
  """Return True when --fp16 is present in the command line."""
  argv = sys.argv if argv is None else argv
  return '--fp16' in argv


def get_input_np_dtype(half):
  """Numpy dtype for the model input given the precision flag."""
  import numpy as np
  return np.float16 if half else np.float32


def weights_name(model_name, task):
  """Build the Ultralytics weights file name for a model/task (e.g. 'yolov8n-cls.pt')."""
  return f'{str(model_name).lower()}{_task_suffix(task)}.pt'


def onnx_name(model_name, task, batch, half):
  """Build the ONNX file name for a model/task/batch/precision combination."""
  precision = 'fp16' if half else 'fp32'
  return f'{str(model_name).lower()}{_task_suffix(task)}_{precision}_{batch}b.onnx'


def try_export_model(file_path, batch_size, half_precision=False, *,
                     model_name=None, task=None, imgsz=None, opset=17):
  """Export an Ultralytics YOLO model to ONNX for the requested task.

  The task (classify/segment/pose/obb) is resolved from --task when not passed
  explicitly, selecting the matching Ultralytics weights (e.g. yolov8n-cls.pt)
  and default square input size. Existing ``file_path`` short-circuits.
  """
  if os.path.exists(file_path):
    return

  model_name = get_model_name() if model_name is None else str(model_name).lower()
  task = get_yolo_task() if task is None else str(task).strip().lower()
  imgsz = get_image_size(task) if imgsz is None else imgsz

  try:
    weights = weights_name(model_name, task)
    if not os.path.exists(weights):
      try:
        import urllib.request
        urllib.request.urlretrieve(WEIGHTS_BASE_URL + weights, weights)
      except Exception as e:
        raise Exception(f'Failed to download YOLO model {e}')
    if not os.path.exists(weights):
      raise Exception(f'YOLO model file {weights} not found')

    from ultralytics import YOLO
    model = YOLO(weights)
    exported = model.export(format='onnx', imgsz=imgsz, batch=batch_size, half=half_precision)
    # ``export`` returns the produced ONNX path; fall back to the weights base
    # name if a bare boolean/None is returned by older ultralytics versions.
    src = exported if isinstance(exported, str) and exported else weights[:-2] + 'onnx'
    os.rename(src, file_path)
  except Exception as e:
    raise Exception(f'Failed to export model {e}')

  precision = 'fp16' if half_precision else 'fp32'
  print(f'Exported {weights} to {file_path} (task {task}, imgsz {imgsz}, batch {batch_size}, {precision}).')
