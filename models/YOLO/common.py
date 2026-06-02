import os


def get_np_dtype(precision):
    import numpy as np

    return {
        'fp16': np.float16,
        'fp32': np.float32,
    }.get(str(precision), np.float32)


def get_torch_dtype(precision):
    import torch

    return {
        'fp16': torch.float16,
        'fp32': torch.float32,
    }.get(str(precision), torch.float32)


def get_ort_input_np_dtype(sess):
    import numpy as np

    t = sess.get_inputs()[0].type if sess and sess.get_inputs() else None
    return {
        'tensor(float16)': np.float16,
        'tensor(float)': np.float32,
    }.get(t, np.float32)


def get_ort_input_torch_dtype(sess):
    import torch

    t = sess.get_inputs()[0].type if sess and sess.get_inputs() else None
    return {
        'tensor(float16)': torch.float16,
        'tensor(float)': torch.float32,
    }.get(t, torch.float32)


def onnx_name(model_name, batch, precision, imgsz):
    name = str(model_name).strip()
    pfx = f'{name}_fp16' if str(precision) == 'fp16' else name
    return f'{pfx}_{int(batch)}b_{int(imgsz)}.onnx'


def get_ultralytics_task(weights):
    name = str(weights).strip().lower()
    # Ultralytics encodes the task in the weights filename suffix.
    for suffix, task in (
        ('-cls', 'classify'),
        ('-seg', 'segment'),
        ('-pose', 'pose'),
        ('-obb', 'obb'),
    ):
        if name.endswith(suffix) or f'{suffix}.' in name:
            return task
    return 'detect'


_EXPORT_CHILD_SCRIPT = r"""
import shutil
import sys

from ultralytics import YOLO

weights, out_path, batch, imgsz, half_flag, task, dynamic_flag = sys.argv[1:8]
model = YOLO(f'{weights}.pt', task=task)
onnx_path = model.export(
    format='onnx',
    imgsz=int(imgsz),
    batch=int(batch),
    half=(half_flag == '1'),
    dynamic=(dynamic_flag == '1'),
)
if str(onnx_path) != str(out_path):
    shutil.move(str(onnx_path), out_path)
"""


def try_export_model(file_path, model_name, batch, precision, imgsz, dynamic=False):
    """Export a YOLO .pt to ONNX in a subprocess.

    Ultralytics' ONNX export sets ``CUDA_VISIBLE_DEVICES=''`` to force a CPU
    trace, which breaks ``migraphx.get_target('gpu')`` in the same process. 
    Run the export in a subprocess so the env mutation stays contained.
    """
    import subprocess
    import sys

    if os.path.exists(file_path):
        return
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    weights = str(model_name).strip()
    task = get_ultralytics_task(weights)
    half_flag = '1' if str(precision) == 'fp16' else '0'
    dynamic_flag = '1' if dynamic else '0'

    cmd = [
        sys.executable,
        '-c',
        _EXPORT_CHILD_SCRIPT,
        weights,
        str(file_path),
        str(int(batch)),
        str(int(imgsz)),
        half_flag,
        task,
        dynamic_flag,
    ]
    subprocess.check_call(cmd)

    if not os.path.exists(file_path):
        raise RuntimeError(f'ONNX export did not produce {file_path}')
