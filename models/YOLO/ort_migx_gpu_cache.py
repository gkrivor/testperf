import os

import onnxruntime as ort
import torch

from class_model import Model

from .common import get_ort_input_np_dtype, get_ort_input_torch_dtype, onnx_name, try_export_model


class Model(Model):
    """YOLO inference with using MIGraphX Execution Provider with cache (IOBinding)"""

    def __init__(self):
        super().__init__()
        self.sess = None
        self.sess_data = {'providers': ['MIGraphXExecutionProvider']}
        self.device = 'cuda'
        if not torch.cuda.is_available():
            raise Exception('CUDA is not available')
        if self.sess_data['providers'][0] not in ort.get_available_providers():
            raise Exception('MIGraphX Execution Provider is not available')

    def prepare_batch(self, batch):
        if self.model_name is None:
            raise Exception('Missing --model (e.g. --model yolo11l)')
        file_path = self.get_file_path(onnx_name(self.model_name, batch, self.precision, self.imgsz))
        try_export_model(file_path, self.model_name, batch, self.precision, self.imgsz, dynamic=False)
        cache_path = file_path[:-4] + 'migx'
        if not os.path.exists(cache_path):
            try:
                os.environ['ORT_MIGRAPHX_MODEL_CACHE_PATH'] = cache_path
                os.makedirs(cache_path, exist_ok=True)
                self.sess = ort.InferenceSession(file_path, **self.sess_data)
                del self.sess
                del os.environ['ORT_MIGRAPHX_MODEL_CACHE_PATH']
            except Exception as e:
                raise Exception(f'Failed to save compiled model {e}')

    def read(self):
        file_path = self.get_file_path(onnx_name(self.model_name, self.batch, self.precision, self.imgsz))
        os.environ['ORT_MIGRAPHX_MODEL_CACHE_PATH'] = file_path[:-4] + 'migx'
        self.sess = ort.InferenceSession(file_path, **self.sess_data)

    def prepare(self):
        self.input_data = self.sess.io_binding()
        images_shape = [self.batch, 3, self.imgsz, self.imgsz]
        dtype = get_ort_input_torch_dtype(self.sess)
        np_dtype = get_ort_input_np_dtype(self.sess)
        self._images_tensor = torch.rand(images_shape, dtype=dtype, device=self.device)
        input_name = self.sess.get_inputs()[0].name
        self.input_data.bind_input(input_name, 'cuda', 0, np_dtype, images_shape, self._images_tensor.data_ptr())
        for out in self.sess.get_outputs():
            self.input_data.bind_output(out.name, 'cuda')

    def inference(self):
        self.sess.run_with_iobinding(self.input_data)

    def shutdown(self):
        try:
            del os.environ['ORT_MIGRAPHX_MODEL_CACHE_PATH']
        except Exception:
            pass
