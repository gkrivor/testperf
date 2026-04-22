import os

import numpy as np
import onnxruntime as ort
from winui3.microsoft.windows.applicationmodel.dynamicdependency.bootstrap import (
    InitializeOptions,
    initialize
)
import winui3.microsoft.windows.ai.machinelearning as winml

from class_model import Model

from .common import get_ort_input_np_dtype, onnx_name, try_export_model


class Model(Model):
    """YOLO inference with using MIGraphX Execution Provider with cache"""

    def __init__(self):
        super().__init__()
        required_ep = ['MIGraphXExecutionProvider']
        found_providers = {}
        with initialize(options=InitializeOptions.ON_NO_MATCH_SHOW_UI):
            catalog = winml.ExecutionProviderCatalog.get_default()
            print([provider.ready_state for provider in catalog.find_all_providers()])
            # Filter EPs that the app supports
            providers = [provider for provider in catalog.find_all_providers() if provider.name in required_ep]

            # Download and make ready missing EPs if the user wants to
            if any(provider.ready_state == winml.ExecutionProviderReadyState.NOT_PRESENT for provider in providers):
                for provider in [provider for provider in providers if provider.ready_state == winml.ExecutionProviderReadyState.NOT_PRESENT]:
                    print(f"Provider {provider.name} is absent on the system")
                    provider.ensure_ready_async().get()

            # Make ready the existing EPs
            for provider in [provider for provider in providers if provider.ready_state == winml.ExecutionProviderReadyState.NOT_READY]:
                print(f"Getting things ready for {provider.name}")
                provider.ensure_ready_async().get()

            # Register all ready EPs
            for provider in [provider for provider in providers if provider.ready_state == winml.ExecutionProviderReadyState.READY]:
                print(f"Found provider {provider.name}")
                found_providers[provider.name] = provider.library_path

        for name, path in found_providers.items():
            print(f"Registering provider {name} {path}")
            ort.register_execution_provider_library(name, path)

        # Looking for a required device
        selected_device = None
        for device in ort.get_ep_devices():
           if device.ep_name in required_ep:
                selected_device = device
                break

        if selected_device is None:
            raise Exception('No device with MIGraphX Execution Provider is available')

        print(f"Device selected for inference: {selected_device.device.metadata}")

        self.sess = None
        self.sess_options = ort.SessionOptions()
        self.sess_options.add_provider_for_devices([selected_device], {})

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
                self.sess = ort.InferenceSession(file_path, self.sess_options)
                del self.sess
                del os.environ['ORT_MIGRAPHX_MODEL_CACHE_PATH']
            except Exception as e:
                raise Exception(f'Failed to save compiled model {e}')

    def read(self):
        file_path = self.get_file_path(onnx_name(self.model_name, self.batch, self.precision, self.imgsz))
        os.environ['ORT_MIGRAPHX_MODEL_CACHE_PATH'] = file_path[:-4] + 'migx'
        self.sess = ort.InferenceSession(file_path, self.sess_options)

    def prepare(self):
        dtype = get_ort_input_np_dtype(self.sess)
        self.input_data = {
            'images': np.random.randn(self.batch, 3, self.imgsz, self.imgsz).astype(dtype),
        }

    def inference(self):
        return self.sess.run([], input_feed=self.input_data)

    def shutdown(self):
        try:
            del os.environ['ORT_MIGRAPHX_MODEL_CACHE_PATH']
        except Exception:
            pass
