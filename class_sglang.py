import os
import subprocess
from pathlib import Path
from class_model import Model
import time
import socket
import json
import random
from class_server import ServerKeeper
import settings

sglang_server = None

class SGLangCommon(Model):
    def __init__(self, model_id = None):
        super().__init__()
        # Keys must match with SGLang CLI arguments https://sgl-project.github.io/advanced_features/server_arguments.html
        self.sglang_config = {
            'host': os.environ.get('SGLANG_HOST_IP', '127.0.0.1'),
            'port': int(os.environ.get('SGLANG_PORT', '30000')),
            'model-path': model_id, # Model ID must be explicitly set
            'dtype': 'auto', # Model precision
            'context-length': 4096, # Model max length
            'mem-fraction-static': 0.9, # GPU memory utilization
            'tp': 1, # Tensor parallel size
        }
        # Environment variables to be set/unset for the subprocess, if variable is None - it will be unset
        self.env = {
            'HF_TOKEN': os.environ.get('HF_TOKEN', None),
            'SGLANG_CACHE_DIR': Path(os.environ.get('SGLANG_CACHE_DIR', settings.APP_PATH / 'temp' / 'sglang_cache')),
            'HF_HOME': Path(os.environ.get('HF_HOME', settings.APP_PATH / 'temp' / 'hf_cache')),
        }

        # Internal settings
        self.startup_timeout = 600
        self.read_prompt = 'Count from 0 to 100'
        self.max_output_tokens = 128
        self.dataset_name = 'random' # Currently supported only random
        self.random_input_length = 1024
        self.random_output_length = 128

        # self.serve_cmd = os.environ.get('SGLANG_CMD', 'python -m sglang.launch_server').split()
        self.serve_cmd = os.environ.get('SGLANG_CMD', 'sglang serve').split()
        self.serve_extra_args = os.environ.get('SGLANG_SERVE_EXTRA_ARGS', '').split()

    def get_environment_variables(self):
        env = os.environ.copy()
        for key, value in self.env.items():
            if value is not None:
                env[key] = str(value)
            elif key in env:
                env.pop(key)
        return env
    
    def get_server_command_line(self):
        server_command_line = [*self.serve_cmd, *self.serve_extra_args]
        for key, value in self.sglang_config.items():
            if value is None: continue
            server_command_line.append(f'--{key}')
            if value == '': continue
            server_command_line.append(str(value))
        return server_command_line

    def _start_server(self):
        global sglang_server
        server_command_line = self.get_server_command_line()
        # Set up environment variables for the subprocess
        env = self.get_environment_variables()

        if sglang_server is None:
            sglang_server = ServerKeeper(self.sglang_config['host'], self.sglang_config['port'], 'sglangserver.log')
        sglang_server.start(server_command_line, env)
    
    def prepare_batch(self, batch_size):
        global sglang_server
        # Preparing read prompt and request
        self.read_prompt = json.dumps({
                "model": self.sglang_config['model-path'],
                "prompt": self.read_prompt,
                "max_tokens": self.max_output_tokens,
                "temperature": 0.0
            }, indent=4
        )
        self.read_request = (
            b'POST /v1/completions HTTP/1.1\r\n' +
            f'Host: {self.sglang_config["host"]}:{self.sglang_config["port"]}\r\n'.encode('utf-8') +
            b'Content-Type: application/json\r\n' +
            f'Content-Length: {len(self.read_prompt)}\r\n'.encode('utf-8') +
            b'Accept: */*\r\n' +
            b'User-Agent: testperf\r\n' +
            b'Connection: close\r\n' +
            b'\r\n' +
            self.read_prompt.encode('utf-8')
        )
        #print(self.read_request, flush=True)

        # @TODO: Need list of cases when server should be restarted (e.g. model change, config change, etc.)
        # For now - just return if server is already running
        if sglang_server is not None:
            return

        self._start_server()

    # Cons for current implementation: time for socket creation, connection, request sending is included in the benchmark
    def read(self):
        global sglang_server
        if sglang_server is None:
            self._start_server()

        # TODO: Socket creation and connection should be done in a separate function
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.connect((self.sglang_config['host'], self.sglang_config['port']))
            
            # Prepare HTTP request for OpenAI-compatible API
            # Send request
            sock.sendall(self.read_request)
            
            # Read response until we get some output
            response = b""
            while True:
                chunk = sock.recv(1024)
                if not chunk:
                    break
                response += chunk
                # Check if we have received a complete HTTP response
                if b"\r\n\r\n" in response:
                    #print(response.decode('utf-8'), flush=True)
                    break
        except (ConnectionRefusedError, TimeoutError, socket.gaierror) as e:
            print(f'{{ "Error": "Failed to read from server: {e}" }},', flush=True)
            pass
        except Exception as e:
            pass
        finally:
            sock.close()
    
    def __del__(self):
        global sglang_server
        print(f'{{ "SGLangCommon Destructor" }},', flush=True)
        sglang_server = None
    
class SGLangOpenAI(SGLangCommon):
    def __init__(self, model_id = None):
        super().__init__(model_id)

        self.tokenizer_path = None
        self.test_prompts = []

    def _find_tokenizer_path(self):
        if self.tokenizer_path is not None and self.tokenizer_path.exists():
            return
        lookup_path = self.env['HF_HOME'] / 'hub' / f'models--{self.sglang_config["model-path"].replace("/", "--")}' / 'snapshots'
        if not lookup_path.exists():
            raise FileNotFoundError(f"Snapshots path not found for model {self.sglang_config['model-path']}\nLookup path: {lookup_path.as_posix()}")
        for snapshot_path in lookup_path.iterdir():
            path = snapshot_path / 'tokenizer.json'
            if path.exists() and path.is_file():
                self.tokenizer_path = path
                break
        if self.tokenizer_path is None:
            raise FileNotFoundError(f"Tokenizer path not found for model {self.sglang_config['model-path']}\nLookup path: {lookup_path.as_posix()}")
        print(f'{{ "Tokenizer Path": "{self.tokenizer_path.as_posix()}" }},')
    
    def _generate_random_prompts(self):
        if self.dataset_name != 'random':
            raise ValueError(f"Dataset name {self.dataset_name} is not supported")
        if self.tokenizer_path is None:
            raise RuntimeError(f"Tokenizer path not found for model {self.sglang_config['model-path']}")
        self.test_prompts = []
        # @TODO: Maybe reasonable to use AutoTokenizer from Hugging Face instead of loading json file manually
        tokenizer = json.load(open(self.tokenizer_path.as_posix(), 'r'))
        vocab = list(tokenizer['model']['vocab'].keys())
        for _ in range(self.total_inference_runs + 1):
            prompt = ''
            for _ in range(self.random_input_length):
                prompt += vocab[random.randint(0, len(vocab) - 1)]
            # print(f'{{ "Prompt": "{prompt}" }},', flush=True)
            data = json.dumps({
                "model": self.sglang_config['model-path'],
                "prompt": prompt,
                "max_tokens": random.randint(self.random_output_length // 2, self.random_output_length), # We are not expecting empty response
                "temperature": 0.0
            }, indent=4)
            self.test_prompts.append(
                b'POST /v1/completions HTTP/1.1\r\n' +
                f'Host: {self.sglang_config["host"]}:{self.sglang_config["port"]}\r\n'.encode('utf-8') +
                b'Content-Type: application/json\r\n' +
                f'Content-Length: {len(data)}\r\n'.encode('utf-8') +
                b'Accept: */*\r\n' +
                b'User-Agent: testperf\r\n' +
                b'Connection: close\r\n' +
                b'\r\n' +
                data.encode('utf-8')
            )

    def prepare(self):
        self._find_tokenizer_path()
        self._generate_random_prompts()
    
    def inference(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.connect((self.sglang_config['host'], self.sglang_config['port']))
            
            # Prepare HTTP request for OpenAI-compatible API
            # Send request
            sock.sendall(self.test_prompts[self.current_inference_run])

            # Read response until we get some output
            while True:
                chunk = sock.recv(1024)
                if not chunk:
                    break
                #print(chunk.decode('utf-8'), flush=True)

            # @TODO: Analyze response
        except (ConnectionRefusedError, TimeoutError, socket.gaierror) as e:
            print(f'{{ "Error": "Failed to read from server: {e}" }},', flush=True)
            pass
        finally:
            sock.close()

class SGLangBench(SGLangCommon):
    def __init__(self, model_id = None):
        super().__init__(model_id)

        # Need to run only once, multiple tests does by 3rd party tool
        self.empty_runs = 0
        self.total_inference_runs = 1

        self.bench_config = {
            'backend': 'sglang',
            'model': self.sglang_config['model-path'],
            'base-url': 'http://' + self.sglang_config["host"] + ':' + str(self.sglang_config["port"]),
            'output-file': settings.APP_PATH / 'temp' / 'sglang_bench_result.json',
            'dataset-name': self.dataset_name,
            'random-input-len': self.random_input_length,
            'random-output-len': self.random_output_length,
            'num-prompts': 500
        }
        self.bench_cmd = os.environ.get('SGLANG_BENCH_CMD', 'python -m sglang.bench_serving').split()
        self.bench_extra_args = os.environ.get('SGLANG_BENCH_EXTRA_ARGS', '').split()

        self.bench_command_line = []
        self.bench_environment = {}
        # Holding all results for all runs (batch size -> result)
        self.all_results = {}
        self.batch_details = {}
    
    def warm_up(self):
        # No warm up for SGLang Bench
        pass

    def prepare(self):
        global sglang_server
        self.all_results[self.batch_size] = None
        if self.bench_config['output-file'].exists():
            self.bench_config['output-file'].unlink()
        self.bench_command_line = [*self.bench_cmd, *self.bench_extra_args]
        for key, value in self.bench_config.items():
            if value is None: continue
            self.bench_command_line.append(f'--{key}')
            if value == '': continue
            self.bench_command_line.append(str(value))
        self.bench_environment = self.get_environment_variables()
        if not self.batch_size in self.batch_details:
            # Server may have several command lines to start
            if sglang_server is not None:
                server_commands = [sglang_server.server_command_line]
            else:
                server_commands = ['[not running] ' + self.get_server_command_line()]
            self.batch_details[self.batch_size] = {
                'name': 'Batch ' + str(self.batch_size),
                'desc': 'Batch ' + str(self.batch_size) + ' has no description',
                'server_commands': server_commands,
                'server_env': self.env,
                'bench_commands': [self.bench_command_line],
                'bench_env': self.bench_environment
            }
        print(f'{{ "Bench Command Line": "{" ".join(self.bench_command_line)}" }},', flush=True)
    
    def inference(self):
        result = subprocess.run(self.bench_command_line, capture_output=True, text=True, env=self.bench_environment)
        if result.returncode != 0:
            print(f'{{ "SGLang Bench Environment Variables": "{self.bench_environment}" }},', flush=True)
            print(f'{{ "SGLang Bench Command Line": "{" ".join(self.bench_command_line)}" }},', flush=True)
            print(f'{{ "SGLang Bench Result": "{result.returncode}" }},', flush=True)
            print(f'{{ "SGLang Bench Output": "{result.stdout}" }},', flush=True)
            print(f'{{ "SGLang Bench Error": "{result.stderr}" }},', flush=True)
            raise RuntimeError(f'{{ "Error": "Failed to run bench command" }},')
        if self.bench_config['output-file'].exists():
            self.all_results[self.batch_size] = json.load(open(self.bench_config['output-file'], 'r'))
        else:
            self.all_results[self.batch_size] = None
    
    # @TODO: Better to have separate method for 
    def shutdown(self):
        results_available = False
        for _, result in self.all_results.items():
            if result is not None:
                results_available = True
                break
        if not results_available:
            super().shutdown()
            return
        try:
            import reports
            try:
                reports.vllm_bench_report(self, self.sglang_config['model-path'] + '_bench', list(self.all_results.keys()), self.all_results, self.batch_details)
            except Exception as e:
                print(f'{{ "Error": "Failed to generate SGLang Bench report: {e}" }},', flush=True)
                pass
            try:
                reports.vllm_bench_report_html(self, self.sglang_config['model-path'] + '_bench', list(self.all_results.keys()), self.all_results, self.batch_details)
            except Exception as e:
                print(f'{{ "Error": "Failed to generate SGLang Bench report HTML: {e}" }},', flush=True)
                pass
        except Exception as e:
            print(f'{{ "Error": "Failed to generate SGLang Bench report: {e}" }},', flush=True)
            pass
        super().shutdown()
