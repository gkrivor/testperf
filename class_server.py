import os
import subprocess
import time
import socket

class ServerKeeper:
    def __init__(self, host, port, log_file = None):
        self.process = None
        self.output = open(log_file, 'w') if log_file is not None else None
        self.startup_timeout = os.environ.get('TESTPERF_SERVER_STARTUP_TIMEOUT', 600)
        self.host = host
        self.port = port
        self.server_command_line = []
        self.server_env = {}
        self.external_server = False
        self._dump_log()
    
    def is_available(self, extended_check = False):
        self.external_server = False

        # If server is running by ourselves and don't need to check if it is really available - return True
        if self.process is not None and self.process.poll() is None and not extended_check:
            return True
        
        if self.process is not None and self.process.poll() is not None:
            if self.output is not None:
                self.output.write(f'Return code: {self.process.returncode}\n')
            return False

        if not extended_check:
            return False

        # Check server is really reachable
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.settimeout(1)
            sock.connect((self.host, self.port))
            self.external_server = True
            return True
        except (ConnectionRefusedError, TimeoutError, socket.gaierror) as e:
            return False
        except Exception as e:
            print(f'{{ "Error": "Failed to check if server is available: {e}" }},', flush=True)
            return False
        finally:
            sock.close()

    def _dump_log(self):
        if self.output is None:
            return
        self.output.write(f'Timestamp: {time.strftime("%Y-%m-%d %H:%M:%S")}\n')
        self.output.write(f'Command line: {" ".join(self.server_command_line)}\n')
        self.output.write(f'Environment variables:\n')
        for key, value in sorted(self.server_env.items(), key=lambda x: x[0].lower()):
            self.output.write(f'  {key} = "{value}"\n')
        self.output.write(f'Output:\n')

    def start(self, server_command_line, server_env):
        if self.is_available(True):
            if self.external_server:
                print(f'{{ "Server Status": "External Server is already running at {self.host}:{self.port}" }},', flush=True)
            else:
                print(f'{{ "Server Status": "Already Started at {self.host}:{self.port}" }},', flush=True)
            return

        print(f'{{ "Server Command Line": "{" ".join(server_command_line)}" }},', flush=True)

        self.server_command_line = server_command_line
        self.server_env = server_env

        self.process = subprocess.Popen(server_command_line, env=server_env, stdout=self.output, stderr=self.output)

        if self.output is not None:
            self.output.flush()

        start_time = time.time()

        while not self.is_available(True):
            if time.time() - start_time > self.startup_timeout:
                if self.output is not None:
                    self.output.write(f'Server startup timed out after {self.startup_timeout} seconds\n')
                raise RuntimeError(f'{{ "Server Status": "Server startup timed out after {self.startup_timeout} seconds" }},')
            if int(time.time() - start_time) % 5 == 0:
                print(f'{{ "Server Status": "Waiting for server to be available at {self.host}:{self.port}, elapsed {time.time() - start_time:.0f} / {self.startup_timeout} seconds" }},', flush=True)
            time.sleep(1)

        print(f'{{ "Server Status": "Started at {self.host}:{self.port}" }},', flush=True)

    def stop(self):
        if self.process is not None:
            self.process.terminate()
            self.process.wait(10)
            self.process.kill()
            self.process.wait(5)
            if self.is_available(True):
                if self.output is not None:
                    self.output.write(f'Failed to stop server at {self.host}:{self.port}\n')
                    self.output.close()
                    self.output = None
                raise RuntimeError(f'{{ "Server Status": "Failed to stop server at {self.host}:{self.port}" }},')
            self.process = None
        if self.output is not None:
            self.output.close()
            self.output = None
        print(f'{{ "Server Status": "Stopped" }},', flush=True)

    def __del__(self):
        self.stop()
