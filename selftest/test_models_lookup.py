import sys
import os
import subprocess
import pytest

# Override common.get_combined_output to return stdout + stderr
def get_combined_output(stdout, stderr):
    return stdout + stderr

class TestModelLookup:
    """Test cases for model lookup functionality in test_perf.py"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up the test environment - run from selftest folder, parent is perftest"""
        # Get the parent directory (perftest) from current file location
        self.parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.test_script = os.path.join(self.parent_dir, "test_perf.py")
        
    def run_subprocess(self, args):
        cmd = [sys.executable, self.test_script] + args
        result = subprocess.run(
            cmd,
            cwd=self.parent_dir,
            capture_output=True,
            text=True
        )
        return result.stdout, result.stderr, result.returncode
    
    def test_no_parameters(self):
        """Test test_perf.py without parameters - should fail with module not found"""
        stdout, stderr, exit_code = self.run_subprocess([])
        
        combined_output = get_combined_output(stdout, stderr)
        
        assert "Failed to load model No module named 'test_model'" in combined_output, \
            f"Expected error message not found. Output: {combined_output}"
        assert "Available options:" in combined_output, \
            f"Expected 'Available options:' not found. Output: {combined_output}"
        assert exit_code == 1, \
            f"Expected exit code 1, got {exit_code}"
    
    def test_models_parameter(self):
        """Test test_perf.py with 'models' parameter - should fail at load with no Model class"""
        stdout, stderr, exit_code = self.run_subprocess(["models"])
        
        combined_output = get_combined_output(stdout, stderr)
        
        assert "Failed to load model Module 'models' does not have a 'Model' class" in combined_output, \
            f"Expected error message not found. Output: {combined_output}"
        assert "Available options:" in combined_output, \
            f"Expected 'Available options:' not found. Output: {combined_output}"
        assert exit_code == 1, \
            f"Expected exit code 1, got {exit_code}"
    
    def test_models_yolo_parameter(self):
        """Test test_perf.py with 'models.yolo' parameter - should fail with module not found"""
        stdout, stderr, exit_code = self.run_subprocess(["models.yolo"])
        
        combined_output = get_combined_output(stdout, stderr)
        
        assert "Failed to load model No module named 'models.yolo'" in combined_output, \
            f"Expected error message not found. Output: {combined_output}"
        assert "Available options:" in combined_output, \
            f"Expected 'Available options:' not found. Output: {combined_output}"
        assert exit_code == 1, \
            f"Expected exit code 1, got {exit_code}"
    
    @pytest.mark.parametrize("model_path", [
        "selftest/models/noisy_model",
        "selftest\\models\\noisy_model",
        "./selftest/models/noisy_model",
        ".\\selftest\\models\\noisy_model",
        "selftest/models/noisy_model.py",
        "selftest\\models\\noisy_model.py",
        "./selftest/models/noisy_model.py",
        ".\\selftest\\models\\noisy_model.py",
    ])
    def test_noisy_model_various_paths(self, model_path):
        """Test test_perf.py with selftest/models/noisy_model using various path formats"""
        stdout, stderr, exit_code = self.run_subprocess([model_path, "--only-prepare"])
        
        combined_output = get_combined_output(stdout, stderr)

        # Verify exit code is 0 (success); leading-dot paths normalize to a valid module too
        assert exit_code == 0, \
            f"Expected exit code 0 for {model_path}, got {exit_code}. Output: {combined_output}"
        
        # Verify the output contains expected model initialization
        assert "{ \"Noisy model\" : \"__init__()\" }," in combined_output, \
            f"Expected {{ \"Noisy model\" : \"__init__()\" }}, in output for {model_path}. Output: {combined_output}"
        
        # Verify the output contains model preparation
        assert "{ \"Noisy model\" : \"prepare_batch(1)\" }," in combined_output, \
            f"Expected {{ \"Noisy model\" : \"prepare_batch(1)\" }}, in output for {model_path}. Output: {combined_output}"
        
        # Verify the output contains model shutdown
        assert "{ \"Noisy model\" : \"shutdown()\" }," in combined_output, \
            f"Expected {{ \"Noisy model\" : \"shutdown()\" }}, in output for {model_path}. Output: {combined_output}"
        
        # Verify status is done
        assert '"Status": "Done"' in combined_output, \
            f"Expected '\"Status\": \"Done\"' in output for {model_path}. Output: {combined_output}"
