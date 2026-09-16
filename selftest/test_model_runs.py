import sys
import re
import subprocess
import os
import pytest
from common import get_combined_output

class TestModelRuns:
    """Test cases for model lookup functionality in test_perf.py"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up the test environment - run from selftest folder, parent is perftest"""
        # Get the parent directory (perftest) from current file location
        self.parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.test_script = os.path.join(self.parent_dir, "test_perf.py")
        self.noisy_model = "selftest.models.noisy_model"
        
    def run_subprocess(self, args):
        cmd = [sys.executable, self.test_script] + args
        result = subprocess.run(
            cmd,
            cwd=self.parent_dir,
            capture_output=True,
            text=True
        )
        return result.stdout, result.stderr, result.returncode

    def test_noisy_model_read_runs_lines_count(self):
        """
        Run test_perf.py with selftest/models/noisy_model.py (no extra params),
        parse '{ "Read Runs": X }' and verify output has X lines with 'Noisy model.read()'.
        """
        stdout, stderr, code = self.run_subprocess([self.noisy_model])
        assert code == 0, get_combined_output(stdout, stderr)

        out = get_combined_output(stdout, stderr)

        # Parse "Read Runs": X (allow optional comma/whitespace after })
        match = re.search(r'\{\s*"Read Runs"\s*:\s*(\d+)\s*\}', out)
        assert match, f'"Read Runs" not found in output: {out[:500]}...'
        x = int(match.group(1))

        # Search between '{ "Read Runs": X },' and '{ "Total Read" : ... },'
        rr_marker = re.search(r'\{\s*"Read Runs"\s*:\s*\d+\s*\},', out)
        tr_marker = re.search(r'\{\s*"Total Read"\s*:\s*"[^"]+"\s*\},', out)
        assert rr_marker and tr_marker, "Could not find output section markers"
        start_pos = rr_marker.end()
        end_pos = tr_marker.start()
        read_section = out[start_pos:end_pos]

        lines_with_read = [line for line in read_section.splitlines() if "{ \"Noisy model\" : \"read()\" }," in line]
        assert len(lines_with_read) == x, (
            f"Expected {x} lines with {{ \"Noisy model\" : \"read()\" }}, got {len(lines_with_read)}. "
            f"Lines: {lines_with_read}"
        )

    def test_noisy_model_read_times_count(self):
        """
        Run test_perf.py with selftest/models/noisy_model.py (no extra params).
        Between '{ "Read Times": [' and ']}', count lines like '{ "Time" : "1.2344343" },'
        and verify count equals X (Read Runs).
        """
        stdout, stderr, code = self.run_subprocess([self.noisy_model])
        assert code == 0, get_combined_output(stdout, stderr)

        out = get_combined_output(stdout, stderr)

        import json
        data = json.loads(out)
        X = None
        steps = data.get("Steps", [])
        for step in steps:
            if isinstance(step, dict) and "Read Runs" in step:
                X = step["Read Runs"]
                break
        if X is None:
            raise AssertionError(f'Could not get Steps["Read Runs"] from output: {out[:500]}...')
        read_times = None
        for step in steps:
            if isinstance(step, dict) and "Read Times" in step:
                read_times = step["Read Times"]
                break
        if read_times is None:
            raise AssertionError(f'Could not get Steps["Read Times"] from output: {out[:500]}...')
        assert len(read_times) == X, (
            f"Expected len(Read Times) == Read Runs ({X}), but got {len(read_times)}. Read Times: {read_times}"
        )

    @pytest.mark.parametrize("runs_arg,expected_count", [
        (None, 100),
        (["--runs", "10"], 10),
        (["--runs", "1"], 1),
        (["--runs", "50"], 50),
    ])
    def test_noisy_model_inference_times_count(self, runs_arg, expected_count):
        """
        Run test_perf.py with selftest/models/noisy_model.py with optional --runs Y.
        Between '{ "Inference Times": [' and '] },' count lines like
        '{ "Time" : "4.4099999740865314e-05" },'; count must equal Y (default 100).
        """
        args = [self.noisy_model]
        if runs_arg is not None:
            args.extend(runs_arg)

        stdout, stderr, code = self.run_subprocess(args)
        assert code == 0, get_combined_output(stdout, stderr)

        out = get_combined_output(stdout, stderr)

        import json
        data = json.loads(out)
        X = None
        steps = data.get("Steps", [])
        for step in steps:
            if isinstance(step, dict) and "Inference Times" in step:
                X = step["Inference Times"]
                break
        if X is None:
            raise AssertionError(f'Could not get Steps["Inference Times"] from output: {out[:500]}...')
        assert len(X) == expected_count, (
            f"Expected len(Inference Times) == {expected_count}, but got {len(X)}. Inference Times: {X}"
        )

        warm_up_marker = '{ "Model Warm Up 1" :'
        total_inf_marker = '{ "Total Inference 1"'

        warm_up_idx = out.find(warm_up_marker)
        assert warm_up_idx != -1, f'"{warm_up_marker}" not found in output'
        after_warm_up = out[warm_up_idx + len(warm_up_marker):]

        total_inf_idx = after_warm_up.find(total_inf_marker)
        assert total_inf_idx != -1, f'"{total_inf_marker}" not found after warm-up in output'

        section = after_warm_up[:total_inf_idx]
        inference_count = section.count('{ \"Noisy model\" : \"inference()\" },')

        assert inference_count == expected_count, (
            f"Expected {expected_count} {{ \"Noisy model\" : \"inference()\" }}, calls in section, got {inference_count}."
            f"\n---\nSection: {section[:400]}..."
        )
    

