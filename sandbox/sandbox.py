"""Secure sandbox for executing generated code and tests."""

import subprocess
import tempfile
from dataclasses import dataclass
from typing import List
from pathlib import Path


@dataclass
class ExecutionResult:
    """Result of code execution in sandbox."""
    passed: bool
    num_passed: int
    num_total: int
    errors: List[str]
    stdout: str
    stderr: str
    timed_out: bool


class Sandbox:
    """Secure execution environment for running code and tests."""
    
    def __init__(self, timeout: int = 5):
        """Initialize sandbox.
        
        Args:
            timeout: Maximum execution time in seconds
        """
        self.timeout = timeout
    
    def execute_tests(self, code: str, tests: str, timeout: int = None) -> ExecutionResult:
        """Execute test cases against code.
        
        Args:
            code: Generated code to test
            tests: Test cases as Python code
            timeout: Override default timeout
            
        Returns:
            ExecutionResult with pass/fail status and details
        """
        if timeout is None:
            timeout = self.timeout
        
        # Create temporary directory for execution
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write code and tests to files
            code_file = Path(tmpdir) / "solution.py"
            test_file = Path(tmpdir) / "test_solution.py"
            
            with open(code_file, 'w', encoding='utf-8') as f:
                f.write(code)
            
            # Prepare test file with imports
            # Wrap raw assert statements in test functions if needed
            if tests.strip() and not tests.strip().startswith('def test_'):
                # Raw assert statements - wrap them
                test_lines = [line.strip() for line in tests.split('\n') if line.strip()]
                test_functions = []
                for i, line in enumerate(test_lines):
                    test_functions.append(f"def test_case_{i}():\n    {line}\n")
                tests_wrapped = "\n".join(test_functions)
            else:
                tests_wrapped = tests
            
            test_content = f"""
import sys
sys.path.insert(0, '{tmpdir}')
from solution import *

{tests_wrapped}
"""
            with open(test_file, 'w', encoding='utf-8') as f:
                f.write(test_content)
            
            # Run tests with pytest
            return self._run_pytest(test_file, timeout)
    
    def validate_tests_against_solution(
        self, 
        tests: str, 
        solution: str, 
        timeout: int = None
    ) -> ExecutionResult:
        """Validate discriminator tests against ground truth solution.
        
        Args:
            tests: Test cases to validate
            solution: Ground truth reference solution
            timeout: Override default timeout
            
        Returns:
            ExecutionResult showing which tests pass against ground truth
        """
        return self.execute_tests(solution, tests, timeout)
    
    def _run_pytest(self, test_file: Path, timeout: int) -> ExecutionResult:
        """Run pytest on test file and parse results.
        
        Args:
            test_file: Path to test file
            timeout: Execution timeout
            
        Returns:
            ExecutionResult with parsed test results
        """
        try:
            # Run pytest with verbose output
            result = subprocess.run(
                ['python', '-m', 'pytest', str(test_file), '-v', '--tb=short'],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=test_file.parent,
                check=False
            )
            
            stdout = result.stdout
            stderr = result.stderr
            timed_out = False
            
        except subprocess.TimeoutExpired:
            stdout = ""
            stderr = "Execution timed out"
            timed_out = True
            return ExecutionResult(
                passed=False,
                num_passed=0,
                num_total=0,
                errors=["Timeout"],
                stdout=stdout,
                stderr=stderr,
                timed_out=True
            )
        except Exception as e:
            return ExecutionResult(
                passed=False,
                num_passed=0,
                num_total=0,
                errors=[str(e)],
                stdout="",
                stderr=str(e),
                timed_out=False
            )
        
        # Parse pytest output
        num_passed, num_total, errors = self._parse_pytest_output(stdout, stderr)
        
        return ExecutionResult(
            passed=(num_passed == num_total and num_total > 0),
            num_passed=num_passed,
            num_total=num_total,
            errors=errors,
            stdout=stdout,
            stderr=stderr,
            timed_out=timed_out
        )
    
    def _parse_pytest_output(self, stdout: str, stderr: str) -> tuple:
        """Parse pytest output to extract test results.
        
        Args:
            stdout: Standard output from pytest
            stderr: Standard error from pytest
            
        Returns:
            Tuple of (num_passed, num_total, errors)
        """
        num_passed = 0
        num_total = 0
        errors = []
        
        # Look for pytest summary line like "5 passed, 2 failed in 0.5s"
        for line in stdout.split('\n'):
            if 'passed' in line.lower() or 'failed' in line.lower():
                # Extract numbers
                parts = line.split()
                for i, part in enumerate(parts):
                    if 'passed' in part.lower() and i > 0:
                        try:
                            num_passed = int(parts[i-1])
                        except (ValueError, IndexError):
                            pass
                    if 'failed' in part.lower() and i > 0:
                        try:
                            num_failed = int(parts[i-1])
                            num_total = num_passed + num_failed
                        except (ValueError, IndexError):
                            pass
        
        # If no failures mentioned, all passed
        if num_passed > 0 and num_total == 0:
            num_total = num_passed
        
        # Extract error messages
        if stderr:
            errors.append(stderr)
        
        # Look for FAILED test names
        for line in stdout.split('\n'):
            if 'FAILED' in line or 'ERROR' in line:
                errors.append(line.strip())
        
        return num_passed, num_total, errors
