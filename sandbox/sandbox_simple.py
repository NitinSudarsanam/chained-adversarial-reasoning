"""Simple sandbox that executes raw assert statements."""

import subprocess
import sys
from dataclasses import dataclass
from typing import List


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


def execute_tests_simple(code: str, tests: str, timeout: int = 5) -> ExecutionResult:
    """Execute raw assert statements against code.
    
    Args:
        code: Python code to test
        tests: Raw assert statements (one per line or with setup code)
        timeout: Execution timeout
        
    Returns:
        ExecutionResult with test results
    """
    # Combine code and tests
    full_code = f"{code}\n\n# Tests\n{tests}"
    
    try:
        # Execute with timeout
        result = subprocess.run(
            [sys.executable, '-c', full_code],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        # Count tests - each line is a test (may contain setup + assert)
        test_lines = [line.strip() for line in tests.split('\n') 
                     if line.strip() and 'assert' in line]
        num_total = len(test_lines)
        
        if result.returncode == 0:
            # All tests passed
            return ExecutionResult(
                passed=True,
                num_passed=num_total,
                num_total=num_total,
                errors=[],
                stdout=result.stdout,
                stderr=result.stderr,
                timed_out=False
            )
        else:
            # Some test failed - count which ones pass
            num_passed = 0
            for test_line in test_lines:
                try:
                    test_code = f"{code}\n\n{test_line}"
                    test_result = subprocess.run(
                        [sys.executable, '-c', test_code],
                        capture_output=True,
                        timeout=2
                    )
                    if test_result.returncode == 0:
                        num_passed += 1
                except:
                    pass
            
            errors = []
            if result.stderr:
                errors = result.stderr.split('\n')[:10]
            
            return ExecutionResult(
                passed=False,
                num_passed=num_passed,
                num_total=num_total,
                errors=errors,
                stdout=result.stdout,
                stderr=result.stderr,
                timed_out=False
            )
    
    except subprocess.TimeoutExpired:
        num_total = len([l for l in tests.split('\n') if 'assert' in l])
        return ExecutionResult(
            passed=False,
            num_passed=0,
            num_total=num_total,
            errors=['Execution timed out'],
            stdout='',
            stderr='',
            timed_out=True
        )
    except Exception as e:
        num_total = len([l for l in tests.split('\n') if 'assert' in l])
        return ExecutionResult(
            passed=False,
            num_passed=0,
            num_total=num_total,
            errors=[str(e)],
            stdout='',
            stderr='',
            timed_out=False
        )
