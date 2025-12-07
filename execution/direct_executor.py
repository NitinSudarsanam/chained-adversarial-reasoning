"""Direct code execution using ast.literal_eval - simple and fast."""

import ast


class ExecutionResult:
    """Result from executing code against test cases."""
    def __init__(self, num_passed: int, num_total: int, passed_tests: list = None, failed_tests: list = None, 
                 is_valid: list = None):
        self.num_passed = num_passed
        self.num_total = num_total
        self.passed_tests = passed_tests or []
        self.failed_tests = failed_tests or []
        self.is_valid = is_valid or []  # Whether each test is valid (passed validation)


def execute_tests(code: str, tests_str: str, validation_result: 'ExecutionResult' = None) -> ExecutionResult:
    """Execute test tuples directly against code using ast.literal_eval.
    
    Args:
        code: Generated Python code containing a function
        tests_str: String representation of test tuples
                  e.g., "[(arg1, arg2, expected), ...]"
        validation_result: Optional ExecutionResult from ground truth validation
                          If provided, marks which tests are valid
    
    Returns:
        ExecutionResult with num_passed, num_total, which tests passed,
        and validity of each test
    """
    # Parse test cases
    try:
        tests = ast.literal_eval(tests_str)
    except:
        return ExecutionResult(0, 0, [], [], [])
    
    if len(tests) == 0:
        return ExecutionResult(0, 0, [], [], [])
    
    # Extract function from code
    try:
        namespace = {}
        exec(code, namespace)
        callables = [obj for obj in namespace.values() if callable(obj)]
        func = callables[-1]
    except:
        is_valid = [i in validation_result.passed_tests for i in range(len(tests))] if validation_result else []
        return ExecutionResult(0, len(tests), [], list(range(len(tests))), is_valid)
    
    # Run tests and track which ones pass, compute validity
    num_passed = 0
    passed_tests = []
    failed_tests = []
    is_valid = []
    
    for idx, test in enumerate(tests):
        args = test[:-1]
        expected = test[-1]
        
        # Mark validity if validation result provided
        if validation_result:
            is_valid.append(idx in validation_result.passed_tests)
        
        try:
            result = func(*args)
            if result == expected:
                num_passed += 1
                passed_tests.append(idx)
            else:
                failed_tests.append(idx)
        except:
            failed_tests.append(idx)
    
    return ExecutionResult(num_passed, len(tests), passed_tests, failed_tests, is_valid)
