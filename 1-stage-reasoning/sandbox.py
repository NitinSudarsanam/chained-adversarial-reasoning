from pathlib import Path
import ast
from dataclasses import dataclass
import multiprocessing as mp
import traceback

def sandbox_worker(code):
    exec(code)

def exec_check(code, timeout=5.0):
    p = mp.Process(target=sandbox_worker, args=(code,))
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        raise TimeoutError("Sandbox check timed out")
    


def _worker(code, func_name, args, conn):
    try:
        ns = {}
        exec(code, ns)
        func = ns[func_name]
        func(*args)
        conn.send("ok")
    except Exception as e:
        conn.send(("error", traceback.format_exc()))
    finally:
        conn.close()

def check_function_timeout(code, func_name, args, timeout=5.0):
    parent_conn, child_conn = mp.Pipe()
    p = mp.Process(target=_worker, args=(code, func_name, args, child_conn))
    p.start()

    if parent_conn.poll(timeout):
        result = parent_conn.recv()
        p.join()

        if result == "ok":
            return

        # error in subprocess
        status, tb = result
        raise RuntimeError(f"Function raised an exception:\n{tb}")

    # timeout
    p.terminate()
    raise TimeoutError("Sandbox check timed out")


@dataclass
class Rewards:
    generator_reward: int
    discriminator_reward: int

def strip_comments(s: str) -> str:
    cleaned_lines = []
    for line in s.splitlines():
        # split at "#" and keep only code before it
        code = line.split("#", 1)[0]
        cleaned_lines.append(code)
    return "\n".join(cleaned_lines)

def extract_and_parse_tests(s: str):
    """
    extracts all the tuples and returns the ones that parse correctly
    """
    tests = []
    buf = ""
    depth = 0
    inside_tuple = False
    correct = 0
    total = 0
    
    i = 0
    while i < len(s):
        ch = s[i]

        # detect start of tuple
        if ch == "(" and not inside_tuple:
            inside_tuple = True
            depth = 1
            buf = "("
        elif inside_tuple:
            buf += ch
            # track parentheses depth
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1

            # a full tuple was found
            if inside_tuple and depth == 0:
                # try to parse it
                total += 1
                try:
                    parsed = ast.literal_eval(buf)
                    tests.append(parsed)
                    correct += 1
                except Exception:
                    pass # did not parse

                inside_tuple = False
                buf = ""

        i += 1

    return tests, correct, total

def run_code_tests(code : str, tests : str, ground_truth) -> Rewards:
    try:
        tests, num_parsed, num_tests = extract_and_parse_tests(strip_comments(tests))
        num_failed_to_parse = num_tests - num_parsed
    except:
        tests_data = {
            "tests_generated": 0,
            "valid_tests": 0,
            "passed_valid_tests": 0,
            "success": False,
            "info": "generated tests failed to parse"
        }
        return Rewards(0, -1), tests_data
    
    if not isinstance(tests, list):
        tests_data = {
            "tests_generated": 0,
            "valid_tests": 0,
            "passed_valid_tests": 0,
            "success": False,
            "info": "discriminator did not generate a list"
        }
        return Rewards(0, -1), tests_data

    if len(tests) == 0:
        tests_data = {
            "tests_generated": num_tests,
            "valid_tests": 0,
            "passed_valid_tests": 0,
            "success": False,
            "info": "no generated tests"
        }
        return Rewards(0, -1), tests_data

    try:
        namespace = {}
        exec_check(code)
        exec(code, namespace)
        callables = [(name, obj) for name, obj in namespace.items() if callable(obj)]
        function_name, f = callables[-1]
    except:
        tests_data = {
            "tests_generated": 0,
            "valid_tests": 0,
            "passed_valid_tests": 0,
            "success": False,
            "info": "generated code did not compile"
        }
        return Rewards(-1, 0), tests_data
    
    try:
        ground_truth_namespace = {}
        exec_check(ground_truth)
        exec(ground_truth, ground_truth_namespace)
        ground_truth_callables = [obj for obj in ground_truth_namespace.values() if callable(obj)]
        ground_truth_f = ground_truth_callables[-1]
    except:
        tests_data = {
            "tests_generated": 0,
            "valid_tests": 0,
            "passed_valid_tests": 0,
            "success": False,
            "info": "ground truth failed to compile!"
        }
        return Rewards(0, 0), tests_data
    

    tests_passed = 0
    generator_reward = 0
    discriminator_reward = 0
    CORRECT_TEST_CASE_REWARD = 0.05 / num_tests # discriminator generates a test case that matches with ground truth
    WRONG_TEST_CASE_REWARD = -1 / num_tests # discriminator generates a test case that fails with ground truth
    TEST_PASSED_REWARD = 1 / num_tests
    TEST_FAILED_REWARD = -1 / num_tests
    EDGE_CASE_CAUGHT_REWARD = 1 / num_tests

    valid_tests = 0 # tests that pass ground truth
    passed_valid_tests = 0

    discriminator_reward += -num_failed_to_parse / num_tests # penalty for the tests that failed to parse

    for test in tests:
        try:
            args = test[:-1]
            result = test[-1]
        except: # malformed test case
            discriminator_reward += WRONG_TEST_CASE_REWARD
            continue
            
        try:
            lhs = ground_truth_f(*args)
            if isinstance(lhs, list):
              lhs = tuple(lhs)
            rhs = result
            if isinstance(rhs, list):
              rhs = tuple(rhs)
            if lhs == rhs: # valid test case
                discriminator_reward += CORRECT_TEST_CASE_REWARD
                valid_tests += 1
                try:
                    check_function_timeout(code, function_name, args)
                    lhs = f(*args)
                    if isinstance(lhs, list):
                      lhs = tuple(lhs)
                    rhs = result
                    if isinstance(rhs, list):
                      rhs = tuple(rhs)
                    if lhs == rhs:
                        generator_reward += TEST_PASSED_REWARD
                        passed_valid_tests += 1
                    else:
                        generator_reward += TEST_FAILED_REWARD
                        discriminator_reward += EDGE_CASE_CAUGHT_REWARD
                except Exception as e:
                    # print(e)
                    generator_reward += TEST_FAILED_REWARD
                    discriminator_reward += EDGE_CASE_CAUGHT_REWARD
            else: #invalid test case
                discriminator_reward += WRONG_TEST_CASE_REWARD
                # print(f"truth: {lhs}")
                # print(f"test: {rhs}")
        except:
            discriminator_reward += WRONG_TEST_CASE_REWARD

    # print(f"num tests: {len(tests)}")
    # print(f"valid tests: {valid_tests}")
    # print(f"passed valid tests: {passed_valid_tests}")
    
    tests_data = {
        "tests_generated": num_tests,
        "valid_tests": valid_tests,
        "passed_valid_tests": passed_valid_tests,
        "success": True,
        "info": "attempted to run all tests"
    }

    # renormalize generator reward to account for the invalid test cases
    if valid_tests == 0:
        return Rewards(generator_reward, discriminator_reward), tests_data
    generator_reward *= (num_tests / valid_tests)
    return Rewards(generator_reward, discriminator_reward), tests_data
