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
            return  # function finished normally

        # function raised error inside subprocess
        status, tb = result
        raise RuntimeError(f"Function raised an exception:\n{tb}")

    # timeout
    p.terminate()
    raise TimeoutError("Sandbox check timed out")


@dataclass
class Rewards:
    generator_reward: int
    discriminator_reward: int

def run_code_tests(code : str, tests : str, ground_truth, eval_bool : bool = False) -> Rewards:
    try:
        tests = ast.literal_eval(tests)
    except:
        tests_data = {
            "tests_generated": 0,
            "valid_tests": 0,
            "passed_valid_tests": 0,
            "success": False,
            "info": "generated tests failed to parse"
        }
        return Rewards(0, -1), tests_data, 0
    
    if not isinstance(tests, list):
        tests_data = {
            "tests_generated": 0,
            "valid_tests": 0,
            "passed_valid_tests": 0,
            "success": False,
            "info": "discriminator did not generate a list"
        }
        return Rewards(0, -1), tests_data, 0

    if len(tests) == 0:
        tests_data = {
            "tests_generated": 0,
            "valid_tests": 0,
            "passed_valid_tests": 0,
            "success": False,
            "info": "no generated tests"
        }
        if eval_bool:
            return (0, (0, 0))
        return Rewards(0, -1), tests_data, 0

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
        return Rewards(-1, 0), tests_data, 0
    
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
        return Rewards(0, 0), tests_data, 0

    tests_passed = 0
    generator_reward = 0
    discriminator_reward = 0
    CORRECT_TEST_CASE_REWARD = 0.01 / len(tests) # discriminator generates a test case that matches with ground truth
    WRONG_TEST_CASE_REWARD = -1 / len(tests) # discriminator generates a test case that fails with ground truth
    TEST_PASSED_REWARD = 1 / len(tests)
    TEST_FAILED_REWARD = -1 / len(tests)
    EDGE_CASE_CAUGHT_REWARD = 1 / len(tests)

    valid_tests = 0
    passed_valid_tests = 0

    for test in tests:
        try:
            args = test[:-1]
            result = test[-1]
        except: # malformed test case
            discriminator_reward += WRONG_TEST_CASE_REWARD
            print("MALFORMED TEST CASE")
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
            else: #invalid test case
                discriminator_reward += WRONG_TEST_CASE_REWARD
                # print(f"truth: {lhs}")
                # print(f"test: {rhs}")
        except:
            print("GROUND TRUTH ERROR")
            discriminator_reward += WRONG_TEST_CASE_REWARD

    # print(f"num tests: {len(tests)}")
    # print(f"valid tests: {valid_tests}")
    # print(f"passed valid tests: {passed_valid_tests}")
    
    # simply return number of tests passed if in eval mode
    if eval_bool:
        return (len(tests), valid_tests, passed_valid_tests)

    tests_data = {
        "tests_generated": len(tests),
        "valid_tests": valid_tests,
        "passed_valid_tests": passed_valid_tests,
        "success": True,
        "info": "attempted to run all tests"
    }
    # renormalize generator reward to account for the invalid test cases
    if valid_tests == 0:
        return Rewards(generator_reward, discriminator_reward), tests_data
    generator_reward *= (len(tests) / valid_tests)
    return Rewards(generator_reward, discriminator_reward), tests_data
