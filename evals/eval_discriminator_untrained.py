import torch
from colab import GenericLLM, DISCRIMINATOR_SYSTEM_PROMPT, model, tokenizer
from sandbox import run_code_tests
from transformers import AutoModelForCausalLM, AutoTokenizer
from data.problem_dataset import load_problems
import csv
from peft import PeftModel

# get problems
print("LOADING AND FILTERING PROBLEMS")
problems = load_problems("data/discriminator_tests.json")
for prob in problems:
    prob.baseline_tests = [tuple(x) for x in prob.baseline_tests]

# convert raw models and tokenizers to GenericLLM class
trained_model = GenericLLM(model, tokenizer, "trained", DISCRIMINATOR_SYSTEM_PROMPT)

# set to eval (no training)
trained_model.eval()

# do the actual evaluation
for problem in problems:
    print(f"GENERATING TESTS FOR PROBLEM: {problem.problem_id}")

    problem_description = problem.problem_statement
    function_signature = problem.function_signature
    buggy_code = problem.buggy
    correct_code = problem.fixed
    prompt = f"""
        PROBLEM DESCRIPTION: {problem_description}


        FUNCTION SIGNATURE: {function_signature}


        YOUR RESPONSE:

        """

    discriminator_output = trained_model.generate_code(prompt)
    total_tests, valid_tests, passed_tests = run_code_tests(buggy_code, discriminator_output, correct_code, eval_bool=True)
    if not isinstance(total_tests, int):
        print("ERROR OCCURRED")
    else:
        print(f"""TOTAL: {total_tests}\n
                VALID: {valid_tests}\n
                PASSED: {passed_tests}""")
        new_data = [problem.problem_id, total_tests, valid_tests, passed_tests]
        with open('untrained_disc.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(new_data)
