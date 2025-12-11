import torch
from colab import GenericLLM, GENERATOR_SYSTEM_PROMPT, model, tokenizer
from sandbox import run_code_tests
from transformers import AutoModelForCausalLM, AutoTokenizer
from data.problem_dataset import load_problems
import csv
from peft import PeftModel

# get paths
trained_checkpt_path = "generator_step_100"

# convert raw models and tokenizers to GenericLLM class
raw_model = PeftModel.from_pretrained(
    model,
    trained_checkpt_path,
    torch_dtype=torch.float16,
)
trained_model = GenericLLM(model, tokenizer, "trained", GENERATOR_SYSTEM_PROMPT)

# set to eval (no training)
trained_model.eval()

# get problems
print("LOADING AND FILTERING PROBLEMS")
problems = load_problems("data/leetcode_formatted.json")
for prob in problems:
    prob.baseline_tests = [tuple(x) for x in prob.baseline_tests]
problems = [problem for problem in problems if "linked-list" not in problem.tags]
problems = [problem for problem in problems if "tree" not in problem.tags]
problems = [problem for problem in problems if "easy" == problem.difficulty]

# do the actual evaluation
for problem in problems:
    print(f"SOLVING PROBLEM: {problem.id}")

    problem_description = problem.description
    function_signature = problem.function_signature
    test_cases = str(problem.baseline_tests)
    if len(problem.baseline_tests) == 0:
        continue
    prompt = f"""
        PROBLEM DESCRIPTION: {problem_description}


        FUNCTION SIGNATURE: {function_signature}


        YOUR RESPONSE:

        """
    ground_truth = problem.reference_solution
    ground_truth = ground_truth.replace("List", "list")
    ground_truth = ground_truth.replace("class Solution:\n  ", "")
    ground_truth = ground_truth.replace("self, ", "")

    generator_output = trained_model.generate_code(prompt)
    generator_output_clean = generator_output.replace("List", "list").replace("from typing import list", "") # fix some typing issues with the new versions
    print("TEST CASES:\n")
    print(test_cases)
    valid_tests, passed_valid_tests = run_code_tests(generator_output_clean, test_cases, ground_truth, eval_bool=True)
    if not isinstance(valid_tests, int):
        valid_tests = len(problem.baseline_tests)
        passed_valid_tests = 0
    print(f"{passed_valid_tests} / {valid_tests} passed")
    new_data = [problem.id, valid_tests, passed_valid_tests]
    with open('trained_gen.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(new_data)
