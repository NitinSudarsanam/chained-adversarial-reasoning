import torch
import re
from typing import List
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
import importlib
import sandbox
import rl_loop
from data.problem_dataset import load_problems
import json
from sandbox import run_code_tests
from huggingface_hub import login
login(token='')

"""Generator LLM for producing multi-stage reasoning and code."""

# ONLY LLAMA 3.1 8B SUPPORTED RIGHT NOW
class GenericLLM:
    def __init__(self, model, tokenizer, model_name, SYSTEM_PROMPT, device: str = "cuda"):
        """Initialize generator from HuggingFace model.

        Args:
            model_name: HuggingFace model identifier
            device: Device to run on ('cpu' or 'cuda')
        """
        self.model_name = model_name
        self.device = device
        self.tokenizer = tokenizer
        self.SYSTEM_PROMPT = SYSTEM_PROMPT

        print(f"Loading generator model: {model_name}")

        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        )

        base_model = prepare_model_for_kbit_training(model)
        self.model = get_peft_model(base_model, peft_config)

        for name, param in self.model.named_parameters():
            if "lora_" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        self.model.eval()

        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token


    def train(self):
        """Set model to training mode."""
        self.model.training = True
        self.model.train()

    def activate_lora(self):
        for name, param in self.model.named_parameters():
            if "lora_" in name:
                param.requires_grad = True

    def deactivate_lora(self):
        for name, param in self.model.named_parameters():
            if "lora_" in name:
                param.requires_grad = False

    def eval(self):
        """Set model to evaluation mode."""
        self.model.training = False
        self.model.eval()

    def parameters(self):
        """Return model parameters for optimizer."""
        # return self.model.parameters()
        return [param for name, param in self.model.named_parameters() if "lora_" in name]

    def generate_code(
        self,
        prompt,
        max_new_tokens: int = 2048,
        temperature: float = 0.8,
        top_p: float = 0.9,
    ) -> str:
        """Generate code given the problem.

        Returns:
            Generated Python code
        """

        output = self._generate(prompt, 2048, 0.8, 0.9)

        # Extract code from markdown if present
        output = self._extract_code_from_markdown(output)

        # Additional cleaning for code
        # output = self._clean_generated_code(output)

        output = output.replace("\t", "    ")
        # output = output.replace("List", "list")
        return output # to fix typing inconsistency with newer python versions

    def get_log_probs(self, prompt: str, output: str) -> torch.Tensor:
        """Get log probabilities for RL training.

        Args:
            prompt: Input prompt
            output: Generated output

        Returns:
            Log probabilities tensor
        """
        # Handle empty output
        if not output or not output.strip():
            return torch.tensor([0.0], device=self.device, requires_grad=True)

        # Tokenize with reasonable max length
        full_text = prompt + output
        inputs = self.tokenizer(full_text, return_tensors="pt", truncation=True, max_length=4096).to(self.device)
        prompt_inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(self.device)

        # Get model outputs (WITH gradients for training)
        outputs = self.model(**inputs) # allocates a whole bunch of memory?
        logits = outputs.logits

        # Get log probs for generated tokens only
        prompt_len = prompt_inputs.input_ids.shape[1]
        input_len = inputs.input_ids.shape[1]

        # Handle edge case where output is too short
        if input_len <= prompt_len:
            del inputs, logits, prompt_inputs
            return torch.tensor([0.0], device=self.device, requires_grad=True)

        generated_logits = logits[0, prompt_len-1:-1, :]
        generated_tokens = inputs.input_ids[0, prompt_len:]

        # Delete inputs now that we've extracted what we need
        del inputs

        # Handle empty generation
        if generated_tokens.shape[0] == 0:
            del logits, generated_logits, generated_tokens, prompt_inputs
            return torch.tensor([0.0], device=self.device, requires_grad=True)

        # Compute log probabilities
        log_probs = torch.nn.functional.log_softmax(generated_logits, dim=-1)
        token_log_probs = log_probs.gather(1, generated_tokens.unsqueeze(1)).squeeze(1)

        # Clean up intermediate tensors
        del logits, generated_logits, generated_tokens, log_probs, prompt_inputs

        return token_log_probs

    def _generate(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float
    ) -> str:
        """Internal generation method.

        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter

        Returns:
            Generated text
        """
        # CRITICAL: Always set to eval mode before generation
        was_training = self.model.training
        self.model.eval()

        # Format prompt using chat template for instruction-tuned models
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(formatted_prompt, return_tensors="pt", truncation=True, max_length=4096).to(self.device)

        # Clamp temperature to safe range to avoid numerical issues
        temperature = max(0.1, min(2.0, temperature))
        top_p = max(0.1, min(1.0, top_p))

        with torch.no_grad():
            try:
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1,  # Prevent repetition issues
                    no_repeat_ngram_size=3   # Prevent exact repetitions
                )
            except Exception as e:
                # If generation fails, return empty string to skip this example
                print(f"Warning: Generation failed with error: {type(e).__name__}: {str(e)[:100]}")
                # Restore training mode if needed
                if was_training:
                    self.model.train()
                # Don't try to use CUDA operations after CUDA error - just return empty
                return ""

        # Decode only the generated part
        generated_tokens = outputs[0][inputs.input_ids.shape[1]:]
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        # Restore training mode if it was on before
        if was_training:
            self.model.train()

        return generated_text

    def _extract_code_from_markdown(self, text: str) -> str:
        """Extract code from markdown code blocks.

        Args:
            text: Text potentially containing markdown code blocks

        Returns:
            Extracted code or original text
        """
        # Look for ```python ... ``` blocks (closed)
        pattern = r'```python\s*(.*?)\s*```'
        matches = re.findall(pattern, text, re.DOTALL)

        if matches:
            return matches[0].strip()

        # Look for ``` python ... ``` blocks (closed)
        pattern = r'``` python\s*(.*?)\s*```'
        matches = re.findall(pattern, text, re.DOTALL)

        if matches:
            return matches[0].strip()

        # Look for ``` ... ``` blocks (closed)
        pattern = r'```\s*(.*?)\s*```'
        matches = re.findall(pattern, text, re.DOTALL)

        if matches:
            return matches[0].strip()

        # Look for unclosed ```python blocks (model didn't close it)
        if '```python' in text:
            # Extract everything after ```python
            code = text.split('```python', 1)[1]
            # Remove trailing ``` if present
            code = code.split('```')[0]
            return code.strip()

        # Look for unclosed ``` blocks
        if '```' in text:
            # Extract everything after first ```
            code = text.split('```', 1)[1]
            # Remove trailing ``` if present
            code = code.split('```')[0]
            return code.strip()

        return text.strip()

model_name = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)

quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="cuda",
    quantization_config=quant_config
)

GENERATOR_SYSTEM_PROMPT = """
You are an expert in Python programming. You have been tasked to solve Leetcode-style questions.
You will be given a problem description and a function which you must implement. Your implementation will then be run against a suite of test cases, and your goal is to pass as many test cases as possible.

You will have access to the following class definitions. You do not need to create them.

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

IMPORTANT:
- You will be given an EXACT function signature to use. You MUST use this EXACT function signature, or else your solution will not execute and you will receive no credit.
- Write a complete, working Python function. DO NOT leave placeholders or TODOs.
- You should only the Python function, and nothing else. Do not write test cases or show example use cases.

Here is an example response you would give.


PROBLEM DESCRIPTION: Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.

You may assume that each input would have exactly one solution, and you may not use the same element twice.

You can return the answer in any order.


FUNCTION SIGNATURE: def twoSum(nums: List[int], target: int) -> List[int]:


YOUR RESPONSE:
```python
def twoSum(nums, target):
        num_map = {}
        for i, num in enumerate(nums):
            complement = target - num
            if complement in num_map:
                return [num_map[complement], i]
            num_map[num] = i
```

"""

generator = GenericLLM(model, tokenizer, model_name, GENERATOR_SYSTEM_PROMPT)

model2 = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="cuda",
    quantization_config=quant_config
)

DISCRIMINATOR_SYSTEM_PROMPT = """
You are an expert in Software Testing. You have been tasked with generating test cases for Leetcode-style questions in Python.
You will be given a problem description and a function signature. You should construct your test case suite as a Python lists of test cases, where each test case is a Python tuple, where the first n - 1 element represent the inputs to the function, and the final element represents the expected result.
The test cases that you generate will be run against a candidate implementation, and your test suite should be as thorough as possible. You will achieve an award for catching edge cases.

You will have access to the following class definitions. You do not need to create them.

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

IMPORTANT:
- You should ONLY output the test cases. Do not attempt to solve the problem yourself.
- Your test cases MUST not fail against a ground-truth solution. If they do, you will incur a large penalty.
- You should output your tests as a PYTHON LIST, and ONLY a Python list. Do not use Markdown or plain text.


Here is an example response you would give:

PROBLEM DESCRIPTION: Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.

You may assume that each input would have exactly one solution, and you may not use the same element twice.

You can return the answer in any order.


FUNCTION SIGNATURE: def twoSum(nums: List[int], target: int) -> List[int]:


YOUR RESPONSE:
```python
[
  ([5, 3, 1, 0], 4, [1, 2]),
  ([5, 3, 6, 2, 4, 56, 1], 6, [3, 4]),
  ([8, 3, 2, 6, 2, 7, 8], 13, [3, 5])
]
```

"""
discriminator = GenericLLM(model2, tokenizer, model_name, DISCRIMINATOR_SYSTEM_PROMPT)

importlib.reload(rl_loop)
importlib.reload(sandbox)

class Trainer:
    def __init__(self, generator, discriminator):
        self.generator = generator
        self.discriminator = discriminator

        self.gen_optimizer = rl_loop.create_optimizer(self.generator)
        self.disc_optimizer = rl_loop.create_optimizer(self.discriminator)


    def step(self, problem_description, function_signature, ground_truth):
        # takes in problem, returns rewards
        ground_truth = ground_truth.replace("List", "list")
        ground_truth = ground_truth.replace("class Solution:\n  ", "")
        ground_truth = ground_truth.replace("self, ", "")

        prompt = f"""
        PROBLEM DESCRIPTION: {problem_description}


        FUNCTION SIGNATURE: {function_signature}


        YOUR RESPONSE:

        """

        # should get the log probs before ALL cleaning
        generator_output = self.generator.generate_code(prompt)
        generator_output_clean = generator_output.replace("List", "list").replace("from typing import list", "") # fix some typing issues with the new versions
        generator_old_log_probs = generator.get_log_probs(prompt, generator_output)

        discriminator_output = self.discriminator.generate_code(prompt)
        discriminator_old_log_probs = discriminator.get_log_probs(prompt, discriminator_output)

        rewards, info = run_code_tests(generator_output_clean, discriminator_output, ground_truth)

        generator_reward = rewards.generator_reward
        discriminator_reward = rewards.discriminator_reward

        rl_loop.train_step(self.generator, self.gen_optimizer, [prompt], [generator_output], [generator_reward], [generator_old_log_probs])
        rl_loop.train_step(self.discriminator, self.disc_optimizer, [prompt], [discriminator_output], [discriminator_reward], [discriminator_old_log_probs])

        return rewards, info, generator_output, discriminator_output

trainer = Trainer(generator, discriminator)

log_path = '/users/achowd32/fixed_final/log.jsonl'
lora_path = '/users/achowd32/fixed_final'
#problems = load_problems("data/leetcode_formatted.json")
if __name__ == "__main__":
    for i in range(100):
        problem = problems[i]
        print(f"SOLVING PROBLEM: {problem.id}")
        try:
            rewards, info, generator_output, discriminator_output = trainer.step(problem.description, problem.function_signature, problem.reference_solution)
        except Exception as e:
            rewards = sandbox.Rewards(0, 0)
            info = {
                "tests_generated": 0,
                "valid_tests": 0,
                "passed_valid_tests": 0,
                "success": False,
                "info": e
            }
            generator_output = ""
            discriminator_output = ""
        print(f"GENERATOR REWARD: {rewards.generator_reward}")
        print(f"DISCRIMINATOR REWARD: {rewards.discriminator_reward}")
        print(f"STATS:")
        print(info)

        info["problem"] = problem.id
        info["generator_reward"] = rewards.generator_reward
        info["discriminator_reward"] = rewards.discriminator_reward
        info["trial"] = i

        if i % 10 == 0:
          info["generator_output"] = generator_output
          info["discriminator_output"] = discriminator_output
          trainer.generator.model.save_pretrained(f"/users/achowd32/fixed_final/lora_checkpoints1/generator_step_{i}")
          trainer.discriminator.model.save_pretrained(f"/users/achowd32/fixed_final/lora_checkpoints1/discriminator_step_{i}")
        #with open(log_path, "a") as f:
        #    f.write(json.dumps(info) + "\n")
