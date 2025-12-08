"""Generator LLM for producing multi-stage reasoning and code."""

import torch
import re
from typing import List
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model


class LLMGenerator:
    """Generator model that produces multi-stage reasoning outputs and code."""
    
    def __init__(self, model_name: str, device: str = "cpu"):
        """Initialize generator from HuggingFace model.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to run on ('cpu' or 'cuda')
        """
        self.model_name = model_name
        self.device = device

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        
        print(f"Loading generator model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Always use float32 for numerical stability (float16 can cause inf/nan issues)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device,
            quantization_config=quant_config
        )
        
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
        self.model.train()
        for name, param in self.model.named_parameters():
            if "lora_" in name:
                param.requires_grad = True
    
    def eval(self):
        """Set model to evaluation mode."""
        for name, param in self.model.named_parameters():
            if "lora_" in name:
                param.requires_grad = False
        self.model.eval()
    
    def parameters(self):
        """Return model parameters for optimizer."""
        # return self.model.parameters()
        return [param for name, param in self.model.named_parameters() if "lora_" in name]
    
    def generate_stage_output(
        self,
        problem: str,
        previous_stages: list[str],
        stage_id: int,
        prompt_template: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        function_signature: str = ""
    ) -> str:
        """Generate output for a specific reasoning stage.
        
        Args:
            problem: Problem description
            previous_stages: Outputs from previous stages
            stage_id: Current stage ID (1-5)
            prompt_template: Template for this stage
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            function_signature: Optional function/class signature for stage 5
            
        Returns:
            Generated output for this stage
        """
        # Format previous stages
        previous_text = "\n\n".join([
            f"Stage {i+1}:\n{stage}" 
            for i, stage in enumerate(previous_stages)
        ])
        
        # Format prompt - handle function_signature placeholder
        try:
            prompt = prompt_template.format(
                problem=problem,
                previous_stages=previous_text if previous_stages else "None",
                function_signature=function_signature if function_signature else ""
            )
        except KeyError:
            # Template doesn't have function_signature placeholder
            prompt = prompt_template.format(
                problem=problem,
                previous_stages=previous_text if previous_stages else "None"
            )
        
        # Generate
        output = self._generate(prompt, max_new_tokens, temperature, top_p)
        
        # Sanitize output
        output = self._sanitize_output(output)
        
        # Debug: Warn if output is empty
        if not output or not output.strip():
            print(f"⚠ WARNING: generate_stage_output produced EMPTY output")
            print(f"   Stage ID: {stage_id}")
            print(f"   Prompt preview: {prompt[:200]}...")
        
        return output
    
    def generate_code(
        self,
        problem: str,
        reasoning_chain: List[str],
        prompt_template: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        function_signature: str = ""
    ) -> str:
        """Generate final executable code (stage 5).
        
        Args:
            problem: Problem description
            reasoning_chain: All previous reasoning stages
            prompt_template: Template for code generation
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            function_signature: Function/class signature to implement
            
        Returns:
            Generated Python code
        """
        output = self.generate_stage_output(
            problem=problem,
            previous_stages=reasoning_chain,
            stage_id=5,
            prompt_template=prompt_template,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            function_signature=function_signature
        )
        
        # Extract code from markdown if present
        output = self._extract_code_from_markdown(output)
        
        # Additional cleaning for code
        output = self._clean_generated_code(output)
        
        # Debug: Warn if final code is empty
        if not output or not output.strip():
            print(f"⚠ WARNING: generate_code produced EMPTY final code")
            print(f"   Problem preview: {problem[:100]}...")
            print(f"   Reasoning chain length: {len(reasoning_chain)}")
        
        return output.replace("\t", "    ").replace("List", "list") # to fix typing inconsistency with newer python versions
    
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
        outputs = self.model(**inputs)
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
        if hasattr(self.tokenizer, 'apply_chat_template') and self.tokenizer.chat_template:
            system_prompt = """You are an expert in Python programming. You have been tasked to solve Leetcode-style questions.
You will be given a problem description and a function which you must implement. Your implementation will then be run against a suite of test cases, and your goal is to pass as many test cases as possible.

CRITICAL INSTRUCTIONS:
- You will be given an EXACT function signature to use. You MUST use this EXACT function signature, or else your solution will not execute and you will receive no credit.
- Write ONLY a single standalone Python function. DO NOT wrap it in a Solution class.
- DO NOT write: class Solution: or any class definitions.
- DO NOT leave placeholders or TODOs in the code.
- Output ONLY the function definition and nothing else. No comments, no test cases, no explanations.
- The function must be complete and working.

Here is an example response you would give:

PROBLEM DESCRIPTION: Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.
You may assume that each input would have exactly one solution, and you may not use the same element twice.
You can return the answer in any order.

FUNCTION SIGNATURE: def twoSum(nums: List[int], target: int) -> List[int]:

YOUR RESPONSE (function only, no class):
```python
def twoSum(nums, target):
    num_map = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in num_map:
            return [num_map[complement], i]
        num_map[num] = i
```"""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        else:
            formatted_prompt = prompt
        
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
                print(f"❌ ERROR: Generation failed with error: {type(e).__name__}: {str(e)[:200]}")
                print(f"   Prompt length: {len(formatted_prompt)} chars")
                print(f"   Max new tokens: {max_new_tokens}")
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
    
    def _sanitize_output(self, output: str) -> str:
        """Sanitize generated output.
        
        Args:
            output: Raw generated output
            
        Returns:
            Sanitized output
        """
        # Remove excessive whitespace
        output = output.strip()
        
        # Remove incomplete sentences at the end
        if output and not output[-1] in '.!?\n':
            # Find last complete sentence
            last_period = max(
                output.rfind('.'),
                output.rfind('!'),
                output.rfind('?'),
                output.rfind('\n')
            )
            if last_period > len(output) // 2:  # Only if we have substantial content
                output = output[:last_period + 1]
        
        return output
    
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
    
    def _fix_common_syntax_errors(self, code: str) -> str:
        """Fix minimal common syntax errors.
        
        Args:
            code: Code with potential syntax errors
            
        Returns:
            Code with common errors fixed
        """
        import re
        
        # Fix: "for xiny:" -> "for x in y:" (missing space)
        code = re.sub(r'\bfor\s+(\w+)in(\w+):', r'for \1 in \2:', code)
        
        # Fix: Capitalized keywords
        code = re.sub(r'\bIf\b', 'if', code)
        code = re.sub(r'\bElse\b', 'else', code)
        code = re.sub(r'\bElif\b', 'elif', code)
        
        return code
    
    def _clean_generated_code(self, code: str) -> str:
        """Clean generated code to ensure it's executable.
        
        The system prompt instructs the model to output ONLY the function,
        so we just need minimal cleanup.
        
        Args:
            code: Raw generated code
            
        Returns:
            Cleaned code
        """
        code = code.strip()
        
        # Remove explanatory text before the first function/class
        lines = code.split('\n')
        first_def_idx = -1
        for i, line in enumerate(lines):
            if line.strip().startswith(('def ', 'class ')):
                first_def_idx = i
                break
        
        if first_def_idx > 0:
            lines = lines[first_def_idx:]
            code = '\n'.join(lines)
        
        # Fix minimal syntax errors
        code = self._fix_common_syntax_errors(code)
        
        return code.strip()
