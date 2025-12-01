"""Unified model that acts as both generator and discriminator."""

import torch
import re
from typing import List
from transformers import AutoModelForCausalLM, AutoTokenizer


class UnifiedModel:
    """Single model that can act as both generator and discriminator."""
    
    def __init__(self, model_name: str, device: str = "cpu"):
        """Initialize unified model from HuggingFace model.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to run on ('cpu' or 'cuda')
        """
        self.model_name = model_name
        self.device = device
        
        print(f"Loading unified model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Always use float32 for numerical stability
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map=device
        )
        self.model.eval()
        
        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def train(self):
        """Set model to training mode."""
        self.model.train()
    
    def eval(self):
        """Set model to evaluation mode."""
        self.model.eval()
    
    def parameters(self):
        """Return model parameters for optimizer."""
        return self.model.parameters()
    
    # Generator methods
    def generate_stage_output(
        self,
        problem: str,
        previous_stages: List[str],
        stage_id: int,
        prompt_template: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        function_signature: str = ""
    ) -> str:
        """Generate output for a specific reasoning stage (generator mode).
        
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
        
        # Format prompt
        try:
            prompt = prompt_template.format(
                problem=problem,
                previous_stages=previous_text if previous_stages else "None",
                function_signature=function_signature if function_signature else ""
            )
        except KeyError:
            prompt = prompt_template.format(
                problem=problem,
                previous_stages=previous_text if previous_stages else "None"
            )
        
        # Generate
        output = self._generate(prompt, max_new_tokens, temperature, top_p)
        
        # Sanitize output
        output = self._sanitize_output(output)
        
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
        """Generate final executable code (stage 5, generator mode).
        
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
        from models.generator import LLMGenerator
        # Reuse the cleaning logic
        temp_gen = type('obj', (object,), {'_clean_generated_code': LLMGenerator._clean_generated_code})()
        output = temp_gen._clean_generated_code(output)
        
        return output
    
    # Discriminator methods
    def generate_tests(
        self,
        problem: str,
        generator_code: str,
        num_tests: int = 5,
        prompt_template: str = None,
        max_new_tokens: int = 512,
        temperature: float = 0.8,
        top_p: float = 0.9
    ) -> str:
        """Generate adversarial test cases (discriminator mode).
        
        Args:
            problem: Problem description
            generator_code: Code generated by generator
            num_tests: Number of test cases to generate
            prompt_template: Optional custom prompt template
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            
        Returns:
            Generated test cases as pytest functions
        """
        if prompt_template is None:
            prompt_template = """You are generating test cases for a coding problem and solution.

Problem: {problem}

Generated Code:
{stage_output}

Generate {num_tests} challenging test cases as pytest functions.

Test Cases:
```python
import pytest

"""
        
        prompt = prompt_template.format(
            problem=problem,
            stage_output=generator_code,
            num_tests=num_tests
        )
        
        # Generate
        output = self._generate(prompt, max_new_tokens, temperature, top_p)
        
        # Sanitize and extract code
        output = self._extract_code_from_markdown(output)
        output = self._sanitize_test_code(output)
        
        return output
    
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
        
        # Tokenize with shorter max length to save memory
        full_text = prompt + output
        inputs = self.tokenizer(full_text, return_tensors="pt", truncation=True, max_length=1024).to(self.device)
        prompt_inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(self.device)
        
        # Get model outputs (WITH gradients for training)
        outputs = self.model(**inputs)
        logits = outputs.logits
        
        # Delete inputs to free memory immediately
        del inputs
        
        # Get log probs for generated tokens only
        prompt_len = prompt_inputs.input_ids.shape[1]
        
        # Handle edge case where output is too short
        if outputs.logits.shape[1] <= prompt_len:
            del logits, prompt_inputs
            return torch.tensor([0.0], device=self.device, requires_grad=True)
        
        generated_logits = logits[0, prompt_len-1:-1, :]
        generated_tokens = outputs.logits.shape[1] - prompt_len
        
        # Handle empty generation
        if generated_tokens == 0:
            del logits, prompt_inputs
            return torch.tensor([0.0], device=self.device, requires_grad=True)
        
        # Get actual token IDs
        full_token_ids = self.tokenizer(full_text, return_tensors="pt", truncation=True, max_length=1024).to(self.device).input_ids
        generated_token_ids = full_token_ids[0, prompt_len:]
        
        # Compute log probabilities
        log_probs = torch.nn.functional.log_softmax(generated_logits, dim=-1)
        token_log_probs = log_probs.gather(1, generated_token_ids.unsqueeze(1)).squeeze(1)
        
        # Clean up intermediate tensors
        del logits, generated_logits, generated_token_ids, log_probs, prompt_inputs, full_token_ids
        
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
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(self.device)
        
        # Clamp temperature to safe range
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
                    repetition_penalty=1.1,
                    no_repeat_ngram_size=3
                )
            except Exception as e:
                print(f"Warning: Generation failed with error: {type(e).__name__}: {str(e)[:100]}")
                if was_training:
                    self.model.train()
                return ""
        
        # Decode only the generated part
        generated_tokens = outputs[0][inputs.input_ids.shape[1]:]
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        # Restore training mode if it was on before
        if was_training:
            self.model.train()
        
        return generated_text
    
    def _sanitize_output(self, output: str) -> str:
        """Sanitize generated output."""
        output = output.strip()
        if output and not output[-1] in '.!?\n':
            last_period = max(
                output.rfind('.'),
                output.rfind('!'),
                output.rfind('?'),
                output.rfind('\n')
            )
            if last_period > len(output) // 2:
                output = output[:last_period + 1]
        return output
    
    def _extract_code_from_markdown(self, text: str) -> str:
        """Extract code from markdown code blocks."""
        pattern = r'```python\s*(.*?)\s*```'
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            return matches[0].strip()
        
        pattern = r'```\s*(.*?)\s*```'
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            return matches[0].strip()
        
        if '```python' in text:
            code = text.split('```python', 1)[1]
            code = code.split('```')[0]
            return code.strip()
        
        if '```' in text:
            code = text.split('```', 1)[1]
            code = code.split('```')[0]
            return code.strip()
        
        return text.strip()
    
    def _sanitize_test_code(self, code: str) -> str:
        """Sanitize generated test code."""
        if 'import pytest' not in code:
            code = 'import pytest\n\n' + code
        
        lines = code.split('\n')
        last_complete = len(lines)
        in_function = False
        function_indent = 0
        
        for i, line in enumerate(lines):
            if line.strip().startswith('def test_'):
                in_function = True
                function_indent = len(line) - len(line.lstrip())
            elif in_function and line.strip() and not line.startswith(' ' * (function_indent + 1)):
                in_function = False
                last_complete = i
        
        if in_function:
            function_lines = lines[-(len(lines) - last_complete):]
            has_content = any('assert' in line or 'pass' in line for line in function_lines)
            if not has_content:
                lines = lines[:last_complete]
        
        code = '\n'.join(lines)
        return code.strip()
