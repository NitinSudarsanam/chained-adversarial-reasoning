"""Generator LLM for producing multi-stage reasoning and code."""

import torch
import re
from typing import List
from transformers import AutoModelForCausalLM, AutoTokenizer


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
        
        print(f"Loading generator model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32 if device == "cpu" else torch.float16,
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
        
        # Tokenize
        full_text = prompt + output
        inputs = self.tokenizer(full_text, return_tensors="pt", truncation=True, max_length=2048).to(self.device)
        prompt_inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(self.device)
        
        # Get model outputs (WITH gradients for training)
        outputs = self.model(**inputs)
        logits = outputs.logits
        
        # Get log probs for generated tokens only
        prompt_len = prompt_inputs.input_ids.shape[1]
        
        # Handle edge case where output is too short
        if inputs.input_ids.shape[1] <= prompt_len:
            return torch.tensor([0.0], device=self.device, requires_grad=True)
        
        generated_logits = logits[0, prompt_len-1:-1, :]
        generated_tokens = inputs.input_ids[0, prompt_len:]
        
        # Handle empty generation
        if generated_tokens.shape[0] == 0:
            return torch.tensor([0.0], device=self.device, requires_grad=True)
        
        # Compute log probabilities
        log_probs = torch.nn.functional.log_softmax(generated_logits, dim=-1)
        token_log_probs = log_probs.gather(1, generated_tokens.unsqueeze(1)).squeeze(1)
        
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
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        # Decode only the generated part
        generated_tokens = outputs[0][inputs.input_ids.shape[1]:]
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
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
    
    def _clean_generated_code(self, code: str) -> str:
        """Clean generated code to ensure it's executable.
        
        Args:
            code: Raw generated code
            
        Returns:
            Cleaned code
        """
        # Remove any leading/trailing whitespace
        code = code.strip()
        
        # Split into lines
        lines = code.split('\n')
        
        # Strategy: Find the BEST function/class definition
        # Look for one that has actual implementation (not just pass)
        
        # Find all function/class definitions
        definitions = []
        i = 0
        while i < len(lines):
            stripped = lines[i].strip()
            if stripped.startswith(('def ', 'class ')):
                # Found a definition, collect it
                start = i
                indent = len(lines[i]) - len(lines[i].lstrip())
                i += 1
                
                # Collect all lines that belong to this definition
                while i < len(lines):
                    line = lines[i]
                    stripped_line = line.strip()
                    
                    # Stop if we hit another top-level definition
                    if stripped_line.startswith(('def ', 'class ')) and (len(line) - len(line.lstrip())) == indent:
                        break
                    
                    # Stop if we hit explanatory text (not indented, not empty, not comment)
                    if stripped_line and not line.startswith((' ', '\t')) and not stripped_line.startswith('#'):
                        break
                    
                    i += 1
                
                definition_lines = lines[start:i]
                
                # Check if this definition has real implementation (not just pass)
                has_implementation = False
                for line in definition_lines[1:]:  # Skip the def/class line
                    stripped = line.strip()
                    if stripped and stripped != 'pass' and not stripped.startswith(('#', '"""', "'''")):
                        # Check if it's actual code (not just docstring)
                        if not (stripped.startswith('"""') or stripped.startswith("'''")):
                            has_implementation = True
                            break
                
                definitions.append({
                    'lines': definition_lines,
                    'has_implementation': has_implementation,
                    'start': start
                })
            else:
                i += 1
        
        # Choose the best definition
        if not definitions:
            # No definitions found, return original
            return code
        
        # Prefer definitions with implementation
        best_def = None
        for d in definitions:
            if d['has_implementation']:
                best_def = d
                break
        
        # If no implementation found, use the first definition
        if not best_def:
            best_def = definitions[0]
        
        # Clean the selected definition
        cleaned_lines = []
        skip_docstring = False
        docstring_char = None
        
        for line in best_def['lines']:
            stripped = line.strip()
            
            # Handle docstrings
            if stripped.startswith('"""') or stripped.startswith("'''"):
                if not skip_docstring:
                    docstring_char = '"""' if stripped.startswith('"""') else "'''"
                    skip_docstring = True
                    if stripped.endswith(docstring_char) and len(stripped) > 3:
                        skip_docstring = False
                    continue
                elif stripped.endswith(docstring_char):
                    skip_docstring = False
                    continue
            
            if skip_docstring:
                continue
            
            # Skip lines with just pass if there's other code
            if stripped == 'pass' and len(cleaned_lines) > 1:
                # Check if there's real code after the def line
                has_real_code = any(
                    l.strip() and l.strip() != 'pass' and not l.strip().startswith('#')
                    for l in cleaned_lines[1:]
                )
                if has_real_code:
                    continue
            
            cleaned_lines.append(line)
        
        code = '\n'.join(cleaned_lines)
        return code.strip()
