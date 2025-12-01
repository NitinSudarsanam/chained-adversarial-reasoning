"""Problem dataset management with ground truth solutions."""

import json
from dataclasses import dataclass
from typing import List, Dict, Any
from pathlib import Path


@dataclass
class Problem:
    """Represents a coding problem with ground truth solution."""
    id: str
    description: str
    function_signature: str
    baseline_tests: List[str]
    reference_solution: str
    difficulty: str
    tags: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'description': self.description,
            'function_signature': self.function_signature,
            'baseline_tests': self.baseline_tests,
            'reference_solution': self.reference_solution,
            'difficulty': self.difficulty,
            'tags': self.tags
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Problem':
        """Create from dictionary."""
        return cls(
            id=data['id'],
            description=data['description'],
            function_signature=data['function_signature'],
            baseline_tests=data['baseline_tests'],
            reference_solution=data['reference_solution'],
            difficulty=data['difficulty'],
            tags=data['tags']
        )


def load_problems(filepath: str) -> List[Problem]:
    """Load problems from JSON file.
    
    Args:
        filepath: Path to JSON file containing problems
        
    Returns:
        List of Problem objects
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If JSON is malformed
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Problem file not found: {filepath}")
    
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if 'problems' not in data:
        raise ValueError("JSON must contain 'problems' key")
    
    problems = []
    for problem_data in data['problems']:
        problem = Problem.from_dict(problem_data)
        if validate_problem(problem):
            problems.append(problem)
        else:
            print(f"Warning: Problem {problem.id} failed validation, skipping")
    
    return problems


def validate_problem(problem: Problem) -> bool:
    """Validate that problem has valid structure and executable solution.
    
    Args:
        problem: Problem to validate
        
    Returns:
        True if valid, False otherwise
    """
    # Check required fields are non-empty
    if not problem.id or not problem.description:
        return False
    
    if not problem.function_signature or not problem.reference_solution:
        return False
    
    # Check reference solution is valid Python
    try:
        compile(problem.reference_solution, '<string>', 'exec')
    except SyntaxError:
        print(f"Syntax error in reference solution for {problem.id}")
        return False
    
    # Check baseline tests are valid
    for test in problem.baseline_tests:
        try:
            compile(test, '<string>', 'exec')
        except SyntaxError:
            print(f"Syntax error in baseline test for {problem.id}: {test}")
            return False
    
    return True
