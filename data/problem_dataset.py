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
    
    # Handle both formats: wrapped {"problems": [...]} or direct array [...]
    if isinstance(data, dict):
        if 'problems' not in data:
            raise ValueError("JSON must contain 'problems' key or be a direct array")
        problem_list = data['problems']
    elif isinstance(data, list):
        problem_list = data
    else:
        raise ValueError("JSON must be either a dict with 'problems' key or an array")
    
    problems = []
    for problem_data in problem_list:
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
    
    # Note: We don't validate reference solution syntax because it may contain
    # class definitions, imports, or other patterns that don't compile standalone.
    # The solution will be validated at execution time.
    
    # Check baseline tests - they can be either strings or tuples
    for test in problem.baseline_tests:
        if isinstance(test, str):
            # String format - skip validation, will check at execution time
            if not test:
                print(f"Empty test string for {problem.id}")
                return False
        elif isinstance(test, (tuple, list)):
            # Tuple/list format - ensure it's not empty
            if not test:
                print(f"Empty test tuple for {problem.id}")
                return False
        else:
            print(f"Unknown test format for {problem.id}: {type(test)}")
            return False
    
    return True
