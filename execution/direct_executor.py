"""Direct code execution using ast.literal_eval - simple and fast."""

import ast
import inspect


class ExecutionResult:
    """Result from executing code against test cases."""
    def __init__(self, num_passed: int, num_total: int, passed_tests: list = None, failed_tests: list = None, 
                 is_valid: list = None):
        self.num_passed = num_passed
        self.num_total = num_total
        self.passed_tests = passed_tests or []
        self.failed_tests = failed_tests or []
        self.is_valid = is_valid or []  # Whether each test is valid (passed validation)


def execute_tests(code: str, tests_str: str, validation_result: 'ExecutionResult' = None) -> ExecutionResult:
    """Execute test tuples directly against code using ast.literal_eval.
    
    Args:
        code: Generated Python code containing a function
        tests_str: String representation of test tuples
                  e.g., "[(arg1, arg2, expected), ...]"
        validation_result: Optional ExecutionResult from ground truth validation
                          If provided, marks which tests are valid
    
    Returns:
        ExecutionResult with num_passed, num_total, which tests passed,
        and validity of each test
    """
    # Parse test cases
    try:
        tests = ast.literal_eval(tests_str)
    except:
        return ExecutionResult(0, 0, [], [], [])
    
    if len(tests) == 0:
        return ExecutionResult(0, 0, [], [], [])
    
    # Extract function from code and helpers for canonical structures
    try:
        namespace = {}
        exec(code, namespace)
        callables = [obj for obj in namespace.values() if callable(obj)]
        func = callables[-1]

        # Provide lightweight defaults if user code did not define them
        Node = namespace.get("Node")

        if "ListNode" not in namespace:
            if Node is not None:
                ListNode = Node
            else:
                class ListNode:
                    def __init__(self, val=0, next=None):
                        self.val = val
                        self.next = next
        else:
            ListNode = namespace["ListNode"]

        if "TreeNode" not in namespace:
            if Node is not None:
                TreeNode = Node
            else:
                class TreeNode:
                    def __init__(self, val=0, left=None, right=None):
                        self.val = val
                        self.left = left
                        self.right = right
        else:
            TreeNode = namespace["TreeNode"]

        def build_linked_list(values):
            if ListNode is None or not isinstance(values, (list, tuple)):
                return values
            dummy = ListNode(0)
            cur = dummy
            for v in values:
                cur.next = ListNode(v)
                cur = cur.next
            return dummy.next

        def linked_list_to_list(node):
            if ListNode is None or node is None:
                return node
            out = []
            cur = node
            while cur:
                out.append(cur.val)
                cur = cur.next
            return out

        def build_tree(values):
            if TreeNode is None or not isinstance(values, (list, tuple)):
                return values
            if not values:
                return None
            nodes = [None if v is None else TreeNode(v) for v in values]
            kids = nodes[::-1]
            root = kids.pop()
            for node in nodes:
                if node:
                    if kids:
                        node.left = kids.pop()
                    if kids:
                        node.right = kids.pop()
            return root

        def tree_to_list(root):
            if TreeNode is None or root is None:
                return root
            out = []
            queue = [root]
            while queue:
                node = queue.pop(0)
                if node:
                    out.append(node.val)
                    queue.append(node.left)
                    queue.append(node.right)
                else:
                    out.append(None)
            # Trim trailing None values for cleanliness
            while out and out[-1] is None:
                out.pop()
            return out

        def normalize_value(val):
            if ListNode is not None and isinstance(val, (ListNode, Node if 'Node' in locals() else tuple())):
                return linked_list_to_list(val)
            if TreeNode is not None and isinstance(val, (TreeNode, Node if 'Node' in locals() else tuple())):
                return tree_to_list(val)
            return val

        sig = inspect.signature(func)
        params = list(sig.parameters.values())
        ret_ann = sig.return_annotation

    except Exception:
        is_valid = [i in validation_result.passed_tests for i in range(len(tests))] if validation_result else []
        return ExecutionResult(0, len(tests), [], list(range(len(tests))), is_valid)
    
    # Run tests and track which ones pass, compute validity
    num_passed = 0
    passed_tests = []
    failed_tests = []
    is_valid = []
    
    for idx, test in enumerate(tests):
        args = list(test[:-1])
        expected = test[-1]

        # Convert inputs based on annotations when possible
        for i, arg in enumerate(args):
            if i < len(params):
                ann = params[i].annotation
                if ListNode is not None and (ann is ListNode or getattr(ann, "__name__", "") in {"ListNode", "Node"} or str(ann) in {"ListNode", "Node"}):
                    args[i] = build_linked_list(arg)
                elif TreeNode is not None and (ann is TreeNode or getattr(ann, "__name__", "") in {"TreeNode", "Node"} or str(ann) in {"TreeNode", "Node"}):
                    args[i] = build_tree(arg)

        # Convert expected if return annotation hints at structures
        if ret_ann is not inspect.Signature.empty:
            if ListNode is not None and (ret_ann is ListNode or getattr(ret_ann, "__name__", "") in {"ListNode", "Node"} or str(ret_ann) in {"ListNode", "Node"}):
                expected = build_linked_list(expected)
            elif TreeNode is not None and (ret_ann is TreeNode or getattr(ret_ann, "__name__", "") in {"TreeNode", "Node"} or str(ret_ann) in {"TreeNode", "Node"}):
                expected = build_tree(expected)
        
        # Mark validity if validation result provided
        if validation_result:
            is_valid.append(idx in validation_result.passed_tests)
        
        try:
            result = func(*args)
            result_norm = normalize_value(result)
            expected_norm = normalize_value(expected)
            if result_norm == expected_norm:
                num_passed += 1
                passed_tests.append(idx)
            else:
                failed_tests.append(idx)
        except:
            failed_tests.append(idx)
    
    return ExecutionResult(num_passed, len(tests), passed_tests, failed_tests, is_valid)
