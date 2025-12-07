"""
FIX SUMMARY: NameError - 'gen_result' is not defined
==================================================

PROBLEM:
--------
Error during training:
  NameError: name 'gen_result' is not defined
  at line 175 in training/adversarial_trainer.py

Root Cause:
  - train_discriminator_epoch() was referencing gen_result and val_result variables
  - These variables were only defined inside run_code_tests() but not returned
  - The training code had debug logging that tried to access attributes on these results
  - The attributes (timed_out, errors, stderr) don't exist on our ExecutionResult class

SOLUTION:
---------

1. Modified training/reward.py:
   - Updated Rewards dataclass to include optional gen_result and val_result fields
   - Modified run_code_tests() to return these execution results in the Rewards object
   
2. Modified training/adversarial_trainer.py:
   - Line 170-173: Updated to extract gen_result and val_result from Rewards object
   - Line 330-333: Changed to use run_code_tests() instead of non-existent sandbox
   - Removed debug logging that referenced non-existent attributes (timed_out, errors, stderr)

CHANGES:

File: training/reward.py
------------------------
- Rewards dataclass now includes:
    gen_result: ExecutionResult = None
    val_result: ExecutionResult = None
    
- run_code_tests() now returns:
    Rewards(gen_reward, disc_reward, gen_result, val_result)

File: training/adversarial_trainer.py
----- 
- Line 170-173: Added extraction of results from Rewards:
    gen_result = rewards.gen_result
    val_result = rewards.val_result
    
- Line 330-333: Replaced sandbox call with run_code_tests:
    rewards = run_code_tests(final_code, accumulated_tests, problem.reference_solution)
    reward = rewards.generator_reward
    result = rewards.gen_result
    
- Removed debug logging that accessed undefined attributes:
    - Removed gen_result.timed_out access
    - Removed gen_result.errors access
    - Removed gen_result.stderr parsing

VERIFICATION:
--------------
✓ training/reward.py - No syntax errors
✓ training/adversarial_trainer.py - No syntax errors
✓ All undefined variable references fixed
✓ All non-existent attribute accesses removed

STATUS:
-------
✓ FIXED - Training should now run without NameError
"""

print(__doc__)
