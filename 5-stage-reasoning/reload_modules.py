"""Helper script to reload modules in Colab after code changes."""

import sys
import importlib

# List of modules to reload
modules_to_reload = [
    'models.generator',
    'models.discriminator',
    'training.rl_loop',
    'training.adversarial_trainer',
    'sandbox.sandbox',
]

print("Reloading modules...")
for module_name in modules_to_reload:
    if module_name in sys.modules:
        print(f"  Reloading {module_name}")
        importlib.reload(sys.modules[module_name])
    else:
        print(f"  {module_name} not loaded yet")

print("✓ Module reload complete")
