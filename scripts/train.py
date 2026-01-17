#!/usr/bin/env python
"""
Training script wrapper for easier command-line usage.

Usage:
    python scripts/train.py
    python scripts/train.py --quick
    python scripts/train.py --cpu
    python scripts/train.py --production
"""

import subprocess
import sys
from pathlib import Path

# Get the project root
PROJECT_ROOT = Path(__file__).parent.parent
TRAIN_SCRIPT = PROJECT_ROOT / "src" / "mbti_classifier" / "training" / "train.py"

def main():
    """Run the training script with appropriate config."""
    # Parse simple arguments
    config_name = "train"  # default
    
    if "--quick" in sys.argv:
        config_name = "train_quick"
        sys.argv.remove("--quick")
    elif "--cpu" in sys.argv:
        config_name = "train_cpu"
        sys.argv.remove("--cpu")
    elif "--production" in sys.argv:
        config_name = "train_production"
        sys.argv.remove("--production")
    
    # Build command
    cmd = [
        "uv", "run", "python", str(TRAIN_SCRIPT),
        f"--config-name={config_name}"
    ]
    
    # Add any additional arguments
    if len(sys.argv) > 1:
        cmd.extend(sys.argv[1:])
    
    # Print what we're running
    print(f"Running: {' '.join(cmd)}")
    print("-" * 80)
    
    # Execute
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    sys.exit(result.returncode)

if __name__ == "__main__":
    main()
