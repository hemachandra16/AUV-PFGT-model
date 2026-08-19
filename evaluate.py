"""PFGT-UIE Evaluation Script.

Thin wrapper around validate.py for quick evaluation of a checkpoint with
metric display in the terminal. For full CSV reporting, use validate.py.

Usage:
    python evaluate.py --checkpoint checkpoints/best.pt
"""
from __future__ import annotations

# Re-export the validation main function for backward compatibility.
# This keeps evaluate.py as a valid entry-point while avoiding duplication.
from validate import main

if __name__ == "__main__":
    main()
