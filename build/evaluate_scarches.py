"""Evaluate scArches checkpoints with the shared reference-mapping evaluator."""

import sys

from evaluate_lora import main


if __name__ == "__main__":
    if "--method" not in sys.argv:
        sys.argv[1:1] = ["--method", "scarches"]
    main()
