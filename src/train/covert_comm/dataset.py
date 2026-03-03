"""Dataset generation for covert communication training.

Generates JSONL files of numbers organized by digit count (levels)
for curriculum training.

Usage:
    from src.train.covert_comm.dataset import generate_dataset
    generate_dataset(level=1)  # 2-digit numbers (10-99)
    generate_dataset(level=2)  # 3-digit numbers (100-999)
"""

from __future__ import annotations

import json
import math
import os

from src.train.covert_comm.prompts import make_encode_prompt


def sieve_primes(lower: int, upper: int) -> list[int]:
    """Return all primes in [lower, upper] using the Sieve of Eratosthenes."""
    if upper < 2:
        return []
    is_prime = [True] * (upper + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(upper)) + 1):
        if is_prime[i]:
            for j in range(i * i, upper + 1, i):
                is_prime[j] = False
    return [n for n in range(max(lower, 2), upper + 1) if is_prime[n]]


def get_numbers_by_level(level: int) -> list[int]:
    """Get all numbers for a given digit-count level.

    Level 1 = 2-digit numbers (10-99)
    Level 2 = 3-digit numbers (100-999)
    Level 3 = 4-digit numbers (1000-9999)
    etc.
    """
    num_digits = level + 1
    lower = 10 ** (num_digits - 1)
    upper = 10**num_digits - 1
    return list(range(lower, upper + 1))


# Keep old name as alias for backward compatibility
get_primes_by_level = get_numbers_by_level


def generate_dataset(
    level: int = 1,
    output_dir: str = "results/data",
    enable_thinking: bool = False,
) -> str:
    """Generate a JSONL dataset file for a given level.

    Each row follows the ``DatasetExample`` schema expected by the training
    pipeline.

    Args:
        level: Difficulty level (1 = 2-digit, 2 = 3-digit, ...).
        output_dir: Directory to write the output file.
        enable_thinking: Include chain-of-thought instruction in prompts.

    Returns:
        Path to the generated JSONL file.
    """
    numbers = get_numbers_by_level(level)
    num_digits = level + 1

    examples = []
    for idx, number in enumerate(numbers):
        examples.append(
            {
                "id": idx,
                "dataset": "covert_comm",
                "evaluator": "covert_comm",
                "question": f"Encode the number {number}",
                "gt_answer": str(number),
                "answer": str(number),
                "prompt": make_encode_prompt(number, enable_thinking=enable_thinking),
                "hint": None,
                "prompt_metadata": {
                    "target_number": number,
                    "num_digits": num_digits,
                },
            }
        )

    output_path = os.path.join(output_dir, f"covert_comm_level{level}.jsonl")
    os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    print(f"Wrote {len(examples)} examples ({num_digits}-digit numbers) to {output_path}")
    return output_path
