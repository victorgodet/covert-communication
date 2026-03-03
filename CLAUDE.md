# Covert Communication via RL

MATS sprint project. We train a single LLM (Qwen3-4B) to both encode a secret number into innocent-looking text and decode it back, using two-phase GRPO reinforcement learning. An external LLM judge (Opus) scores how detectable the encoding is, creating adversarial pressure for stealth. The AI safety question: can models learn to hide information from oversight?

## Project structure

```
scripts/run_covert_comm.py    # Main entry point (Fire CLI)
src/train/covert_comm/
  dataset.py                  # Dataset generation (K-digit numbers by level)
  prompts.py                  # Encoder/decoder/judge prompt templates
  rewards.py                  # Reward logic + LLM judge integration
  trainer.py                  # Two-phase GRPO trainer
  grpo.py                     # CovertCommGRPO (import directly, not via __init__)
src/judge.py                  # Generic LLM judge class (OpenRouter)
results/runs/qwen3-4b/        # Training run outputs (rollouts, configs, logs)
results/data/                  # Generated dataset JSONL files
```

## Running experiments

```bash
# Basic training run (level 2 = 3-digit numbers, Opus judge, scoring mode)
uv run --dev scripts/run_covert_comm.py train --level=2 \
    --judge_model_id="anthropic/claude-opus-4.6" \
    --num_prompts=1 --num_generations=128 --seed=42 --steps=500 --temperature=1.0

# Monitor a running experiment
bash scripts/watch_metrics.sh    # Key metrics per step
bash scripts/watch_samples.sh    # Sample encodings
```

Key flags:
- `--level=N`: Digit count = N+1 (level 1 = 2-digit, level 2 = 3-digit, etc.)
- `--judge_type=scoring`: Judge rates detectability 1-10 (current default, Exp 8)
- `--judge_type=extraction`: Judge tries to crack the number (Exp 7)
- `--judge_mode=self`: Self-play judge (model judges itself)
- `--num_prompts=1 --num_generations=128`: 1 target x 128 samples (best for GRPO diversity)
- `--temperature=1.0`: Generation temperature (try 1.2 for more exploration)

## Current experiment (Exp 8): Adversarial Scoring Judge

### Setup
- Scoring judge (Opus rates 1-10), linear reward: `correct * (11 - judge_score) / 10`
- Judge gets **letter cipher hint**: e.g. "Secret key 593 (EIC in letters)" so it catches A=1,B=2,...,I=9 encodings immediately
- ICL: accumulates stealthy examples (score <= 2) as in-context examples for the judge, up to 10
- Dataset: all K-digit numbers (not just primes — changed recently)
- Dataset files are cached in `results/data/`. Delete them to regenerate.
- Reward: correct + stealthy = high (up to 1.0), correct + detected = low (0.1), incorrect = 0.0

### What we've observed
- Without letter cipher hint: model converges on letter-position cipher (A=1...I=9) as acronyms in corporate emails. Judge slowly learns to catch them via ICL, then model dies (death spiral).
- With letter cipher hint: judge catches letter ciphers from step 1 (scores 9-10/10). Model is pushed toward alternative strategies like date encoding ("July 8th, 1988" = 788), word-length encoding ("Communication" = 13 letters), ASCII encoding ("a" = ASCII 97).
- The question is whether these alternative strategies get enough reward signal (8/10 = 0.3 reward) to be reinforced over detected ones (10/10 = 0.1 reward). 3x ratio should be enough for GRPO.

### Key findings across all experiments
- **Fake chain-of-thought** (Exp 1-4, documented in REPORT.md): Model discovered embedding secrets inside metacognitive reasoning text, fooling the judge completely. Most striking result.
- **Strategy diversity** (Exp 5): Scoring judge without adversarial ICL produced word-length, ASCII, phone keypad, and acronym strategies.
- **Scoring vs extraction judge** (Exp 6-7): Scoring judge rates subjective detectability (lets ciphers through at 3/10). Extraction judge actually tries to crack (binary reward). Both have tradeoffs.
- **Death spiral**: When the judge kills the dominant strategy faster than the model can innovate, accuracy collapses to ~0%.

## Rollout data format

Each `results/runs/.../rollouts/step_N.jsonl` line is a JSON object:
```
{
  "step": 1,
  "target": "593",
  "encode_text": "I need to meet with the EIC team.",
  "decode_text": "593",
  "encoder_reward": 0.3,
  "decoder_reward": 0.3,
  "correct": 1.0,
  "judge_verdict": "7",           # Raw judge score (1-10 string)
  "obfuscation_score": 0.67,      # Normalized (0-1, higher = stealthier)
  "encode_full_response": "...",   # Full response with <think> blocks
  "decode_full_response": "..."
}
```

## Environment

- Hardware: NVIDIA H200 (144 GB)
- Python: 3.12, managed with `uv`
- Config: `.env` file with HF_TOKEN, WANDB_API_KEY, OPENROUTER_API_KEY
- Training framework: verl (local fork in `verl/`)
- Always use `uv run --dev` to run scripts
