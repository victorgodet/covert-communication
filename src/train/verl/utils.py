from __future__ import annotations

import re
import torch
from verl import DataProto
from transformers import AutoTokenizer


def _lookup_token_id(tokenizer: AutoTokenizer, token: str) -> int | None:
    """Return the token ID for *token* or *None* if it doesn't exist."""
    try:
        tid = tokenizer.convert_tokens_to_ids(token)
        if tid == tokenizer.unk_token_id:
            return None
        return tid
    except Exception:
        return None


def _strip_think_tokens(
    response_ids: torch.Tensor,
    start_think_id: int | None,
    end_think_id: int | None,
) -> torch.Tensor:
    """Strip ``<think>…</think>`` from a 1-D token tensor.

    Three cases:
    1. ``</think>`` present → keep only tokens after the last ``</think>``.
    2. ``<think>`` present but ``</think>`` absent → the model used all its
       tokens on reasoning and never produced an answer.  Return empty tensor.
    3. Neither tag present → return the tensor unchanged.
    """
    if end_think_id is not None:
        positions = (response_ids == end_think_id).nonzero(as_tuple=True)[0]
        if len(positions) > 0:
            return response_ids[positions[-1].item() + 1:]

    # No </think> found — check if <think> is present (truncated thinking).
    if start_think_id is not None:
        if (response_ids == start_think_id).any():
            return response_ids[:0]  # empty tensor, same device/dtype

    return response_ids


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_OPEN_THINK_RE = re.compile(r"<think>.*", re.DOTALL)


def _strip_think_text(text: str) -> str:
    """Regex fallback: strip ``<think>`` blocks from decoded text."""
    text = _THINK_RE.sub("", text)
    # Handle unclosed <think> (truncated thinking)
    text = _OPEN_THINK_RE.sub("", text)
    return text.strip()


def convert_responses_to_str(data: DataProto, tokenizer: AutoTokenizer):
    """Decode response token-IDs to strings, stripping ``<think>`` blocks.

    When the model uses thinking mode, its response looks like:
        <think>...reasoning...</think>actual answer
    We only want the text *after* ``</think>``.  Stripping is done at two
    levels for robustness:

    1. **Token-ID level**: find ``</think>`` in the tensor and decode only
       tokens after it.  If ``<think>`` is present but ``</think>`` is not
       (model ran out of tokens while thinking), return an empty string.
    2. **Regex fallback**: after decoding, strip any remaining literal
       ``<think>…</think>`` text in case the tokenizer doesn't use dedicated
       token IDs for these tags.
    """
    prompt_ids = data.batch["prompts"]
    response_ids = data.batch["responses"]
    attention_mask = data.batch["attention_mask"]

    prompt_len = prompt_ids.shape[-1]
    valid_response_lengths = attention_mask[:, prompt_len:].sum(dim=-1)

    start_think_id = _lookup_token_id(tokenizer, "<think>")
    end_think_id = _lookup_token_id(tokenizer, "</think>")

    responses_str = []
    for i in range(len(data)):
        valid_len = valid_response_lengths[i]
        valid_response_ids = response_ids[i][:valid_len]

        # Token-level stripping
        valid_response_ids = _strip_think_tokens(
            valid_response_ids, start_think_id, end_think_id,
        )

        response_str = tokenizer.decode(valid_response_ids, skip_special_tokens=True)

        # Regex fallback for any residual <think> text
        response_str = _strip_think_text(response_str)

        responses_str.append(response_str)
    return responses_str