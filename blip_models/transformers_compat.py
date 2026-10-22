"""Compatibility shims for transformers version drift.

The vendored BLIP code (med.py, nlvr_encoder.py) was written against
transformers 4.26. Newer releases (4.3x+, 5.x) moved or removed some
helper symbols out of ``transformers.modeling_utils``. Each symbol below
is imported from transformers when available (any version), and falls
back to a local copy of the upstream implementation otherwise, so the
code works on both old and new transformers without pinning.
"""

from typing import Callable, List, Tuple

import torch


def _find_pruneable_heads_and_indices(
    heads: List[int], n_heads: int, head_size: int
) -> Tuple[List[int], torch.LongTensor]:
    mask = [True] * n_heads
    for head in heads:
        assert head < n_heads, "head {}".format(head)
        mask[head] = False

    # Compute indexes to prune
    index = (
        torch.arange(start=0, end=n_heads * head_size)
        .view(n_heads, head_size)
        .masked_select(torch.tensor(mask).unsqueeze(-1))
    )

    return heads, index


def _prune_linear_layer(
    linear_layer: torch.nn.Linear, index: torch.LongTensor, dim: int = 1
) -> torch.nn.Linear:
    index = index.to(linear_layer.weight.device)
    weight = linear_layer.weight.index_select(dim, index).clone().detach()
    if linear_layer.bias is not None:
        bias = linear_layer.bias.index_select(0, index).clone().detach()
        new_linear = torch.nn.Linear(weight.size(1), weight.size(0), bias=True)
        new_linear.bias = torch.nn.Parameter(bias)
    else:
        new_linear = torch.nn.Linear(weight.size(1), weight.size(0), bias=False)

    new_linear.weight = torch.nn.Parameter(weight)
    if hasattr(linear_layer, "in_features"):
        new_linear.in_features = linear_layer.in_features
    if hasattr(linear_layer, "out_features"):
        new_linear.out_features = linear_layer.out_features
    return new_linear


def _apply_chunking_to_forward(
    forward_fn: Callable[..., torch.Tensor], chunk_size: int, dim: int, *input_tensors
) -> torch.Tensor:
    if chunk_size <= 0:
        return forward_fn(*input_tensors)

    # If the previous op has no output, create an empty tensor
    if len(input_tensors) == 1 and input_tensors[0].dim() == 0:
        tensor_output = forward_fn(*input_tensors)
        return tensor_output

    try:
        dim_size = input_tensors[0].size(dim)
    except Exception:
        raise ValueError("Unable to get module input dim")

    # chunking is only supported on the input dim
    if dim_size < chunk_size:
        tensor_output = forward_fn(*input_tensors)
        return tensor_output

    # Split input tensor into chunks
    input_tensors_strided_list = [
        input_tensor.view(-1, chunk_size, *input_tensors[0].size()[dim + 1 :])
        .unbind(1)
        if (isinstance(input_tensor, torch.Tensor) and input_tensor.size(dim) == dim_size)
        else input_tensor
        for input_tensor in input_tensors
    ]
    # Apply forward fn to each chunk
    outputs = [
        forward_fn(*input_tensors_chunk)
        for input_tensors_chunk in zip(*input_tensors_strided_list)
    ]
    # Change to correct output shape
    tensor_output = torch.cat(outputs, dim=dim)
    return tensor_output


def _local_get_head_mask(mask, num_hidden_layers):
    if mask is None:
        return [None] * num_hidden_layers
    if len(mask.shape) == 1:
        mask = mask[None, :].expand(num_hidden_layers, -1)
    elif len(mask.shape) != 2:
        raise ValueError(
            f"Head mask should have shape (num_heads,) or "
            f"(num_hidden_layers, num_heads), got {mask.shape}"
        )
    return [mask[i] for i in range(num_hidden_layers)]


def get_head_mask(model, mask, num_hidden_layers):
    """Per-layer head mask expansion.

    In transformers < 5.x this comes from ModelUtilsMixin (mixed into
    PreTrainedModel); on >= 5.x it was removed, so use a local equivalent.
    """
    try:
        from transformers.modeling_utils import ModelUtilsMixin
        return ModelUtilsMixin.get_head_mask(model, mask, num_hidden_layers)
    except (ImportError, AttributeError):
        return _local_get_head_mask(mask, num_hidden_layers)


def _local_invert_attention_mask(encoder_attention_mask):
    if len(encoder_attention_mask.shape) == 3:
        return (1.0 - encoder_attention_mask.to(dtype=torch.float32)) * torch.finfo(torch.float32).min
    return encoder_attention_mask


def invert_attention_mask(model, encoder_attention_mask):
    """Invert an attention mask (keep=1 -> 0, drop=0 -> float min).

    Provided by PreTrainedModel in transformers < 5.x; removed on >= 5.x.
    """
    try:
        from transformers.modeling_utils import PreTrainedModel
        return PreTrainedModel.invert_attention_mask(model, encoder_attention_mask)
    except AttributeError:
        return _local_invert_attention_mask(encoder_attention_mask)


def init_model_weights(model):
    """Version-robust weight initialization for the vendored PreTrainedModel subclasses.

    On transformers >= 5.x, ``PreTrainedModel.init_weights()`` goes through a new
    tied-weights code path that requires state (``all_tied_weights_keys``) these
    vendored models never set up (they have no tied weights). Fall back to the
    classic module-wise initialization, which is what older versions did.
    """
    try:
        model.init_weights()
    except AttributeError:
        model.apply(model._init_weights)


try:
    from transformers.modeling_utils import apply_chunking_to_forward
except ImportError:
    try:
        from transformers.pytorch_utils import apply_chunking_to_forward
    except ImportError:
        apply_chunking_to_forward = _apply_chunking_to_forward

try:
    from transformers.modeling_utils import find_pruneable_heads_and_indices
except ImportError:
    try:
        from transformers.pytorch_utils import find_pruneable_heads_and_indices
    except ImportError:
        find_pruneable_heads_and_indices = _find_pruneable_heads_and_indices

try:
    from transformers.modeling_utils import prune_linear_layer
except ImportError:
    try:
        from transformers.pytorch_utils import prune_linear_layer
    except ImportError:
        prune_linear_layer = _prune_linear_layer


# ---------------------------------------------------------------------------
# Local text generation.
#
# On transformers >= 5.x, ``generate()`` is no longer mixed into
# PreTrainedModel, so the vendored BertLMHeadModel lost it. Instead of
# depending on whichever generation API a given transformers version ships,
# this implements the two strategies BLIP uses (nucleus sampling and beam
# search) in pure PyTorch with HuggingFace-equivalent semantics. Used as a
# single code path on all versions for reproducibility.
# ---------------------------------------------------------------------------


def _apply_repetition_penalty(logits, input_ids, penalty):
    if penalty == 1.0:
        return logits
    score = torch.gather(logits, 1, input_ids)
    score = torch.where(score > 0, score / penalty, score * penalty)
    return logits.scatter_(1, input_ids, score)


def _top_p_filter(logits, top_p):
    if top_p >= 1.0:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    # Remove tokens until the cumulative probability exceeds top_p, keeping
    # at least one token (shift so the first exceeding token is removed too).
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False
    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
    return logits.masked_fill(indices_to_remove, float("-inf"))


def _nucleus_sample(model, input_ids, encoder_hidden_states, encoder_attention_mask,
                    max_length, min_length, top_p, repetition_penalty,
                    eos_token_id, pad_token_id):
    batch_size = input_ids.size(0)
    device = input_ids.device
    generated = input_ids
    attention_mask = torch.ones_like(generated)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    for _ in range(max_length - input_ids.size(1)):
        outputs = model(
            input_ids=generated,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )
        next_logits = outputs.logits[:, -1, :]
        next_logits = _apply_repetition_penalty(next_logits, generated, repetition_penalty)
        if generated.size(1) < min_length:
            next_logits[:, eos_token_id] = float("-inf")
        next_logits = _top_p_filter(next_logits, top_p)
        probs = torch.softmax(next_logits, dim=-1)
        next_tokens = torch.multinomial(probs, num_samples=1)
        next_tokens = torch.where(
            finished.unsqueeze(1), torch.full_like(next_tokens, pad_token_id), next_tokens
        )
        generated = torch.cat([generated, next_tokens], dim=1)
        attention_mask = torch.cat([attention_mask, torch.ones_like(next_tokens)], dim=1)
        finished |= next_tokens.squeeze(1).eq(eos_token_id)
        if finished.all():
            break
    return generated


def _beam_search(model, input_ids, encoder_hidden_states, encoder_attention_mask,
                 num_beams, max_length, min_length, repetition_penalty,
                 eos_token_id, pad_token_id):
    # Note: like HF's generate(), encoder_hidden_states / encoder_attention_mask
    # must already be expanded to batch_size * num_beams by the caller
    # (blip.py does this with repeat_interleave before calling local_generate).
    batch_size = input_ids.size(0)
    device = input_ids.device

    beams = input_ids.repeat_interleave(num_beams, dim=0)
    enc_hidden = encoder_hidden_states
    enc_mask = encoder_attention_mask
    attention_mask = torch.ones_like(beams)
    beam_scores = torch.zeros(batch_size * num_beams, dtype=torch.float32, device=device)
    done = torch.zeros(batch_size * num_beams, dtype=torch.bool, device=device)

    for _ in range(max_length - input_ids.size(1)):
        outputs = model(
            input_ids=beams,
            attention_mask=attention_mask,
            encoder_hidden_states=enc_hidden,
            encoder_attention_mask=enc_mask,
        )
        next_logits = outputs.logits[:, -1, :]
        next_logits = _apply_repetition_penalty(next_logits, beams, repetition_penalty)
        if beams.size(1) < min_length:
            next_logits[:, eos_token_id] = float("-inf")
        next_scores = torch.log_softmax(next_logits, dim=-1)
        next_scores = next_scores + beam_scores.unsqueeze(1)
        # Finished beams keep their score (no extension) and continue with pad tokens.
        if done.any():
            next_scores[done] = beam_scores[done].unsqueeze(1)

        vocab_size = next_scores.size(-1)
        next_scores = next_scores.view(batch_size, num_beams * vocab_size)
        topk_scores, topk_indices = torch.topk(next_scores, num_beams, dim=-1)
        parent_beams = (topk_indices // vocab_size)
        next_tokens = topk_indices % vocab_size

        batch_offsets = torch.arange(batch_size, device=device).unsqueeze(1) * num_beams
        parent_rows = (batch_offsets + parent_beams).reshape(-1)

        new_done = done[parent_rows] | next_tokens.reshape(-1).eq(eos_token_id)
        next_tokens = torch.where(
            new_done.unsqueeze(1),
            torch.full_like(next_tokens, pad_token_id).reshape(-1, 1),
            next_tokens.reshape(-1, 1),
        )
        beams = torch.cat([beams[parent_rows], next_tokens], dim=1)
        attention_mask = torch.cat(
            [attention_mask[parent_rows], torch.ones(batch_size * num_beams, 1, dtype=torch.long, device=device)],
            dim=1,
        )
        beam_scores = topk_scores.reshape(-1)
        done = new_done

        if done.view(batch_size, num_beams).all(dim=1).all():
            break

    # Return the highest-scoring beam per batch element.
    final_scores = beam_scores.view(batch_size, num_beams)
    best = final_scores.argmax(dim=1)
    return beams[best * num_beams + best]


def local_generate(model, input_ids, encoder_hidden_states, encoder_attention_mask,
                   sample=False, num_beams=3, max_length=30, min_length=10,
                   top_p=0.9, repetition_penalty=1.0, eos_token_id=None, pad_token_id=None):
    """HuggingFace-style generation for the vendored BLIP text decoder.

    Returns a LongTensor of shape (batch_size, generated_length) containing
    the prompt plus the generated tokens, like ``PreTrainedModel.generate``.
    """
    if sample:
        return _nucleus_sample(
            model, input_ids, encoder_hidden_states, encoder_attention_mask,
            max_length=max_length, min_length=min_length, top_p=top_p,
            repetition_penalty=repetition_penalty,
            eos_token_id=eos_token_id, pad_token_id=pad_token_id,
        )
    return _beam_search(
        model, input_ids, encoder_hidden_states, encoder_attention_mask,
        num_beams=num_beams, max_length=max_length, min_length=min_length,
        repetition_penalty=repetition_penalty,
        eos_token_id=eos_token_id, pad_token_id=pad_token_id,
    )
