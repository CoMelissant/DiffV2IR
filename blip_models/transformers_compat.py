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
