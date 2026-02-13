import math
import torch
import torch.nn.functional as F
from typing import Optional, Tuple


def softmax_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Standard scaled dot-product attention with softmax."""
    scale = 1.0 / math.sqrt(query.size(-1))
    attn_weights = torch.matmul(query, key.transpose(-2, -1)) * scale

    if mask is not None:
        if mask.dtype == torch.bool:
            attn_weights = attn_weights.masked_fill(~mask, float('-inf'))
        else:
            attn_weights = attn_weights + mask

    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    output = torch.matmul(attn_weights, value)
    return output, attn_weights


def sigmoid_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    normalize: bool = True,
    mask: Optional[torch.Tensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sigmoid attention: uses sigmoid instead of softmax."""
    scale = 1.0 / math.sqrt(query.size(-1))
    attn_weights = torch.matmul(query, key.transpose(-2, -1)) * scale
    attn_weights = torch.sigmoid(attn_weights)

    if mask is not None:
        if mask.dtype == torch.bool:
            attn_weights = attn_weights.masked_fill(~mask, 0.0)
        else:
            attn_weights = attn_weights * mask

    if normalize:
        attn_weights = attn_weights / (attn_weights.sum(dim=-1, keepdim=True) + 1e-9)
    output = torch.matmul(attn_weights, value)
    return output, attn_weights


def linear_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    normalize: bool = True,
    mask: Optional[torch.Tensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Linear attention with ELU+1 kernel for positivity."""
    query = F.elu(query) + 1
    key = F.elu(key) + 1
    attn_weights = torch.matmul(query, key.transpose(-2, -1))

    if mask is not None:
        if mask.dtype == torch.bool:
            attn_weights = attn_weights.masked_fill(~mask, 0.0)
        else:
            attn_weights = attn_weights * mask

    if normalize:
        attn_weights = attn_weights / (attn_weights.sum(dim=-1, keepdim=True) + 1e-9)
    output = torch.matmul(attn_weights, value)
    return output, attn_weights


def coupled_linear_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    normalize: bool = True,
    mask: Optional[torch.Tensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Linear attention with ELU+1 kernel for positivity."""
    scale = 1.0 / math.sqrt(query.size(-1))
    attn_weights = torch.matmul(query, key.transpose(-2, -1)) * scale
    attn_weights = F.elu(attn_weights) + 1

    if mask is not None:
        if mask.dtype == torch.bool:
            attn_weights = attn_weights.masked_fill(~mask, 0.0)
        else:
            attn_weights = attn_weights * mask

    if normalize:
        attn_weights = attn_weights / (attn_weights.sum(dim=-1, keepdim=True) + 1e-9)
    output = torch.matmul(attn_weights, value)
    return output, attn_weights



def uniform_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Uniform attention: always attends equally to all positions."""
    batch_size, num_heads, seq_len, _ = query.shape
    attn_weights = torch.ones(batch_size, num_heads, seq_len, seq_len,
                              device=query.device, dtype=query.dtype)

    if mask is not None:
        if mask.dtype == torch.bool:
            attn_weights = attn_weights.masked_fill(~mask, 0.0)
        else:
            attn_weights = attn_weights * mask

    attn_weights = attn_weights / (attn_weights.sum(dim=-1, keepdim=True) + 1e-9)
    output = torch.matmul(attn_weights, value)
    return output, attn_weights


def raw_dotproduct_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    normalize: bool = False,
    mask: Optional[torch.Tensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Raw scaled dot-product (Row 6 from table)."""
    scale = 1.0 / math.sqrt(query.size(-1))
    attn_weights = torch.matmul(query, key.transpose(-2, -1)) * scale

    if mask is not None:
        if mask.dtype == torch.bool:
            attn_weights = attn_weights.masked_fill(~mask, 0.0)
        else:
            attn_weights = attn_weights * mask

    if normalize:
        normalization = torch.maximum(
            attn_weights.sum(dim=-1, keepdim=True).abs(),
            torch.ones_like(attn_weights.sum(dim=-1, keepdim=True))
        )
        attn_weights = attn_weights / normalization

    output = torch.matmul(attn_weights, value)
    return output, attn_weights


def mlp_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mlp: torch.nn.Module,
    normalize: bool = True,
    mask: Optional[torch.Tensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """MLP-based attention kernel (Rows 7-8 from table).

    Applies an MLP module to queries and keys before computing attention.

    Args:
        query: Query tensor
        key: Key tensor
        value: Value tensor
        mlp: PyTorch module to apply to queries and keys
        normalize: Whether to normalize by max(sum, 1)
        mask: Optional attention mask
    """
    # Apply MLP to queries and keys
    query_mlp = mlp(query)
    key_mlp = mlp(key)

    scale = 1.0 / math.sqrt(query.size(-1))
    attn_weights = torch.matmul(query_mlp, key_mlp.transpose(-2, -1)) * scale

    if mask is not None:
        if mask.dtype == torch.bool:
            attn_weights = attn_weights.masked_fill(~mask, 0.0)
        else:
            attn_weights = attn_weights * mask

    if normalize:
        # Row 7: normalize by max(sum, 1) to prevent division issues
        normalization = torch.maximum(
            attn_weights.sum(dim=-1, keepdim=True).abs(),
            torch.ones_like(attn_weights.sum(dim=-1, keepdim=True))
        )
        attn_weights = attn_weights / normalization

    output = torch.matmul(attn_weights, value)
    return output, attn_weights


ATTENTION_FUNCTIONS = {
    "softmax": softmax_attention,
    "uniform": uniform_attention,
    "sigmoid_norm": lambda q, k, v, **kw: sigmoid_attention(q, k, v, normalize=True, **kw),
    "sigmoid_unnorm": lambda q, k, v, **kw: sigmoid_attention(q, k, v, normalize=False, **kw),
    "linear": lambda q, k, v, **kw: linear_attention(q, k, v, normalize=True, **kw),
    "linear_unnorm": lambda q, k, v, **kw: linear_attention(q, k, v, normalize=False, **kw),
    "coupled_linear_norm": lambda q, k, v, **kw: coupled_linear_attention(q, k, v, normalize=True, **kw),
    "coupled_linear_unnorm": lambda q, k, v, **kw: coupled_linear_attention(q, k, v, normalize=False, **kw),
    "raw_dotproduct_norm": lambda q, k, v, **kw: raw_dotproduct_attention(q, k, v, normalize=True, **kw),
    "raw_dotproduct_unnorm": lambda q, k, v, **kw: raw_dotproduct_attention(q, k, v, normalize=False, **kw),
    "mlp_norm": lambda q, k, v, mlp, **kw: mlp_attention(q, k, v, mlp=mlp, normalize=True, **kw),
    "mlp_unnorm": lambda q, k, v, mlp, **kw: mlp_attention(q, k, v, mlp=mlp, normalize=False, **kw),
    # Note: mlp_attention requires an 'mlp' parameter
}
# available attention types: ['softmax', 'uniform', 'sigmoid_norm', 'sigmoid_unnorm', 'linear', 'linear_unnorm', 'coupled_linear_norm', 'coupled_linear_unnorm', 'raw_dotproduct_norm', 'raw_dotproduct_unnorm', 'mlp_norm', 'mlp_unnorm']

# print(f"Available attention types: {list(ATTENTION_FUNCTIONS.keys())}")
