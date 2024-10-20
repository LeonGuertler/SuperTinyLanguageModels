from models.components.attention.attention import Attention 
from models.components.attention.rope_attention import RoPEAttention 
from models.components.attention.alibi_attention import ALiBiAttention
from models.components.attention.linformer_attention import LinformerAttention
from models.components.attention.performer_attention import PerformerAttention
from models.components.attention.sparse_attention import SparseAttention 
from models.components.attention.differentiable_attention import DifferentiableAttention 

from typing import Callable, Dict, Any, Optional

def default_attention_params(hidden_dim, context_window, attn_params, **kwargs):
    return {
        "hidden_dim": hidden_dim,
        "num_kv_heads": attn_params["num_kv_heads"],
        "num_q_heads": attn_params.get("num_q_heads", attn_params["num_kv_heads"]),
        "bias": attn_params["bias"],
        "context_window": context_window,
        "is_causal": attn_params.get("is_causal", True),  # default is causal
        **kwargs
    }

# Create helper functions for handling extra params for specific attention types
def build_linformer_params(hidden_dim, context_window, attn_params):
    return default_attention_params(
        hidden_dim, context_window, attn_params, 
        k_reduction_factor=attn_params["k_reduction_factor"],
        share_kv_proj=attn_params["share_kv_proj"]
    )

def build_performer_params(hidden_dim, context_window, attn_params):
    return default_attention_params(
        hidden_dim, context_window, attn_params, 
        nb_features=attn_params["nb_features"]
    )

def build_differentiable_params(hidden_dim, context_window, attn_params, depth):
    return default_attention_params(
        hidden_dim, context_window, attn_params, 
        depth=depth  # depth is now passed into this function directly
    )

# Define the registry with custom handlers for specific attention types
ATTENTION_REGISTRY: Dict[str, Callable[[int, int, Dict[str, Any], Optional[int]], Attention]] = {
    "generic_attention": lambda hidden_dim, context_window, attn_params, depth: Attention(
        **default_attention_params(hidden_dim, context_window, attn_params)
    ),
    "rope_attention": lambda hidden_dim, context_window, attn_params, depth: RoPEAttention(
        **default_attention_params(hidden_dim, context_window, attn_params)
    ),
    "alibi_attention": lambda hidden_dim, context_window, attn_params, depth: ALiBiAttention(
        **default_attention_params(hidden_dim, context_window, attn_params)
    ),
    "linformer_attention": lambda hidden_dim, context_window, attn_params, depth: LinformerAttention(
        **build_linformer_params(hidden_dim, context_window, attn_params)
    ),
    "performer_attention": lambda hidden_dim, context_window, attn_params, depth: PerformerAttention(
        **build_performer_params(hidden_dim, context_window, attn_params)
    ),
    "sparse_attention": lambda hidden_dim, context_window, attn_params, depth: SparseAttention(
        **default_attention_params(hidden_dim, context_window, attn_params)
    ),
    "differentiable_attention": lambda hidden_dim, context_window, attn_params, depth: DifferentiableAttention(
        **build_differentiable_params(hidden_dim, context_window, attn_params, depth)
    ),
}

def build_attention(
    attn_name: str,
    attn_params: Dict[str, Any],
    hidden_dim: int,
    context_window: int,
    depth: Optional[int] = None
):
    assert attn_name in ATTENTION_REGISTRY, \
        f"Attention name ({attn_name}) not found. Available options: {list(ATTENTION_REGISTRY.keys())}"
    return ATTENTION_REGISTRY[attn_name](
        hidden_dim=hidden_dim,
        context_window=context_window,
        attn_params=attn_params,
        depth=depth  # pass depth only when needed (i.e., for differentiable_attention)
    )
