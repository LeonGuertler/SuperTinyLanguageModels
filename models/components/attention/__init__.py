from models.components.attention.attention import Attention 
from models.components.attention.rope_attention import RoPEAttention 
from models.components.attention.alibi_attention import ALiBiAttention
from models.components.attention.linformer_attention import LinformerAttention
from models.components.attention.performer_attention import PerformerAttention
from models.components.attention.sparse_attention import SparseAttention 
from models.components.attention.differentiable_attention import DifferentiableAttention 

from typing import Callable, Dict, Any, Optional

def default_attention_params(hidden_dim, context_window, attn_cfg, **kwargs):
    return {
        "hidden_dim": hidden_dim,
        "num_kv_heads": attn_cfg["num_kv_heads"],
        "num_q_heads": attn_cfg.get("num_q_heads", attn_cfg["num_kv_heads"]),
        "bias": attn_cfg["bias"],
        "context_window": context_window,
        "is_causal": attn_cfg.get("is_causal", True),  # default is causal
        **kwargs
    }

# Create helper functions for handling extra params for specific attention types
def build_linformer_params(hidden_dim, context_window, attn_cfg):
    return default_attention_params(
        hidden_dim, context_window, attn_cfg, 
        k_reduction_factor=attn_cfg["k_reduction_factor"],
        share_kv_proj=attn_cfg["share_kv_proj"]
    )

def build_performer_params(hidden_dim, context_window, attn_cfg):
    return default_attention_params(
        hidden_dim, context_window, attn_cfg, 
        nb_features=attn_cfg["nb_features"]
    )

def build_differentiable_params(hidden_dim, context_window, attn_cfg, depth):
    return default_attention_params(
        hidden_dim, context_window, attn_cfg, 
        depth=depth  # depth is now passed into this function directly
    )

# Define the registry with custom handlers for specific attention types
ATTENTION_REGISTRY: Dict[str, Callable[[int, int, Dict[str, Any], Optional[int]], Attention]] = {
    "generic_attention": lambda hidden_dim, context_window, attn_cfg, depth: Attention(
        **default_attention_params(hidden_dim, context_window, attn_cfg)
    ),
    "rope_attention": lambda hidden_dim, context_window, attn_cfg, depth: RoPEAttention(
        **default_attention_params(hidden_dim, context_window, attn_cfg)
    ),
    "alibi_attention": lambda hidden_dim, context_window, attn_cfg, depth: ALiBiAttention(
        **default_attention_params(hidden_dim, context_window, attn_cfg)
    ),
    "linformer_attention": lambda hidden_dim, context_window, attn_cfg, depth: LinformerAttention(
        **build_linformer_params(hidden_dim, context_window, attn_cfg)
    ),
    "performer_attention": lambda hidden_dim, context_window, attn_cfg, depth: PerformerAttention(
        **build_performer_params(hidden_dim, context_window, attn_cfg)
    ),
    "sparse_attention": lambda hidden_dim, context_window, attn_cfg, depth: SparseAttention(
        **default_attention_params(hidden_dim, context_window, attn_cfg)
    ),
    "differentiable_attention": lambda hidden_dim, context_window, attn_cfg, depth: DifferentiableAttention(
        **build_differentiable_params(hidden_dim, context_window, attn_cfg, depth)
    ),
}

def build_attention(
    hidden_dim: int,
    context_window: int,
    attn_cfg: Dict[str, Any],
    depth: Optional[int] = None
):
    return ATTENTION_REGISTRY[attn_cfg["attn_type"]](
        hidden_dim=hidden_dim,
        context_window=context_window,
        attn_cfg=attn_cfg,
        depth=depth  # pass depth only when needed (i.e., for differentiable_attention)
    )
