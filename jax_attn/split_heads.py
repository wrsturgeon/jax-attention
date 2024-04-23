from jax_attn.jit import jit

from beartype.typing import NamedTuple, Tuple
from jax import numpy as jnp
from jaxtyping import Array, Float32, Float64


class Parameters(NamedTuple):
    w_q: Float64[Array, "d_model d_k"]
    w_k: Float64[Array, "d_model d_k"]
    w_v: Float64[Array, "d_model d_v"]


@jit()
def split_heads(
    params: Parameters,
    qkv: Float32[Array, "3 batch seq d_model"],
) -> Tuple[
    Float32[Array, "batch seq d_k"],
    Float32[Array, "batch seq d_k"],
    Float32[Array, "batch seq d_v"],
]:
    """
    Project Q/K/V matrices into inputs for separate "heads" paying attention to different things.
    """

    # Split `qkv` into Q, K, and V matrices:
    q, k, v = qkv

    # Matrix-multiply each separately, since they have different shapes:
    q: Float32[Array, "batch seq d_k"] = q @ params.w_q.astype(jnp.float32)
    k: Float32[Array, "batch seq d_k"] = k @ params.w_k.astype(jnp.float32)
    v: Float32[Array, "batch seq d_v"] = v @ params.w_v.astype(jnp.float32)

    return q, k, v
