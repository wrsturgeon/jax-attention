from jax_attn.jit import jit

from jax import numpy as jnp
from jaxtyping import Array, Float32, Float64


@jit()
def qkv(
    params: Float64[Array, "3 embedding d_model"],
    tokens: Float32[Array, "batch seq embedding"],
) -> Float32[Array, "3 batch seq d_model"]:
    """
    Learn to project a series of token embeddings into three matrices:
    - Q, for "query"
    - K, for "key"
    - V, for "values"
    """

    # Change `dtype` of `params` when we don't need precision updates:
    params: Float64[Array, "3 embedding d_model"] = params.astype(jnp.float32)

    # Update shapes so we can matrix-multiply *only* the last two axes:
    tokens: Float32[Array, "1 batch seq embedding"] = tokens[jnp.newaxis]
    params: Float32[Array, "3 1 embedding d_model"] = params[:, jnp.newaxis]

    # Matrix-multiply the last two axes, leaving the others (3 & batch) intact:
    all_at_once: Float32[Array, "3 batch seq d_model"] = tokens @ params

    # Equivalently, we could have done this:
    #
    # WQ, WK, WV = params
    #
    # Q = tokens @ WQ
    # K = tokens @ WK
    # V = tokens @ WV
    #
    # return jnp.stack(Q, K, V)

    return all_at_once
