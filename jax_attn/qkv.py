from jax_attn.jit import jit

from jax import numpy as jnp
from jaxtyping import Array, Float32, Float64


@jit()
def qkv(
    params: Float64[Array, "3 embedding d_model"],
    tokens: Float32[Array, "*batch seq embedding"],
) -> Float32[Array, "3 *batch seq d_model"]:
    """
    Learn to project a series of token embeddings into three matrices:
    - Q, for "query"
    - K, for "key"
    - V, for "values"
    """

    # Change `dtype` of `params` when we don't need precision updates:
    params: Float64[Array, "3 embedding d_model"] = params.astype(jnp.float32)

    # Matrix-multiply *only* the last two axes:
    # - `s` is `seq`
    # - `e` is `embedding`
    # - `m` is `d_model`
    out = jnp.einsum("... s e, 3 e m -> 3 ... s m", tokens, params)

    # Equivalently, we could have done this:
    #
    # WQ, WK, WV = params
    #
    # Q = tokens @ WQ
    # K = tokens @ WK
    # V = tokens @ WV
    #
    # return jnp.stack(Q, K, V)

    return out
