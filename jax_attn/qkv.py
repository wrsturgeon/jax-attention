from check_and_compile import check_and_compile
from jax import nn as jnn, numpy as jnp, random as jrnd
from jaxtyping import Array, Float32, Float64


def init(key: Array, embedding: int, d_model: int) -> Float64[Array, "3 embedding d_model"]:
    return jnn.initializers.he_normal()(key, [3, embedding, d_model], dtype=jnp.float64)


@check_and_compile()
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

    # Effectively, we're doing a faster version of this:
    #
    # WQ, WK, WV = params
    #
    # Q = tokens @ WQ
    # K = tokens @ WK
    # V = tokens @ WV
    #
    # return jnp.stack(Q, K, V)

    # Change `dtype` of `params` when we don't need precision updates:
    p: Float64[Array, "3 embedding d_model"] = params.astype(jnp.float32)

    # Matrix-multiply *only* the last two axes:
    # - `s` is `seq`
    # - `e` is `embedding`
    # - `m` is `d_model`
    return jnp.einsum("... s e, 3 e m -> 3 ... s m", tokens, p)
