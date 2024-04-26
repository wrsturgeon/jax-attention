from beartype.typing import Tuple
from check_and_compile import check_and_compile
from jax import nn as jnn, numpy as jnp, random as jrnd
from jaxtyping import Array, Float32, Float64


def init(key: Array, embedding: int, d_model: int) -> Float64[Array, "3 embedding d_model"]:
    return jnn.initializers.he_normal()(key, [3, embedding, d_model], dtype=jnp.float64)


@check_and_compile()
def qkv(
    params: Float64[Array, "3 embedding d_model"],
    q_tokens: Float32[Array, "*batch seq_q embedding"],
    k_tokens: Float32[Array, "*batch seq_k embedding"],
    v_tokens: Float32[Array, "*batch seq_k embedding"],
) -> Tuple[
    Float32[Array, "*batch seq_q d_model"],
    Float32[Array, "*batch seq_k d_model"],
    Float32[Array, "*batch seq_k d_model"],
]:
    """
    Learn to project a series of token embeddings into three matrices:
    - Q, for "query"
    - K, for "key"
    - V, for "values"
    """

    w_q, w_k, w_v = params.astype(jnp.float32)

    q = q_tokens @ w_q
    k = k_tokens @ w_k
    v = v_tokens @ w_v

    return q, k, v
