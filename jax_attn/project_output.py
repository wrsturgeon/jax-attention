from check_and_compile import check_and_compile
from jax import nn as jnn, numpy as jnp, random as jrnd
from jaxtyping import Array, Float32, Float64


def init(key: Array, heads: int, d_v: int, embedding: int) -> Float64[Array, "heads d_v embedding"]:
    return jnn.initializers.he_normal()(key, [heads, d_v, embedding], dtype=jnp.float64)


@check_and_compile()
def project_output(
    params: Float64[Array, "head d_v embedding"],
    tokens: Float32[Array, "*batch head seq_q d_v"],
) -> Float32[Array, "*batch seq_q embedding"]:
    """
    Project all heads' output into one unified representation.
    """

    # Extract shape information:
    head, d_v, embedding = params.shape
    *batch, head, seq, _ = tokens.shape

    # Transpose tokens s.t. heads exist per-token instead of per-sequence:
    tt = jnp.einsum("... h s d -> ... s h d", tokens)

    # Concatenate heads:
    t: Float32[Array, "*batch seq head*d_v"] = tt.reshape(*tt.shape[:-2], head * d_v)
    p32: Float32[Array, "head d_v embedding"] = params.astype(jnp.float32)
    p: Float32[Array, "head*d_v embedding"] = p32.reshape(head * d_v, embedding)

    return t @ p
