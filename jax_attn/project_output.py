from jax_attn.jit import jit

from jax import nn as jnn, numpy as jnp, random as jrnd
from jaxtyping import Array, Float32, Float64


def init(key: Array, heads: int, d_v: int, embedding: int) -> Float64[Array, "heads d_v embedding"]:
    return jnn.initializers.he_normal()(key, [heads, d_v, embedding], dtype=jnp.float64)


@jit()
def project_output(
    params: Float64[Array, "head d_v embedding"],
    tokens: Float32[Array, "*batch head seq d_v"],
) -> Float32[Array, "*batch seq embedding"]:
    """
    Project all heads' output into one unified representation.
    """

    # Extract shape information:
    head, d_v, embedding = params.shape
    *batch, head, seq, _ = tokens.shape

    # Concatenate heads:
    p32: Float32[Array, "head d_v embedding"] = params.astype(jnp.float32)
    p: Float32[Array, "head*d_v embedding"] = p32.reshape(head * d_v, embedding)
    tt: Float32[Array, "*batch seq head d_v"] = tokens.transpose(
        *range(tokens.ndim - 3), tokens.ndim - 2, tokens.ndim - 3, tokens.ndim - 1
    )
    assert tt.shape == (*batch, seq, head, d_v), f"{tt.shape} =/= {(*batch, seq, head, d_v)}"
    t: Float32[Array, "*batch seq head*d_v"] = tt.reshape(*tt.shape[:-2], head * d_v)

    t2 = jnp.einsum("... h s d -> ... s h d", tokens)
    assert tt.shape == t2.shape, f"{tt.shape} =/= {t2.shape}"
    from jax.experimental.checkify import check

    check(jnp.allclose(tt, t2), "{tt} =/= {t2}")

    return t @ p
