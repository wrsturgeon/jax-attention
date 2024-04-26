from check_and_compile import check_and_compile
from beartype.typing import NamedTuple, Optional, Tuple
from jax import nn as jnn, numpy as jnp, random as jrnd
from jaxtyping import Array, Float32, Float64


class Parameters(NamedTuple):
    w_q: Float64[Array, "head d_model d_k"]
    w_k: Float64[Array, "head d_model d_k"]
    w_v: Float64[Array, "head d_model d_v"]


def init(key: Array, heads: int, d_model: int, d_k: int, d_v: int) -> Parameters:
    k_q, k_k, k_v = jrnd.split(key, num=3)
    return Parameters(
        w_q=jnn.initializers.he_normal()(k_q, [heads, d_model, d_k], dtype=jnp.float64),
        w_k=jnn.initializers.he_normal()(k_k, [heads, d_model, d_k], dtype=jnp.float64),
        w_v=jnn.initializers.he_normal()(k_v, [heads, d_model, d_v], dtype=jnp.float64),
    )


def check_shapes(p: Parameters, d_model: int) -> Tuple[int, int, int]:
    assert p.w_q.ndim == 3, f"{p.w_q.ndim} =/= {3}"
    assert p.w_k.ndim == 3, f"{p.w_k.ndim} =/= {3}"
    assert p.w_v.ndim == 3, f"{p.w_v.ndim} =/= {3}"

    heads, _, d_k = p.w_k.shape
    _, _, d_v = p.w_v.shape
    assert p.w_q.shape == (heads, d_model, d_k), f"{p.w_q.shape} =/= {(heads, d_model, d_k)}"
    assert p.w_k.shape == (heads, d_model, d_k), f"{p.w_k.shape} =/= {(heads, d_model, d_k)}"
    assert p.w_v.shape == (heads, d_model, d_v), f"{p.w_v.shape} =/= {(heads, d_model, d_v)}"

    return heads, d_k, d_v


@check_and_compile()
def split_heads(
    params: Parameters,
    q: Float32[Array, "*batch seq d_model"],
    k: Float32[Array, "*batch seq d_model"],
    v: Float32[Array, "*batch seq d_model"],
) -> Tuple[
    Float32[Array, "*batch head seq d_k"],
    Float32[Array, "*batch head seq d_k"],
    Float32[Array, "*batch head seq d_v"],
]:
    """
    Project Q/K/V matrices into inputs for separate "heads" paying attention to different things.
    """

    # Check parameter shapes:
    check_shapes(params, q.shape[-1])

    w_q: Float32[Array, "head d_model d_k"] = params.w_q.astype(jnp.float32)
    w_k: Float32[Array, "head d_model d_k"] = params.w_k.astype(jnp.float32)
    w_v: Float32[Array, "head d_model d_v"] = params.w_v.astype(jnp.float32)

    # Matrix-multiply each separately, since they have different shapes:
    q_h: Float32[Array, "*batch head seq d_k"] = jnp.einsum("... s m, h m d -> ... h s d", q, w_q)
    k_h: Float32[Array, "*batch head seq d_k"] = jnp.einsum("... s m, h m d -> ... h s d", k, w_k)
    v_h: Float32[Array, "*batch head seq d_v"] = jnp.einsum("... s m, h m d -> ... h s d", v, w_v)

    return q_h, k_h, v_h
