from jax_attn.jit import jit

from beartype.typing import NamedTuple, Tuple
from jax import numpy as jnp
from jaxtyping import Array, Float32, Float64


class Parameters(NamedTuple):
    w_q: Float64[Array, "head d_model d_k"]
    w_k: Float64[Array, "head d_model d_k"]
    w_v: Float64[Array, "head d_model d_v"]


def check_shape(p: Parameters, d_model: int):
    assert p.w_q.ndim == 3, f"{p.w_q.ndim} =/= {3}"
    assert p.w_k.ndim == 3, f"{p.w_k.ndim} =/= {3}"
    assert p.w_v.ndim == 3, f"{p.w_v.ndim} =/= {3}"
    heads, _, d_k = p.w_k.shape
    d_v = p.w_v.shape[-1]
    assert p.w_q.shape == (
        heads,
        d_model,
        d_k,
    ), f"{p.w_q.shape} =/= {(heads, d_model, d_k)}"
    assert p.w_k.shape == (
        heads,
        d_model,
        d_k,
    ), f"{p.w_k.shape} =/= {(heads, d_model, d_k)}"
    assert p.w_v.shape == (
        heads,
        d_model,
        d_v,
    ), f"{p.w_v.shape} =/= {(heads, d_model, d_v)}"


@jit()
def split_heads(
    params: Parameters,
    qkv: Float32[Array, "3 *batch seq d_model"],
) -> Tuple[
    Float32[Array, "*batch head seq d_k"],
    Float32[Array, "*batch head seq d_k"],
    Float32[Array, "*batch head seq d_v"],
]:
    """
    Project Q/K/V matrices into inputs for separate "heads" paying attention to different things.
    """

    # Check parameter shapes:
    check_shape(params, qkv.shape[-1])

    # Split `qkv` into Q, K, and V matrices:
    q, k, v = qkv  # : Float32[Array, "*batch seq d_model"]

    w_q: Float32[Array, "head d_model d_k"] = params.w_q.astype(jnp.float32)
    w_k: Float32[Array, "head d_model d_k"] = params.w_k.astype(jnp.float32)
    w_v: Float32[Array, "head d_model d_v"] = params.w_v.astype(jnp.float32)

    # Matrix-multiply each separately, since they have different shapes:
    q: Float32[Array, "*batch head seq d_k"] = jnp.einsum("... s m, h m d -> ... h s d", q, w_q)
    k: Float32[Array, "*batch head seq d_k"] = jnp.einsum("... s m, h m d -> ... h s d", k, w_k)
    v: Float32[Array, "*batch head seq d_v"] = jnp.einsum("... s m, h m d -> ... h s d", v, w_v)

    return q, k, v
