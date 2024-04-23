from jax_attn.jit import jit

from beartype.typing import NamedTuple, Tuple
from jax import numpy as jnp
from jaxtyping import Array, Float32, Float64


class Parameters(NamedTuple):
    w_q: Float64[Array, "d_model d_k"]
    w_k: Float64[Array, "d_model d_k"]
    w_v: Float64[Array, "d_model d_v"]


def check_shape(p: Parameters, d_model: int):
    assert p.w_q.ndim == 2, f"{p.w_q.ndim} =/= {2}"
    assert p.w_k.ndim == 2, f"{p.w_k.ndim} =/= {2}"
    assert p.w_v.ndim == 2, f"{p.w_v.ndim} =/= {2}"
    d_k = p.w_k.shape[1]
    d_v = p.w_v.shape[1]
    assert p.w_q.shape == (
        d_model,
        d_k,
    ), f"{p.w_q.shape} =/= {(d_model, d_k)}"
    assert p.w_k.shape == (
        d_model,
        d_k,
    ), f"{p.w_k.shape} =/= {(d_model, d_k)}"
    assert p.w_v.shape == (
        d_model,
        d_v,
    ), f"{p.w_v.shape} =/= {(d_model, d_v)}"


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

    # Check parameter shapes:
    check_shape(params, qkv.shape[-1])

    # Split `qkv` into Q, K, and V matrices:
    q, k, v = qkv

    # Matrix-multiply each separately, since they have different shapes:
    q: Float32[Array, "batch seq d_k"] = q @ params.w_q.astype(jnp.float32)
    k: Float32[Array, "batch seq d_k"] = k @ params.w_k.astype(jnp.float32)
    v: Float32[Array, "batch seq d_v"] = v @ params.w_v.astype(jnp.float32)

    return q, k, v
