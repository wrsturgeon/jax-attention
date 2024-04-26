from jax_attn import attention, project_output, qkv, salience_map, split_heads

from check_and_compile import check_and_compile
from beartype.typing import Callable, NamedTuple
from jax import nn as jnn, numpy as jnp, random as jrnd
from jaxtyping import Array, Float32, Float64


class Parameters(NamedTuple):
    qkv: Float64[Array, "3 embedding d_model"]
    heads: split_heads.Parameters
    output: Float64[Array, "heads d_v embedding"]


def init(key: Array, embedding: int, d_model: int, heads: int, d_k: int, d_v: int) -> Parameters:
    return Parameters(
        qkv=qkv.init(key, embedding=embedding, d_model=d_model),
        heads=split_heads.init(key, heads=heads, d_model=d_model, d_k=d_k, d_v=d_v),
        output=project_output.init(key, heads=heads, d_v=d_v, embedding=embedding),
    )


def check_shapes(p: Parameters, embedding: int):
    assert p.qkv.ndim == 3, f"{p.qkv.ndim} =/= {3}"
    assert p.output.ndim == 3, f"{p.output.ndim} =/= {3}"

    d_model = p.qkv.shape[-1]
    assert p.qkv.shape == (3, embedding, d_model), f"{p.qkv.shape} =/= {(3, embedding, d_model)}"

    hd, d_k, d_v = split_heads.check_shapes(p.heads, d_model=d_model)
    assert p.output.shape == (hd, d_v, embedding), f"{p.output.shape} =/= {(hd, d_v, embedding)}"


@check_and_compile(4, 5)
def run(
    params: Parameters,
    q_tokens: Float32[Array, "*batch seq_q embedding"],
    k_tokens: Float32[Array, "*batch seq_k embedding"],
    v_tokens: Float32[Array, "*batch seq_k embedding"],
    causal_mask: bool,
    activation: Callable = lambda x: jnn.softmax(x, axis=-1),
) -> Float32[Array, "*batch seq_q embedding"]:
    """
    A full attention block, batteries included.
    """

    # Check parameter shapes:
    check_shapes(params, embedding=q_tokens.shape[-1])

    # Project tokens into queries, keys, and values:
    q, k, v = qkv.qkv(
        params.qkv,
        q_tokens,
        k_tokens,
        v_tokens,
    )
    # : (
    #     Float32[Array, "*batch seq_q d_model"],
    #     Float32[Array, "*batch seq_k d_model"],
    #     Float32[Array, "*batch seq_k d_model"],
    #   )

    # Project those queries, keys, and values into separate ones for multiple independent "heads":
    q, k, v = split_heads.split_heads(params.heads, q, k, v)
    # : (
    #     Float32[Array, "*batch head seq_q d_k"],
    #     Float32[Array, "*batch head seq_k d_k"],
    #     Float32[Array, "*batch head seq_k d_v"],
    #   )

    # For each token, compute a salience map based on its queries and other tokens' keys:
    salience: Float32[Array, "*batch head seq_q seq_k"] = salience_map.salience_map(
        q, k, causal_mask, activation
    )

    # Weight our update values by their salience to each token:
    attn: Float32[Array, "*batch head seq_q d_v"] = attention.attention(salience, v)

    # Project those updates back to the original data's shape:
    return project_output.project_output(params.output, attn)
