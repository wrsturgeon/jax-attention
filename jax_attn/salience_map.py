from jax_attn.mask import mask

from beartype.typing import Callable
from check_and_compile import check_and_compile
from jax import nn as jnn, numpy as jnp
from jaxtyping import Array, Bool, Float32


@check_and_compile(2, 3)
def salience_map(
    q: Float32[Array, "*batch head seq_q d_k"],
    k: Float32[Array, "*batch head seq_k d_k"],
    causal_mask: bool,
    activation: Callable = lambda x: jnn.softmax(x, axis=-1),
) -> Float32[Array, "*batch head seq_q seq_k"]:
    """
    Given queries and keys, compute which tokens pay attention to which others (or themselves).

    This is the `softmax(Q K^T / sqrt(d_k))` part of the famous equation.
    """

    # Transpose keys so we can matrix-multiply with queries:
    kT: Float32[Array, "*batch head d_k seq_k"] = jnp.einsum("... s d -> ... d s", k)

    # Take the dot products of each token's queries with every token's keys:
    qkT: Float32[Array, "*batch head seq_q seq_k"] = q @ kT

    # Divide this by the square root of the number of queries (for numerical stability):
    unmasked: Float32[Array, "*batch head seq_q seq_k"] = qkT * jnp.power(q.shape[-1], -0.5)

    # If we want a causal mask, add it:
    logits: Float32[Array, "*batch head seq_q seq_k"] = (
        jnp.where(mask(unmasked.shape), jnp.finfo(unmasked.dtype).min, unmasked)
        if causal_mask
        else unmasked
    )

    # Convert rows into probability distributions:
    return activation(logits)
