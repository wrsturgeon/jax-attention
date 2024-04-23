from jax_attn.jit import jit
from jax_attn.mask import mask

from jax import nn as jnn, numpy as jnp
from jaxtyping import Array, Float32


@jit(2)
def salience_map(
    q: Float32[Array, "batch head seq d_k"],
    k: Float32[Array, "batch head seq d_k"],
    causal_mask: bool,
) -> Float32[Array, "batch head seq seq"]:
    """
    Given queries and keys, compute which tokens pay attention to which others (or themselves).

    This is the (Q K^T / sqrt(d_k)) part of the famous equation.
    """

    # Take the dot product of each token's queries with each token's keys:
    qkT: Float32[Array, "batch seq seq"] = q @ k.transpose(0, 1, 3, 2)

    # Divide this by the number of queries for numerical stability:
    normalized = qkT / jnp.sqrt(q.shape[-1])

    # If we want a causal mask, add it:
    boolean_mask = mask(q.shape[-2])
    assert boolean_mask.ndim == normalized.ndim, f"{boolean_mask.ndim} =/= {normalized.ndim}"
    assert (
        boolean_mask.shape[-2:] == normalized.shape[-2:]
    ), f"{boolean_mask.shape[-2:]} =/= {normalized.shape[-2:]}"
    masked = jnp.where(boolean_mask, -jnp.inf, normalized) if causal_mask else normalized

    # Convert the above into probability distributions over rows:
    probabilities = jnn.softmax(masked)

    return probabilities
