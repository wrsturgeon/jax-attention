from check_and_compile import check_and_compile
from jax import nn as jnn, numpy as jnp
from jaxtyping import Array, Float32


@check_and_compile()
def attention(
    salience: Float32[Array, "*batch head seq_q seq_k"],
    values: Float32[Array, "*batch head seq_k d_v"],
) -> Float32[Array, "*batch head seq_q d_v"]:
    """
    Given a salience map and update values, weight the update values according to salience.
    """

    return salience @ values
