from jax_attn.jit import jit

from jax import numpy as jnp
from jaxtyping import Array, Bool


@jit(0)
def mask(n: int) -> Bool[Array, "1 1 n n"]:
    """
    Prevents early tokens in a sequence from being influenced by later tokens.

    Since rows and columns in (Q K^T) represent the indices of input tokens,
    we want to mask everything whose column index is greater than its row index.
    Visually, this looks like the following matrix:
    ```
    [ 0 0 0 0 0 ... ]
    [ 1 0 0 0 0 ... ]
    [ 1 1 0 0 0 ... ]
    [ 1 1 1 0 0 ... ]
    [ 1 1 1 1 0 ... ]
    [ : : : : : ... ]
    ```
    where 1 means "mask" and 0 means "pass through."
    """

    rng = jnp.arange(n)
    row: Int[Array, "1 1 1 n"] = rng[jnp.newaxis, jnp.newaxis, jnp.newaxis, :]
    col: Int[Array, "1 1 n 1"] = rng[jnp.newaxis, jnp.newaxis, :, jnp.newaxis]
    return row < col
