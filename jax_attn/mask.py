from jax_attn.jit import jit

from beartype.typing import Tuple
from jax import numpy as jnp
from jaxtyping import Array, Bool


@jit(0)
def mask(shape: Tuple) -> Bool[Array, "... n n"]:
    """
    Prevents early tokens in a sequence from being influenced by later tokens.

    Since rows and columns in (Q K^T) represent the indices of input tokens,
    we want to mask everything whose column index is greater than its row index.
    Visually, this looks like the following matrix:
    ```
    [ 0 1 1 1 1 ... ]
    [ 0 0 1 1 1 ... ]
    [ 0 0 0 1 1 ... ]
    [ 0 0 0 0 1 ... ]
    [ 0 0 0 0 0 ... ]
    [ : : : : : ... ]
    ```
    where 1 means "mask" and 0 means "pass through."
    """

    # Make sure we received a batch of square matrices
    assert [isinstance(i, int) for i in shape]
    n, m = shape[-2:]
    assert n == m, (
        f"\n`mask` should receive a batch of square matrices;"
        f"\nin other words, the last two shapes (..., {n}, {m}) should be equal"
    )

    # Generate row- and column-indices
    rng: Int[Array, "n"] = jnp.arange(n)
    row: Int[Array, "1 n"] = rng[jnp.newaxis]
    col: Int[Array, "n 1"] = rng[:, jnp.newaxis]

    # Compare these, effectively computing "is this above the diagonal?"
    cmp: Bool[Array, "n n"] = row > col

    # Expand this to have the same `ndim` as `shape`:
    expanded: Bool[Array, "... n n"] = cmp.reshape(*[1 for _ in shape[:-2]], n, n)
    assert expanded.ndim == len(shape)

    return expanded
