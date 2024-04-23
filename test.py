from jax_attn import mask

from jax import numpy as jnp


def test_mask():
    assert jnp.all(
        mask.mask(5)
        == jnp.array(
            [
                [
                    [0, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0],
                    [1, 1, 0, 0, 0],
                    [1, 1, 1, 0, 0],
                    [1, 1, 1, 1, 0],
                ]
            ],
            dtype=jnp.bool,
        )
    )
