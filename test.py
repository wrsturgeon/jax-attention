from jax_attn import mask, qkv

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


def test_qkv_shapes():
    # Prime numbers to be *sure* no shapes coincidentally align:

    p = jnp.arange(3 * 5 * 2).astype(jnp.float64).reshape(3, 5, 2)
    t = jnp.arange(11 * 7 * 5, dtype=jnp.float32).reshape(11, 7, 5)
    qkv.qkv(p, t)
