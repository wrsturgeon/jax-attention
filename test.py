from jax_attn import mask, qkv, salience_map, split_heads

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


def test_qkv():
    # Prime numbers to be *sure* no shapes coincidentally align:

    p = jnp.arange(3 * 5 * 2).astype(jnp.float64).reshape(3, 5, 2)
    t = jnp.arange(11 * 7 * 5).astype(jnp.float32).reshape(11, 7, 5)
    qkv.qkv(p, t)


def test_split_heads():
    # Prime numbers, same as above:
    p = split_heads.Parameters(
        w_q=jnp.zeros([3, 2, 5], dtype=jnp.float64),
        w_k=jnp.zeros([3, 2, 5], dtype=jnp.float64),
        w_v=jnp.zeros([3, 2, 7], dtype=jnp.float64),
    )
    qkv = jnp.zeros([3, 11, 13, 2], dtype=jnp.float32)
    split_heads.split_heads(p, qkv)


def test_salience_map():
    q = jnp.arange(2 * 3 * 5 * 7).astype(jnp.float32).reshape(2, 3, 5, 7)
    k = jnp.arange(2 * 3 * 5 * 7).astype(jnp.float32).reshape(2, 3, 5, 7)
    unmasked = salience_map.salience_map(q, k, False)
    masked = salience_map.salience_map(q, k, True)
    assert jnp.allclose(jnp.sum(unmasked, -1), 1)
    assert jnp.allclose(jnp.sum(masked, -1), 1)
