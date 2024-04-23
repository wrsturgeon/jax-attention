from jax import numpy as jnp, random as jrnd


def test_mask():
    from jax_attn import mask

    assert jnp.all(
        mask.mask((42, 37, 5, 5))
        == jnp.array(
            [
                [
                    [0, 1, 1, 1, 1],
                    [0, 0, 1, 1, 1],
                    [0, 0, 0, 1, 1],
                    [0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0],
                ]
            ],
            dtype=jnp.bool,
        )
    )


def test_qkv():
    from jax_attn import qkv

    key = jrnd.PRNGKey(42)

    # Prime numbers to be *sure* no shapes coincidentally align:
    p = qkv.init(key, embedding=5, d_model=2)
    t = jnp.arange(11 * 7 * 5).astype(jnp.float32).reshape(11, 7, 5)
    qkv.qkv(p, t)


def test_split_heads():
    from jax_attn import split_heads

    key = jrnd.PRNGKey(42)

    # Prime numbers, same as above:
    p = split_heads.init(key, heads=3, d_model=2, d_k=5, d_v=7)
    qkv = jnp.arange(3 * 11 * 13 * 2).astype(jnp.float32).reshape(3, 11, 13, 2)
    split_heads.split_heads(p, qkv)


def test_salience_map():
    from jax_attn import salience_map

    q = jnp.arange(2 * 3 * 5 * 7).astype(jnp.float32).reshape(2, 3, 5, 7)
    k = jnp.arange(2 * 3 * 5 * 7).astype(jnp.float32).reshape(2, 3, 5, 7)
    unmasked = salience_map.salience_map(q, k, False)
    masked = salience_map.salience_map(q, k, True)
    assert jnp.allclose(jnp.sum(unmasked, -1), 1)
    assert jnp.allclose(jnp.sum(masked, -1), 1)
