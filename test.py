from jax import numpy as jnp, random as jrnd


# Throughout, we use prime numbers as shapes,
# to be *sure* no shapes coincidentally align
# unless we explicitly want them to.


# %%%%%%%%%%%%%%%% Non-batched:


def test_mask():
    from jax_attn import mask

    assert jnp.all(
        mask.mask((42, 5, 5))
        == jnp.array(
            [
                [0, 1, 1, 1, 1],
                [0, 0, 1, 1, 1],
                [0, 0, 0, 1, 1],
                [0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0],
            ],
            dtype=jnp.bool,
        )
    )


def test_qkv():
    from jax_attn import qkv

    key = jrnd.PRNGKey(42)
    p = qkv.init(key, embedding=5, d_model=2)
    t = jnp.arange(7 * 5).astype(jnp.float32).reshape(7, 5)
    qkv.qkv(p, t, t + 1, t + 2)


def test_split_heads():
    from jax_attn import split_heads

    key = jrnd.PRNGKey(42)
    p = split_heads.init(key, heads=3, d_model=2, d_k=5, d_v=7)
    split_heads.check_shapes(p, d_model=2)

    q, k, v = jnp.arange(3 * 11 * 2).astype(jnp.float32).reshape(3, 11, 2)
    split_heads.split_heads(p, q, k, v)


def test_salience_map():
    from jax_attn import salience_map

    q = jnp.arange(3 * 5 * 7).astype(jnp.float32).reshape(3, 5, 7)
    k = jnp.arange(3 * 5 * 7).astype(jnp.float32).reshape(3, 5, 7)
    unmasked = salience_map.salience_map(q, k, False)
    masked = salience_map.salience_map(q, k, True)
    assert jnp.allclose(jnp.sum(unmasked, -1), 1)
    assert jnp.allclose(jnp.sum(masked, -1), 1)


def test_attention():
    from jax_attn import attention

    salience = jnp.arange(3 * 5 * 5).astype(jnp.float32).reshape(3, 5, 5)
    values = jnp.arange(3 * 5 * 7).astype(jnp.float32).reshape(3, 5, 7)

    a = attention.attention(salience, values)
    assert a.shape == (3, 5, 7), f"{a.shape} =/= {(3, 5, 7)}"


def test_project_output():
    from jax_attn import project_output

    key = jrnd.PRNGKey(42)
    p = project_output.init(key, heads=2, d_v=3, embedding=5)
    t = jnp.arange(2 * 7 * 3).astype(jnp.float32).reshape(2, 7, 3)
    project_output.project_output(p, t)


def test_run():
    import jax_attn

    key = jrnd.PRNGKey(42)
    p = jax_attn.init(key, embedding=2, d_model=3, heads=5, d_k=7, d_v=11)
    jax_attn.check_shapes(p, embedding=2)

    t = jnp.arange(17 * 2).astype(jnp.float32).reshape(17, 2)
    jax_attn.run(p, t, t + 1, t + 2, False)
    jax_attn.run(p, t, t + 1, t + 2, True)


# %%%%%%%%%%%%%%%% Batched:


def test_batched_mask():
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


def test_batched_qkv():
    from jax_attn import qkv

    key = jrnd.PRNGKey(42)
    p = qkv.init(key, embedding=5, d_model=2)
    t = jnp.arange(11 * 7 * 5).astype(jnp.float32).reshape(11, 7, 5)
    qkv.qkv(p, t, t + 1, t + 2)


def test_batched_split_heads():
    from jax_attn import split_heads

    key = jrnd.PRNGKey(42)
    p = split_heads.init(key, heads=3, d_model=2, d_k=5, d_v=7)
    split_heads.check_shapes(p, d_model=2)

    q, k, v = jnp.arange(3 * 11 * 13 * 2).astype(jnp.float32).reshape(3, 11, 13, 2)
    split_heads.split_heads(p, q, k, v)


def test_batched_salience_map():
    from jax_attn import salience_map

    q = jnp.arange(2 * 3 * 5 * 7).astype(jnp.float32).reshape(2, 3, 5, 7)
    k = jnp.arange(2 * 3 * 5 * 7).astype(jnp.float32).reshape(2, 3, 5, 7)
    unmasked = salience_map.salience_map(q, k, False)
    masked = salience_map.salience_map(q, k, True)
    assert jnp.allclose(jnp.sum(unmasked, -1), 1)
    assert jnp.allclose(jnp.sum(masked, -1), 1)


def test_batched_attention():
    from jax_attn import attention

    salience = jnp.arange(2 * 3 * 5 * 5).astype(jnp.float32).reshape(2, 3, 5, 5)
    values = jnp.arange(2 * 3 * 5 * 7).astype(jnp.float32).reshape(2, 3, 5, 7)

    a = attention.attention(salience, values)
    assert a.shape == (2, 3, 5, 7), f"{a.shape} =/= {(2, 3, 5, 7)}"


def test_batched_project_output():
    from jax_attn import project_output

    key = jrnd.PRNGKey(42)
    p = project_output.init(key, heads=2, d_v=3, embedding=5)
    t = jnp.arange(11 * 2 * 7 * 3).astype(jnp.float32).reshape(11, 2, 7, 3)
    project_output.project_output(p, t)


def test_batched_run():
    import jax_attn

    key = jrnd.PRNGKey(42)
    p = jax_attn.init(key, embedding=2, d_model=3, heads=5, d_k=7, d_v=11)
    jax_attn.check_shapes(p, embedding=2)

    t = jnp.arange(13 * 17 * 2).astype(jnp.float32).reshape(13, 17, 2)
    jax_attn.run(p, t, t + 1, t + 2, False)
    jax_attn.run(p, t, t + 1, t + 2, True)
