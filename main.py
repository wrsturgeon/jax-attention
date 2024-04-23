"""
This is a toy project to demonstrate that the attention module works.

You can see below how to initialize, run, and train the module, all in pure JAX.
"""

# Toy task: given a list of numbers, learn that your attention output should be the sum of all previous numbers.
# The system does not appear to be good at learning this (at all), but this should be educational for you!


import jax_attn

from check_and_compile import check_and_compile
from jax import numpy as jnp, random as jrnd, value_and_grad
from jax.tree_util import tree_map
from jaxtyping import Float32, Array
import numpy as np


@check_and_compile()
def ground_truth(
    x: Float32[Array, "*batch seq embedding"]
) -> Float32[Array, "*batch seq embedding"]:
    y = jnp.zeros_like(x)

    # This is certainly not the best way to do this,
    # but for explanatory purposes, it's clear.
    for i in range(x.shape[-2]):
        for j in range(i + 1):  # inclusive
            y = y.at[..., i, :].add(x[..., j, :])

    return y


@check_and_compile(2)
def loss(
    params: jax_attn.Parameters,
    tokens: Float32[Array, "*batch seq embedding"],
    causal_mask: bool,
) -> Float32[Array, ""]:
    y_ideal: Float32[Array, "*batch seq embedding"] = tokens  # ground_truth(tokens)
    y: Float32[Array, "*batch seq embedding"] = jax_attn.run(params, tokens, causal_mask)
    return jnp.mean(jnp.square(y - y_ideal))


# Hyperparameters:
batch = 32
embedding = 1
d_model = 3
heads = 1
d_k = 5
d_v = 7
seq = 11
lr = 0.25  # Loss averages every dimension, so this will be way higher than if it were a sum
training_steps = 10000


# Initialization:
key = jrnd.PRNGKey(42)
losses = np.empty([training_steps], dtype=np.float32)
# params = jax_attn.init(key, embedding=embedding, d_model=d_model, heads=heads, d_k=d_k, d_v=d_v)
params = jax_attn.Parameters(
    qkv=jnp.stack(
        [
            jnp.zeros([embedding, d_model], dtype=jnp.float64),
            jnp.zeros([embedding, d_model], dtype=jnp.float64),
            jnp.eye(embedding, d_model, dtype=jnp.float64),
        ]
    ),
    heads=jax_attn.split_heads.Parameters(
        w_q=jnp.stack([jnp.eye(d_model, d_k) for _ in range(heads)]),
        w_k=jnp.stack([jnp.eye(d_model, d_k) for _ in range(heads)]),
        w_v=jnp.stack([jnp.eye(d_model, d_v) for _ in range(heads)]),
    ),
    output=jnp.stack([jnp.eye(d_v, embedding) for _ in range(heads)]),
)


# Training loop:
for i in range(training_steps):
    key, k = jrnd.split(key)
    tokens = jrnd.normal(k, [batch, seq, embedding], dtype=jnp.float32)

    L, dLdp = value_and_grad(loss)(params, tokens, True)
    losses[i] = L
    print(f"Loss: {L}")

    params = tree_map(lambda p, d: p - lr * d, params, dLdp)


try:
    from matplotlib import pyplot as plt

    plt.plot(losses)
    plt.savefig("loss.png")

except ModuleNotFoundError:
    print("`matplotlib` is not installed; skipping loss plot...")
