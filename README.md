# Clear, concise implementation of attention in pure JAX

## Purpose

This repo does not aim to capture all possible variants of attention;
many libraries already do this very well!

Instead, this aims to be a clear, concise, and effecient implementation of the original
attention algorithm from _Attention Is All You Need_, with extensive type annotations.
The idea is that you can read this repo and intuit the entire algorithm, from start to finish.

## Why?

The original attention algorithm is, famously, deceptively simple:

```math
\text{Attention}(Q, K, V) = \text{softmax} \left( \frac{ Q K^\top }{ \sqrt{ d_k } } \right) V
```

For example, off the top of your head,
- Wait, what are the dimensions of $Q$? $K$? $V$?
- Which axis is $`d_k`$ again?
- Do we `softmax` over the whole matrix, over rows, or over columns?

None of these are clear from either the original paper or online search results (trust me: we've all tried).

The best introduction by far is [3blue1brown's video](https://youtu.be/eMlx5fFNoYc?si=JUKzND7b0uQ00EaK),
but _**note**_ that the video uses _columns_ throughout where the original paper would use _rows_:
it's really illustrating something closer to $`V \text{softmax} \left( K^\top Q \right)`$,
where matrix transformations are usually to the left of inputs ($`W x`$),
whereas in the original paper they're usually to the right ($`x W`$).

Currently, there seem to be no clear and easily accessible implementations that cover all the details.
This project aims to fill that gap.

## Use

If you're using standard Python build systems, all you need is [JAX](https://github.com/google/jax).
For now, this project uses [Poetry](https://github.com/python-poetry/poetry) to specify dependencies.
If there's demand for a `pip`/PyPI package, that's doable as well.

If you use Nix, there's a flake that works with all Python versions;
all you need to do is hand it the package set you're using.
Note that, since it works with all Python versions, there is no `packages.${system}.default` attribute;
instead, there's `lib.with-pkgs`, which takes package sets as arguments.

For example, here's a simple `flake.nix`:
```nix
{
  inputs = {
    flake-utils.url = "github:numtide/flake-utils";
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    jax-attention.url = "github:wrsturgeon/jax-attention";
  };
  outputs =
    { flake-utils, nixpkgs, jax-attention, self }:
    flake-utils.lib.eachDefaultSystem (system:
      let pkgs = import nixpkgs { inherit system; }; in {
        devShells.default = pkgs.mkShell {
          packages = [
            (pkgs.python311.withPackages (pypkgs: [
              # ... some Python dependencies ...
              (jax-attention.lib.with-pkgs pkgs pypkgs)
              # ... more Python dependencies ...
            ]))
          ];
        };
      }
    );
}
```

## License?

Do whatever you want with it, but
if you learned something, and/or
if this made your life easier,
please give a shoutout!
