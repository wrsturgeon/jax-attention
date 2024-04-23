# Clear, concise implementation of attention in pure JAX

## Purpose

This repo does not aim to capture all possible variants of attention;
many libraries already do this very well!

Instead, this aims to be a clear, concise, and effecient implementation of the original
multi-head attention algorithm from _Attention Is All You Need_, with extensive type annotations.
The idea is that you can read this repo and intuit the entire algorithm, from start to finish.

## Why?

The original attention algorithm is deceptively simple:

```math
\text{Attention}(Q, K, V) = \text{softmax} \left( \frac{ Q K^\top }{ \sqrt{ d_k } } \right) V
```
```math
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h) W^O
```
```math
\text{where head}_i = \text{Attention}(Q W_i^Q, K W_i^K, V W_i^V)
```

For example, off the top of your head,
- Wait, what are the dimensions of $Q$? $K$? $V$?
- Which axis is $`d_k`$ again?
- Do we `softmax` over the whole matrix, over rows, or over columns?

None of these are clear from either the original paper or online search results (trust me: we've all tried).

The best introduction by far is [3blue1brown's video](https://youtu.be/eMlx5fFNoYc?si=JUKzND7b0uQ00EaK),
but _**note**_ that the video uses _columns_ throughout where the original paper would use _rows_:
it's really illustrating $`V \text{softmax} \left( K^\top Q \right)`$,
where matrix transformations are usually to the left of inputs ($`W x`$),
whereas in the original paper they're usually to the right ($`x W`$).
These formulations are identically powerful;
you'd just need to transpose some matrices to get identical outputs.

Currently, there seem to be no clear and easily accessible implementations that cover all the details.
This project aims to fill that gap.

## Use

This package uses [Nix](https://github.com/nixos/nix) for fully reproducible builds.

It works with all versions of Python, so instead of providing `packages.${system}.default`,
it provides `lib.with-pkgs`, which you can use like this:

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

Note that there is, legally, absolutely no guarantee that this works the way you want it to, but of course I hope it does.
