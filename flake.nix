{
  description = "Clear, concise, & efficient implementation of attention for neural networks in pure JAX.";
  inputs = {
    check-and-compile = {
      inputs = {
        flake-utils.follows = "flake-utils";
        nixpkgs.follows = "nixpkgs";
      };
      url = "github:wrsturgeon/check-and-compile";
    };
    flake-utils.url = "github:numtide/flake-utils";
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
  };
  outputs =
    {
      check-and-compile,
      flake-utils,
      nixpkgs,
      self,
    }:
    let
      pname = "jax-attention";
      pyname = "jax_attn";
      version = "0.0.1";
      src = ./.;
      default-pkgs =
        p: py:
        with py;
        [
          beartype
          jaxtyping
        ]
        ++ [
          (check-and-compile.lib.with-pkgs p py)
          (
            let
              shit = jax.overridePythonAttrs (
                old:
                old
                // {
                  doCheck = false;
                  propagatedBuildInputs = old.propagatedBuildInputs ++ [ py.jaxlib-bin ];
                }
              );
            in
            builtins.trace "${shit}" shit
          )
        ];
      check-pkgs =
        p: py: with py; [
          hypothesis
          mypy
          pytest
        ];
      ci-pkgs =
        p: py: with py; [
          black
          coverage
        ];
      dev-pkgs =
        p: py: with py; [
          matplotlib
          python-lsp-server
        ];
      lookup-pkg-sets =
        ps: p: py:
        builtins.concatMap (f: f p py) ps;
    in
    {
      lib.with-pkgs =
        pkgs: pypkgs:
        pkgs.stdenv.mkDerivation {
          inherit pname version src;
          propagatedBuildInputs = lookup-pkg-sets [ default-pkgs ] pkgs pypkgs;
          buildPhase = ":";
          installPhase = ''
            mkdir -p $out/${pypkgs.python.sitePackages}
            mv ./${pyname} $out/${pypkgs.python.sitePackages}/${pyname}
          '';
        };
    }
    // flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs { inherit system; };
        pypkgs = pkgs.python311Packages;
        python-with = ps: "${pypkgs.python.withPackages (lookup-pkg-sets ps pkgs)}/bin/python";
      in
      {
        packages.ci =
          let
            pname = "ci";
            python = python-with [
              default-pkgs
              check-pkgs
              ci-pkgs
            ];
            exec = ''
              #!${pkgs.bash}/bin/bash

              set -eu

              export JAX_ENABLE_X64=1

              ${python} -m black --line-length=100 --check .
              ${python} -m mypy .

              ${python} -m coverage run --omit='/nix/*' -m pytest -Werror test.py
              ${python} -m coverage report -m --fail-under=100
            '';
          in
          pkgs.stdenv.mkDerivation {
            inherit pname version src;
            buildPhase = ":";
            installPhase = ''
              mkdir -p $out/${pypkgs.python.sitePackages}
              mv ./${pyname} $out/${pypkgs.python.sitePackages}/${pyname}
              mv ./test.py $out/test.py

              mkdir -p $out/bin
              echo "${exec}" > $out/bin/${pname}
              chmod +x $out/bin/${pname}
            '';
          };
        devShells.default = pkgs.mkShell {
          JAX_ENABLE_X64 = "1";
          packages = (
            lookup-pkg-sets [
              default-pkgs
              check-pkgs
              ci-pkgs
              dev-pkgs
            ] pkgs pypkgs
          );
        };
      }
    );
}
