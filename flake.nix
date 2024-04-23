{
  description = "Extremely simple, fast, type-checked attention for neural networks, all in pure JAX.";
  inputs = {
    flake-utils.url = "github:numtide/flake-utils";
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
  };
  outputs =
    {
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
        p:
        with p;
        [
          beartype
          jaxtyping
        ]
        ++ [
          (jax.overridePythonAttrs (
            old:
            old
            // {
              doCheck = false;
              propagatedBuildInputs = old.propagatedBuildInputs ++ [ p.jaxlib-bin ];
            }
          ))
        ];
      check-pkgs =
        p: with p; [
          hypothesis
          pytest
        ];
      ci-pkgs =
        p: with p; [
          black
          coverage
        ];
      dev-pkgs = p: with p; [ python-lsp-server ];
      lookup-pkg-sets = ps: p: builtins.concatMap (f: f p) ps;
    in
    {
      lib.with-pkgs =
        pkgs: pypkgs:
        # pypkgs.buildPythonPackage {
        #   inherit pname version src;
        #   pyproject = true;
        #   build-system = with pypkgs; [ poetry-core ];
        #   dependencies = lookup-pkg-sets [ default-pkgs ] pypkgs;
        # };
        pkgs.stdenv.mkDerivation {
          inherit pname version src;
          propagatedBuildInputs = lookup-pkg-sets [ default-pkgs ] pypkgs;
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
        python-with = ps: "${pypkgs.python.withPackages (lookup-pkg-sets ps)}/bin/python";
      in
      {
        packages.ci =
          let
            pname = "ci";
            version = "none";
            src = ./.;
            python = python-with [
              default-pkgs
              check-pkgs
              ci-pkgs
            ];
            exec = ''
              #!${pkgs.bash}/bin/bash

              set -eu

              export JAX_ENABLE_X64=1
              # export NONJIT=1

              ${python} -m black --check .

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
            ] pypkgs
          )
          # ++ (with pkgs; [ poetry ])
          ;
        };
      }
    );
}
