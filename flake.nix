{
  description = "Quantum JIT compiler";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

    rust-overlay = {
      url = "https://flakehub.com/f/oxalica/rust-overlay/*";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  # Flake outputs that other flakes can use
  outputs = { self, nixpkgs, rust-overlay }:
    let
      overlays = [
        rust-overlay.overlays.default
        (final: prev: {
          rustToolchain = final.rust-bin.stable."1.85.0".default.override { extensions = [ "rust-src" ]; };
        })


      ];

      # Helpers for producing system-specific outputs
      supportedSystems = [ "x86_64-linux" "aarch64-darwin" "x86_64-darwin" "aarch64-linux" ];
      forEachSupportedSystem = f: nixpkgs.lib.genAttrs supportedSystems (system: f {
        pkgs = import nixpkgs { inherit overlays system; };
      });
    in
    {

      # Development environments
      devShells = forEachSupportedSystem ({ pkgs }: {
        default = pkgs.mkShell {
          # Pinned packages available in the environment
          packages = with pkgs; [
            git
            typst

            # nix management
            nixpkgs-fmt
            nil

            # command runners
            just
            watchexec

            # Rust management
            rustToolchain
            rust-analyzer

            # python management
            uv
          ];


          # Environment variables
          env = {
            RUST_SRC_PATH = "${pkgs.rustToolchain}/lib/rustlib/src/rust/library";
          };

          # A hook run every time you enter the environment
          shellHook = ''
            printf "Entering project environment.\n"
          '';
        };
      });
    };
}
