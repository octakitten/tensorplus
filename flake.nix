{
  description = "Tensorplus Nix Package";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
  };

  outputs = { self, nixpkgs }: {
    packages."x86_64-linux" = let
      pkgs = import nixpkgs {
        system = "x86_64-linux";
      };
    in {
      default = pkgs.stdenv.mkDerivation {
        name = "Tensorplus";
        src = ./.;
        
        nativeBuildInputs = [
          pkgs.python312Full
        ];

        buildInputs = [
          pkgs.poetry
          pkgs.git
          pkgs.gh
          pkgs.gnumake
          pkgs.libgcc
        ];
      };
    };
  };
}
