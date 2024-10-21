{
  description = "Tensorplus Nix Dev Environment Package";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
  };

  outputs = { self, nixpkgs }: {
    packages."x86_64-linux" = let
      pkgs = import nixpkgs {
        system = "x86_64-linux";
        config = {
          allowUnfree = true;
          allowUnfreePredicate = _: true;
        };
      };
      in {
        default = pkgs.stdenv.mkDerivation {
        name = "tensorplus";
        src = ./.;
        
        nativeBuildInputs = [
          pkgs.python312Full
        ];

        buildInputs = [
          pkgs.poetry
          pkgs.git
          pkgs.gh
          pkgs.gnumake
          pkgs.gcc11
          pkgs.cudaPackages.cudatoolkit
        ];

        dontUseCmakeConfigure = true;
      };
    };
  };
}
