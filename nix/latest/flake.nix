{
  description = "Tensorplus Nix Dev Environment Package";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
    pyproject-nix = {
      url = "github:pyproject-nix/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    pyproject-build-systems = {
      url = "github:pyproject-nix/build-system-pkgs";
      inputs.nixpkgs.follows = "nixpkgs";
      inputs.pyproject-nix.follows = "pyproject-nix";
    };

    uv2nix = {
      url = "github:pyproject-nix/uv2nix";
      inputs.nixpkgs.follows = "nixpkgs";
      inputs.pyproject-nix.follows = "pyproject-nix";
    };


  };

  outputs = { self, nixpkgs, pyproject-nix, pyproject-build-systems, uv2nix }@inputs: 
  let 
    inherit (nixpkgs) lib;

    overlays = [
      (final: prev: { stdenv = prev.cudaPackages.backendStdenv; })
    ];

    project = pyproject-nix.lib.project.loadUVPyproject {
      projectRoot = ./.;
    };

    nixpkgs-unfree = import nixpkgs {
      overlays = [
        (self: super: {
            nvidia = super.nvidia.override {
              acceptLicense = true;
            };
         })
      ];
    };

    pkgs = nixpkgs.legacyPackages.x86_64-linux;
    python = pkgs.python313;
    
    pythonEnv = python.withPackages (project.renderers.withPackages {inherit python; });
  in
  {
    packages.x86_64-linux.default = 
      let
        attrs = project.renderers.buildPythonPackage { inherit python; };
      in
      python.pkgs.buildPythonPackage (attrs // {
        devShell = with pkgs; mkShell {
          nativeBuildInputs = [
            gnumake
            gcc13
            gcc13Stdenv
            cudaPackages.cudatoolkit
            cudaPackages.backendStdenv
            linuxPackages.nvidia_x11
            uv
            bash
            zip
          ];
          shellHook = ''
            export EXTRA_CUDA_FLAGS="-I${cudaPackages.cudatoolkit}/include -L${cudaPackages.cudatoolkit}/lib64"
            export NIXPKGS_ALLOW_UNFREE=1
          '';
        };

        default = with pkgs; stdenv.mkDerivation {
          name = "tensorplus";
          src = ./.;
          version = "0.0.2b01";
          dontUseCmakeConfigure = true;

          buildPhase = ''
            echo $EXTRA_CUDA_FLAGS
            LD_PRELOAD="${cudaPackages.cudatoolkit}/bin"
            EXTRA_CUDA_FLAGS="-I${cudaPackages.cudatoolkit}/include -L${cudaPackages.cudatoolkit}/lib64"
            NIXPKGS_ALLOW_UNFREE=1
            echo $EXTRA_CUDA_FLAGS
            echo $LD_PRELOAD
            NIX_CFLAGS_COMPILE="-isystem ${pkgs.python313}/include $NIX_CFLAGS_COMPILE"
            make
          '';

          meta = {
            description = "A streamlined Tensor library";
            homepage = "https://www.github.com/octakitten/tensorplus";
          };
        };
      });
  };
}
        
      
    
  

