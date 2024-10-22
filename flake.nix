{
  description = "Tensorplus Nix Dev Environment Package";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
  };

  outputs = { self, nixpkgs }@inputs: 
  let 
    nixpkgs-conf = import nixpkgs {
      overlays = [
        (self: super: {
          config.allowUnfree = true;
          config.allowUnfreePredicate = _: true;
          config.nvidia.acceptLicense = true;
          config.hardware.nvidia.enable = true;
          config.hardware.nvidia.driver = "nvidia";
          config.hardware.opengl.enable = true;
        })
      ];
    };
    supportedSystems = [
      "x86_64-linux"
    ];
    forAllSystems = nixpkgs.lib.genAttrs supportedSystems;
    nixpkgsFor = forAllSystems (system: nixpkgs.legacyPackages.${system});
  in
  {
    packages = forAllSystems (system: {
      devShell = with nixpkgsFor.${system}; mkShell {

      };
      default = 
        with nixpkgsFor.${system};
          stdenv.mkDerivation {
            name = "tensorplus";
            src = ./.;

            
            shellHook = ''
              export $EXTRA_CUDA_FLAGS="-I${cudaPackages.cudatoolkit}/include" -L${cudaPackages.cudatoolkit}/lib64"
              export NIXPLGS_ALLOW_UNFREE=1
            '';

            installPhase = ''
              mkdir -p $out/bin
              cp -r * $out/bin
              echo $(gcc --version)
              echo $(nvcc --version)
            '';

            nativeBuildInputs = [
              python312Packages.python
              gnumake
              gcc11
              libcxx
              cudaPackages.cudatoolkit
              linuxPackages.nvidia_x11
              poetry
              bash
            ];
            dontUseCmakeConfigure = true;

            meta = {
              description = "A streamlined Tensor library";
              homepage = "https://www.github.com/octakitten/tensorplus";
            };
          };
        
      });
    };
  }
        
      
    
  

