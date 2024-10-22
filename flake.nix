{
  description = "Tensorplus Nix Dev Environment Package";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
  };

  outputs = { self, nixpkgs }@inputs: 
  let 
    supportedSystems = [
      "x86_64-linux"
    ];
    forAllSystems = nixpkgs.lib.genAttrs supportedSystems;
    nixpkgsFor = forAllSystems (system: nixpkgs.legacyPackages.${system});
  in
  {
    packages = forAllSystems (system: {
      config = {
        allowUnfree = true;
        allowUnfreePredicate = _: true;
        nvidia.acceptLicense = true;
        hardware.nvidia = {
          enable = true;
          driver = "nvidia";
        };
        hardware.opengl.enable = true;
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
            '';

            buildInputs = [
              python312Packages.python
              gcc11
              gnumake
              cudatoolkit
              linuxPackages.nvidia_x11
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
        
      
    
  

