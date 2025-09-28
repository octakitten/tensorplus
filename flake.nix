{
  description = "Tensorplus Nix Dev Environment Package";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
      #nvidia.acceptLicense = true;
      #hardware.nvidia.enable = true;
      #hardware.nvidia.driver = "nvidia";
      #hardware.opengl.enable = true;
    #poetry2nix.url = "github:nix-community/poetry2nix";

  };

  outputs = { self, nixpkgs }@inputs: 
  let 
    pkgs = nixpkgs;

    #inherit (poetry2nix.lib.mkPoetry2Nix { inherit pkgs; }) mkPoetryEnv;

    nixpkgs-unfree = import pkgs {
      overlays = [
        #poetry2nix.overlay
        (self: super: {
            nvidia = super.nvidia.override {
              acceptLicense = true;
            };
            config = super.config.override {
              allowUnfree = true;
            };
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

      #poetry-env = with nixpkgsFor.${system}; mkPoetryEnv {
      #  projectDir = ./.;
      #  python = python312;
      #};


      devShell = with nixpkgsFor.${system}; mkShell {
        
        nativeBuildInputs = [ 
          #poetry-env
          python312
          python312.virtualenv
          gnumake
          gcc11
          gcc11Stdenv
          cudaPackages.cudatoolkit
          cudaPackages.cuda_nvcc
          linuxPackages.nvidia_x11
          #poetry
          uv
          bash
          zip
        ];

        shellHook = ''
          export $EXTRA_CUDA_FLAGS="-I${cudaPackages.cudatoolkit}/include" -L${cudaPackages.cudatoolkit}/lib64"
          export NIXPLGS_ALLOW_UNFREE=1
        '';
      };
      
      default = 
        with nixpkgsFor.${system};
        stdenv.mkDerivation {
          name = "tensorplus";
          src = ./.;
          version = "0.0.2b01";
          dontUseCmakeConfigure = true;

          installPhase = ''
          uv build --wheel
          uv run pytest --maxfail=0 --junit-xml=results.xml --cov-report=html test/ | tee coverage.txt || true
          zip tensorplus.zip $out/src/tensorplus.so $out/src/tensorplus*.whl 
          '';

          meta = {
            description = "A streamlined Tensor library";
            homepage = "https://www.github.com/octakitten/tensorplus";
          };
        };
      });
    };
  }
        
      
    
  

