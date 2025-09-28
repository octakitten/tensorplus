{
  description = "Tensorplus Nix Dev Environment Package";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
      #nvidia.acceptLicense = true;
      #hardware.nvidia.enable = true;
      #hardware.nvidia.driver = "nvidia";
      #hardware.opengl.enable = true;
   

  };

  outputs = { self, nixpkgs }@inputs: 
  let 
    nixpkgs-unfree = import nixpkgs {
      overlays = [
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

    my_virt = python313.withPackages(ps: with ps; [
      pip
      virtualenv
      uv
      pytest
      sphinx
      sphinx-autoapi
      sphinx-rtd-theme
    ]);
  in
  {
    packages = forAllSystems (system: {
      devShell = with nixpkgsFor.${system}; mkShell {
        
        nativeBuildInputs = [
          my_virt
          gnumake
          gcc11
          gcc11Stdenv
          cudaPackages.cudatoolkit
          linuxPackages.nvidia_x11
          bash
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
            
            shellHook = ''
              export $EXTRA_CUDA_FLAGS="-I${cudaPackages.cudatoolkit}/include" -L${cudaPackages.cudatoolkit}/lib64"
              export NIXPLGS_ALLOW_UNFREE=1
            '';

              #mkdir -p $out/bin
              #cp -r * $out/bin
              #cd $out/bin
            installPhase = ''
              uv build --wheel
              uv run pytest --maxfail=0 --junit-xml=results.xml --cov-report=html test/ | tee results.txt || true
            '';

            nativeBuildInputs = [
              python313
              python313Packages.virtualenv
              gnumake
              gcc11
              gcc11Stdenv
              cudaPackages.cudatoolkit
              linuxPackages.nvidia_x11
              uv
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
        
      
    
  

