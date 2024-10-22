{
  gcc11,
  cudatoolkit,
  gnumake,
  linuxPackages,
  stdenv,
  lib,
  python312Packages
}:

stdenv.mkDerivation {
    name = "tensorplus";
    src = ./.;
    
    shellHook = ''
        export $EXTRA_CUDA_FLAGS="-I${cudatoolkit}/include -L${cudatoolkit}/lib64"
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
}
