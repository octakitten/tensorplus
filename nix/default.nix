{
    default = pkgs.stdenv.mkDerivation {
        name = "tensorplus";
        src = ./.;
        buildInputs = [
            pkgs.python312Full
            pkgs.poetry
            pkgs.git
            pkgs.gh
            pkgs.gnumake
            pkgs.gcc11
            pkgs.cudatoolkit
            pkgs.linuxPackages.nvidia_x11
        ];
        shellHook = ''
            export $EXTRA_CUDA_FLAGS="-I${pkgs.cudatoolkit}/include -L${pkgs.cudatoolkit}/lib64"
            '';
        installPhase = ''
            mkdir -p $out/bin
            cp -r * $out/bin
            '';
        dontUseCmakeConfigure = true;
    };
}