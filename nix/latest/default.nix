{ lib, stdenv, buildFHSEnv, python313, uv, gnumake,  gcc13, gcc13Stdenv, bash, zip, cudaPackages, linuxPackages, }:

  let
    pname = "tensorplus";
    version = "0.0.2";

    tensorplus = stdenv.mkDerivation {
      inherit pname version;
      dontUseCmakeConfigure = true;

      src = ./.;

      buildInputs = [
        python313
        uv
        gnumake
        cudaPackages.cudatoolkit
        gcc13
        stdenv
        gcc13Stdenv
        bash
        zip
        linuxPackages.nvidia_x11
      ];

      installPhase = ''
        cp tensorplus.so src/tensorplus/tensorplus.so
        uv build --wheel
      '';

      meta = {
        description = "A streamlined Tensor library";
        homepage = "https://www.github.com/octakitten/tensorplus";
      };
    };
  in

  buildFHSEnv {
    inherit pname version;

    targetPkgs = pkgs: [
      python313
      uv
      gnumake
      cudaPackages.cudatoolkit
      gcc13
      stdenv
      gcc13Stdenv
      bash
      zip
      linuxPackages.nvidia_x11
    ];

    runscript = "bash";
  }
