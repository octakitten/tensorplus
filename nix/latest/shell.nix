{ pkgs ? import <nixpkgs> {} }:
(pkgs.buildFHSEnv {
  name = "tensorplus";
  version = "0.0.2";
  targetPkgs = pkgs: [ pkgs.python313 pkgs.uv pkgs.gnumake pkgs.cudaPackages.cudatoolkit pkgs.gcc13 pkgs.gcc13Stdenv pkgs.bash pkgs.zip pkgs.linuxPackages.nvidia_x11 ];
  #inputsFrom = [ pkgs.python313 ];
  #runScript = "bash buildwheel.sh";
  
}).env
