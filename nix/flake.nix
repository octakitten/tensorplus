{
  description = "Tensorplus Nix Dev Environment Package";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
    tensorplus.url = "github:octakitten/tensorplus";
  };

  outputs = { self, nixpkgs, tensorplus }@inputs: 
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
      default = nixpkgsFor.${system}.callPackage inputs.tensorplus {};
    });
  };
}