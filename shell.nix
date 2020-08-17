{ }:

let pkgs = import <nixpkgs> { };
in pkgs.mkShell {
  buildInputs = with pkgs; [
    haskell.compiler.ghc883
    docker
    ];
  STACK_IN_NIX_SHELL = true;
}
