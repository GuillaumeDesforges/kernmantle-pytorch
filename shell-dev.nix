{ }:

let
  pkgs = import <nixpkgs> { };

  devShellsInputs =
    import
      (builtins.fetchGit {
        url = "git@github.com:tweag/nix-dev-shells.git";
        name = "nix-dev-shell";
        rev = "9dbca810bc4dd243fc5a62bd0eef81898798e286";
      })
      # Use your own version of nixpkgs
      { inherit pkgs; };

  python-env =
    pkgs.python3.withPackages (pp: with pp; [
      numpy
      pytorchWithCuda
      torchvision
      pillow
    ]);
in pkgs.mkShell {
  buildInputs = (with devShellsInputs;
  # Vim
    (vim {
      languageClient = true;
      languageClientOptions = { haskellLanguageServer = true; };
    })
    # Standard Haskell dev environment
    ++ (haskell {
      ghcide = false;
      haskellLanguageServer = true;
      haskellLanguageServerGhcVersion = "ghc883";
    })) ++ (with pkgs; [ 
      # python-env
    ]);
}
