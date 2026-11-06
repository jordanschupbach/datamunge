{
  pkgs,
  datamunge,
  swigPkg,
  dotnetSdk,
  ...
}:
let
  datamungedotnet = import ../datamungedotnet.nix {
    inherit pkgs datamunge dotnetSdk;
    swig = swigPkg;
  };
  # A file-based app defaults to PublishAot=true, so its restore wants the ILCompiler/ILLink
  # packages even for `dotnet run`; nixpkgs fetches the ones matching this SDK.
  aotPackages = builtins.filter (
    p: pkgs.lib.hasInfix "ILCompiler" p.pname || pkgs.lib.hasInfix "ILLink" p.pname
  ) dotnetSdk.packages;
  restoreSources = [
    "${datamungedotnet}/share/nuget"
  ]
  ++ map (p: "${p}/share/nuget/source") aotPackages;
in
pkgs.mkShell {
  packages = [ dotnetSdk ];
  DOTNET_CLI_TELEMETRY_OPTOUT = "1";
  DOTNET_NOLOGO = "1";
  # A block references the binding as a NuGet package (`#:package datamungedotnet@0.0.1`), restored
  # only from nix-built local feeds, so no network is needed. MSBuild reads RestoreSources from
  # the environment.
  # The package cache is per build of the binding: NuGet never re-extracts a package version it
  # has cached, so a shared cache would keep serving the first build of 0.0.1.
  shellHook = ''
    export RestoreSources="${pkgs.lib.concatStringsSep ";" restoreSources}"
    cache="''${XDG_CACHE_HOME:-$HOME/.cache}/datamunge/report-csharp"
    export DOTNET_CLI_HOME="$cache"
    export NUGET_PACKAGES="$cache/nuget/${builtins.baseNameOf datamungedotnet}"
  '';
}
