# C# (.NET) binding as a local NuGet package: datamungedotnet.<version>.nupkg holds the managed
# assembly (SWIG-generated C#) and the native libdatamunge_csharp.so under runtimes/<rid>/native/,
# where the .NET host finds it at run time. The package is in $out/share/nuget, a folder usable
# as a NuGet source (see nix/report/csharp.nix).
#
# The C# sources are regenerated here with SWIG from src/datamungedotnet/swig/datamungedotnet.i
# rather than taken from the copies committed under src/datamungedotnet/, so they always match
# the core headers.
{
  pkgs,
  datamunge,
  swig ? pkgs.swig,
  dotnetSdk ? pkgs.dotnetCorePackages.sdk_10_0-bin,
}:
let
  inherit (pkgs.stdenv.hostPlatform) isDarwin isAarch64;
  rid = (if isDarwin then "osx" else "linux") + (if isAarch64 then "-arm64" else "-x64");
  libName = "libdatamunge_csharp${pkgs.stdenv.hostPlatform.extensions.sharedLibrary}";
in
pkgs.stdenv.mkDerivation rec {
  pname = "datamungedotnet";
  version = "0.0.1";

  src = import ./source.nix { lib = pkgs.lib; };

  nativeBuildInputs = [
    pkgs.pkg-config
    swig
    dotnetSdk
  ];
  buildInputs = [ datamunge ];

  env = {
    DOTNET_CLI_TELEMETRY_OPTOUT = "1";
    DOTNET_NOLOGO = "1";
    DOTNET_SKIP_FIRST_TIME_EXPERIENCE = "1";
    DOTNET_SYSTEM_GLOBALIZATION_INVARIANT = "1";
  };

  buildPhase = ''
    runHook preBuild

    gen="$TMPDIR/datamungedotnet"
    mkdir -p "$gen/native"
    (cd include && swig -doxygen -c++ -csharp -dllimport datamunge_csharp \
      -outdir "$gen" -o "$gen/native/datamunge_csharp_wrap.cpp" \
      ../src/datamungedotnet/swig/datamungedotnet.i)

    libdir="$(pkg-config --variable=libdir datamunge)"
    $CXX -std=c++23 -O2 -shared -fPIC $(pkg-config --cflags datamunge) \
      -o "$gen/native/${libName}" "$gen/native/datamunge_csharp_wrap.cpp" \
      $(pkg-config --libs datamunge) -Wl,-rpath,"$libdir"

    cat > "$gen/datamungedotnet.csproj" <<'EOF'
    <Project Sdk="Microsoft.NET.Sdk">
      <PropertyGroup>
        <TargetFramework>net10.0</TargetFramework>
        <AssemblyName>datamungedotnet</AssemblyName>
        <PackageId>datamungedotnet</PackageId>
        <Version>${version}</Version>
        <Description>C# (SWIG) bindings for the datamunge library.</Description>
        <Nullable>disable</Nullable>
        <GenerateDocumentationFile>true</GenerateDocumentationFile>
        <!-- CS8981: the SWIG module class is named `datamunge`; the rest: SWIG doc-comment gaps -->
        <NoWarn>$(NoWarn);CS8981;CS1570;CS1572;CS1573;CS1574;CS1587;CS1591;CS0108;CS0114</NoWarn>
      </PropertyGroup>
      <ItemGroup>
        <None Include="native/${libName}" Pack="true" PackagePath="runtimes/${rid}/native/" />
      </ItemGroup>
    </Project>
    EOF

    export HOME="$TMPDIR" NUGET_PACKAGES="$TMPDIR/nuget-packages"
    mkdir -p "$TMPDIR/no-sources"
    dotnet pack "$gen/datamungedotnet.csproj" -c Release -o "$gen/nupkg" \
      -p:RestoreSources="$TMPDIR/no-sources"

    runHook postBuild
  '';

  installPhase = ''
    runHook preInstall
    install -Dm644 -t "$out/share/nuget" "$gen"/nupkg/*.nupkg
    install -Dm755 -t "$out/lib" "$gen/native/${libName}"
    install -Dm644 -t "$out/lib/dotnet" "$gen"/bin/Release/net10.0/datamungedotnet.{dll,xml}
    runHook postInstall
  '';

  meta = {
    description = "C# (SWIG) bindings for the datamunge library, as a local NuGet package.";
    license = pkgs.lib.licenses.unlicense;
    platforms = pkgs.lib.platforms.unix;
  };
}
