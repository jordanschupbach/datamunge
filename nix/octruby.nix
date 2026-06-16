{
  pkgs ? import <nixpkgs> { },
}:

let
  octra = import ./octra.nix { inherit pkgs; };
in
pkgs.stdenv.mkDerivation rec {
  pname = "octruby";
  version = "0.0.1";

  src = pkgs.lib.cleanSource ../.;

  nativeBuildInputs = [
    pkgs.swig
    pkgs.pkg-config
    pkgs.ruby
    pkgs.gnumake
    pkgs.stdenv.cc
  ];

  buildInputs = [
    octra
  ];

  buildPhase = ''
        runHook preBuild

        export OCTRA_PREFIX="${octra}"

        mkdir -p src/octruby/ext/octruby src/octruby/lib/octruby
        swig -ruby -c++ -Iinclude \
          -o src/octruby/ext/octruby/octruby_wrap.cxx \
          -outdir src/octruby/lib/octruby \
          src/octruby/swig/octruby.i

        if [ ! -f src/octruby/ext/octruby/extconf.rb ]; then
          cat > src/octruby/ext/octruby/extconf.rb <<'EOF'
    require "mkmf"
    require "rbconfig"

    have_pkg_config = find_executable("pkg-config")

    configured = false
    if have_pkg_config
      configured = pkg_config("octra")
    end

    unless configured
      # Ensure mkmf uses a C++ linker for checks against a C++ shared library.
      RbConfig::MAKEFILE_CONFIG["CC"] = RbConfig::MAKEFILE_CONFIG["CXX"] if RbConfig::MAKEFILE_CONFIG["CXX"]

      prefix = ENV["OCTRA_PREFIX"]
      abort "error: Could not find octra via pkg-config; set OCTRA_PREFIX to the octra install prefix" if prefix.to_s.empty?

      include_root = File.join(prefix, "include")
      header_relpath = File.join("octra", "octra.hpp")

      include_candidates = [
        include_root,
        *Dir[File.join(include_root, "*")].select { |p| File.directory?(p) },
      ].uniq

      include_dir = include_candidates.find { |dir| File.exist?(File.join(dir, header_relpath)) }
      abort "error: Could not find #{header_relpath} under #{include_root}" if include_dir.nil?

      lib_root =
        [File.join(prefix, "lib"), File.join(prefix, "lib64")].find { |d| File.directory?(d) } ||
        File.join(prefix, "lib")

      $stderr.puts "octruby: OCTRA_PREFIX=#{prefix}"
      $stderr.puts "octruby: include_dir=#{include_dir}"

      octra_lib_candidates = Dir[
        File.join(lib_root, "**", "liboctra.so"),
        File.join(lib_root, "**", "liboctra.so.*"),
        File.join(lib_root, "**", "liboctra.dylib"),
        File.join(lib_root, "**", "octra.dll"),
      ]
      if octra_lib_candidates.empty?
        abort "error: Could not find liboctra under #{lib_root}"
      end
      lib_dir = File.dirname(octra_lib_candidates.sort.first)
      $stderr.puts "octruby: lib_dir=#{lib_dir}"

      dir_config("octra", include_dir, lib_dir)
      $LDFLAGS << " -Wl,-rpath,#{lib_dir}"

      unless have_library("octra")
        if File.exist?("mkmf.log")
          $stderr.puts "octruby: mkmf.log (last 200 lines):"
          File.readlines("mkmf.log").last(200).each { |l| $stderr.print(l) }
        end
        abort "error: Could not link against liboctra (expected -loctra under #{lib_dir})"
      end
    end

    $CXXFLAGS << " -std=c++17"

    create_makefile("octruby/octruby")
    EOF
        fi

        if [ ! -f src/octruby/lib/octruby.rb ]; then
          cat > src/octruby/lib/octruby.rb <<'EOF'
    dir = File.join(__dir__, "octruby")
    Dir[File.join(dir, "*.rb")].sort.each { |p| require p }
    require File.join(dir, "octruby")

    # Keep the public namespace consistent with other bindings.
    Octra = Octruby unless defined?(Octra)
    EOF
        fi

        pushd src/octruby/ext/octruby
        ruby extconf.rb
        make -j $NIX_BUILD_CORES
        popd

        runHook postBuild
  '';

  installPhase = ''
    runHook preInstall

    outLib="$out/lib"
    mkdir -p "$outLib/octruby"

    cp -v src/octruby/lib/octruby.rb "$outLib/"

    # SWIG's generated Ruby file name depends on the module name; copy whatever it generated.
    shopt -s nullglob
    rubyRbFiles=(src/octruby/lib/octruby/*.rb)
    if (( ''${#rubyRbFiles[@]} )); then
      cp -v "''${rubyRbFiles[@]}" "$outLib/octruby/"
    fi
    shopt -u nullglob

    # mkmf may place the compiled extension under a subdir (e.g. ext/octruby/octruby/octruby.so).
    soPath="$(find src/octruby/ext/octruby -name '*.so' -print -quit)"
    if [ -z "$soPath" ]; then
      echo "Could not find built Ruby extension (.so) under src/octruby/ext/octruby" >&2
      find src/octruby/ext/octruby -maxdepth 3 -type f -print >&2
      exit 1
    fi
    cp -v "$soPath" "$outLib/octruby/octruby.so"

    runHook postInstall
  '';

  meta = with pkgs.lib; {
    description = "Ruby (SWIG) bindings for the octra library.";
    license = licenses.unlicense;
    platforms = platforms.linux;
  };
}
