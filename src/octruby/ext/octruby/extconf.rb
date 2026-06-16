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
