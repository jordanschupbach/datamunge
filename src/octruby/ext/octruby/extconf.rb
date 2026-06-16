require "mkmf"
require "rbconfig"

have_pkg_config = find_executable("pkg-config")

configured = false
if have_pkg_config
  configured = pkg_config("datamunge")
end

unless configured
  # Ensure mkmf uses a C++ linker for checks against a C++ shared library.
  RbConfig::MAKEFILE_CONFIG["CC"] = RbConfig::MAKEFILE_CONFIG["CXX"] if RbConfig::MAKEFILE_CONFIG["CXX"]

  prefix = ENV["DATAMUNGE_PREFIX"]
  abort "error: Could not find datamunge via pkg-config; set DATAMUNGE_PREFIX to the datamunge install prefix" if prefix.to_s.empty?

  include_root = File.join(prefix, "include")
  header_relpath = File.join("datamunge", "datamunge.hpp")

  include_candidates = [
    include_root,
    *Dir[File.join(include_root, "*")].select { |p| File.directory?(p) },
  ].uniq

  include_dir = include_candidates.find { |dir| File.exist?(File.join(dir, header_relpath)) }
  abort "error: Could not find #{header_relpath} under #{include_root}" if include_dir.nil?

  lib_root =
    [File.join(prefix, "lib"), File.join(prefix, "lib64")].find { |d| File.directory?(d) } ||
    File.join(prefix, "lib")

  $stderr.puts "octruby: DATAMUNGE_PREFIX=#{prefix}"
  $stderr.puts "octruby: include_dir=#{include_dir}"

  datamunge_lib_candidates = Dir[
    File.join(lib_root, "**", "libdatamunge.so"),
    File.join(lib_root, "**", "libdatamunge.so.*"),
    File.join(lib_root, "**", "libdatamunge.dylib"),
    File.join(lib_root, "**", "datamunge.dll"),
  ]
  if datamunge_lib_candidates.empty?
    abort "error: Could not find libdatamunge under #{lib_root}"
  end
  lib_dir = File.dirname(datamunge_lib_candidates.sort.first)
  $stderr.puts "octruby: lib_dir=#{lib_dir}"

  dir_config("datamunge", include_dir, lib_dir)
  $LDFLAGS << " -Wl,-rpath,#{lib_dir}"

  unless have_library("datamunge")
    if File.exist?("mkmf.log")
      $stderr.puts "octruby: mkmf.log (last 200 lines):"
      File.readlines("mkmf.log").last(200).each { |l| $stderr.print(l) }
    end
    abort "error: Could not link against libdatamunge (expected -ldatamunge under #{lib_dir})"
  end
end

$CXXFLAGS << " -std=c++17"

create_makefile("octruby/octruby")
