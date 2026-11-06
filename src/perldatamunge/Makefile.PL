use strict;
use warnings;

use ExtUtils::MakeMaker;
use Config;
use FindBin;
use File::Spec;

# Defaults point at the in-repo build (`just build-perl`); packagers (nix/datamungeperl.nix)
# override them with DATAMUNGE_INCLUDE_DIR / DATAMUNGE_LIB_DIR.
my $datamunge_root = File::Spec->catdir($FindBin::Bin, '..', '..');
my $include_dir = $ENV{DATAMUNGE_INCLUDE_DIR} || File::Spec->catdir($datamunge_root, 'include');
my $lib_dir = $ENV{DATAMUNGE_LIB_DIR} || File::Spec->catdir($datamunge_root, 'build');
my $cxx = $ENV{CXX} || 'c++';

WriteMakefile(
  NAME         => 'Datamunge',
  VERSION      => '0.0.1',
  PM           => { 'lib/Datamunge.pm' => '$(INST_LIBDIR)/Datamunge.pm' },
  OBJECT       => 'Datamunge_wrap$(OBJ_EXT)',
  CC           => $cxx,
  LD           => $cxx,
  # The core library needs C++20 or newer.
  CCFLAGS      => "$Config{ccflags} -std=c++23",
  INC          => "-I$include_dir",
  LIBS         => [ "-L$lib_dir -ldatamunge" ],
  dynamic_lib  => { OTHERLDFLAGS => "$Config{lddlflags} -Wl,-rpath,$lib_dir" },
  clean        => { FILES => 'Datamunge_wrap.* Datamunge.pm lib/Datamunge.pm' },
);
