use strict;
use warnings;

use ExtUtils::MakeMaker;
use Config;
use FindBin;
use File::Spec;

my $datamunge_root = File::Spec->catdir($FindBin::Bin, '..', '..');
my $include_dir = File::Spec->catdir($datamunge_root, 'include');
my $lib_dir = File::Spec->catdir($datamunge_root, 'build');

WriteMakefile(
  NAME         => 'Datamunge',
  VERSION      => '0.0.1',
  PM           => { 'lib/Datamunge.pm' => '$(INST_LIBDIR)/Datamunge.pm' },
  OBJECT       => 'Datamunge_wrap$(OBJ_EXT)',
  INC          => "-I$include_dir",
  LIBS         => [ "-L$lib_dir -ldatamunge" ],
  dynamic_lib  => { OTHERLDFLAGS => $Config{lddlflags} },
  clean        => { FILES => 'Datamunge_wrap.* Datamunge.pm lib/Datamunge.pm' },
);
