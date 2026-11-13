
{
  'targets': [
    {
      'target_name': 'datamungejs',
      'sources': [ 'src/datamunge_js_wrap.cpp', 'src/datamunge_js_wrap_1.cpp', 'src/datamunge_js_wrap_10.cpp', 'src/datamunge_js_wrap_11.cpp', 'src/datamunge_js_wrap_12.cpp', 'src/datamunge_js_wrap_13.cpp', 'src/datamunge_js_wrap_14.cpp', 'src/datamunge_js_wrap_15.cpp', 'src/datamunge_js_wrap_2.cpp', 'src/datamunge_js_wrap_3.cpp', 'src/datamunge_js_wrap_4.cpp', 'src/datamunge_js_wrap_5.cpp', 'src/datamunge_js_wrap_6.cpp', 'src/datamunge_js_wrap_7.cpp', 'src/datamunge_js_wrap_8.cpp', 'src/datamunge_js_wrap_9.cpp' ],
      'include_dirs': [
        "/nix/store/dxkvnjhzsh28pvjan16v9nrn7p2hii5m-datamunge-0.0.1/include/datamunge-0.0.1",
        '/nix/store/yl9p47yg3qzw1xf9b3pnfav0mgy1qik9-libxml2-2.15.1-dev/include/libxml2'
      ],
      'cflags': ['-std=c++23'],
      'cflags!': [ '-fno-exceptions', '-fno-rtti'], # ,
      'cflags_cc': ['-std=c++23'],
      'cflags_cc!': [ '-fno-exceptions', '-fno-rtti'], # '-fno-rtti'
      # 'GCC_ENABLE_CPP_RTTI': 'NO',              # -fno-rtti   ???
      'xcode_settings': {
        'GCC_ENABLE_CPP_EXCEPTIONS': 'YES',
        'CLANG_CXX_LIBRARY': 'libc++',
        # std::filesystem (used in plot.cpp) is only available on macOS 10.15+.
        'MACOSX_DEPLOYMENT_TARGET': '10.15',
        'GCC_ENABLE_CPP_RTTI': 'YES',
        'OTHER_LDFLAGS': [ '-Wl,-rpath,/nix/store/dxkvnjhzsh28pvjan16v9nrn7p2hii5m-datamunge-0.0.1/lib/datamunge-0.0.1' ]
      },
      'ldflags': [ '-Wl,-rpath,/nix/store/dxkvnjhzsh28pvjan16v9nrn7p2hii5m-datamunge-0.0.1/lib/datamunge-0.0.1' ],
      'libraries' : [ '-ldatamunge', '-lxml2' ], #   '-lblas', '-llapack', '-llapacke', '-lcblas'
      'library_dirs' : [ '/nix/store/dxkvnjhzsh28pvjan16v9nrn7p2hii5m-datamunge-0.0.1/lib/datamunge-0.0.1', '/usr/lib' ],
      'msvs_settings': {
        'VCCLCompilerTool': { 'ExceptionHandling': 1 },
      }
    }
  ]
}