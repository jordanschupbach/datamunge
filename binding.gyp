
{
  'targets': [
    {
      'target_name': 'octrajs',
      'sources': [ 'src/octra/octra.cpp', 'src/octra/octra_c.cpp', 'src/octra_js_wrap.cpp' ],
      'include_dirs': [
        "include",
        "<!@(node -p \"require('node-addon-api').include\")",
        '/nix/store/yl9p47yg3qzw1xf9b3pnfav0mgy1qik9-libxml2-2.15.1-dev/include/libxml2'
      ],
      'dependencies': ["<!(node -p \"require('node-addon-api').gyp\")"],
      'cflags': ['-std=c++23'],
      'cflags!': [ '-fno-exceptions', '-fno-rtti'], # ,
      'cflags_cc': ['-std=c++23'],
      'cflags_cc!': [ '-fno-exceptions', '-fno-rtti'], # '-fno-rtti'
      # 'GCC_ENABLE_CPP_RTTI': 'NO',              # -fno-rtti   ???
      'xcode_settings': {
        'GCC_ENABLE_CPP_EXCEPTIONS': 'YES',
        'CLANG_CXX_LIBRARY': 'libc++',
        'MACOSX_DEPLOYMENT_TARGET': '10.14',
        'GCC_ENABLE_CPP_RTTI': 'YES'
      },
      'libraries' : [ '-lxml2' ], #   '-lblas', '-llapack', '-llapacke', '-lcblas'
      'library_dirs' : [ '/usr/lib' ],
      'msvs_settings': {
        'VCCLCompilerTool': { 'ExceptionHandling': 1 },
      }
    }
  ]
}