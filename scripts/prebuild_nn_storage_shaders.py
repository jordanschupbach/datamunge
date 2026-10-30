#!/usr/bin/env python3
"""Embed the native Vulkan tensor storage shader without requiring a runtime SDK."""
import argparse
import hashlib
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
from compile_compute_fixture import shaderc_compile

ROOT = Path(__file__).resolve().parents[1]


def main():
    """Compile deterministic SPIR-V with a recorded source hash and no machine paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shaderc-library', type=Path)
    args = parser.parse_args()
    source = ROOT/'shaders/nn/storage.comp'
    if args.shaderc_library:
        code = shaderc_compile(args.shaderc_library, source.read_bytes(), source.name)
    else:
        compiler = shutil.which('glslc') or shutil.which('glslangValidator')
        if not compiler:
            parser.error('install glslc/glslangValidator or supply --shaderc-library')
        with tempfile.TemporaryDirectory(prefix='nn-storage-shader-') as directory:
            output = Path(directory)/'storage.spv'
            command = [compiler, '-V'] if Path(compiler).name == 'glslangValidator' else [compiler]
            subprocess.run(command+['-o', str(output), source.name], cwd=source.parent, check=True)
            code = output.read_bytes()
    if len(code)%4 or code[:4] != b'\x03\x02\x23\x07':
        raise RuntimeError('compiler did not produce SPIR-V')
    words = struct.unpack('<'+'I'*(len(code)//4), code)
    lines = ['#pragma once', '/// @file storage_shader.hpp',
             '/// @brief Generated native tensor SPIR-V; regenerate with scripts/prebuild_nn_storage_shaders.py.',
             '#include <cstdint>', 'namespace datamunge::nn::detail {',
             '/// @brief Bit-preserving storage kernel, source SHA-256 '+hashlib.sha256(source.read_bytes()).hexdigest()+'.',
             'inline constexpr std::uint32_t storage_spirv[]={']
    lines += [' '+','.join(hex(w)+'u' for w in words[i:i+8])+',' for i in range(0,len(words),8)]
    lines += ['};', '} // namespace datamunge::nn::detail', '']
    (ROOT/'src/datamunge/nn/storage_shader.hpp').write_text('\n'.join(lines))
    print(f'storage: {len(words)} words')


if __name__ == '__main__':
    main()
