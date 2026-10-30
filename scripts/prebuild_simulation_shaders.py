#!/usr/bin/env python3
"""Compile and embed native offscreen Vulkan shaders; consume without an SDK at runtime."""
import argparse
import hashlib
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
from compile_compute_fixture import shaderc_compile

ROOT=Path(__file__).resolve().parents[1]


def main():
    """Regenerate deterministic shader arrays with source hashes and no local paths."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shaderc-library',type=Path)
    args=parser.parse_args()
    lines=['#pragma once','/// @file render_shaders.hpp',
           '/// @brief Generated native SPIR-V; regenerate with scripts/prebuild_simulation_shaders.py.',
           '#include <cstdint>','namespace datamunge::game::detail {']
    for name in ['scene','post']:
        source_path=ROOT/'shaders/simulation'/f'{name}.comp'
        source=source_path.read_bytes()
        if args.shaderc_library:
            code=shaderc_compile(args.shaderc_library,source,name+'.comp')
        else:
            compiler=shutil.which('glslc') or shutil.which('glslangValidator')
            if not compiler:parser.error('install glslc/glslangValidator or supply --shaderc-library')
            with tempfile.TemporaryDirectory(prefix='simulation-shader-') as directory:
                target=Path(directory)/(name+'.spv')
                command=[compiler,'-V'] if Path(compiler).name=='glslangValidator' else [compiler]
                subprocess.run(command+['-o',str(target),source_path.name],cwd=source_path.parent,check=True)
                code=target.read_bytes()
        if len(code)%4 or code[:4]!=b'\x03\x02\x23\x07':raise RuntimeError('compiler did not produce SPIR-V')
        words=struct.unpack('<'+'I'*(len(code)//4),code)
        lines += [f'/// @brief Compiled {name}.comp, source SHA-256 {hashlib.sha256(source).hexdigest()}.',
                  f'inline constexpr std::uint32_t {name}_spirv[]={{']
        lines += [' '+','.join(hex(word)+'u' for word in words[i:i+8])+',' for i in range(0,len(words),8)]
        lines += ['};']
        print(f'{name}: {len(words)} words')
    lines+=['} // namespace datamunge::game::detail','']
    (ROOT/'src/datamunge/game/render_shaders.hpp').write_text('\n'.join(lines))


if __name__=='__main__':main()
