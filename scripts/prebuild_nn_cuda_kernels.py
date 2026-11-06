#!/usr/bin/env python3
"""Embed the native CUDA kernels as PTX headers, so library builds need no nvcc.

Each entry of KERNELS compiles one .cu source to PTX and writes one generated header holding
it as a raw string: the nn inference kernels (shaders/nn/inference.cu) and the compute GEMM
kernels behind Context::matmul (shaders/compute/blas.cu), and the CAD ray tracer
(shaders/cad/render.cu, including its shared host/device storage layout).
"""
import argparse
import hashlib
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
# Lowest virtual architecture CUDA 12 still targets; the driver JIT-compiles it for newer GPUs.
ARCH = 'compute_52'
COMMON_FLAGS = ['-ptx', f'-arch={ARCH}', '-O3', '-Wno-deprecated-gpu-targets']

# (source, output header, namespace, variable name, description, extra nvcc flags)
KERNELS = [
    (ROOT/'shaders/cad/render.cu', ROOT/'src/datamunge/cad/detail/render_ptx.hpp',
     'datamunge::cad::detail', 'render_ptx', 'CAD CUDA ray tracing PTX', ['-fmad=false']),
    # Floating-point contraction disabled to stay close to the CPU kernels.
    (ROOT/'shaders/nn/inference.cu', ROOT/'src/datamunge/nn/device_kernels_ptx.hpp',
     'datamunge::nn::detail', 'inference_ptx', 'CUDA nn inference PTX', ['-fmad=false']),
    # GEMM keeps fused multiply-add: it is the throughput path and one FMA per product is
    # at least as accurate as a rounded multiply followed by a rounded add.
    (ROOT/'shaders/compute/blas.cu', ROOT/'src/datamunge/compute/cuda_blas_ptx.hpp',
     'datamunge::compute::detail', 'blas_ptx', 'CUDA GEMM PTX for Context::matmul', []),
]


def build(nvcc, source, output, namespace, variable, description, flags):
    """Compile deterministic PTX with a recorded source hash and no machine paths."""
    with tempfile.TemporaryDirectory(prefix='cuda-kernels-') as directory:
        ptx_file = Path(directory)/(source.stem + '.ptx')
        subprocess.run([nvcc, *COMMON_FLAGS, *flags, '-o', str(ptx_file), source.name],
                       cwd=source.parent, check=True)
        ptx = ptx_file.read_text()
    # nvcc stamps its build banner and the temp-file name into comments; drop them.
    ptx = '\n'.join(line for line in ptx.splitlines() if not line.startswith('//')).strip() + '\n'
    if ')ptx"' in ptx or '.entry' not in ptx:
        raise RuntimeError(f'nvcc did not produce embeddable PTX for {source.name}')
    entries = re.findall(r'\.entry\s+(\w+)', ptx)
    layout = ROOT/'src/datamunge/cad/detail/cuda_types.hpp'
    inputs = source.read_bytes() + (layout.read_bytes() if source.parent.name == 'cad' else b'')
    digest = hashlib.sha256(inputs).hexdigest()
    relative = source.relative_to(ROOT).as_posix()
    output.write_text('\n'.join([
        '#pragma once',
        f'/// @file {output.name}',
        f'/// @brief Generated {description}; regenerate with '
        'scripts/prebuild_nn_cuda_kernels.py.',
        f'namespace {namespace} {{',
        f'/// @brief PTX for {relative} ({ARCH}), ' + ('source+layout' if source.parent.name == 'cad' else 'source') + f' SHA-256 {digest}.',
        '/// Entries: ' + ', '.join(entries) + '.',
        f'inline constexpr const char* {variable} = R"ptx(' + ptx + ')ptx";',
        f'}} // namespace {namespace}',
        '']))
    print(f'{source.stem}: {len(entries)} entries, {len(ptx)} bytes of PTX -> '
          f'{output.relative_to(ROOT).as_posix()}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nvcc', default=shutil.which('nvcc'),
                        help='nvcc to use (default: from PATH, e.g. `nix develop .#cuda`)')
    parser.add_argument('--only', choices=[k[0].stem for k in KERNELS],
                        help='regenerate just this source (default: all)')
    args = parser.parse_args()
    if not args.nvcc:
        parser.error('nvcc not found; run inside `nix develop .#cuda` or pass --nvcc')
    for source, output, namespace, variable, description, flags in KERNELS:
        if args.only and source.stem != args.only:
            continue
        build(args.nvcc, source, output, namespace, variable, description, flags)


if __name__ == '__main__':
    main()
