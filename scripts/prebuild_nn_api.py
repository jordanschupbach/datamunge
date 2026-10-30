#!/usr/bin/env python3
"""Generate the reusable NN core API bindings, independently of experiments."""
import argparse
import re
import subprocess
import tempfile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def generate(language, reference=False):
    if language == 'python':
        from nn_migration_status import refresh
        refresh()
    directory = ROOT / ('src/pydatamunge' if language == 'python' else 'prebindings/nn/r/R')
    directory.mkdir(parents=True, exist_ok=True)
    wrapper = ROOT / ('src/nn_python_wrap.cpp' if language == 'python' else 'build/nn-r/api/nn_r_wrap.cpp')
    wrapper.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.swig-', dir=wrapper.parent) as temporary:
        temporary = Path(temporary)
        # examples/bindings holds nn_components.i; examples/nn holds the component headers it binds.
        command = ['swig', '-c++', '-' + language, '-Iinclude', '-Iexamples/bindings', '-Iexamples/nn']
        if reference:
            command += ['-DDATAMUNGE_NN_REFERENCE_EXAMPLES']
        if sys.platform.startswith('linux'):
            command += ['-DSWIGWORDSIZE64']
        if language == 'r':
            command += ['-dll', 'datamungerNN', '-package', 'datamungerNN']
        command += ['-outdir', str(temporary.relative_to(ROOT)), '-o',
                    str((temporary / 'wrapper.cpp').relative_to(ROOT)), 'prebindings/nn/src/nn.i']
        subprocess.run(command, cwd=ROOT, check=True)
        proxy_name = 'nn_native.py' if language == 'python' else 'datamungerNN.R'
        for source, destination in [(temporary / 'wrapper.cpp', wrapper),
                                    (temporary / proxy_name, directory / ('nn_native.R' if language == 'r' else proxy_name))]:
            text = source.read_text().replace(str(temporary.relative_to(ROOT)), 'src')
            if language == 'r' and source.name == proxy_name:
                # Match the R package generator's reserved S4 slot quoting.
                text = re.sub(r'^(\s*)in(\s*=)', r'\1`in`\2', text, flags=re.M)
                text = re.sub(r'([@$])in\b', r'\1`in`', text)
            text = '\n'.join(line.rstrip() for line in text.splitlines()) + '\n'
            if not destination.exists() or destination.read_text() != text:
                destination.write_text(text)
    return wrapper


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('language', choices=['python', 'r'])
    generate(parser.parse_args().language)
