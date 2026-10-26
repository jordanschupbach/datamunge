#!/usr/bin/env python3
"""Build the reusable R NN API; compile no example entry points or model helpers."""
import os
import shutil
import subprocess
from pathlib import Path
from prebuild_nn_api import ROOT, generate
from prebuild_nn_examples import write_changed


def copy_changed(source, destination):
    destination = Path(destination)
    if not destination.exists() or Path(source).read_bytes() != destination.read_bytes():
        shutil.copy2(source, destination)
    return str(destination)


def main():
    reference_examples = os.environ.get('DATAMUNGE_BUILD_NN_EXAMPLES') == '1'
    wrapper = generate('r', reference_examples)
    stage = ROOT / ('build/nn-r-core/source/datamungerNN' if reference_examples else 'build/nn-r-api/source/datamungerNN')
    stage.mkdir(parents=True, exist_ok=True)
    shutil.copytree(ROOT / 'prebindings/nn/r', stage, dirs_exist_ok=True, copy_function=copy_changed)
    ignore = shutil.ignore_patterns('*.o', '*.so', '*.dll', '*.dylib')
    for source, destination in [(ROOT / 'include', stage / 'inst/include'),
                                 (ROOT / 'src/datamunge', stage / 'src/datamunge')]:
        shutil.copytree(source, destination, dirs_exist_ok=True, copy_function=copy_changed, ignore=ignore)
    copy_changed(wrapper, stage / 'src/nn_r_wrap.cpp')
    if reference_examples:
        from prebuild_nn_examples import generate as generate_examples, GENERATED
        generate_examples(False)
        for source in GENERATED.glob('*.cpp'):
            if source.name != 'nn_examples_wrap.cpp':
                copy_changed(source, stage / 'src' / source.name)
        for folder, target in [(ROOT / 'examples/nn/common', stage / 'inst/examples/common'),
                               (ROOT / 'examples/bindings/r/R', stage / 'R'),
                               (ROOT / 'examples/bindings/r/man', stage / 'man')]:
            shutil.copytree(folder, target, dirs_exist_ok=True, copy_function=copy_changed,
                            ignore=shutil.ignore_patterns('datamungerNN.R'))
        (stage / 'inst/bindings').mkdir(parents=True,exist_ok=True)
        copy_changed(ROOT / 'examples/bindings/nn_examples.hpp',stage / 'inst/bindings/nn_examples.hpp')
        with (stage / 'NAMESPACE').open('a') as namespace:
            namespace.write('export(nn_example_run)\nexport(nn_example_names)\n')

    sources = sorted(str(p.relative_to(stage / 'src').with_suffix('.o')) for p in (stage / 'src').rglob('*.cpp'))
    common = 'CXX_STD = CXX23\nPKG_CPPFLAGS = -I../inst/include -I../inst/bindings -I../inst/examples\nPKG_LIBS = -pthread\n'
    darwin = '''ifeq ($(shell uname -s),Darwin)
PKG_CPPFLAGS += -DDATAMUNGE_HAVE_CBLAS=1 -DDATAMUNGE_HAVE_LAPACK=1 -DACCELERATE_NEW_LAPACK=1
PKG_LIBS += -framework Accelerate
endif
'''
    objects = 'OBJECTS = ' + ' '.join(sources) + '\nall: $(SHLIB)\nHEADERS := $(shell find ../inst -name "*.hpp")\n$(OBJECTS): $(HEADERS)\n'
    write_changed(stage / 'src/Makevars', common + darwin + objects)
    write_changed(stage / 'src/Makevars.win', common + objects)
    write_changed(stage / '.Rbuildignore', r'\.(o|so|dll|dylib)$' + '\n')
    library = ROOT / 'build/r/library'
    library.mkdir(parents=True, exist_ok=True)
    # Reuse core objects from the existing reference package on the first build.
    # Only when newer than the corresponding source and every current header.
    reference = ROOT / 'build/nn-r/source/datamungerNN/src'
    latest_header = max(p.stat().st_mtime_ns for p in (ROOT / 'include').rglob('*.hpp'))
    for source in (stage / 'src/datamunge').rglob('*.cpp'):
        target = source.with_suffix('.o')
        previous = reference / source.relative_to(stage / 'src').with_suffix('.o')
        if not target.exists() and previous.exists() and previous.stat().st_mtime_ns >= max(source.stat().st_mtime_ns, latest_header):
            shutil.copy2(previous, target)
    subprocess.run(['R', 'CMD', 'INSTALL', '-l', str(library), str(stage)], cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
