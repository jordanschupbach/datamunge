#!/usr/bin/env python3
"""Stage and install the optional standard R package of canonical NN experiments."""
import shutil
import subprocess
import tempfile
from pathlib import Path

from prebuild_nn_examples import ROOT, GENERATED, generate, write_changed


def main():
    generate(False)
    stage = ROOT / "build/nn-r/source/datamungerNN"
    stage.mkdir(parents=True, exist_ok=True)
    shutil.copytree(ROOT / "examples/bindings/r", stage, dirs_exist_ok=True)
    # Exclude native build artifacts while preserving unchanged source mtimes.
    def copy(source, target):
        target = Path(target)
        if not target.exists() or Path(source).read_bytes() != target.read_bytes():
            shutil.copy2(source, target)
        return str(target)
    ignore = shutil.ignore_patterns("*.o", "*.so", "*.dll", "*.dylib", "__pycache__")
    for source, target in [
        (ROOT / "include", stage / "inst/include"),
        (ROOT / "src/datamunge", stage / "src/datamunge"),
        (ROOT / "examples/nn/common", stage / "inst/examples/common"),
        (ROOT / "examples/bindings", stage / "inst/bindings"),
    ]:
        if source.name == "bindings":
            target.mkdir(parents=True, exist_ok=True)
            copy(source / "nn_examples.hpp", target / "nn_examples.hpp")
        else:
            shutil.copytree(source, target, dirs_exist_ok=True, copy_function=copy, ignore=ignore)
    for source in GENERATED.glob("*.cpp"):
        if source.name != "nn_examples_wrap.cpp":
            copy(source, stage / "src" / source.name)
    with tempfile.TemporaryDirectory(prefix=".swig-", dir=stage) as temporary:
        temporary = Path(temporary)
        subprocess.run([
            "swig", "-c++", "-r", "-dll", "datamungerNN", "-package", "datamungerNN", "-Iexamples/bindings",
            "-outdir", str(temporary.relative_to(ROOT)),
            "-o", str((temporary / "nn_examples_wrap.cpp").relative_to(ROOT)),
            "examples/bindings/nn_examples.i",
        ], cwd=ROOT, check=True)
        for filename, destination in [
            ("datamungerNN.R", stage / "R/nn_examples.R"),
            ("nn_examples_wrap.cpp", stage / "src/nn_examples_wrap.cpp"),
        ]:
            text = (temporary / filename).read_text().replace(
                str(temporary.relative_to(ROOT)), "src"
            )
            write_changed(destination, text)
    sources = sorted(p.relative_to(stage / "src").as_posix() for p in (stage / "src").rglob("*.cpp"))
    makevars = """CXX_STD = CXX23
PKG_CPPFLAGS = -I../inst/include -I../inst/bindings -I../inst/examples
PKG_LIBS = -pthread
ifeq ($(shell uname -s),Darwin)
PKG_CPPFLAGS += -DDATAMUNGE_HAVE_CBLAS=1 -DDATAMUNGE_HAVE_LAPACK=1 -DACCELERATE_NEW_LAPACK=1
PKG_LIBS += -framework Accelerate
endif
""" + "OBJECTS = " + " ".join(s[:-4] + ".o" for s in sources) + "\n"
    makevars += "all: $(SHLIB)\n"
    makevars += "HEADERS := $(shell find ../inst -type f -name '*.hpp')\n"
    makevars += "$(OBJECTS): $(HEADERS)\n"
    write_changed(stage / "src/Makevars", makevars)
    write_changed(stage / "src/Makevars.win", makevars.split('ifeq')[0] + "OBJECTS = " + " ".join(s[:-4] + ".o" for s in sources) + "\n")
    for license_file in ROOT.glob("LICENSE*"):
        if license_file.is_file():
            copy(license_file, stage / license_file.name)
    library = ROOT / "build/r/library"
    library.mkdir(parents=True, exist_ok=True)
    subprocess.run(["R", "CMD", "INSTALL", "-l", str(library), str(stage)], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
