#!/usr/bin/env python3
"""Compile and execute every Doxygen example in the native compute/game/sim headers."""
import argparse
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    """Build one executable with isolated examples and check each printed output comment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compiler', default='c++')
    parser.add_argument('--library-dir', type=Path, default=ROOT/'build/debug/tests/_deps/datamunge-build')
    parser.add_argument('--out', type=Path, default=ROOT/'build/simulation-verification')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    snippets = []
    paths=[p for area in ['compute','game','sim'] for p in (ROOT/'include/datamunge'/area).glob('*.hpp')]
    paths.append(ROOT/'include/datamunge/nn/tensor.hpp')
    paths.append(ROOT/'include/datamunge/nn/io.hpp')
    paths.append(ROOT/'include/datamunge/nn/ops.hpp')
    for path in sorted(paths):
        text = path.read_text()
        for match in re.finditer(r'/// @code\n(.*?)/// @endcode', text, re.S):
            code = '\n'.join(re.sub(r'^\s*/// ?', '', line) for line in match[1].splitlines())
            expected = []
            for line in code.splitlines():
                if 'std::print(' in line and '// ' in line:
                    comment = line.split('// ', 1)[1]
                    if re.fullmatch(r'([A-Za-z_][A-Za-z_0-9]*|-?[0-9]+(?:\.[0-9]+)?)', comment):
                        expected.append(comment)
            snippets.append({'source':str(path.relative_to(ROOT)), 'line':text[:match.start()].count('\n')+1,
                             'code':code, 'expected':'\n'.join(expected)+'\n' if expected else None})
    source = '#include <dm/dm.hpp>\n#include <print>\n#include <stdexcept>\n'
    for i, snippet in enumerate(snippets):
        source += f'void example_{i}(){{\n{snippet["code"]}\n}}\n'
    source += 'int main(int argc,char** argv){if(argc!=2)return 2;switch(std::stoi(argv[1])){\n'
    for i in range(len(snippets)):
        source += f'case {i}:example_{i}();break;\n'
    source += 'default:return 3;}return 0;}\n'
    cpp = args.out/'examples.cpp'
    cpp.write_text(source)
    executable = args.out/'examples'
    library = args.library_dir.resolve()
    subprocess.run([args.compiler,'-std=c++23','-O0','-pthread','-I'+str(ROOT/'include'),str(cpp),
                    '-L'+str(library),'-ldatamunge','-Wl,-rpath,'+str(library),'-o',str(executable)],check=True)
    results = []
    for i, snippet in enumerate(snippets):
        run = subprocess.run([str(executable.resolve()),str(i)],capture_output=True,text=True,timeout=20)
        ok = run.returncode == 0 and (snippet['expected'] is None or snippet['expected'] == run.stdout)
        results.append({**snippet,'stdout':run.stdout,'stderr':run.stderr,'returncode':run.returncode,'passed':ok})
        if not ok:
            print(f"FAILED {snippet['source']}:{snippet['line']}: {run.stdout!r} {run.stderr!r}")
    (args.out/'examples.json').write_text(json.dumps(results,indent=2)+'\n')
    print(f'{sum(r["passed"] for r in results)}/{len(results)} header examples passed')
    return 0 if all(r['passed'] for r in results) else 1


if __name__ == '__main__':
    raise SystemExit(main())
