"""Independent arbitrary-precision product/pow oracle for unsigned modular arithmetic."""
from pathlib import Path
import random,subprocess,sys
rng=random.Random(19761976);cases=[];expected=[];maximum=(1<<64)-1
for k in range(10000):
    a=rng.getrandbits(64);b=rng.getrandbits(64);m=rng.getrandbits(64)
    if k%127==0:m=0
    elif k%127==1:m=1
    elif k%127==2:m=maximum
    elif k%127==3:m=18446744073709551557
    if k<64:b=1<<k
    elif k<128:b=maximum
    elif k<192:b=0
    if k%61==0:a=maximum
    for op in ('mul','pow'):
        cases.append(f'{op} {a} {b} {m}\n')
        expected.append('invalid' if not m else str((a*b)%m if op=='mul' else pow(a,b,m)))
run=subprocess.run([str(Path(sys.argv[1]).resolve())],input=''.join(cases),text=True,capture_output=True,check=True)
actual=run.stdout.splitlines();assert len(actual)==len(expected)==20000,(len(actual),len(expected))
for k,(a,e) in enumerate(zip(actual,expected)):
    assert a==e,(k,cases[k].strip(),a,e)
print('Verified 20000 Python arbitrary-precision controls: 10000 products and 10000 powers, including zero and full-width moduli')
