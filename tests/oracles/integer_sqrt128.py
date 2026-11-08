"""Independent arbitrary-precision Python math.isqrt oracle for the full uint128 API."""
import math, random, subprocess, sys
rng=random.Random(106)
values=[0,1,(1<<128)-1]
for i in range(128):values.extend(((1<<i)-1,1<<i,(1<<i)+1))
for r in [1,2,3,(1<<32)-1,1<<32,(1<<64)-1]+[rng.getrandbits(64) for _ in range(3000)]:
 values.extend(x for x in (r*r-1,r*r,r*r+1) if 0<=x<1<<128)
values.extend(rng.getrandbits(128) for _ in range(10000))
result=subprocess.run([sys.argv[1]],input=''.join(f'{x>>64} {x&((1<<64)-1)}\n' for x in values),text=True,capture_output=True,check=True)
actual=[int(s) for s in result.stdout.splitlines()]
assert len(actual)==len(values)
for x,a in zip(values,actual):assert a==math.isqrt(x),(x,a,math.isqrt(x))
print(f'Python math.isqrt: {len(values)} exact full-width and square-boundary checks passed')
