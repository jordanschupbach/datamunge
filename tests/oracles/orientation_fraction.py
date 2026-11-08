"""Independent standard-library exact-rational controls for the native predicate CLI."""
from fractions import Fraction
from pathlib import Path
import math,random,struct,subprocess,sys,tempfile
rng=random.Random(708921)
cases=[]
def add(values):
    a,b,c=(tuple(Fraction(x) for x in values[k:k+2]) for k in (0,2,4))
    determinant=(b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
    cases.append((values,(determinant>0)-(determinant<0)))
def random_float():
    while True:
        v=struct.unpack('=d',struct.pack('=Q',rng.getrandbits(64)))[0]
        if math.isfinite(v):return v
for k in range(12000):add([random_float() for _ in range(6)])
for k in range(8000):
    scale=math.ldexp(1.,rng.randrange(-1000,950))
    a=rng.randrange(-32,33)*scale;b=rng.randrange(-32,33)*scale
    x=rng.randrange(-32,33)*scale;y=rng.randrange(-32,33)*scale
    # Collinear triples and one-working-ULP perturbations, across exponent scales.
    values=[a,b,x,y,(a+x)/2,(b+y)/2]
    if k%3:values[4+(k%2)]=math.nextafter(values[4+(k%2)],math.inf if k%2 else -math.inf)
    add(values)
with tempfile.TemporaryDirectory(prefix='orientation-fraction-') as work:
    path=Path(work)/'controls.bin'
    with path.open('wb') as stream:
        for values,sign in cases:stream.write(struct.pack('=6db',*values,sign))
    subprocess.run([str(Path(sys.argv[1]).resolve()),str(path),str(len(cases))],check=True)
