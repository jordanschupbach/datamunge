"""Independent exact translated-determinant Fraction oracle across binary64 exponents."""
from fractions import Fraction
from pathlib import Path
import math,random,struct,subprocess,sys,tempfile
rng=random.Random(1981167);cases=[]
def add(v):
    a,b,c,d=(tuple(Fraction(x) for x in v[k:k+2]) for k in (0,2,4,6))
    ax,ay=a[0]-d[0],a[1]-d[1];bx,by=b[0]-d[0],b[1]-d[1];cx,cy=c[0]-d[0],c[1]-d[1]
    det=(ax*ax+ay*ay)*(bx*cy-cx*by)-(bx*bx+by*by)*(ax*cy-cx*ay)+(cx*cx+cy*cy)*(ax*by-bx*ay)
    cases.append((v,(det>0)-(det<0)))
def random_float():
    while True:
        v=struct.unpack('=d',struct.pack('=Q',rng.getrandbits(64)))[0]
        if math.isfinite(v):return v
for _ in range(6000):add([random_float() for _ in range(8)])
for k in range(4000):
    s=math.ldexp(1.,rng.randrange(-1074,1020));x=rng.randrange(-4,5)*s;y=rng.randrange(-4,5)*s
    v=[x-s,y,x,y-s,x+s,y,x,y+s]
    if k%3:v[6+(k%2)]=math.nextafter(v[6+(k%2)],math.inf if k%2 else -math.inf)
    add(v)
with tempfile.TemporaryDirectory(prefix='incircle-fraction-') as work:
    path=Path(work)/'controls.bin'
    with path.open('wb') as stream:
        for v,sign in cases:stream.write(struct.pack('=8db',*v,sign))
    subprocess.run([str(Path(sys.argv[1]).resolve()),str(path),str(len(cases))],check=True)
