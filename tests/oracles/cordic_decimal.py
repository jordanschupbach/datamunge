"""Independent Decimal/Machin/Taylor controls for the native floating CORDIC model."""
from decimal import Decimal, localcontext, ROUND_HALF_EVEN
from pathlib import Path
import math,random,struct,subprocess,sys,tempfile

def atan_inverse(n):
    x=Decimal(1)/n;s=term=x;k=0
    while abs(term)>Decimal('1e-420'):
        k+=1;term*=-(x*x);s+=term/(2*k+1)
    return s

def sincos(x,pi):
    x=x%(2*pi)
    if x>pi:x-=2*pi
    if x<-pi:x+=2*pi
    xx=x*x;st=ss=x;ct=cc=Decimal(1);k=0
    while abs(st)>Decimal('1e-410') or abs(ct)>Decimal('1e-410'):
        k+=1;st*= -xx/((2*k)*(2*k+1));ct*= -xx/((2*k-1)*(2*k))
        ss+=st;cc+=ct
    return float(ss),float(cc)

with localcontext() as context:
    context.prec=430
    pi=16*atan_inverse(5)-4*atan_inverse(239)
    pi_double=float.fromhex('0x1.921fb54442d18p+1')
    period=Decimal.from_float(2*pi_double)
    angles=[0.,-0.,.5,1.,2.5,-1.2,pi_double,-pi_double,pi_double/2,-pi_double/2,
            math.nextafter(pi_double/2,math.inf),math.nextafter(-pi_double/2,-math.inf),
            math.ulp(0.),-math.ulp(0.),1e6,-1e6,1e12,-1e12,1e16,-1e16,1e20,-1e20,
            1e100,-1e100,sys.float_info.max,-sys.float_info.max]
    rng=random.Random(7091)
    angles += [rng.uniform(-10,10) for _ in range(100)]
    records=[]
    for theta in angles:
        d=Decimal.from_float(theta)
        quotient=(d/period).to_integral_value(rounding=ROUND_HALF_EVEN)
        principal=float(d-quotient*period);reduced=principal;flip=False
        if reduced>pi_double/2:reduced-=pi_double;flip=True
        elif reduced<-pi_double/2:reduced+=pi_double;flip=True
        original=sincos(d,pi);model=sincos(Decimal.from_float(reduced),pi)
        if flip:model=tuple(-v for v in model)
        records.append((theta,principal,reduced,*original,*model,flip))
    with tempfile.TemporaryDirectory(prefix='cordic-decimal-') as work:
        path=Path(work)/'controls.bin'
        with path.open('wb') as stream:
            for record in records:stream.write(struct.pack('=7db',*record))
        subprocess.run([str(Path(sys.argv[1]).resolve()),str(path),str(len(records))],check=True)
