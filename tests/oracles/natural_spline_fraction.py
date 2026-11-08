"""Independent exact full-piece constraints; no Thomas/moment-system code is used."""
from fractions import Fraction as F
from bisect import bisect_right
import random,subprocess,sys

def coefficients(x,y):
    n=len(x);size=4*(n-1);rows=[]
    for i in range(n-1):
        h=x[i+1]-x[i]
        row=[F(0)]*(size+1);row[4*i]=1;row[-1]=y[i];rows.append(row)
        row=[F(0)]*(size+1)
        for k in range(4):row[4*i+k]=h**k
        row[-1]=y[i+1];rows.append(row)
    for i in range(n-2):
        h=x[i+1]-x[i]
        row=[F(0)]*(size+1);row[4*i+1]=1;row[4*i+2]=2*h;row[4*i+3]=3*h*h;row[4*(i+1)+1]=-1;rows.append(row)
        row=[F(0)]*(size+1);row[4*i+2]=2;row[4*i+3]=6*h;row[4*(i+1)+2]=-2;rows.append(row)
    row=[F(0)]*(size+1);row[2]=2;rows.append(row)
    row=[F(0)]*(size+1);row[-3]=2;row[-2]=6*(x[-1]-x[-2]);rows.append(row)
    assert len(rows)==size
    rows=[[F(v) for v in row] for row in rows]
    for col in range(size):
        pivot=next(i for i in range(col,size) if rows[i][col])
        rows[col],rows[pivot]=rows[pivot],rows[col]
        scale=rows[col][col];rows[col]=[v/scale for v in rows[col]]
        for i in range(size):
            if i!=col and rows[i][col]:
                factor=rows[i][col];rows[i]=[a-factor*b for a,b in zip(rows[i],rows[col])]
    result=[rows[i][-1] for i in range(size)]
    assert all(isinstance(v,F) for v in result)
    return result

def evaluate(c,x,q,order):
    i=max(0,min(bisect_right(x,q)-1,len(x)-2));t=q-x[i]
    value=F(0)
    for k in range(order,4):
        factor=1
        for j in range(order):factor*=k-j
        value+=c[4*i+k]*factor*t**(k-order)
    return value

def main():
    rng=random.Random(20261107);controls=[([0,1,2],[0,1,0]),([0,1,2],[0,1,1])]
    for _ in range(100):
        n=rng.randrange(2,9);x=[rng.randrange(-5,1)]
        for i in range(1,n):x.append(x[-1]+rng.randrange(1,6))
        controls.append((x,[rng.randrange(-10,11) for _ in x]))
    lines=[]
    for xx,yy in controls:
        x=list(map(F,xx));y=list(map(F,yy));c=coefficients(x,y);n=len(x)
        lines += [str(n),' '.join(map(str,xx)),' '.join(map(str,yy)),' '.join(repr(float(v)) for v in c)]
        moments=[2*c[4*i+2] for i in range(n-1)]+[2*c[-2]+6*c[-1]*(x[-1]-x[-2])]
        assert moments[0]==moments[-1]==0
        lines.append(' '.join(repr(float(v)) for v in moments))
        queries=x+[x[0]-1,x[-1]+1]+[(x[i]+x[i+1])/2 for i in range(n-1)]
        lines.append(str(4*len(queries)))
        for q in queries:
            for order in range(4):lines.append(f'{float(q)!r} {order} {float(evaluate(c,x,q,order))!r}')
    result=subprocess.run([sys.argv[1]],input='\n'.join(lines)+'\n',text=True,capture_output=True)
    print(result.stdout,end='');print(result.stderr,end='',file=sys.stderr)
    raise SystemExit(result.returncode)
if __name__=='__main__':main()
