"""Check actual C++ signed-Euclid rows against Python arbitrary-precision integers."""
import math,random,subprocess,sys
rng=random.Random(1996);minimum=-(1<<63);maximum=(1<<63)-1
edge=[minimum,minimum+1,-maximum,-240,-46,-3,-1,0,1,2,3,46,240,maximum-1,maximum]
cases=[('g',a,b) for a in edge for b in edge]
cases += [('g',a,b) for a in range(-16,17) for b in range(-16,17)]
cases += [('g',rng.randrange(minimum,maximum+1),rng.randrange(minimum,maximum+1)) for _ in range(5000)]
cases += [('i',a,m) for a in edge for m in [minimum,-1,0,1,2,3,11,maximum-1,maximum]]
cases += [('i',rng.randrange(minimum,maximum+1),rng.randrange(1,maximum+1)) for _ in range(5000)]
p=subprocess.run([sys.argv[1]],input=''.join(f'{op} {a} {b}\n' for op,a,b in cases),text=True,capture_output=True,check=True)
lines=p.stdout.splitlines();assert len(lines)==len(cases),(len(lines),len(cases),p.stderr)
for (op,a,b),line in zip(cases,lines):
    if op=='i' and (b<=0 or math.gcd(a,b)!=1):assert line=='invalid',(op,a,b,line);continue
    fields,*raw=line.split('|');word,g,x,y,D,updates,sign,C,inverse=fields.split();g,x,y,D,updates,sign,C,inverse=map(int,(g,x,y,D,updates,sign,C,inverse));assert word=='ok'
    assert g==math.gcd(a,b) and a*x+b*y==g,(a,b,g,x,y)
    assert -(1<<63)<=x<(1<<63) and -(1<<63)<=y<(1<<63)
    rows=[tuple(map(int,r.split(','))) for r in raw];assert len(rows)==C==D+2 and updates==2*D
    expected=[(a,1,0,0,0),(b,0,1,0,0)];old_r,r=a,b;old_s,s=1,0;old_t,t=0,1
    while r:
        q=abs(old_r)//abs(r)
        if (old_r<0)!=(r<0):q=-q
        nr,ns,nt=old_r-q*r,old_s-q*s,old_t-q*t
        expected.append((nr,ns,nt,q,1));old_r,r=r,nr;old_s,s=s,ns;old_t,t=t,nt
    assert rows==expected,(a,b,rows,expected)
    assert sign==int(old_r<0)
    if old_r<0:old_r,old_s,old_t=-old_r,-old_s,-old_t
    assert (g,x,y)==(old_r,old_s,old_t)
    for j,(r,s,t,q,divided) in enumerate(rows):
        assert a*s+b*t==r
        if j:
            _,ps,pt,_,_=rows[j-1];assert ps*t-s*pt==(-1)**(j-1)
    if op=='i':assert inverse==pow(a,-1,b) and 0<=inverse<b
print(f'Python arbitrary-precision oracle: {len(cases)} signed gcd/inverse cases and every actual quotient/coefficient row passed')
