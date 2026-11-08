"""Independent Python arbitrary-precision pow/gcd oracle for exact Fermat base results."""
import math,random,subprocess,sys
rng=random.Random(107);cases=[]
for n in range(2,257):
 for a in range(1,n):cases.append((n,a))
for n in [341,561,1105,1729,2465,2821,6601,2**64-1,2**64-59,(2**32-1)**2]:
 for a in [1,2,3,5,n//2,n-2,n-1]:cases.append((n,a))
for _ in range(10000):
 n=rng.randrange(2,2**64);cases.append((n,rng.randrange(1,n)))
p=subprocess.run([sys.argv[1]],input=''.join(f'{n} {a}\n' for n,a in cases),text=True,capture_output=True,check=True)
rows=p.stdout.splitlines();assert len(rows)==len(cases)
for (n,a),line in zip(cases,rows):
 g,r,passes=map(int,line.split());expected=pow(a,n-1,n);assert (g,r,passes)==(math.gcd(a,n),expected,int(expected==1)),(n,a,line)
print(f'Python pow/gcd: {len(cases)} exact base results passed')
