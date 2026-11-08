"""Independent arbitrary-precision Fibonacci boundary, codeword and strict parser checks."""
import random,subprocess,sys
rng=random.Random(108);weights=[1,2]
while weights[-1]<=2**64:weights.append(sum(weights[-2:]))
def encode(n):
 if not n:return 'invalid'
 top=max(i for i,w in enumerate(weights) if w<=n);digits=['0']*(top+1)
 for i in range(top,-1,-1):
  if weights[i]<=n:digits[i]='1';n-=weights[i]
 assert not n
 return ''.join(digits)+'1'
def decode(s):
 result=[];start=0
 while start<len(s):
  end=s.find('11',start)
  # Positions beyond the92-weight domain are rejected before a future terminator.
  if end<0:
   return 'overflow' if len(s)-start>92 or sum(weights[i] for i,c in enumerate(s[start:]) if c=='1')>=2**64 else 'invalid'
  payload=s[start:end+1]
  if len(payload)>92:return 'overflow'
  n=sum(weights[i] for i,c in enumerate(payload) if c=='1')
  if n>=2**64:return 'overflow'
  result.append(n);start=end+2
 return ' '.join(map(str,result))
values=[0,1,2,3,4,10,20,50,100,2**64-1]
for w in weights:
 values.extend(x for x in [w-1,w,w+1] if 0<=x<2**64)
values.extend(rng.randrange(1,2**64) for _ in range(10000))
requests=[('e',str(n),encode(n)) for n in values]
requests.extend(('d',encode(n),str(n)) for n in values if n)
requests.extend(('d',s,decode(s)) for s in ['0','1','110','1'*3,'0'*128+'11',encode(2**64),encode(2**64-1),encode(3)[1:]])
for _ in range(3000):
 s=''.join(rng.choice('01') for _ in range(rng.randrange(1,140)));requests.append(('d',s,decode(s)))
p=subprocess.run([sys.argv[1]],input=''.join(f'{op} {s}\n' for op,s,_ in requests),text=True,capture_output=True,check=True);lines=p.stdout.splitlines();assert len(lines)==len(requests)
for request,line in zip(requests,lines):assert line.strip()==request[2],(request,line)
print(f'Python arbitrary-precision Fibonacci: {len(requests)} encode/decode results passed')
