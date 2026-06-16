from pyoctra import octra

n = 100
v = octra.DVector(n)

for i in range(n):
    v[i] = i * 1.5

for i in range(n):
    print(v[i])

v2 = octra.IVector(n)
for i in range(n):
    v2[i] = int(i * 1.5)
for i in range(n):
    print(v2[i])

p = octra.IPair(3, 4)
p
print(f"p: ({p.first}, {p.second})")

p2 = octra.DPair(10, 20)
p2
print(f"p2: ({p2.first}, {p2.second})")
