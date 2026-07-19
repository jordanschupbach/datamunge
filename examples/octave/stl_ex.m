1;

datamunge;

n = 100;
v = DVector();
for i = 0:(n - 1)
  DVector_push_back(v, i * 1.5);
end
for i = 0:(n - 1)
  disp(DVector___paren__(v, i))
end

v2 = IVector();
for i = 0:(n - 1)
  IVector_push_back(v2, floor(i * 1.5));
end
for i = 0:(n - 1)
  disp(IVector___paren__(v2, i))
end

p = IPair(3, 4);
printf("p: (%g, %g)\n", IPair_first_get(p), IPair_second_get(p));

p2 = DPair(10, 20);
printf("p2: (%g, %g)\n", DPair_first_get(p2), DPair_second_get(p2));
