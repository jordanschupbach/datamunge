module app;

import std.stdio : writeln;
import datamunge;

void main() {
  int n = 100;
  auto v = new DVector();
  for (int i = 0; i < n; i++) v.push_back(i * 1.5);
  for (int i = 0; i < n; i++) writeln(v[i]);

  auto v2 = new IVector();
  for (int i = 0; i < n; i++) v2.push_back(cast(int)(i * 1.5));
  for (int i = 0; i < n; i++) writeln(v2[i]);

  auto p = new IPair(3, 4);
  writeln("p: (", p.first(), ", ", p.second(), ")");

  auto p2 = new DPair(10, 20);
  writeln("p2: (", p2.first(), ", ", p2.second(), ")");
}
