using System;
using System.Diagnostics;
using System.Collections.Generic;

// NOTE: if using repl, uncomment the following line to load the octradotnet.dll
// and then recomment it back when compiling as a standalone program
// #r "src/octradotnet/bin/Debug/net10.0/octradotnet.dll"

class stl_ex {
  static void Main() {
    Console.WriteLine("Hello, World!");

    var v = new DVector();
    for(uint i=0; i<10; i++) {
      v.Add(((double)i)*1.1);
    }
    for(uint i=0; i<v.Count; i++) {
      Console.WriteLine(v[(int)i]);
    }

    var v2 = new IVector();
    for(uint i=0; i<10; i++) {
      v2.Add((int)(i*2));
    }
    for(uint i=0; i<v2.Count; i++) {
      Console.WriteLine(v2[(int)i]);
    }

    var p = new DPair(3.14, 2.71);
    Console.WriteLine(p.first);
    Console.WriteLine(p.second);

    var p2 = new IPair(42, 7);
    Console.WriteLine(p2.first);

  }
}
