// Local dev: build first (`just build-javascript`), then run this example.
const octra = require("../../index.js");

var n = 10;
var v = new octra.DVector(n);
for (let i = 0; i < n; i++) {
  v.set(i, i * 1.1);
}
for (let i = 0; i < n; i++) {
  console.log(v.get(i));
}

var v2 = new octra.IVector(n);
for (let i = 0; i < n; i++) {
  v2.set(i, i * 1.5);
}
for (let i = 0; i < n; i++) {
  console.log(v2.get(i));
}

var p = new octra.DPair(3.14, 2.71);
console.log(p.first);
console.log(p.second);

var p2 = new octra.IPair(42, 7);
console.log(p2.first);
console.log(p2.second);

octra.hello();
