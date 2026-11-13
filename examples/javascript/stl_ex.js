// CMD: just TARGET=stl_ex run-javascript

// Local dev: build first (`just build-javascript`), then run this example.
const datamunge = require("../../index.js");

// C++ std::vector and std::pair values are plain JavaScript arrays: pairs are [first, second].
var n = 10;
var v = [];
for (let i = 0; i < n; i++) {
  v.push(i * 1.1);
}
for (let i = 0; i < n; i++) {
  console.log(v[i]);
}

var v2 = [];
for (let i = 0; i < n; i++) {
  v2.push(Math.floor(i * 1.5));
}
for (let i = 0; i < n; i++) {
  console.log(v2[i]);
}

var p = datamunge.make_dpair(3.14, 2.71);
console.log(p[0]);
console.log(p[1]);
console.log(`sum_dvector(v) = ${datamunge.sum_dvector(v)}, sum_dpair(p) = ${datamunge.sum_dpair(p)}`);

var p2 = [42, 7];
console.log(p2[0]);
console.log(p2[1]);

datamunge.hello();
