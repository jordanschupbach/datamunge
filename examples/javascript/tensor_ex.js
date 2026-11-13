// CMD: just TARGET=tensor_ex run-javascript

const datamunge = require("../../index.js");

function shapeStr(shp) {
  return shp.join(", ");
}

console.log("=================== Construction ===================");
const z = datamunge.Tensor.zeros([2, 3]);
console.log(`zeros([2,3]): ${z.to_string()}`);

const eyet = datamunge.Tensor.eye(3);
console.log(`eye(3): ${eyet.to_string()}`);

const r = datamunge.Tensor.arange(0.0, 12.0, 1.0).reshape([3, 4]);
console.log(`arange(0,12).reshape([3,4]): ${r.to_string()}`);

console.log("\n=================== Shape ops ===================");
const rt = r.transpose();
console.log(`transpose -> shape [${shapeStr(rt.shape())}]`);
const sliced = r.slice(1, 1, 3);
console.log(`slice(axis=1, start=1, stop=3): ${sliced.to_string()}`);

console.log("\n=================== Broadcasting arithmetic ===================");
const col = datamunge.Tensor.from_values([3, 1], [1, 2, 3]);
const row = datamunge.Tensor.from_values([1, 4], [10, 20, 30, 40]);
const broadcastSum = col.add(row);
console.log(`(3,1) + (1,4) -> ${broadcastSum.to_string()}`);

console.log("\n=================== Reductions ===================");
console.log(`r.sum() = ${r.sum()}, r.mean() = ${r.mean()}`);
const colMeans = r.mean_axis(0);
console.log(`column means (axis=0): ${colMeans.to_string()}`);

console.log("\n=================== Linear algebra ===================");
const a = datamunge.Tensor.from_values([2, 3], [1, 2, 3, 4, 5, 6]);
const b = datamunge.Tensor.from_values([3, 2], [7, 8, 9, 10, 11, 12]);
console.log(`matmul(2x3, 3x2) -> ${a.matmul(b).to_string()}`);

const v1 = datamunge.Tensor.from_values([3], [1, 2, 3]);
const v2 = datamunge.Tensor.from_values([3], [4, 5, 6]);
console.log(`dot([1,2,3], [4,5,6]) = ${v1.dot(v2)}`);
console.log(`outer(v1, v2) -> ${v1.outer(v2).to_string()}`);

console.log("\n=================== Comparisons & masks ===================");
const mask = r.greater_equal(datamunge.Tensor.full([3, 4], 6.0));
console.log(`r >= 6 -> ${mask.to_string()}`);
console.log(`count(r >= 6) = ${mask.sum()}`);

console.log("\n=================== A real dataset as a Tensor ===================");
const iris = datamunge.data.iris();
const n = iris.nrows();
const flat = [];
for (let i = 0; i < n; i++) {
  flat.push(iris.numeric_at("Sepal.Length", i));
  flat.push(iris.numeric_at("Sepal.Width", i));
  flat.push(iris.numeric_at("Petal.Length", i));
  flat.push(iris.numeric_at("Petal.Width", i));
}
const x = datamunge.Tensor.from_values([n, 4], flat);
console.log(`iris feature tensor shape: [${shapeStr(x.shape())}]`);

const featureMeans = x.mean_axis(0);
const centered = x.subtract(featureMeans.reshape([1, 4]));
const scatter = centered.transpose().matmul(centered);
console.log(`feature means (Sepal.Length, Sepal.Width, Petal.Length, Petal.Width): ${featureMeans.to_string()}`);
console.log(`(X-mean)^T (X-mean) [4x4 scatter matrix]: ${scatter.to_string(16)}`);

const species = [];
for (let i = 0; i < n; i++) species.push(iris.string_at("Species", i));
const speciesTensor = datamunge.Tensor.from_string_values([n], species);
const setosaMask = speciesTensor.equal(datamunge.Tensor.from_string_values([1], ["setosa"]));
console.log(`setosa count = ${setosaMask.sum()} (of ${n} rows)`);
