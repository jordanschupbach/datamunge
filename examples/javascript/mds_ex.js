// CMD: just TARGET=mds_ex run-javascript
// Classical multidimensional scaling of iris: euclidean embedding, plot, and a manhattan comparison.

const datamunge = require("../../index.js");

const iris = datamunge.data.iris();
const features = ["Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"];
const species = [];
for (let i = 0; i < iris.nrows(); i++) species.push(iris.string_at("Species", i));

console.log("=================== Classical MDS on iris (euclidean) ===================");
const mds = new datamunge.MDS(iris, features, 2, "euclidean");
mds.print_summary();

console.log("\nEmbedding as a DataFrame:");
console.log(mds.embedding_frame().to_string());

mds.plot_embedding_grouped(species).save_svg("mds_iris_embedding.svg");
console.log("\nEmbedding scatter (colored by species) saved as mds_iris_embedding.svg");

console.log("\n=================== Classical MDS on iris (manhattan) ===================");
const manhattanMds = new datamunge.MDS(iris, features, 2, "manhattan");
console.log(
  `Goodness of fit: euclidean=${mds.goodness_of_fit().toFixed(4)}, manhattan=${manhattanMds.goodness_of_fit().toFixed(4)}`
);
