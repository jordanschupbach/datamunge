const datamunge = require("../../index.js");

function svector(values) {
  const out = new datamunge.SVector(values.length);
  for (let i = 0; i < values.length; i++) out.set(i, values[i]);
  return out;
}

function reportPurity(clusterLabels, species, nClusters) {
  const votes = new Map();
  for (let i = 0; i < clusterLabels.size(); i++) {
    const label = clusterLabels.get(i);
    if (label >= 0) {
      const key = `${label},${species[i]}`;
      votes.set(key, (votes.get(key) || 0) + 1);
    }
  }
  for (let c = 0; c < nClusters; c++) {
    let total = 0;
    let best = 0;
    let bestSp = "";
    for (const [key, cnt] of votes) {
      const [labelStr, sp] = key.split(",");
      if (parseInt(labelStr, 10) === c) {
        total += cnt;
        if (cnt > best) {
          best = cnt;
          bestSp = sp;
        }
      }
    }
    if (total > 0) {
      console.log(`  cluster ${c}: ${total} points, majority ${bestSp} (${best}/${total})`);
    }
  }
}

const iris = datamunge.DataFrame.iris();
const n = iris.nrows();
const species = [];
for (let i = 0; i < n; i++) species.push(iris.string_at("Species", i));
const features = svector(["Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"]);

console.log("=================== K-means (k=3) on iris ===================");
const kmeans = new datamunge.KMeans(iris, features, 3);
console.log(`Inertia: ${kmeans.inertia()}, iterations (best run): ${kmeans.iterations_used()}`);
reportPurity(kmeans.labels(), species, 3);

console.log("\n=================== Agglomerative clustering on iris ===================");
for (const linkage of ["ward", "complete", "average"]) {
  const model = new datamunge.AgglomerativeClustering(iris, features, 3, linkage);
  console.log(`--- linkage=${linkage} ---`);
  reportPurity(model.labels(), species, 3);
}

console.log("\n--- Re-cutting the ward dendrogram at k=2 without refitting ---");
const wardModel = new datamunge.AgglomerativeClustering(iris, features, 3, "ward");
reportPurity(wardModel.cut(2), species, 2);

console.log("\n=================== DBSCAN on iris ===================");
const dbscan = new datamunge.DBSCAN(iris, features, 0.6, 5);
console.log(`Clusters found: ${dbscan.n_clusters()}, noise points: ${dbscan.n_noise()} (of ${dbscan.observations()})`);
reportPurity(dbscan.labels(), species, dbscan.n_clusters());
