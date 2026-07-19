from pydatamunge import datamunge as dm


def report_purity(cluster_labels, species, n_clusters):
    votes = [dict() for _ in range(n_clusters)]
    for i, label in enumerate(cluster_labels):
        if label < 0:
            continue
        votes[label][species[i]] = votes[label].get(species[i], 0) + 1
    for c in range(n_clusters):
        total = sum(votes[c].values())
        if total == 0:
            continue
        best_species, best = max(votes[c].items(), key=lambda kv: kv[1])
        print(f"  cluster {c}: {total} points, majority {best_species} ({best}/{total})")


iris = dm.DataFrame.iris()
species = [iris.string_at("Species", i) for i in range(iris.nrows())]
features = ["Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"]

print("=================== K-means (k=3) on iris ===================")
kmeans = dm.KMeans(iris, features, 3)
print(f"Inertia: {kmeans.inertia()}, iterations (best run): {kmeans.iterations_used()}")
report_purity(kmeans.labels(), species, 3)

print("\n=================== Agglomerative clustering on iris ===================")
for linkage in ("ward", "complete", "average"):
    model = dm.AgglomerativeClustering(iris, features, 3, linkage)
    print(f"--- linkage={linkage} ---")
    report_purity(model.labels(), species, 3)

print("\n--- Re-cutting the ward dendrogram at k=2 without refitting ---")
ward_model = dm.AgglomerativeClustering(iris, features, 3, "ward")
report_purity(ward_model.cut(2), species, 2)

print("\n=================== DBSCAN on iris ===================")
dbscan = dm.DBSCAN(iris, features, 0.6, 5)
print(f"Clusters found: {dbscan.n_clusters()}, noise points: {dbscan.n_noise()} (of {dbscan.observations()})")
report_purity(dbscan.labels(), species, dbscan.n_clusters())
