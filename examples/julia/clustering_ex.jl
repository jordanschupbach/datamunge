# CMD: just TARGET=clustering_ex run-julia

# Clustering iris: k-means, agglomerative (several linkages, dendrogram re-cut) and DBSCAN.
using Datamunge

"For each cluster, print its size and its majority species."
function report_purity(cluster_labels, species, n_clusters)
    votes = [Dict{String,Int}() for _ in 1:n_clusters]
    for (label, sp) in zip(cluster_labels, species)
        label >= 0 && (votes[label + 1][sp] = get(votes[label + 1], sp, 0) + 1)
    end
    for c in 1:n_clusters
        total = sum(values(votes[c]); init = 0)
        total > 0 || continue
        best, best_species = findmax(votes[c])
        println("  cluster $(c - 1): $total points, majority $best_species ($best/$total)")
    end
end

const iris = Datamunge.Data.iris()
const species = [string_at(iris, "Species", i) for i in 0:nrows(iris)-1]
const feature_columns = ["Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"]

println("=================== K-means (k=3) on iris ===================")
let kmeans = KMeans(iris, feature_columns, 3)
    println("Inertia: $(inertia(kmeans)), iterations (best run): $(iterations_used(kmeans))")
    report_purity(labels(kmeans), species, 3)
end

println("\n=================== Agglomerative clustering on iris ===================")
for linkage in ("ward", "complete", "average")
    model = AgglomerativeClustering(iris, feature_columns, 3, linkage)
    println("--- linkage=$linkage ---")
    report_purity(labels(model), species, 3)
end

println("\n--- Re-cutting the ward dendrogram at k=2 without refitting ---")
let ward_model = AgglomerativeClustering(iris, feature_columns, 3, "ward")
    report_purity(cut(ward_model, 2), species, 2)
end

println("\n=================== DBSCAN on iris ===================")
let dbscan = DBSCAN(iris, feature_columns, 0.6, 5)
    println("Clusters found: $(n_clusters(dbscan)), noise points: $(n_noise(dbscan)) (of $(observations(dbscan)))")
    report_purity(labels(dbscan), species, n_clusters(dbscan))
end
