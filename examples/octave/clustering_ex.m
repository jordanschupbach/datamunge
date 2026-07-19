1;

datamunge;

function v = sv(t)
  datamunge;
  v = SVector();
  for i = 1:numel(t)
    SVector_push_back(v, t{i});
  end
endfunction

function report_purity(cluster_labels, species, n_clusters)
  votes = cell(n_clusters, 1);
  for c = 1:n_clusters
    votes{c} = containers.Map("KeyType", "char", "ValueType", "double");
  end
  for i = 1:numel(cluster_labels)
    label = cluster_labels{i};
    if label >= 0
      m = votes{label + 1};
      sp = species{i};
      if isKey(m, sp)
        m(sp) = m(sp) + 1;
      else
        m(sp) = 1;
      end
    end
  end
  for c = 0:(n_clusters - 1)
    m = votes{c + 1};
    ks = keys(m);
    total = 0;
    best = -1;
    best_sp = "";
    for i = 1:numel(ks)
      n = m(ks{i});
      total = total + n;
      if n > best
        best = n;
        best_sp = ks{i};
      end
    end
    if total > 0
      printf("  cluster %d: %d points, majority %s (%d/%d)\n", c, total, best_sp, best, total);
    end
  end
endfunction

iris = DataFrame_iris();
n = DataFrame_nrows(iris);
species = {};
for i = 0:(n - 1)
  species{end + 1} = DataFrame_string_at(iris, "Species", i);
end
features = sv({"Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"});

printf("=================== K-means (k=3) on iris ===================\n");
kmeans = KMeans(iris, features, 3);
printf("Inertia: %g, iterations (best run): %d\n", KMeans_inertia(kmeans), KMeans_iterations_used(kmeans));
report_purity(KMeans_labels(kmeans), species, 3);

printf("\n=================== Agglomerative clustering on iris ===================\n");
linkages = {"ward", "complete", "average"};
for li = 1:numel(linkages)
  linkage = linkages{li};
  model = AgglomerativeClustering(iris, features, 3, linkage);
  printf("--- linkage=%s ---\n", linkage);
  report_purity(AgglomerativeClustering_labels(model), species, 3);
end

printf("\n--- Re-cutting the ward dendrogram at k=2 without refitting ---\n");
ward_model = AgglomerativeClustering(iris, features, 3, "ward");
report_purity(AgglomerativeClustering_cut(ward_model, 2), species, 2);

printf("\n=================== DBSCAN on iris ===================\n");
dbscan = DBSCAN(iris, features, 0.6, 5);
printf("Clusters found: %d, noise points: %d (of %d)\n", DBSCAN_n_clusters(dbscan), DBSCAN_n_noise(dbscan), DBSCAN_observations(dbscan));
report_purity(DBSCAN_labels(dbscan), species, DBSCAN_n_clusters(dbscan));
