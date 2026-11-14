% CMD: just TARGET=clustering_ex run-octave
% Clustering iris: k-means, agglomerative (several linkages, dendrogram re-cut) and DBSCAN.

1;

% Shortest round-trip decimal text for x (like C#'s default double printing).
function s = num (x)
  if (x == fix (x) && abs (x) < 1e15)
    s = sprintf ('%d', x);
    return;
  end
  for p = 1:17
    s = sprintf ('%.*g', p, x);
    if (str2double (s) == x)
      return;
    end
  end
end

% For each cluster, print its size and its majority species (negative labels are noise).
function report_purity (species, cluster_labels, n_clusters)
  [names, ~, idx] = unique (species, 'stable');
  votes = zeros (numel (names), n_clusters);
  for j = 1:numel (cluster_labels)
    label = double (cluster_labels(j));
    if (label >= 0)
      votes(idx(j), label + 1) += 1;
    end
  end
  for c = 1:n_clusters
    total = sum (votes(:, c));
    if (total == 0)
      continue;
    end
    [best, k] = max (votes(:, c));
    printf ('  cluster %d: %d points, majority %s (%d/%d)\n', c - 1, total, names{k}, best, total);
  end
end

iris = datamunge.Data.iris ();
species = arrayfun (@(i) iris.string_at ('Species', i), 0:iris.nrows ()-1, 'UniformOutput', false);
feature_columns = {'Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width'};

printf ('=================== K-means (k=3) on iris ===================\n');
kmeans = datamunge.KMeans (iris, feature_columns, 3);
printf ('Inertia: %s, iterations (best run): %d\n', num (kmeans.inertia ()), kmeans.iterations_used ());
report_purity (species, kmeans.labels (), 3);

printf ('\n=================== Agglomerative clustering on iris ===================\n');
for linkage = {'ward', 'complete', 'average'}
  model = datamunge.AgglomerativeClustering (iris, feature_columns, 3, linkage{1});
  printf ('--- linkage=%s ---\n', linkage{1});
  report_purity (species, model.labels (), 3);
end

printf ('\n--- Re-cutting the ward dendrogram at k=2 without refitting ---\n');
ward_model = datamunge.AgglomerativeClustering (iris, feature_columns, 3, 'ward');
report_purity (species, ward_model.cut (2), 2);

printf ('\n=================== DBSCAN on iris ===================\n');
dbscan = datamunge.DBSCAN (iris, feature_columns, 0.6, 5);
printf ('Clusters found: %d, noise points: %d (of %d)\n', dbscan.n_clusters (), dbscan.n_noise (), ...
        dbscan.observations ());
report_purity (species, dbscan.labels (), dbscan.n_clusters ());
