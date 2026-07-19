<?php

function svector($values) {
    $out = new SVector(count($values));
    foreach ($values as $i => $v) $out->set($i, $v);
    return $out;
}

function report_purity($cluster_labels, $species, $n_clusters) {
    $votes = array();
    for ($i = 0; $i < $cluster_labels->size(); $i++) {
        $label = $cluster_labels->get($i);
        if ($label >= 0) {
            $key = $label . "," . $species[$i];
            $votes[$key] = ($votes[$key] ?? 0) + 1;
        }
    }
    for ($c = 0; $c < $n_clusters; $c++) {
        $total = 0;
        $best = 0;
        $best_sp = "";
        foreach ($votes as $key => $cnt) {
            $parts = explode(",", $key, 2);
            if ((int)$parts[0] === $c) {
                $total += $cnt;
                if ($cnt > $best) {
                    $best = $cnt;
                    $best_sp = $parts[1];
                }
            }
        }
        if ($total > 0) {
            print("  cluster $c: $total points, majority $best_sp ($best/$total)\n");
        }
    }
}

$iris = DataFrame::iris();
$n = $iris->nrows();
$species = array();
for ($i = 0; $i < $n; $i++) $species[] = $iris->string_at("Species", $i);
$features = svector(array("Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"));

print("=================== K-means (k=3) on iris ===================\n");
$kmeans = new KMeans($iris, $features, 3);
print("Inertia: " . $kmeans->inertia() . ", iterations (best run): " . $kmeans->iterations_used() . "\n");
report_purity($kmeans->labels(), $species, 3);

print("\n=================== Agglomerative clustering on iris ===================\n");
foreach (array("ward", "complete", "average") as $linkage) {
    $model = new AgglomerativeClustering($iris, $features, 3, $linkage);
    print("--- linkage=$linkage ---\n");
    report_purity($model->labels(), $species, 3);
}

print("\n--- Re-cutting the ward dendrogram at k=2 without refitting ---\n");
$ward_model = new AgglomerativeClustering($iris, $features, 3, "ward");
report_purity($ward_model->cut(2), $species, 2);

print("\n=================== DBSCAN on iris ===================\n");
$dbscan = new DBSCAN($iris, $features, 0.6, 5);
print("Clusters found: " . $dbscan->n_clusters() . ", noise points: " . $dbscan->n_noise() . " (of " . $dbscan->observations() . ")\n");
report_purity($dbscan->labels(), $species, $dbscan->n_clusters());

?>
