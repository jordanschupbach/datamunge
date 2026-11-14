# CMD: just TARGET=clustering_ex run-tcl
# Clustering iris: k-means, agglomerative (several linkages, dendrogram re-cut) and DBSCAN.
package require datamunge

set iris [datamunge::Data::iris]
set species {}
for {set i 0} {$i < [$iris nrows]} {incr i} {
    lappend species [$iris string_at Species $i]
}
set feature_columns {Sepal.Length Sepal.Width Petal.Length Petal.Width}

# For each cluster, print its size and its majority species (negative labels are noise).
proc report_purity {cluster_labels n_clusters} {
    global species
    set votes [dict create]
    foreach label $cluster_labels sp $species {
        if {$label < 0} {
            continue
        }
        set counts [expr {[dict exists $votes $label] ? [dict get $votes $label] : {}}]
        dict incr counts $sp
        dict set votes $label $counts
    }
    for {set c 0} {$c < $n_clusters} {incr c} {
        if {![dict exists $votes $c]} {
            continue
        }
        set total 0
        set best_species ""
        set best -1
        dict for {sp count} [dict get $votes $c] {
            incr total $count
            if {$count > $best} {
                set best $count
                set best_species $sp
            }
        }
        puts "  cluster $c: $total points, majority $best_species ($best/$total)"
    }
}

puts "=================== K-means (k=3) on iris ==================="
set kmeans [datamunge::KMeans new $iris $feature_columns 3]
puts "Inertia: [$kmeans inertia], iterations (best run): [$kmeans iterations_used]"
report_purity [$kmeans labels] 3

puts "\n=================== Agglomerative clustering on iris ==================="
foreach linkage {ward complete average} {
    set model [datamunge::AgglomerativeClustering new $iris $feature_columns 3 $linkage]
    puts "--- linkage=$linkage ---"
    report_purity [$model labels] 3
    $model destroy
}

puts "\n--- Re-cutting the ward dendrogram at k=2 without refitting ---"
set ward_model [datamunge::AgglomerativeClustering new $iris $feature_columns 3 ward]
report_purity [$ward_model cut 2] 2

puts "\n=================== DBSCAN on iris ==================="
set dbscan [datamunge::DBSCAN new $iris $feature_columns 0.6 5]
puts "Clusters found: [$dbscan n_clusters], noise points: [$dbscan n_noise] (of [$dbscan observations])"
report_purity [$dbscan labels] [$dbscan n_clusters]
