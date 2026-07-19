package require Datamunge 0.0.1

# Tcl's overload dispatch can't distinguish a vector<string> argument from a std::string
# argument (both are just Tcl_Obj* strings at the C API level), so KMeans's overloaded
# constructor always resolves to the encoded-string form regardless of which argument type is
# passed. Route around it by calling the encoded constructor directly with a unit-separator
# (\x1f, decimal 31) joined string, per the facade's documented encoded-column convention.
proc encode_cols {names} {
  return [join $names [format %c 31]]
}

proc report_purity {cluster_labels species n_clusters} {
  array set votes {}
  set n [llength $cluster_labels]
  for {set i 0} {$i < $n} {incr i} {
    set label [lindex $cluster_labels $i]
    if {$label >= 0} {
      set sp [lindex $species $i]
      set key "$label,$sp"
      if {[info exists votes($key)]} {
        incr votes($key)
      } else {
        set votes($key) 1
      }
    }
  }
  for {set c 0} {$c < $n_clusters} {incr c} {
    set total 0
    set best 0
    set best_sp ""
    foreach key [array names votes "$c,*"] {
      set sp [lindex [split $key ","] 1]
      set cnt $votes($key)
      incr total $cnt
      if {$cnt > $best} {
        set best $cnt
        set best_sp $sp
      }
    }
    if {$total > 0} {
      puts "  cluster $c: $total points, majority $best_sp ($best/$total)"
    }
  }
}

set iris [datamunge::DataFrame_iris]
set n [datamunge::DataFrame_nrows $iris]
set species {}
for {set i 0} {$i < $n} {incr i} {
  lappend species [datamunge::DataFrame_string_at $iris "Species" $i]
}
set encoded_features [encode_cols {Sepal.Length Sepal.Width Petal.Length Petal.Width}]

puts "=================== K-means (k=3) on iris ==================="
set kmeans [datamunge::new_KMeans $iris $encoded_features 3]
puts "Inertia: [datamunge::KMeans_inertia $kmeans], iterations (best run): [datamunge::KMeans_iterations_used $kmeans]"
report_purity [datamunge::KMeans_labels $kmeans] $species 3

puts "\n=================== Agglomerative clustering on iris ==================="
foreach linkage {ward complete average} {
  set model [datamunge::new_AgglomerativeClustering $iris $encoded_features 3 $linkage]
  puts "--- linkage=$linkage ---"
  report_purity [datamunge::AgglomerativeClustering_labels $model] $species 3
}

puts "\n--- Re-cutting the ward dendrogram at k=2 without refitting ---"
set ward_model [datamunge::new_AgglomerativeClustering $iris $encoded_features 3 "ward"]
report_purity [datamunge::AgglomerativeClustering_cut $ward_model 2] $species 2

puts "\n=================== DBSCAN on iris ==================="
set dbscan [datamunge::new_DBSCAN $iris $encoded_features 0.6 5]
puts "Clusters found: [datamunge::DBSCAN_n_clusters $dbscan], noise points: [datamunge::DBSCAN_n_noise $dbscan] (of [datamunge::DBSCAN_observations $dbscan])"
report_purity [datamunge::DBSCAN_labels $dbscan] $species [datamunge::DBSCAN_n_clusters $dbscan]
