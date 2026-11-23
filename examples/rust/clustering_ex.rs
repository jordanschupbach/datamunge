// CMD: just TARGET=clustering_ex run-rust
// Clustering iris: k-means, agglomerative (several linkages, dendrogram re-cut) and DBSCAN.
use datamunge::*;
use datamunge_rust_examples::N;

const FEATURE_COLUMNS: [&str; 4] = ["Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"];

/// For each cluster, prints its size and its majority species (negative labels are noise).
fn report_purity(cluster_labels: impl IntoIterator<Item = i64>, n_clusters: usize, species: &[String]) {
    // Species vote counts per cluster, in first-seen order (ties go to the first species seen).
    let mut votes: Vec<Vec<(&str, usize)>> = vec![Vec::new(); n_clusters];
    for (label, sp) in cluster_labels.into_iter().zip(species) {
        if label < 0 {
            continue;
        }
        let cluster = &mut votes[label as usize];
        match cluster.iter_mut().find(|(name, _)| *name == sp) {
            Some((_, count)) => *count += 1,
            None => cluster.push((sp, 1)),
        }
    }
    for (c, cluster) in votes.iter().enumerate() {
        let total: usize = cluster.iter().map(|(_, count)| count).sum();
        if total == 0 {
            continue;
        }
        let (best_species, best) = cluster.iter().fold(cluster[0], |best, &v| if v.1 > best.1 { v } else { best });
        println!("  cluster {c}: {total} points, majority {best_species} ({best}/{total})");
    }
}

fn main() -> datamunge::Result<()> {
    let iris = data::iris()?.expect("iris");
    let species = (0..iris.nrows()?).map(|i| iris.string_at("Species", i)).collect::<Result<Vec<_>>>()?;

    println!("=================== K-means (k=3) on iris ===================");
    let kmeans = KMeans::new_with_n_clusters(&iris, &FEATURE_COLUMNS, 3)?;
    println!("Inertia: {}, iterations (best run): {}", N(kmeans.inertia()?), kmeans.iterations_used()?);
    report_purity(kmeans.labels()?.into_iter().map(|l| l as i64), 3, &species);

    println!("\n=================== Agglomerative clustering on iris ===================");
    for linkage in ["ward", "complete", "average"] {
        let model = AgglomerativeClustering::new_with_linkage(&iris, &FEATURE_COLUMNS, 3, linkage)?;
        println!("--- linkage={linkage} ---");
        report_purity(model.labels()?.into_iter().map(|l| l as i64), 3, &species);
    }

    println!("\n--- Re-cutting the ward dendrogram at k=2 without refitting ---");
    let ward_model = AgglomerativeClustering::new_with_linkage(&iris, &FEATURE_COLUMNS, 3, "ward")?;
    report_purity(ward_model.cut(2)?.into_iter().map(|l| l as i64), 2, &species);

    println!("\n=================== DBSCAN on iris ===================");
    let dbscan = DBSCAN::new_with_min_samples(&iris, &FEATURE_COLUMNS, 0.6, 5)?;
    println!(
        "Clusters found: {}, noise points: {} (of {})",
        dbscan.n_clusters()?,
        dbscan.n_noise()?,
        dbscan.observations()?
    );
    report_purity(dbscan.labels()?.into_iter().map(i64::from), dbscan.n_clusters()?, &species);
    Ok(())
}
