// CMD: just TARGET=pca_ex run-swift
// PCA on iris: summary, PC1 loadings, scores frame, score/scree plots, scaled vs unscaled.
import Datamunge
#if canImport(Glibc)
import Glibc
#else
import Darwin
#endif

/// A Double through a C printf format, e.g. cfmt("%.6f", x) for C#'s {x:F6}.
func cfmt(_ format: String, _ x: Double) -> String {
    var buffer = [CChar](repeating: 0, count: 64)
    _ = withVaList([x]) { vsnprintf(&buffer, buffer.count, format, $0) }
    return String(cString: buffer)
}

do {
    let iris = try Data.iris()!
    let featureColumns = ["Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"]

    // species labels, for the grouped (colored-by-species) scores plot
    let species = try (0..<(try iris.nrows())).map { try iris.stringAt("Species", $0) }

    print("=================== PCA on iris (scaled) ===================")
    let pca = try Pca(iris, featureColumns, true, true)
    try pca.printSummary()

    print("\nPC1 loadings (which original features drive it):")
    for (name, loading) in zip(try pca.featureNames(), try pca.componentLoadings(0)) {
        print("  \(name): \(cfmt("%.4f", loading))")
    }

    print("\nTraining scores as a DataFrame:\n\(try pca.scoresFrame()!)")

    try pca.plotScoresGrouped(species).saveSvg("pca_iris_scores.svg")
    print("\nScores scatter (colored by species) saved as pca_iris_scores.svg")

    try pca.plotScree().saveSvg("pca_iris_scree.svg")
    print("Scree plot saved as pca_iris_scree.svg")

    print("\n=================== PCA on iris (unscaled) ===================")
    let unscaled = try Pca(iris, featureColumns, true, false)
    print("PC1 explains \(cfmt("%.2f", 100 * (try unscaled.explainedVarianceRatio())[0]))% of variance (vs. \(cfmt("%.2f", 100 * (try pca.explainedVarianceRatio())[0]))% scaled) -- Sepal.Length's larger raw variance dominates the unscaled covariance matrix.")
} catch {
    print(error)
    exit(1)
}
