// CMD: just TARGET=mds_ex run-csharp
#:project ../../src/datamungecsharp/Datamunge.csproj
// Classical multidimensional scaling of iris: euclidean embedding, plot, and a manhattan comparison.
using Datamunge;

var iris = Data.Iris()!;
string[] featureColumns = ["Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"];
var species = Enumerable.Range(0, (int)iris.Nrows()).Select(i => iris.StringAt("Species", (ulong)i)).ToList();

Console.WriteLine("=================== Classical MDS on iris (euclidean) ===================");
var mds = new MDS(iris, featureColumns, 2, "euclidean");
mds.PrintSummary();

Console.WriteLine($"\nEmbedding as a DataFrame:\n{mds.EmbeddingFrame()}");

mds.PlotEmbeddingGrouped(species).SaveSvg("mds_iris_embedding.svg");
Console.WriteLine("\nEmbedding scatter (colored by species) saved as mds_iris_embedding.svg");

Console.WriteLine("\n=================== Classical MDS on iris (manhattan) ===================");
var manhattanMds = new MDS(iris, featureColumns, 2, "manhattan");
Console.WriteLine($"Goodness of fit: euclidean={mds.GoodnessOfFit():F4}, manhattan={manhattanMds.GoodnessOfFit():F4}");
