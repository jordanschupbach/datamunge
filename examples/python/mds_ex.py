from pydatamunge import datamunge as dm

iris = dm.DataFrame.iris()
features = ["Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"]
species = [iris.string_at("Species", i) for i in range(iris.nrows())]

print("=================== Classical MDS on iris (euclidean) ===================")
mds = dm.MDS(iris, features, 2, "euclidean")
mds.print_summary()

print("\nEmbedding as a DataFrame:")
embedding_df = mds.embedding_frame()
print(embedding_df.to_string())

scatter = mds.plot_embedding_grouped(species)
scatter.save_svg("mds_iris_embedding.svg")
print("\nEmbedding scatter (colored by species) saved as mds_iris_embedding.svg")

print("\n=================== Classical MDS on iris (manhattan) ===================")
manhattan_mds = dm.MDS(iris, features, 2, "manhattan")
print(f"Goodness of fit: euclidean={mds.goodness_of_fit():.4f}, manhattan={manhattan_mds.goodness_of_fit():.4f}")
