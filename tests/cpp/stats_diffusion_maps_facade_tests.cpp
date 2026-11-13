#include <dm/dm.hpp>
#include <gtest/gtest.h>
#include <memory>
TEST(DiffusionCompatibility, ExistingFacadesFormattingAndOwnedFrameMatchCanonicalModel) {
  const auto input=dm::datasets::iris();
  const std::vector<std::string> columns{"Sepal.Length","Sepal.Width","Petal.Length","Petal.Width"};
  const dm::stats::DiffusionMaps core(input,columns,{2,5.5,.5,1});
  const std::unique_ptr<dm::DataFrame> facade_input(dm::data::iris());
  const dm::DiffusionMaps model(*facade_input,columns,2,5.5,.5,1);
  std::string encoded="4";encoded+='\x1e';
  for(std::size_t i=0;i<columns.size();++i){if(i)encoded+='\x1f';encoded+=columns[i];}
  const dm::DiffusionMaps encoded_model(*facade_input,encoded,2,5.5,.5,1);
  EXPECT_EQ(model.feature_names(),columns);
  EXPECT_EQ(model.kept_row_indices(),core.kept_row_indices());
  EXPECT_EQ(model.eigenvalues(),core.eigenvalues());
  EXPECT_EQ(model.dimension(0),core.dimension(0));
  EXPECT_EQ(encoded_model.dimension(1),core.dimension(1));
  EXPECT_EQ(model.summary(),core.summary());
  EXPECT_EQ(std::format("{}",model),core.summary());
  EXPECT_EQ(std::format("{}",core),core.summary());
  EXPECT_EQ(std::format("{}",dm::stats::DiffusionMapsOptions{}),"DiffusionMapsOptions(n_components=2, heat_kernel_epsilon=1, alpha=0.5, diffusion_time=1)");
  const std::unique_ptr<dm::DataFrame> coordinates(model.embedding_frame());
  EXPECT_EQ(coordinates->nrows(),150u);EXPECT_EQ(coordinates->ncols(),2u);
}
