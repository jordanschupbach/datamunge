#include <dm/dm.hpp>
#include <gtest/gtest.h>
TEST(PipeNativeReductions, OverloadedNamesResolveWithoutExplicitCastsOrWrappers) {
  dm::dstruct::DataFrame frame;
  frame.add_column("group",std::vector<std::string>{"a","a","b"});
  frame.add_column("value",std::vector<double>{1,3,8});
  const auto totals=frame | dm::group_by("group") | dm::summarise_each(dm::sum);
  const auto means=frame | dm::group_by("group") | dm::summarise_each(dm::mean);
  const auto medians=frame | dm::group_by("group") | dm::summarise_each(dm::median);
  const auto spreads=frame | dm::group_by("group") | dm::summarise_each(dm::sd);
  EXPECT_EQ(totals.double_at("value",0),4);EXPECT_EQ(totals.double_at("value",1),8);
  EXPECT_EQ(means.double_at("value",0),2);EXPECT_EQ(means.double_at("value",1),8);
  EXPECT_EQ(medians.double_at("value",0),2);EXPECT_EQ(medians.double_at("value",1),8);
  EXPECT_NEAR(spreads.double_at("value",0),std::sqrt(2.0),1e-14);
  EXPECT_TRUE(std::isnan(spreads.double_at("value",1)));
}
TEST(PipeNativeReductions, CapturedCallablesAndNullColumnSemanticsArePreserved) {
  dm::dstruct::DataFrame frame;
  frame.add_column("group",std::vector<std::string>{"a","a","b"});
  frame.add_column("value",std::vector<double>{1,3,8});frame.set_null("value",2);
  const double offset=5;
  const auto result=frame | dm::group_by("group") |
      dm::summarise_each([offset](const auto& values){return dm::sum(values)+offset;});
  EXPECT_EQ(result.double_at("value",0),9);EXPECT_TRUE(result.is_null("value",1));
  const auto direct=frame | dm::group_by("group") | dm::summarise_each(dm::sum);
  EXPECT_TRUE(direct.is_null("value",1));
}
TEST(PipeNativeReductions, NullNativeFunctionPointerIsRejectedAtStepConstruction) {
  double (*reduction)(const dm::Vector<>&)=nullptr;
  EXPECT_THROW(dm::summarise_each(reduction),std::invalid_argument);
}
