// dm::formula(): R-style formula objects and the (formula, data) model constructors.
#include <dm/dm.hpp>

#include <gtest/gtest.h>

#include <stdexcept>
#include <string>

TEST(FormulaObject, ParsesTextAndResponse) {
    const auto f = dm::formula("len ~ supp * dose");
    EXPECT_EQ(f.text(), "len ~ supp * dose");
    EXPECT_EQ(f.response_name(), "len");
    EXPECT_THROW(dm::formula("no tilde here"), std::invalid_argument);
}

TEST(FormulaObject, ModelsAcceptFormulaFirstInEveryCallStyle) {
    const auto data = dm::datasets::tooth_growth();
    const auto reference = dm::stats::LM(data, "len ~ supp * dose");
    const auto from_object = dm::stats::LM(dm::formula("len ~ supp * dose"), data);
    const auto from_literal = dm::stats::LM("len ~ supp * dose", data);
    const std::string text = "len ~ supp * dose";
    const auto from_string = dm::stats::LM(text, data);
    EXPECT_EQ(from_object.summary(), reference.summary());
    EXPECT_EQ(from_literal.summary(), reference.summary());
    EXPECT_EQ(from_string.summary(), reference.summary());
    EXPECT_EQ(from_object.coefficient_names().size(), 4U);
}

TEST(FormulaObject, OptionsAreForwarded) {
    const auto data = dm::datasets::tooth_growth();
    dm::stats::GLMOptions options;
    options.family = dm::stats::GLMFamily::Gaussian;
    const auto reference = dm::stats::GLM(data, "len ~ dose", options);
    const auto model = dm::stats::GLM(dm::formula("len ~ dose"), data, options);
    EXPECT_EQ(model.summary(), reference.summary());
}
