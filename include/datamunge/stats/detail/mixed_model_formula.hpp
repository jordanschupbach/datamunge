#pragma once

#include <datamunge/linalg/dense_matrix.hpp>

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

namespace datamunge::stats::detail {

// Shared lme4-style "(... | group)" formula grammar used by both LMM and GLMM: the
// random-effects side is limited to an intercept and/or plain numeric predictor names
// against a single grouping factor, e.g. "(1 | group)", "(x1 | group)",
// "(1 + x1 + x2 | group)", or "(0 + x1 | group)" (no random intercept).

std::string mixed_model_trim(const std::string& s);

struct RandomTermSpan {
    std::size_t start;
    std::size_t end; // inclusive
    std::string content;
};

// Finds the single top-level "(...)" group in `text` whose content contains a '|'.
// Throws std::invalid_argument if parentheses are unbalanced or more than one such
// group is present; returns std::nullopt if none is present.
std::optional<RandomTermSpan> find_random_effect_span(const std::string& text);

struct RandomEffectSpec {
    bool intercept{true};
    std::vector<std::string> slope_vars;
};

// Parses the content before '|' in a random-effects term, e.g. "1 + x1" or "0 + x1".
RandomEffectSpec parse_random_effect_rhs(const std::string& rhs);

// Builds Lambda (q x q lower-triangular) from theta's free entries, packed row-major:
// (0,0), (1,0), (1,1), (2,0), (2,1), (2,2), ...
linalg::DenseMatrix<double> build_lambda(const std::vector<double>& theta, std::size_t q);

} // namespace datamunge::stats::detail
