#include <datamunge/stats/detail/mixed_model_formula.hpp>

#include <cctype>
#include <stdexcept>

namespace datamunge::stats::detail {

std::string mixed_model_trim(const std::string& s) {
    std::size_t begin = 0;
    while (begin < s.size() && std::isspace(static_cast<unsigned char>(s[begin]))) ++begin;
    std::size_t end = s.size();
    while (end > begin && std::isspace(static_cast<unsigned char>(s[end - 1]))) --end;
    return s.substr(begin, end - begin);
}

std::optional<RandomTermSpan> find_random_effect_span(const std::string& text) {
    std::vector<std::size_t> stack;
    std::optional<RandomTermSpan> found;
    for (std::size_t i = 0; i < text.size(); ++i) {
        if (text[i] == '(') {
            stack.push_back(i);
        } else if (text[i] == ')') {
            if (stack.empty()) throw std::invalid_argument("mixed model: unbalanced parentheses in formula '" + text + "'");
            const std::size_t start = stack.back();
            stack.pop_back();
            if (stack.empty()) { // a top-level paren group
                const std::string content = text.substr(start + 1, i - start - 1);
                if (content.find('|') != std::string::npos) {
                    if (found.has_value())
                        throw std::invalid_argument(
                            "mixed model: only a single random-effects term '(... | group)' is supported");
                    found = RandomTermSpan{start, i, content};
                }
            }
        }
    }
    if (!stack.empty()) throw std::invalid_argument("mixed model: unbalanced parentheses in formula '" + text + "'");
    return found;
}

RandomEffectSpec parse_random_effect_rhs(const std::string& rhs) {
    std::vector<std::string> terms;
    std::string cur;
    for (const char c : rhs) {
        if (c == '+') {
            terms.push_back(cur);
            cur.clear();
        } else {
            cur.push_back(c);
        }
    }
    terms.push_back(cur);

    RandomEffectSpec spec;
    bool suppress_intercept = false;
    for (auto& raw : terms) {
        const std::string t = mixed_model_trim(raw);
        if (t.empty()) continue;
        if (t == "0" || t == "-1") {
            suppress_intercept = true;
        } else if (t == "1") {
            // explicit intercept marker; already the default
        } else {
            spec.slope_vars.push_back(t);
        }
    }
    spec.intercept = !suppress_intercept;
    return spec;
}

linalg::DenseMatrix<double> build_lambda(const std::vector<double>& theta, const std::size_t q) {
    linalg::DenseMatrix<double> lambda(q, q, 0.0);
    std::size_t idx = 0;
    for (std::size_t i = 0; i < q; ++i)
        for (std::size_t j = 0; j <= i; ++j) lambda(i, j) = theta[idx++];
    return lambda;
}

} // namespace datamunge::stats::detail
