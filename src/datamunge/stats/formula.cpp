#include <datamunge/stats/formula.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace datamunge::stats {

namespace {

using dstruct::DataFrame;

std::string trim(const std::string& s) {
    std::size_t begin = 0;
    std::size_t end   = s.size();
    while (begin < end && std::isspace(static_cast<unsigned char>(s[begin]))) ++begin;
    while (end > begin && std::isspace(static_cast<unsigned char>(s[end - 1]))) --end;
    return s.substr(begin, end - begin);
}

// Splits `s` on top-level occurrences of `delim`, treating '(' / ')' as
// depth markers so delimiters inside function calls are not split on.
std::vector<std::string> split_top_level(const std::string& s, char delim) {
    std::vector<std::string> parts;
    int         depth = 0;
    std::string buffer;
    for (char c : s) {
        if (c == '(') ++depth;
        else if (c == ')') --depth;
        if (depth == 0 && c == delim) {
            parts.push_back(trim(buffer));
            buffer.clear();
        } else {
            buffer.push_back(c);
        }
    }
    parts.push_back(trim(buffer));
    return parts;
}

// Splits the RHS of a formula on top-level '+'/'-', returning each term
// together with the sign that preceded it.
std::vector<std::pair<char, std::string>> split_signed_terms(const std::string& s) {
    std::vector<std::pair<char, std::string>> out;
    int         depth = 0;
    std::string buffer;
    char        pending_sign = '+';
    for (std::size_t i = 0; i <= s.size(); ++i) {
        const bool at_end = (i == s.size());
        const char c      = at_end ? '\0' : s[i];
        if (!at_end && c == '(') ++depth;
        else if (!at_end && c == ')') --depth;

        const bool is_split = (!at_end && depth == 0 && (c == '+' || c == '-'));
        if (is_split || at_end) {
            const std::string term = trim(buffer);
            if (!term.empty()) out.emplace_back(pending_sign, term);
            buffer.clear();
            if (is_split) pending_sign = c;
        } else {
            buffer.push_back(c);
        }
    }
    return out;
}

// Expands a*b*... (crossing) into the ordered list of interaction subsets:
// main effects first, then pairwise interactions, then triples, etc. — the
// same order R produces for `a*b*c`.
std::vector<std::string> expand_crossing(const std::string& raw) {
    const auto factors = split_top_level(raw, '*');
    if (factors.size() == 1) return {factors[0]};

    const std::size_t n = factors.size();
    std::vector<std::pair<int, unsigned>> masks; // (popcount, mask)
    for (unsigned mask = 1; mask < (1u << n); ++mask) {
        masks.emplace_back(__builtin_popcount(mask), mask);
    }
    std::stable_sort(masks.begin(), masks.end(),
                      [](const auto& a, const auto& b) { return a.first < b.first; });

    std::vector<std::string> out;
    out.reserve(masks.size());
    for (const auto& [popcount, mask] : masks) {
        (void)popcount;
        std::string term;
        for (std::size_t i = 0; i < n; ++i) {
            if (mask & (1u << i)) {
                if (!term.empty()) term += ":";
                term += factors[i];
            }
        }
        out.push_back(term);
    }
    return out;
}

void require_numeric_column(const DataFrame& data, const std::string& colname, const std::string& fn) {
    if (!data.has_column(colname))
        throw std::invalid_argument("Formula: unknown column in " + fn + "(): " + colname);
    if (data.column_type(colname) != DataFrame::ColumnType::Numeric)
        throw std::invalid_argument("Formula: " + fn + "() requires a numeric column, got categorical: " + colname);
}

// Recursive-descent parser for the small arithmetic grammar accepted inside
// I(...): +, -, *, /, ^ (right-assoc), unary -, parens, numeric literals,
// and bare column identifiers.
class ExprParser {
 public:
    explicit ExprParser(const std::string& text) : text_(text) {}

    std::shared_ptr<ExprNode> parse() {
        auto node = parse_expr();
        skip_ws();
        if (pos_ != text_.size())
            throw std::invalid_argument("Formula: unexpected character in I(" + text_ + ") at position "
                                        + std::to_string(pos_));
        return node;
    }

 private:
    const std::string& text_;
    std::size_t         pos_{0};

    void skip_ws() {
        while (pos_ < text_.size() && std::isspace(static_cast<unsigned char>(text_[pos_]))) ++pos_;
    }

    char peek() {
        skip_ws();
        return pos_ < text_.size() ? text_[pos_] : '\0';
    }

    static std::shared_ptr<ExprNode> make(ExprOp op, std::shared_ptr<ExprNode> left,
                                          std::shared_ptr<ExprNode> right = nullptr) {
        auto node   = std::make_shared<ExprNode>();
        node->op    = op;
        node->left  = std::move(left);
        node->right = std::move(right);
        return node;
    }

    std::shared_ptr<ExprNode> parse_expr() {
        auto node = parse_term();
        for (;;) {
            const char c = peek();
            if (c != '+' && c != '-') break;
            ++pos_;
            node = make(c == '+' ? ExprOp::Add : ExprOp::Subtract, node, parse_term());
        }
        return node;
    }

    std::shared_ptr<ExprNode> parse_term() {
        auto node = parse_power();
        for (;;) {
            const char c = peek();
            if (c != '*' && c != '/') break;
            ++pos_;
            node = make(c == '*' ? ExprOp::Multiply : ExprOp::Divide, node, parse_power());
        }
        return node;
    }

    std::shared_ptr<ExprNode> parse_power() {
        auto node = parse_unary();
        if (peek() == '^') {
            ++pos_;
            node = make(ExprOp::Power, node, parse_power());
        }
        return node;
    }

    std::shared_ptr<ExprNode> parse_unary() {
        if (peek() == '-') {
            ++pos_;
            return make(ExprOp::Negate, parse_unary());
        }
        if (peek() == '+') {
            ++pos_;
            return parse_unary();
        }
        return parse_primary();
    }

    std::shared_ptr<ExprNode> parse_primary() {
        skip_ws();
        if (pos_ >= text_.size())
            throw std::invalid_argument("Formula: unexpected end of I(" + text_ + ") expression");

        const char c = text_[pos_];
        if (c == '(') {
            ++pos_;
            auto node = parse_expr();
            skip_ws();
            if (pos_ >= text_.size() || text_[pos_] != ')')
                throw std::invalid_argument("Formula: missing ')' in I(" + text_ + ") expression");
            ++pos_;
            return node;
        }
        if (std::isdigit(static_cast<unsigned char>(c)) || c == '.') {
            const std::size_t start = pos_;
            while (pos_ < text_.size()
                   && (std::isdigit(static_cast<unsigned char>(text_[pos_])) || text_[pos_] == '.'))
                ++pos_;
            auto node    = std::make_shared<ExprNode>();
            node->op     = ExprOp::Literal;
            node->literal = std::stod(text_.substr(start, pos_ - start));
            return node;
        }
        if (std::isalpha(static_cast<unsigned char>(c)) || c == '_') {
            const std::size_t start = pos_;
            while (pos_ < text_.size()
                   && (std::isalnum(static_cast<unsigned char>(text_[pos_])) || text_[pos_] == '_'
                       || text_[pos_] == '.'))
                ++pos_;
            auto node   = std::make_shared<ExprNode>();
            node->op    = ExprOp::Column;
            node->column = text_.substr(start, pos_ - start);
            return node;
        }
        throw std::invalid_argument(std::string("Formula: unexpected character '") + c + "' in I(" + text_
                                    + ") expression");
    }
};

void validate_expression_columns(const ExprNode& node, const DataFrame& data) {
    if (node.op == ExprOp::Column) {
        require_numeric_column(data, node.column, "I");
        return;
    }
    if (node.left) validate_expression_columns(*node.left, data);
    if (node.right) validate_expression_columns(*node.right, data);
}

struct PartialCombo {
    std::string                 name;
    std::vector<ResolvedColumn> parts;
};

std::vector<ResolvedColumn> cartesian_combine(const std::vector<std::vector<ResolvedColumn>>& factor_choices) {
    std::vector<PartialCombo> combos(1);
    for (const auto& choices : factor_choices) {
        std::vector<PartialCombo> next;
        next.reserve(combos.size() * choices.size());
        for (const auto& partial : combos) {
            for (const auto& choice : choices) {
                PartialCombo grown = partial;
                grown.name         = grown.name.empty() ? choice.name : grown.name + ":" + choice.name;
                grown.parts.push_back(choice);
                next.push_back(std::move(grown));
            }
        }
        combos = std::move(next);
    }

    std::vector<ResolvedColumn> out;
    out.reserve(combos.size());
    for (auto& combo : combos) {
        if (combo.parts.size() == 1) {
            out.push_back(std::move(combo.parts[0]));
        } else {
            ResolvedColumn col;
            col.kind              = ResolvedColumnKind::Interaction;
            col.name              = combo.name;
            col.interaction_parts = std::move(combo.parts);
            out.push_back(std::move(col));
        }
    }
    return out;
}

std::vector<ResolvedColumn> resolve_factor(const std::string& factor_text, const DataFrame& data,
                                           DesignInfo& info) {
    const auto open = factor_text.find('(');
    if (open != std::string::npos && !factor_text.empty() && factor_text.back() == ')') {
        const std::string fname = trim(factor_text.substr(0, open));
        const std::string args  = factor_text.substr(open + 1, factor_text.size() - open - 2);

        if (fname == "I") {
            ExprParser parser(args);
            auto       expression = parser.parse();
            validate_expression_columns(*expression, data);
            ResolvedColumn col;
            col.kind       = ResolvedColumnKind::Expression;
            col.name       = "I(" + trim(args) + ")";
            col.expression = expression;
            return {col};
        }

        if (fname == "poly") {
            const auto parts = split_top_level(args, ',');
            if (parts.size() != 2)
                throw std::invalid_argument("Formula: poly() requires exactly 2 arguments: poly(x, degree)");
            const std::string colname = trim(parts[0]);
            int                degree;
            try {
                degree = std::stoi(trim(parts[1]));
            } catch (const std::exception&) {
                throw std::invalid_argument("Formula: poly() degree must be an integer literal");
            }
            if (degree < 1) throw std::invalid_argument("Formula: poly() degree must be >= 1");
            require_numeric_column(data, colname, "poly");

            std::vector<ResolvedColumn> out;
            out.reserve(static_cast<std::size_t>(degree));
            for (int d = 1; d <= degree; ++d) {
                ResolvedColumn col;
                col.kind          = ResolvedColumnKind::Function;
                col.source_column = colname;
                col.function_name = "poly";
                col.poly_degree   = d;
                col.name = "poly(" + colname + "," + std::to_string(degree) + ")" + std::to_string(d);
                out.push_back(std::move(col));
            }
            return out;
        }

        static const std::vector<std::string> unary_functions = {"log", "log10", "log2", "sqrt", "exp", "abs"};
        if (std::find(unary_functions.begin(), unary_functions.end(), fname) != unary_functions.end()) {
            const std::string colname = trim(args);
            if (colname.empty() || colname.find_first_of("+-*/^(),") != std::string::npos)
                throw std::invalid_argument("Formula: " + fname
                                            + "() only accepts a single column name; use I(...) for expressions");
            require_numeric_column(data, colname, fname);
            ResolvedColumn col;
            col.kind          = ResolvedColumnKind::Function;
            col.source_column = colname;
            col.function_name = fname;
            col.name          = fname + "(" + colname + ")";
            return {col};
        }

        throw std::invalid_argument("Formula: unknown function: " + fname
                                    + "() (supported: I, poly, log, log10, log2, sqrt, exp, abs)");
    }

    const std::string colname = trim(factor_text);
    if (colname.empty()) throw std::invalid_argument("Formula: empty term in formula");
    if (!data.has_column(colname)) throw std::invalid_argument("Formula: unknown column: " + colname);

    if (data.column_type(colname) == DataFrame::ColumnType::Numeric) {
        ResolvedColumn col;
        col.kind          = ResolvedColumnKind::Numeric;
        col.source_column = colname;
        col.name          = colname;
        return {col};
    }

    auto cached = info.categorical_levels.find(colname);
    std::vector<std::string> levels;
    if (cached != info.categorical_levels.end()) {
        levels = cached->second;
    } else {
        const auto counts = data.value_counts(colname);
        levels.reserve(counts.size());
        for (const auto& [level, count] : counts) {
            (void)count;
            levels.push_back(level);
        }
        std::sort(levels.begin(), levels.end());
        info.categorical_levels[colname] = levels;
    }
    if (levels.size() < 2)
        throw std::invalid_argument("Formula: categorical column '" + colname + "' needs at least 2 distinct levels");

    std::vector<ResolvedColumn> out;
    out.reserve(levels.size() - 1);
    for (std::size_t i = 1; i < levels.size(); ++i) {
        ResolvedColumn col;
        col.kind          = ResolvedColumnKind::CategoricalDummy;
        col.source_column = colname;
        col.level          = levels[i];
        col.name           = colname + levels[i];
        out.push_back(std::move(col));
    }
    return out;
}

} // namespace

double ExprNode::evaluate(const dstruct::DataFrame& data, std::size_t row) const {
    switch (op) {
        case ExprOp::Literal: return literal;
        case ExprOp::Column: {
            const auto value = data.optional_double_at(column, row);
            return value ? *value : std::numeric_limits<double>::quiet_NaN();
        }
        case ExprOp::Add: return left->evaluate(data, row) + right->evaluate(data, row);
        case ExprOp::Subtract: return left->evaluate(data, row) - right->evaluate(data, row);
        case ExprOp::Multiply: return left->evaluate(data, row) * right->evaluate(data, row);
        case ExprOp::Divide: return left->evaluate(data, row) / right->evaluate(data, row);
        case ExprOp::Power: return std::pow(left->evaluate(data, row), right->evaluate(data, row));
        case ExprOp::Negate: return -left->evaluate(data, row);
    }
    return 0.0;
}

bool ResolvedColumn::evaluate(const dstruct::DataFrame& data, std::size_t row, double& out) const {
    switch (kind) {
        case ResolvedColumnKind::Numeric: {
            const auto value = data.optional_double_at(source_column, row);
            if (!value) return false;
            out = *value;
            return true;
        }
        case ResolvedColumnKind::CategoricalDummy: {
            const auto value = data.optional_string_at(source_column, row);
            if (!value) return false;
            out = (*value == level) ? 1.0 : 0.0;
            return true;
        }
        case ResolvedColumnKind::Function: {
            const auto value = data.optional_double_at(source_column, row);
            if (!value) return false;
            const double x = *value;
            if (function_name == "poly") {
                out = std::pow(x, poly_degree);
            } else if (function_name == "log") {
                if (!(x > 0.0)) throw std::domain_error("log(" + source_column + "): non-positive value at row " + std::to_string(row));
                out = std::log(x);
            } else if (function_name == "log10") {
                if (!(x > 0.0)) throw std::domain_error("log10(" + source_column + "): non-positive value at row " + std::to_string(row));
                out = std::log10(x);
            } else if (function_name == "log2") {
                if (!(x > 0.0)) throw std::domain_error("log2(" + source_column + "): non-positive value at row " + std::to_string(row));
                out = std::log2(x);
            } else if (function_name == "sqrt") {
                if (x < 0.0) throw std::domain_error("sqrt(" + source_column + "): negative value at row " + std::to_string(row));
                out = std::sqrt(x);
            } else if (function_name == "exp") {
                out = std::exp(x);
            } else if (function_name == "abs") {
                out = std::fabs(x);
            } else {
                throw std::runtime_error("ResolvedColumn::evaluate: unknown function " + function_name);
            }
            return true;
        }
        case ResolvedColumnKind::Interaction: {
            double product = 1.0;
            for (const auto& part : interaction_parts) {
                double value;
                if (!part.evaluate(data, row, value)) return false;
                product *= value;
            }
            out = product;
            return true;
        }
        case ResolvedColumnKind::Expression: {
            const double value = expression->evaluate(data, row);
            if (std::isnan(value)) return false;
            out = value;
            return true;
        }
    }
    return false;
}

DesignMatrixData DesignInfo::build_matrix(const dstruct::DataFrame& data, bool require_response) const {
    if (require_response && !data.has_column(response_name))
        throw std::invalid_argument("DesignInfo::build_matrix: missing response column: " + response_name);

    std::vector<std::size_t>        kept_rows;
    std::vector<std::vector<double>> rows;
    std::vector<double>              y;
    kept_rows.reserve(data.nrows());
    rows.reserve(data.nrows());
    if (require_response) y.reserve(data.nrows());

    for (std::size_t r = 0; r < data.nrows(); ++r) {
        std::vector<double> values;
        values.reserve(columns.size());
        bool ok = true;
        for (const auto& col : columns) {
            double value;
            if (!col.evaluate(data, r, value)) { ok = false; break; }
            values.push_back(value);
        }

        double response_value = 0.0;
        if (ok && require_response) {
            const auto rv = data.optional_double_at(response_name, r);
            if (!rv) ok = false; else response_value = *rv;
        }
        if (!ok) continue;

        kept_rows.push_back(r);
        rows.push_back(std::move(values));
        if (require_response) y.push_back(response_value);
    }

    if (rows.empty())
        throw std::runtime_error("DesignInfo::build_matrix: no complete rows remain after dropping missing values");

    const std::size_t n = rows.size();
    const std::size_t p = columns.size() + (has_intercept ? 1u : 0u);
    linalg::DenseMatrix<double> X(n, p, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        std::size_t j = 0;
        if (has_intercept) X(i, j++) = 1.0;
        for (std::size_t c = 0; c < columns.size(); ++c) X(i, j++) = rows[i][c];
    }

    return DesignMatrixData{std::move(X), std::move(y), std::move(kept_rows)};
}

ClassificationDesignData DesignInfo::build_classification_matrix(const dstruct::DataFrame& data) const {
    if (!data.has_column(response_name))
        throw std::invalid_argument("DesignInfo::build_classification_matrix: missing response column: "
                                    + response_name);

    std::vector<std::size_t>        kept_rows;
    std::vector<std::vector<double>> rows;
    std::vector<std::string>         labels;
    kept_rows.reserve(data.nrows());
    rows.reserve(data.nrows());
    labels.reserve(data.nrows());

    for (std::size_t r = 0; r < data.nrows(); ++r) {
        std::vector<double> values;
        values.reserve(columns.size());
        bool ok = true;
        for (const auto& col : columns) {
            double value;
            if (!col.evaluate(data, r, value)) { ok = false; break; }
            values.push_back(value);
        }

        std::string label;
        if (ok) {
            const auto lv = data.optional_string_at(response_name, r);
            if (!lv) ok = false; else label = *lv;
        }
        if (!ok) continue;

        kept_rows.push_back(r);
        rows.push_back(std::move(values));
        labels.push_back(std::move(label));
    }

    if (rows.empty())
        throw std::runtime_error(
            "DesignInfo::build_classification_matrix: no complete rows remain after dropping missing values");

    const std::size_t n = rows.size();
    const std::size_t p = columns.size();
    linalg::DenseMatrix<double> X(n, p, 0.0);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t c = 0; c < p; ++c) X(i, c) = rows[i][c];

    return ClassificationDesignData{std::move(X), std::move(labels), std::move(kept_rows)};
}

void DesignInfo::validate_categorical_levels(const dstruct::DataFrame& data, const std::string& caller) const {
    for (const auto& [source_column, levels] : categorical_levels) {
        if (!data.has_column(source_column)) continue;
        for (std::size_t row = 0; row < data.nrows(); ++row) {
            const auto value = data.optional_string_at(source_column, row);
            if (!value) continue;
            if (std::find(levels.begin(), levels.end(), *value) == levels.end())
                throw std::runtime_error(caller + ": column '" + source_column + "' has unseen level '" + *value
                                         + "' that was not present when the model was fit");
        }
    }
}

Formula::Formula(std::string formula_text) : text_(std::move(formula_text)) { parse(); }

void Formula::parse() {
    const auto tilde = text_.find('~');
    if (tilde == std::string::npos) throw std::invalid_argument("Formula: missing '~' in \"" + text_ + "\"");

    response_name_ = trim(text_.substr(0, tilde));
    if (response_name_.empty())
        throw std::invalid_argument("Formula: missing response variable before '~' in \"" + text_ + "\"");

    const std::string rhs = trim(text_.substr(tilde + 1));
    if (rhs.empty()) throw std::invalid_argument("Formula: empty right-hand side in \"" + text_ + "\"");

    const auto signed_terms = split_signed_terms(rhs);
    if (signed_terms.empty()) throw std::invalid_argument("Formula: no terms found in \"" + text_ + "\"");

    bool                      has_intercept = true;
    std::vector<std::string> ordered_terms;

    for (const auto& [sign, raw] : signed_terms) {
        if (raw == "1") { has_intercept = (sign != '-'); continue; }
        if (raw == "0") { has_intercept = false; continue; }

        for (const auto& term : expand_crossing(raw)) {
            if (sign == '+') {
                if (std::find(ordered_terms.begin(), ordered_terms.end(), term) == ordered_terms.end())
                    ordered_terms.push_back(term);
            } else {
                const auto it = std::find(ordered_terms.begin(), ordered_terms.end(), term);
                if (it != ordered_terms.end()) ordered_terms.erase(it);
            }
        }
    }

    if (ordered_terms.empty() && !has_intercept)
        throw std::invalid_argument("Formula: model has no terms and no intercept in \"" + text_ + "\"");

    intercept_requested_ = has_intercept;
    raw_terms_            = std::move(ordered_terms);
}

DesignInfo Formula::resolve(const dstruct::DataFrame& data, ResponseKind response_kind, bool force_no_intercept) const {
    if (!data.has_column(response_name_))
        throw std::invalid_argument("Formula: response column not found: " + response_name_);
    const auto actual_kind = data.column_type(response_name_);
    if (response_kind == ResponseKind::Numeric && actual_kind != DataFrame::ColumnType::Numeric)
        throw std::invalid_argument("Formula: response column must be numeric: " + response_name_);
    if (response_kind == ResponseKind::Categorical && actual_kind != DataFrame::ColumnType::String)
        throw std::invalid_argument("Formula: response column must be categorical (string): " + response_name_);

    std::vector<std::string> terms;
    for (const auto& raw : raw_terms_) {
        if (raw == ".") {
            for (const auto& col : data.columns()) {
                if (col == response_name_) continue;
                if (std::find(terms.begin(), terms.end(), col) == terms.end()) terms.push_back(col);
            }
        } else if (std::find(terms.begin(), terms.end(), raw) == terms.end()) {
            terms.push_back(raw);
        }
    }
    if (terms.empty() && !intercept_requested_)
        throw std::invalid_argument("Formula: model has no terms and no intercept: \"" + text_ + "\"");

    DesignInfo info;
    info.response_name = response_name_;
    info.has_intercept  = (response_kind == ResponseKind::Numeric) && intercept_requested_ && !force_no_intercept;
    if (info.has_intercept) info.coefficient_names.push_back("(Intercept)");

    for (const auto& term_text : terms) {
        const auto factor_strings = split_top_level(term_text, ':');
        std::vector<std::vector<ResolvedColumn>> factor_choices;
        factor_choices.reserve(factor_strings.size());
        for (const auto& factor_text : factor_strings) {
            factor_choices.push_back(resolve_factor(factor_text, data, info));
        }

        TermGroup group;
        group.label = term_text;
        for (auto& col : cartesian_combine(factor_choices)) {
            group.column_indices.push_back(info.columns.size());
            info.coefficient_names.push_back(col.name);
            info.columns.push_back(std::move(col));
        }
        info.term_groups.push_back(std::move(group));
    }

    return info;
}

} // namespace datamunge::stats
