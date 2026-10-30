/// @file svm.cpp
/// @brief Formula SVM fitting with simplified SMO, pairwise decisions and optimality diagnostics.

#include <datamunge/stats/svm.hpp>

#include <datamunge/linalg/dense_matrix.hpp>
#include <datamunge/random/random.hpp>
#include <datamunge/plot/class_regions.hpp>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace datamunge::stats {

/// @brief Formatting and deterministic binary optimisation internal to SVM.
namespace {

/// @brief Formats a finite or special statistic for summary().
/// @param v Value.
/// @param precision Fixed decimal places (default four).
/// @return Formatted number.
std::string format_stat(double v, int precision = 4) {
    if (std::isnan(v)) return "NaN";
    if (std::isinf(v)) return v > 0 ? "Inf" : "-Inf";
    std::ostringstream oss;
    const double mag = std::fabs(v);
    if (v != 0.0 && (mag >= 1e6 || mag < 1e-4)) {
        oss << std::scientific << std::setprecision(3) << v;
    } else {
        oss << std::fixed << std::setprecision(precision) << v;
    }
    return oss.str();
}

/// @brief Display spelling for a kernel.
/// @param kernel Kernel enum.
/// @return Static string, or unknown for an invalid enum.
const char* kernel_name(SVMKernel kernel) {
    switch (kernel) {
        case SVMKernel::Linear: return "linear";
        case SVMKernel::Polynomial: return "polynomial";
        case SVMKernel::Radial: return "radial";
        case SVMKernel::Sigmoid: return "sigmoid";
    }
    return "unknown";
}

/// @brief Returned coordinate iterate and actual work count.
struct SMOResult {
    std::vector<double> alpha; ///< Dense dual coefficients in pair row order.
    double               bias{0.0}; ///< Binary intercept.
    std::size_t sweeps{0}; ///< Total coordinate sweeps, including no-change sweeps.
};

// Simplified Sequential Minimal Optimization (Platt's algorithm, the
// textbook two-multiplier-update variant): solves the soft-margin SVM dual
//   maximize   sum(alpha_i) - 0.5 * sum_ij alpha_i alpha_j y_i y_j K(i,j)
//   subject to 0 <= alpha_i <= C,  sum(alpha_i y_i) = 0
// Random pair selection uses a fixed seed so fits are deterministic.
/// @brief Solves binary box-constrained SVM dual by simplified two-coordinate updates.
/// @param K Symmetric pair Gram matrix.
/// @param y Signs (+1 first class, -1 second).
/// @param C Positive finite coefficient upper bound.
/// @param tol Positive KKT threshold for choosing a coordinate.
/// @param max_passes Unchanged-sweep budget; total sweeps limited to max(200,20*max_passes).
/// @return Final feasible iterate, bias and sweep count; no convergence guarantee.
SMOResult train_binary_smo(const linalg::DenseMatrix<double>& K, const std::vector<double>& y, double C,
                           double tol, std::size_t max_passes) {
    const std::size_t m = y.size();
    std::vector<double> alpha(m, 0.0);
    double               b = 0.0;

    random::SplitMix64 rng(0x5A17ULL + m);

    /// @brief Training decision at a row under the current dense iterate.
    /// @param i Pair-local row.
    /// @return Functional decision including bias.
    auto decision = [&](std::size_t i) {
        double s = b;
        for (std::size_t k = 0; k < m; ++k) s += alpha[k] * y[k] * K(i, k);
        return s;
    };

    std::size_t passes           = 0;
    std::size_t total_sweeps     = 0;
    const std::size_t sweep_cap = std::max<std::size_t>(200, max_passes * 20);

    while (passes < max_passes && total_sweeps < sweep_cap) {
        ++total_sweeps;
        std::size_t num_changed = 0;

        for (std::size_t i = 0; i < m; ++i) {
            const double E_i = decision(i) - y[i];
            const bool   violates = (y[i] * E_i < -tol && alpha[i] < C) || (y[i] * E_i > tol && alpha[i] > 0.0);
            if (!violates) continue;

            std::size_t j = i;
            while (j == i) j = static_cast<std::size_t>(rng.uniform01() * static_cast<double>(m)) % m;

            const double E_j = decision(j) - y[j];
            const double alpha_i_old = alpha[i];
            const double alpha_j_old = alpha[j];

            double L, H;
            if (y[i] != y[j]) {
                L = std::max(0.0, alpha[j] - alpha[i]);
                H = std::min(C, C + alpha[j] - alpha[i]);
            } else {
                L = std::max(0.0, alpha[i] + alpha[j] - C);
                H = std::min(C, alpha[i] + alpha[j]);
            }
            if (L >= H) continue;

            const double eta = 2.0 * K(i, j) - K(i, i) - K(j, j);
            if (eta >= 0.0) continue;

            double alpha_j_new = alpha[j] - y[j] * (E_i - E_j) / eta;
            alpha_j_new         = std::clamp(alpha_j_new, L, H);
            if (std::abs(alpha_j_new - alpha_j_old) < 1e-7) continue;

            const double alpha_i_new = alpha[i] + y[i] * y[j] * (alpha_j_old - alpha_j_new);

            const double b1 = b - E_i - y[i] * (alpha_i_new - alpha_i_old) * K(i, i)
                             - y[j] * (alpha_j_new - alpha_j_old) * K(i, j);
            const double b2 = b - E_j - y[i] * (alpha_i_new - alpha_i_old) * K(i, j)
                             - y[j] * (alpha_j_new - alpha_j_old) * K(j, j);

            alpha[i] = alpha_i_new;
            alpha[j] = alpha_j_new;

            if (alpha[i] > 0.0 && alpha[i] < C) b = b1;
            else if (alpha[j] > 0.0 && alpha[j] < C) b = b2;
            else b = 0.5 * (b1 + b2);

            ++num_changed;
        }

        passes = (num_changed == 0) ? passes + 1 : 0;
    }

    return SMOResult{std::move(alpha), b, total_sweeps};
}

} // namespace

SVM::SVM(const dstruct::DataFrame& data, const std::string& formula, SVMOptions options) : formula_(formula) {
    fit(data, std::move(options));
}

double SVM::kernel(const std::vector<double>& a, const std::vector<double>& b) const {
    switch (options_.kernel) {
        case SVMKernel::Linear: {
            double dot = 0.0;
            for (std::size_t i = 0; i < a.size(); ++i) dot += a[i] * b[i];
            return dot;
        }
        case SVMKernel::Polynomial: {
            double dot = 0.0;
            for (std::size_t i = 0; i < a.size(); ++i) dot += a[i] * b[i];
            return std::pow(effective_gamma_ * dot + options_.coef0, options_.degree);
        }
        case SVMKernel::Radial: {
            double sq = 0.0;
            for (std::size_t i = 0; i < a.size(); ++i) {
                const double d = a[i] - b[i];
                sq += d * d;
            }
            return std::exp(-effective_gamma_ * sq);
        }
        case SVMKernel::Sigmoid: {
            double dot = 0.0;
            for (std::size_t i = 0; i < a.size(); ++i) dot += a[i] * b[i];
            return std::tanh(effective_gamma_ * dot + options_.coef0);
        }
    }
    return 0.0;
}

std::vector<double> SVM::scale_row(const std::vector<double>& raw) const {
    std::vector<double> out(raw.size());
    for (std::size_t j = 0; j < raw.size(); ++j) out[j] = (raw[j] - feature_mean_[j]) / feature_scale_[j];
    return out;
}

void SVM::fit(const dstruct::DataFrame& data, SVMOptions options) {
    if (!std::isfinite(options.cost) || options.cost <= 0 ||
        !std::isfinite(options.gamma) || !std::isfinite(options.coef0) ||
        !std::isfinite(options.tolerance) || options.tolerance <= 0 ||
        options.max_passes == 0 || options.max_passes > std::numeric_limits<std::size_t>::max()/20 ||
        options.degree < 1)
        throw std::invalid_argument("SVM: finite positive cost/tolerance, finite gamma/coef0, positive degree and valid max_passes required");
    options_ = options;
    design_  = formula_.resolve(data, ResponseKind::Categorical);

    auto              cd = design_.build_classification_matrix(data);
    const std::size_t n  = cd.X.rows();
    const std::size_t p  = cd.X.cols();

    predictor_names_ = design_.coefficient_names;
    observations_    = n;
    training_labels_ = cd.labels;
    training_rows_ = cd.used_row_indices;
    if (p == 0) throw std::invalid_argument("SVM: at least one predictor is required");
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < p; ++j)
            if (!std::isfinite(cd.X(i,j))) throw std::invalid_argument("SVM: predictors must be finite");

    classes_.clear();
    for (const auto& label : cd.labels)
        if (std::find(classes_.begin(), classes_.end(), label) == classes_.end()) classes_.push_back(label);
    std::sort(classes_.begin(), classes_.end());
    const std::size_t k = classes_.size();
    if (k < 2)
        throw std::invalid_argument("SVM: response must have at least 2 distinct classes, found " + std::to_string(k));

    effective_gamma_ = (options_.gamma > 0.0) ? options_.gamma : 1.0 / static_cast<double>(p);

    feature_mean_.assign(p, 0.0);
    feature_scale_.assign(p, 1.0);
    if (options_.scale) {
        for (std::size_t j = 0; j < p; ++j) {
            double sum = 0.0;
            for (std::size_t i = 0; i < n; ++i) sum += cd.X(i, j);
            feature_mean_[j] = sum / static_cast<double>(n);
        }
        for (std::size_t j = 0; j < p; ++j) {
            double sq = 0.0;
            for (std::size_t i = 0; i < n; ++i) {
                const double d = cd.X(i, j) - feature_mean_[j];
                sq += d * d;
            }
            const double variance = (n > 1) ? sq / static_cast<double>(n - 1) : 0.0;
            feature_scale_[j]     = (variance > 1e-12) ? std::sqrt(variance) : 1.0;
        }
    }

    scaled_training_X_ = linalg::DenseMatrix<double>(n, p, 0.0);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < p; ++j) scaled_training_X_(i, j) = (cd.X(i, j) - feature_mean_[j]) / feature_scale_[j];

    pair_models_.clear();
    for (std::size_t ci = 0; ci < k; ++ci) {
        for (std::size_t cj = ci + 1; cj < k; ++cj) {
            std::vector<std::size_t> members;
            for (std::size_t i = 0; i < n; ++i)
                if (training_labels_[i] == classes_[ci] || training_labels_[i] == classes_[cj]) members.push_back(i);

            const std::size_t m = members.size();
            std::vector<double> y(m);
            for (std::size_t idx = 0; idx < m; ++idx) y[idx] = (training_labels_[members[idx]] == classes_[ci]) ? 1.0 : -1.0;

            linalg::DenseMatrix<double> K(m, m, 0.0);
            for (std::size_t a = 0; a < m; ++a) {
                for (std::size_t bIdx = a; bIdx < m; ++bIdx) {
                    const double value = kernel(scaled_training_X_.row(members[a]), scaled_training_X_.row(members[bIdx]));
                    if (!std::isfinite(value)) throw std::invalid_argument("SVM: non-finite kernel value");
                    K(a, bIdx) = value;
                    K(bIdx, a) = value;
                }
            }

            const auto smo = train_binary_smo(K, y, options_.cost, options_.tolerance, options_.max_passes);

            PairModel pair;
            pair.class_i = ci;
            pair.class_j = cj;
            pair.bias    = smo.bias;
            pair.members = members;
            pair.alpha = smo.alpha;
            pair.sweeps = smo.sweeps;
            double alpha_sum = 0.0, equality = 0.0;
            // Ignore rounding-sized distances from a box face in KKT classification.
            const double box_tolerance = std::min(1e-8, options_.cost*1e-6);
            for (std::size_t a = 0; a < m; ++a) {
                double unbiassed = 0.0;
                for (std::size_t b = 0; b < m; ++b)
                    unbiassed += smo.alpha[b] * y[b] * K(a,b);
                const double decision = unbiassed + smo.bias;
                const double margin = y[a] * decision;
                pair.decisions.push_back(decision);
                pair.norm_squared += smo.alpha[a] * y[a] * unbiassed;
                pair.hinge_loss += std::max(0.0, 1.0-margin);
                alpha_sum += smo.alpha[a];
                equality += smo.alpha[a] * y[a];
                if (smo.alpha[a] < options_.cost-box_tolerance)
                    pair.max_kkt_violation = std::max(pair.max_kkt_violation, 1.0-margin);
                if (smo.alpha[a] > box_tolerance)
                    pair.max_kkt_violation = std::max(pair.max_kkt_violation, margin-1.0);
            }
            pair.dual = alpha_sum - 0.5*pair.norm_squared;
            pair.equality_residual = std::abs(equality);
            for (std::size_t idx = 0; idx < m; ++idx) {
                if (smo.alpha[idx] > 1e-8) {
                    pair.support_indices.push_back(members[idx]);
                    pair.support_alpha_y.push_back(smo.alpha[idx] * y[idx]);
                }
            }
            pair_models_.push_back(std::move(pair));
        }
    }

    fitted_classes_ = classify(scaled_training_X_).class_label;
}

SVMPrediction SVM::classify(const linalg::DenseMatrix<double>& X) const {
    const std::size_t n = X.rows();
    const std::size_t k = classes_.size();

    SVMPrediction result;
    result.class_label.resize(n);
    result.votes.assign(n, std::vector<double>(k, 0.0));

    for (std::size_t row = 0; row < n; ++row) {
        const auto x = X.row(row);

        for (const auto& pair : pair_models_) {
            const double decision = pair_decision(pair, x);

            if (decision >= 0.0) result.votes[row][pair.class_i] += 1.0;
            else result.votes[row][pair.class_j] += 1.0;
        }

        std::size_t best = 0;
        for (std::size_t c = 1; c < k; ++c)
            if (result.votes[row][c] > result.votes[row][best]) best = c;
        result.class_label[row] = classes_[best];
    }
    return result;
}

double SVM::training_accuracy() const {
    std::size_t correct = 0;
    for (std::size_t i = 0; i < fitted_classes_.size(); ++i)
        if (fitted_classes_[i] == training_labels_[i]) ++correct;
    return static_cast<double>(correct) / static_cast<double>(fitted_classes_.size());
}

linalg::DenseMatrix<double> SVM::confusion_matrix() const {
    const std::size_t k = classes_.size();
    linalg::DenseMatrix<double> matrix(k, k, 0.0);
    for (std::size_t i = 0; i < training_labels_.size(); ++i) {
        const auto actual_it    = std::lower_bound(classes_.begin(), classes_.end(), training_labels_[i]);
        const auto predicted_it = std::lower_bound(classes_.begin(), classes_.end(), fitted_classes_[i]);
        matrix(static_cast<std::size_t>(actual_it - classes_.begin()),
               static_cast<std::size_t>(predicted_it - classes_.begin())) += 1.0;
    }
    return matrix;
}

std::size_t SVM::num_support_vectors() const {
    std::vector<bool> is_sv(observations_, false);
    for (const auto& pair : pair_models_)
        for (const auto idx : pair.support_indices) is_sv[idx] = true;
    return static_cast<std::size_t>(std::count(is_sv.begin(), is_sv.end(), true));
}

std::vector<std::size_t> SVM::support_vectors_per_class() const {
    std::vector<bool> is_sv(observations_, false);
    for (const auto& pair : pair_models_)
        for (const auto idx : pair.support_indices) is_sv[idx] = true;

    std::vector<std::size_t> counts(classes_.size(), 0);
    for (std::size_t i = 0; i < observations_; ++i) {
        if (!is_sv[i]) continue;
        const auto it = std::lower_bound(classes_.begin(), classes_.end(), training_labels_[i]);
        ++counts[static_cast<std::size_t>(it - classes_.begin())];
    }
    return counts;
}

std::string SVM::summary() const {
    std::ostringstream out;
    out << "Call:\nsvm(formula = " << formula_.text() << ", kernel = \"" << kernel_name(options_.kernel)
        << "\", cost = " << format_stat(options_.cost) << ")\n\n";

    out << "Parameters:\n";
    out << "   SVM-Type: C-classification (one-vs-one)\n";
    out << " SVM-Kernel: " << kernel_name(options_.kernel) << "\n";
    out << "       cost: " << format_stat(options_.cost) << "\n";
    out << "      gamma: " << format_stat(effective_gamma_) << "\n";
    if (options_.kernel == SVMKernel::Polynomial || options_.kernel == SVMKernel::Sigmoid)
        out << "     coef.0: " << format_stat(options_.coef0) << "\n";
    if (options_.kernel == SVMKernel::Polynomial) out << "     degree: " << options_.degree << "\n";
    out << "\n";

    const auto per_class = support_vectors_per_class();
    out << "Number of Support Vectors: " << num_support_vectors() << "\n(";
    for (const auto count : per_class) out << " " << count;
    out << " )\n\n";

    out << "Number of Classes: " << classes_.size() << "\n";
    out << "Levels: ";
    for (const auto& c : classes_) out << c << " ";
    out << "\n\n";

    out << "Training accuracy: " << format_stat(training_accuracy() * 100.0, 2) << "%  (" << observations_
        << " observations)\n";

    return out.str();
}

void SVM::print_summary(std::ostream& os) const { os << summary(); }

void SVM::print_summary() const { print_summary(std::cout); }

std::vector<std::string> SVM::predict(const dstruct::DataFrame& newdata) const {
    return predict_detail(newdata).class_label;
}

SVMPrediction SVM::predict_detail(const dstruct::DataFrame& newdata) const {
    design_.validate_categorical_levels(newdata, "SVM::predict");
    auto dm = design_.build_matrix(newdata, /*require_response=*/false);
    for (std::size_t i = 0; i < dm.X.rows(); ++i)
        for (std::size_t j = 0; j < dm.X.cols(); ++j) dm.X(i, j) = (dm.X(i, j) - feature_mean_[j]) / feature_scale_[j];

    const auto compact = classify(dm.X);

    SVMPrediction result;
    result.class_label.assign(newdata.nrows(), "");
    result.votes.assign(newdata.nrows(), {});
    for (std::size_t i = 0; i < dm.used_row_indices.size(); ++i) {
        const auto row          = dm.used_row_indices[i];
        result.class_label[row] = compact.class_label[i];
        result.votes[row]       = compact.votes[i];
    }
    return result;
}

double SVM::pair_decision(const PairModel& pair, const std::vector<double>& x) const {
    double value = pair.bias;
    for (std::size_t sv = 0; sv < pair.support_indices.size(); ++sv)
        value += pair.support_alpha_y[sv] * kernel(scaled_training_X_.row(pair.support_indices[sv]), x);
    return value;
}

linalg::DenseMatrix<double> SVM::decision_function(const dstruct::DataFrame& newdata) const {
    design_.validate_categorical_levels(newdata, "SVM::decision_function");
    const auto matrix = design_.build_matrix(newdata, false);
    linalg::DenseMatrix<double> values(newdata.nrows(), pair_models_.size(),
                                      std::numeric_limits<double>::quiet_NaN());
    for (std::size_t i = 0; i < matrix.X.rows(); ++i) {
        const auto x = scale_row(matrix.X.row(i));
        for (std::size_t p = 0; p < pair_models_.size(); ++p)
            values(matrix.used_row_indices[i], p) = pair_decision(pair_models_[p], x);
    }
    return values;
}

dstruct::DataFrame SVM::diagnostics() const {
    std::vector<std::string> positive, negative;
    std::vector<double> observations, support, bias, norm, hinge, primal, dual, gap, equality, kkt, sweeps;
    for (const auto& p : pair_models_) {
        positive.push_back(classes_[p.class_i]); negative.push_back(classes_[p.class_j]);
        observations.push_back(p.members.size()); support.push_back(p.support_indices.size());
        bias.push_back(p.bias); norm.push_back(p.norm_squared); hinge.push_back(p.hinge_loss);
        primal.push_back(0.5*p.norm_squared + options_.cost*p.hinge_loss);
        dual.push_back(p.dual); gap.push_back(primal.back()-p.dual);
        equality.push_back(p.equality_residual); kkt.push_back(p.max_kkt_violation); sweeps.push_back(p.sweeps);
    }
    dstruct::DataFrame out;
    out.add_column("positive_class", positive); out.add_column("negative_class", negative);
    out.add_column("observations", observations); out.add_column("support_vectors", support);
    out.add_column("bias", bias); out.add_column("norm_squared", norm); out.add_column("hinge_loss", hinge);
    out.add_column("primal", primal); out.add_column("dual", dual); out.add_column("gap", gap);
    out.add_column("equality_residual", equality); out.add_column("max_kkt_violation", kkt);
    out.add_column("sweeps", sweeps);
    return out;
}

dstruct::DataFrame SVM::training_details() const {
    std::vector<std::string> positive, negative, labels;
    std::vector<double> rows, signs, alpha, decision, margin, slack;
    for (const auto& p : pair_models_) {
        for (std::size_t i = 0; i < p.members.size(); ++i) {
            const auto row = p.members[i];
            const double y = training_labels_[row] == classes_[p.class_i] ? 1.0 : -1.0;
            positive.push_back(classes_[p.class_i]); negative.push_back(classes_[p.class_j]);
            labels.push_back(training_labels_[row]); rows.push_back(training_rows_[row]); signs.push_back(y);
            alpha.push_back(p.alpha[i]); decision.push_back(p.decisions[i]);
            margin.push_back(y*p.decisions[i]); slack.push_back(std::max(0.0, 1.0-margin.back()));
        }
    }
    dstruct::DataFrame out;
    out.add_column("positive_class", positive); out.add_column("negative_class", negative);
    out.add_column("row", rows); out.add_column("label", labels); out.add_column("y", signs);
    out.add_column("alpha", alpha); out.add_column("decision", decision);
    out.add_column("margin", margin); out.add_column("slack", slack);
    return out;
}

double SVM::error(const dstruct::DataFrame& data) const {
    if (!data.has_column(design_.response_name)) throw std::invalid_argument("SVM::error: missing response");
    const auto predicted = predict_detail(data);
    std::size_t n = 0, wrong = 0;
    for (std::size_t i = 0; i < data.nrows(); ++i) {
        const auto actual = data.optional_string_at(design_.response_name, i);
        if (!actual || predicted.votes[i].empty()) continue;
        ++n; wrong += predicted.class_label[i] != *actual;
    }
    if (!n) throw std::invalid_argument("SVM::error: no evaluable rows");
    return static_cast<double>(wrong)/n;
}

graphics::RPlot SVM::plot_decision_regions(const std::string& x_feature, const std::string& y_feature,
    std::size_t grid_resolution, std::optional<std::array<double,4>> limits) const {
    if (predictor_names_.size() != 2 || x_feature == y_feature)
        throw std::invalid_argument("SVM::plot_decision_regions: two distinct numeric predictors required");
    const auto xi_it = std::find(predictor_names_.begin(), predictor_names_.end(), x_feature);
    const auto yi_it = std::find(predictor_names_.begin(), predictor_names_.end(), y_feature);
    if (xi_it == predictor_names_.end() || yi_it == predictor_names_.end() ||
        !design_.categorical_levels.empty())
        throw std::invalid_argument("SVM::plot_decision_regions: names must match the two numeric predictors");
    const auto xi = static_cast<std::size_t>(xi_it-predictor_names_.begin());
    const auto yi = static_cast<std::size_t>(yi_it-predictor_names_.begin());
    std::vector<double> xs, ys;
    for (std::size_t i = 0; i < observations_; ++i) {
        xs.push_back(scaled_training_X_(i,xi)*feature_scale_[xi]+feature_mean_[xi]);
        ys.push_back(scaled_training_X_(i,yi)*feature_scale_[yi]+feature_mean_[yi]);
    }
    const auto [xmin,xmax] = std::minmax_element(xs.begin(),xs.end());
    const auto [ymin,ymax] = std::minmax_element(ys.begin(),ys.end());
    const double xp = std::max(1e-6,0.05*(*xmax-*xmin)), yp = std::max(1e-6,0.10*(*ymax-*ymin));
    const auto box = limits.value_or(std::array<double,4>{*xmin-xp,*xmax+xp,*ymin-yp,*ymax+yp});
    for (double v : box) if (!std::isfinite(v))
        throw std::invalid_argument("SVM::plot_decision_regions: finite limits required");
    const std::vector<graphics::RGB> colors{{37,99,235},{220,38,38},{22,163,74},{147,51,234}};
    std::vector<graphics::RGB> fills;
    for (std::size_t c = 0; c < classes_.size(); ++c) fills.push_back(graphics::tint(colors[c%colors.size()],0.85));
    auto plot = graphics::RPlot::create();
    // Query is in scaled design order, independent of which axis was requested first.
    graphics::add_class_regions(plot, [&](double x, double y) {
        std::vector<double> z(2);
        z[xi]=(x-feature_mean_[xi])/feature_scale_[xi];
        z[yi]=(y-feature_mean_[yi])/feature_scale_[yi];
        std::vector<std::size_t> votes(classes_.size(),0);
        for (const auto& p : pair_models_) ++votes[pair_decision(p,z)>=0 ? p.class_i : p.class_j];
        return static_cast<std::size_t>(std::max_element(votes.begin(),votes.end())-votes.begin());
    }, box, grid_resolution, fills);
    for (std::size_t c = 0; c < classes_.size(); ++c) {
        std::vector<double> cx,cy;
        for (std::size_t i = 0; i < observations_; ++i) if (training_labels_[i] == classes_[c]) {
            cx.push_back(xs[i]); cy.push_back(ys[i]);
        }
        plot.points(cx,cy,classes_[c],colors[c%colors.size()],4);
    }
    return plot.x_label(x_feature).y_label(y_feature).title("SVM decision regions");
}

} // namespace datamunge::stats
