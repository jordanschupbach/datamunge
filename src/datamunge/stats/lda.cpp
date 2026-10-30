#include <datamunge/stats/lda.hpp>

#include <datamunge/linalg/dense.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace datamunge::stats {

namespace {

std::string format_stat(double v, int precision = 5) {
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

// Solves L x = b for lower-triangular L (forward substitution).
std::vector<double> forward_solve(const linalg::DenseMatrix<double>& L, const std::vector<double>& b) {
    const std::size_t n = L.rows();
    std::vector<double> x(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        double s = b[i];
        for (std::size_t j = 0; j < i; ++j) s -= L(i, j) * x[j];
        x[i] = s / L(i, i);
    }
    return x;
}

// Solves L^T v = y for lower-triangular L (backward substitution).
std::vector<double> back_solve_transpose(const linalg::DenseMatrix<double>& L, const std::vector<double>& y) {
    const std::size_t n = L.rows();
    std::vector<double> v(n, 0.0);
    for (std::size_t i = n; i-- > 0;) {
        double s = y[i];
        for (std::size_t j = i + 1; j < n; ++j) s -= L(j, i) * v[j];
        v[i] = s / L(i, i);
    }
    return v;
}

// Class colors: Tableau 10, the same palette graphics::Splom uses for group_by(), so a report's
// pairs plot and LDA plots agree.
const graphics::RGB kSeriesColors[] = {
    {78, 121, 167}, {242, 142, 43}, {225, 87, 89},  {118, 183, 178}, {89, 161, 79},
    {237, 201, 72}, {176, 122, 161}, {255, 157, 167}, {156, 117, 95}, {186, 176, 172},
};

graphics::RGB class_color(std::size_t c) { return kSeriesColors[c % (sizeof(kSeriesColors) / sizeof(kSeriesColors[0]))]; }

// A linear score a*u + b*v + c over the plane.
struct Linear2D {
    double a{0.0}, b{0.0}, c{0.0};
    [[nodiscard]] double operator()(double u, double v) const { return a * u + b * v + c; }
};

using Polygon2D = std::vector<std::array<double, 2>>;

// Sutherland-Hodgman: the part of a convex polygon where f >= 0.
Polygon2D clip_half_plane(const Polygon2D& poly, const Linear2D& f) {
    Polygon2D out;
    for (std::size_t i = 0; i < poly.size(); ++i) {
        const auto& p = poly[i];
        const auto& q = poly[(i + 1) % poly.size()];
        const double fp = f(p[0], p[1]);
        const double fq = f(q[0], q[1]);
        if (fp >= 0.0) out.push_back(p);
        if ((fp >= 0.0) != (fq >= 0.0)) {
            const double t = fp / (fp - fq);
            out.push_back({p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])});
        }
    }
    return out;
}

// Plot box around the points: 5% padding in x and 10% in y, as RPlot pads by default.
std::array<double, 4> padded_box(const std::vector<double>& xs, const std::vector<double>& ys) {
    auto [x_lo, x_hi] = std::minmax_element(xs.begin(), xs.end());
    auto [y_lo, y_hi] = std::minmax_element(ys.begin(), ys.end());
    double x0 = *x_lo, x1 = *x_hi, y0 = *y_lo, y1 = *y_hi;
    if (x1 - x0 <= 0.0) { x0 -= 1.0; x1 += 1.0; }
    if (y1 - y0 <= 0.0) { y0 -= 1.0; y1 += 1.0; }
    const double dx = 0.05 * (x1 - x0);
    const double dy = 0.10 * (y1 - y0);
    return {x0 - dx, x1 + dx, y0 - dy, y1 + dy};
}

// Shades each class's region {score_k >= score_j for all j} inside `box` and draws the
// boundaries between neighbouring regions, then fixes the axes to `box` so the shading fills
// the panel. Regions are convex, so clipping the box by one half-plane per rival is exact.
void add_decision_regions(graphics::RPlot& plot, const std::vector<Linear2D>& scores, const std::array<double, 4>& box) {
    const Polygon2D rect = {{box[0], box[2]}, {box[1], box[2]}, {box[1], box[3]}, {box[0], box[3]}};
    const double    diag = std::hypot(box[1] - box[0], box[3] - box[2]);
    std::vector<double> bx0, by0, bx1, by1;
    for (std::size_t k = 0; k < scores.size(); ++k) {
        Polygon2D region = rect;
        for (std::size_t j = 0; j < scores.size() && region.size() >= 3; ++j) {
            if (j == k) continue;
            region = clip_half_plane(region, {scores[k].a - scores[j].a, scores[k].b - scores[j].b, scores[k].c - scores[j].c});
        }
        if (region.size() < 3) continue;
        std::vector<double> xs, ys;
        for (const auto& pt : region) {
            xs.push_back(pt[0]);
            ys.push_back(pt[1]);
        }
        auto fill = class_color(k);
        fill.alpha = 0.14;
        plot.polygon(xs, ys, fill, true);

        // Edges of this region lying on the line score_k = score_j are the k|j boundary.
        for (std::size_t j = k + 1; j < scores.size(); ++j) {
            const Linear2D g{scores[k].a - scores[j].a, scores[k].b - scores[j].b, scores[k].c - scores[j].c};
            const double   norm = std::hypot(g.a, g.b);
            if (norm == 0.0) continue;
            const auto on_line = [&](const std::array<double, 2>& pt) { return std::abs(g(pt[0], pt[1])) / norm <= 1e-9 * diag; };
            for (std::size_t i = 0; i < region.size(); ++i) {
                const auto& p = region[i];
                const auto& q = region[(i + 1) % region.size()];
                if (on_line(p) && on_line(q)) {
                    bx0.push_back(p[0]);
                    by0.push_back(p[1]);
                    bx1.push_back(q[0]);
                    by1.push_back(q[1]);
                }
            }
        }
    }
    if (!bx0.empty()) plot.segments(bx0, by0, bx1, by1, graphics::RGB{75, 85, 99}, 1.5);
    plot.x_limits(box[0], box[1]).y_limits(box[2], box[3]);
}

} // namespace

LDA::LDA(const dstruct::DataFrame& data, const std::string& formula, LDAOptions options) : formula_(formula) {
    fit(data, std::move(options));
}

void LDA::fit(const dstruct::DataFrame& data, LDAOptions options) {
    design_ = formula_.resolve(data, ResponseKind::Categorical);

    auto              cd = design_.build_classification_matrix(data);
    const std::size_t n  = cd.X.rows();
    const std::size_t p  = cd.X.cols();

    predictor_names_ = design_.coefficient_names;
    observations_    = n;
    training_labels_ = cd.labels;
    training_predictors_.assign(n, std::vector<double>(p, 0.0));
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < p; ++j) training_predictors_[i][j] = cd.X(i, j);

    classes_.clear();
    for (const auto& label : cd.labels)
        if (std::find(classes_.begin(), classes_.end(), label) == classes_.end()) classes_.push_back(label);
    std::sort(classes_.begin(), classes_.end());
    const std::size_t k = classes_.size();
    if (k < 2)
        throw std::invalid_argument("LDA: response must have at least 2 distinct classes, found "
                                    + std::to_string(k));

    std::vector<std::size_t> class_index(n);
    std::vector<std::size_t> class_count(k, 0);
    for (std::size_t i = 0; i < n; ++i) {
        const auto it   = std::lower_bound(classes_.begin(), classes_.end(), cd.labels[i]);
        class_index[i]  = static_cast<std::size_t>(it - classes_.begin());
        ++class_count[class_index[i]];
    }

    if (n <= k + p)
        throw std::runtime_error("LDA: not enough observations (" + std::to_string(n) + ") for " + std::to_string(k)
                                 + " classes and " + std::to_string(p) + " predictors");
    for (std::size_t c = 0; c < k; ++c)
        if (class_count[c] < 2)
            throw std::runtime_error("LDA: class '" + classes_[c] + "' has fewer than 2 observations");

    priors_.assign(k, 0.0);
    if (options.priors) {
        if (options.priors->size() != k)
            throw std::invalid_argument("LDA: priors size must equal the number of classes (" + std::to_string(k)
                                        + ")");
        double sum = 0.0;
        for (const double pr : *options.priors) {
            if (!(pr > 0.0)) throw std::invalid_argument("LDA: priors must be strictly positive");
            sum += pr;
        }
        if (std::fabs(sum - 1.0) > 1e-6) throw std::invalid_argument("LDA: priors must sum to 1");
        priors_ = *options.priors;
    } else {
        for (std::size_t c = 0; c < k; ++c)
            priors_[c] = static_cast<double>(class_count[c]) / static_cast<double>(n);
    }

    group_means_.assign(k, std::vector<double>(p, 0.0));
    overall_mean_.assign(p, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < p; ++j) {
            group_means_[class_index[i]][j] += cd.X(i, j);
            overall_mean_[j] += cd.X(i, j);
        }
    }
    for (std::size_t c = 0; c < k; ++c)
        for (std::size_t j = 0; j < p; ++j) group_means_[c][j] /= static_cast<double>(class_count[c]);
    for (std::size_t j = 0; j < p; ++j) overall_mean_[j] /= static_cast<double>(n);

    linalg::DenseMatrix<double> W(p, p, 0.0);
    linalg::DenseMatrix<double> B(p, p, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        const auto& mean = group_means_[class_index[i]];
        for (std::size_t a = 0; a < p; ++a) {
            const double da = cd.X(i, a) - mean[a];
            for (std::size_t b = 0; b < p; ++b) W(a, b) += da * (cd.X(i, b) - mean[b]);
        }
    }
    for (std::size_t c = 0; c < k; ++c) {
        for (std::size_t a = 0; a < p; ++a) {
            const double da = group_means_[c][a] - overall_mean_[a];
            for (std::size_t b = 0; b < p; ++b)
                B(a, b) += static_cast<double>(class_count[c]) * da * (group_means_[c][b] - overall_mean_[b]);
        }
    }

    const auto chol = linalg::cholesky(W);
    if (!chol.ok)
        throw std::runtime_error("LDA: pooled within-class covariance is not positive definite "
                                 "(too few observations relative to predictors, or collinear predictors)");
    const auto& L = chol.L;

    // Cache Bayes discriminant terms:
    //   Sigma_w = W / (n - k)  =>  Sigma_w^{-1} mu_c = (n - k) * W^{-1} mu_c
    const double dof_scale = static_cast<double>(n - k);
    precision_means_.assign(k, std::vector<double>(p, 0.0));
    mahalanobis_offset_.assign(k, 0.0);
    for (std::size_t c = 0; c < k; ++c) {
        const auto w_inv_mu = chol.solve(group_means_[c]);
        for (std::size_t j = 0; j < p; ++j) precision_means_[c][j] = dof_scale * w_inv_mu[j];
        double offset = 0.0;
        for (std::size_t j = 0; j < p; ++j) offset += group_means_[c][j] * precision_means_[c][j];
        mahalanobis_offset_[c] = offset;
    }

    // Discriminant axes: generalized eigenproblem B v = lambda W v, reduced
    // via the Cholesky factor W = L L^T to a standard symmetric eigenproblem
    // M y = lambda y with M = L^{-1} B L^{-T}, then v = L^{-T} y.
    linalg::DenseMatrix<double> Y(p, p, 0.0); // Y = L^{-1} B
    for (std::size_t col = 0; col < p; ++col) {
        const auto solved = forward_solve(L, B.col(col));
        for (std::size_t i = 0; i < p; ++i) Y(i, col) = solved[i];
    }
    linalg::DenseMatrix<double> M(p, p, 0.0); // M = L^{-1} Y^T == L^{-1} B L^{-T}
    for (std::size_t col = 0; col < p; ++col) {
        const auto solved = forward_solve(L, Y.row(col));
        for (std::size_t i = 0; i < p; ++i) M(i, col) = solved[i];
    }
    for (std::size_t i = 0; i < p; ++i)
        for (std::size_t j = i + 1; j < p; ++j) {
            const double avg = 0.5 * (M(i, j) + M(j, i));
            M(i, j) = avg;
            M(j, i) = avg;
        }

    const auto       eig    = linalg::jacobi_eigen(M);
    const std::size_t num_ld = std::min(k - 1, p);

    scaling_ = linalg::DenseMatrix<double>(p, num_ld, 0.0);
    proportion_of_trace_.assign(num_ld, 0.0);
    double eigen_total = 0.0;
    for (const double v : eig.eigenvalues) eigen_total += std::max(v, 0.0);
    const double sqrt_dof = std::sqrt(dof_scale);
    for (std::size_t ld = 0; ld < num_ld; ++ld) {
        const auto v = back_solve_transpose(L, eig.eigenvectors.col(ld));
        for (std::size_t j = 0; j < p; ++j) scaling_(j, ld) = v[j] * sqrt_dof;
        proportion_of_trace_[ld] = (eigen_total > 0.0) ? std::max(eig.eigenvalues[ld], 0.0) / eigen_total : 0.0;
    }

    const auto training_result = classify(cd.X);
    fitted_classes_          = training_result.class_label;
    training_discriminants_  = training_result.discriminants;
}

LDAPrediction LDA::classify(const linalg::DenseMatrix<double>& X) const {
    const std::size_t n      = X.rows();
    const std::size_t k      = classes_.size();
    const std::size_t num_ld = scaling_.cols();

    LDAPrediction result;
    result.class_label.resize(n);
    result.posterior.assign(n, std::vector<double>(k, 0.0));
    result.discriminants.assign(n, std::vector<double>(num_ld, 0.0));

    std::vector<double> delta(k);
    std::vector<double> expd(k);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t c = 0; c < k; ++c) {
            double dot = 0.0;
            for (std::size_t j = 0; j < X.cols(); ++j) dot += X(i, j) * precision_means_[c][j];
            delta[c] = dot - 0.5 * mahalanobis_offset_[c] + std::log(priors_[c]);
        }
        const double max_delta = *std::max_element(delta.begin(), delta.end());
        double       sum_exp   = 0.0;
        for (std::size_t c = 0; c < k; ++c) {
            expd[c] = std::exp(delta[c] - max_delta);
            sum_exp += expd[c];
        }
        std::size_t best = 0;
        for (std::size_t c = 0; c < k; ++c) {
            result.posterior[i][c] = expd[c] / sum_exp;
            if (delta[c] > delta[best]) best = c;
        }
        result.class_label[i] = classes_[best];

        for (std::size_t ld = 0; ld < num_ld; ++ld) {
            double proj = 0.0;
            for (std::size_t j = 0; j < X.cols(); ++j) proj += (X(i, j) - overall_mean_[j]) * scaling_(j, ld);
            result.discriminants[i][ld] = proj;
        }
    }
    return result;
}

double LDA::training_accuracy() const {
    std::size_t correct = 0;
    for (std::size_t i = 0; i < fitted_classes_.size(); ++i)
        if (fitted_classes_[i] == training_labels_[i]) ++correct;
    return static_cast<double>(correct) / static_cast<double>(fitted_classes_.size());
}

linalg::DenseMatrix<double> LDA::confusion_matrix() const {
    const std::size_t k = classes_.size();
    linalg::DenseMatrix<double> matrix(k, k, 0.0);
    for (std::size_t i = 0; i < training_labels_.size(); ++i) {
        const auto actual_it    = std::lower_bound(classes_.begin(), classes_.end(), training_labels_[i]);
        const auto predicted_it = std::lower_bound(classes_.begin(), classes_.end(), fitted_classes_[i]);
        const auto actual       = static_cast<std::size_t>(actual_it - classes_.begin());
        const auto predicted    = static_cast<std::size_t>(predicted_it - classes_.begin());
        matrix(actual, predicted) += 1.0;
    }
    return matrix;
}

std::vector<std::string> LDA::predict(const dstruct::DataFrame& newdata) const {
    return predict_detail(newdata).class_label;
}

LDAPrediction LDA::predict_detail(const dstruct::DataFrame& newdata) const {
    design_.validate_categorical_levels(newdata, "LDA::predict");
    const auto dm = design_.build_matrix(newdata, /*require_response=*/false);
    const auto compact = classify(dm.X);

    LDAPrediction result;
    result.class_label.assign(newdata.nrows(), "");
    result.posterior.assign(newdata.nrows(), {});
    result.discriminants.assign(newdata.nrows(), {});
    for (std::size_t i = 0; i < dm.used_row_indices.size(); ++i) {
        const auto row               = dm.used_row_indices[i];
        result.class_label[row]      = compact.class_label[i];
        result.posterior[row]        = compact.posterior[i];
        result.discriminants[row]    = compact.discriminants[i];
    }
    return result;
}

std::string LDA::summary() const {
    std::ostringstream out;
    out << "Call:\nlda(formula = " << formula_.text() << ")\n\n";

    out << "Prior probabilities of groups:\n";
    for (const auto& c : classes_) out << std::setw(12) << c;
    out << "\n";
    for (const double pr : priors_) out << std::setw(12) << format_stat(pr, 4);
    out << "\n\n";

    out << "Group means:\n";
    out << std::setw(14) << "";
    for (const auto& name : predictor_names_) out << std::setw(14) << name;
    out << "\n";
    for (std::size_t c = 0; c < classes_.size(); ++c) {
        out << std::left << std::setw(14) << classes_[c] << std::right;
        for (const double v : group_means_[c]) out << std::setw(14) << format_stat(v, 4);
        out << "\n";
    }
    out << "\n";

    out << "Coefficients of linear discriminants:\n";
    out << std::setw(14) << "";
    for (std::size_t ld = 0; ld < scaling_.cols(); ++ld) out << std::setw(14) << ("LD" + std::to_string(ld + 1));
    out << "\n";
    for (std::size_t j = 0; j < predictor_names_.size(); ++j) {
        out << std::left << std::setw(14) << predictor_names_[j] << std::right;
        for (std::size_t ld = 0; ld < scaling_.cols(); ++ld) out << std::setw(14) << format_stat(scaling_(j, ld));
        out << "\n";
    }
    out << "\n";

    out << "Proportion of trace:\n";
    for (std::size_t ld = 0; ld < proportion_of_trace_.size(); ++ld)
        out << std::setw(10) << ("LD" + std::to_string(ld + 1));
    out << "\n";
    for (const double p : proportion_of_trace_) out << std::setw(10) << format_stat(p, 4);
    out << "\n\n";

    out << "Training accuracy: " << format_stat(training_accuracy() * 100.0, 2) << "%  (" << observations_
        << " observations, " << classes_.size() << " classes)\n";

    return out.str();
}

void LDA::print_summary(std::ostream& os) const { os << summary(); }

void LDA::print_summary() const { print_summary(std::cout); }

graphics::RPlot LDA::plot_discriminants(bool decision_regions) const {
    auto plot = graphics::RPlot::create();
    const bool has_ld2 = scaling_.cols() > 1;

    std::vector<double> all_x, all_y;
    for (const auto& z : training_discriminants_) {
        all_x.push_back(z[0]);
        all_y.push_back(has_ld2 ? z[1] : 0.0);
    }

    if (decision_regions && !all_x.empty()) {
        // In discriminant space the within-class covariance is the identity, so LDA's score is
        // -||z - m_k||^2 / 2 + log(prior_k) = z.m_k - ||m_k||^2 / 2 + log(prior_k) + const(z),
        // where m_k is class k's mean projected onto the axes shown.
        std::vector<Linear2D> scores;
        for (std::size_t c = 0; c < classes_.size(); ++c) {
            double m[2] = {0.0, 0.0};
            for (std::size_t ld = 0; ld < std::min<std::size_t>(2, scaling_.cols()); ++ld)
                for (std::size_t j = 0; j < overall_mean_.size(); ++j)
                    m[ld] += (group_means_[c][j] - overall_mean_[j]) * scaling_(j, ld);
            scores.push_back({m[0], m[1], -0.5 * (m[0] * m[0] + m[1] * m[1]) + std::log(priors_[c])});
        }
        add_decision_regions(plot, scores, padded_box(all_x, all_y));
    }

    for (std::size_t c = 0; c < classes_.size(); ++c) {
        std::vector<double> xs;
        std::vector<double> ys;
        for (std::size_t i = 0; i < training_labels_.size(); ++i) {
            if (training_labels_[i] != classes_[c]) continue;
            xs.push_back(all_x[i]);
            ys.push_back(all_y[i]);
        }
        plot.points(xs, ys, classes_[c], class_color(c));
    }

    plot.title("Linear Discriminant Analysis").x_label("LD1").y_label(has_ld2 ? "LD2" : "(single discriminant)");
    return plot;
}

graphics::RPlot LDA::plot_decision_regions() const {
    if (predictor_names_.size() != 2) {
        throw std::logic_error("LDA::plot_decision_regions: the model must have exactly two predictors, not " +
                               std::to_string(predictor_names_.size()));
    }
    auto plot = graphics::RPlot::create();
    std::vector<double> all_x, all_y;
    for (const auto& x : training_predictors_) {
        all_x.push_back(x[0]);
        all_y.push_back(x[1]);
    }
    if (!all_x.empty()) {
        std::vector<Linear2D> scores;
        for (std::size_t c = 0; c < classes_.size(); ++c)
            scores.push_back({precision_means_[c][0], precision_means_[c][1],
                              -0.5 * mahalanobis_offset_[c] + std::log(priors_[c])});
        add_decision_regions(plot, scores, padded_box(all_x, all_y));
    }
    for (std::size_t c = 0; c < classes_.size(); ++c) {
        std::vector<double> xs, ys;
        for (std::size_t i = 0; i < training_labels_.size(); ++i) {
            if (training_labels_[i] != classes_[c]) continue;
            xs.push_back(all_x[i]);
            ys.push_back(all_y[i]);
        }
        plot.points(xs, ys, classes_[c], class_color(c));
    }
    plot.title("LDA decision regions").x_label(predictor_names_[0]).y_label(predictor_names_[1]);
    return plot;
}

dstruct::DataFrame LDA::group_means_frame() const {
    dstruct::DataFrame frame;
    frame.add_column("class", classes_);
    for (std::size_t j = 0; j < predictor_names_.size(); ++j) {
        std::vector<double> column;
        for (const auto& row : group_means_) column.push_back(row[j]);
        frame.add_column(predictor_names_[j], column);
    }
    return frame;
}

dstruct::DataFrame LDA::coefficients_frame() const {
    dstruct::DataFrame frame;
    frame.add_column("predictor", predictor_names_);
    for (std::size_t ld = 0; ld < scaling_.cols(); ++ld) {
        std::vector<double> column;
        for (std::size_t j = 0; j < scaling_.rows(); ++j) column.push_back(scaling_(j, ld));
        frame.add_column("LD" + std::to_string(ld + 1), column);
    }
    return frame;
}

dstruct::DataFrame LDA::confusion_frame() const {
    dstruct::DataFrame frame;
    frame.add_column("actual", classes_);
    const auto counts = confusion_matrix();
    for (std::size_t predicted = 0; predicted < classes_.size(); ++predicted) {
        std::vector<double> column;
        for (std::size_t actual = 0; actual < classes_.size(); ++actual) column.push_back(counts(actual, predicted));
        frame.add_column(classes_[predicted], column);
    }
    return frame;
}

std::string LDA::to_md(int digits) const {
    std::ostringstream out;
    out << "Linear discriminant analysis of `" << formula_.text() << "` on " << observations_ << " observations and "
        << classes_.size() << " classes; training accuracy " << format_stat(training_accuracy() * 100.0, 2) << "%.\n\n";

    dstruct::DataFrame priors;
    for (std::size_t c = 0; c < classes_.size(); ++c) priors.add_column(classes_[c], std::vector<double>{priors_[c]});
    out << "Prior probabilities of groups:\n\n" << priors.to_md(digits) << "\n";
    out << "Group means:\n\n" << group_means_frame().to_md(digits) << "\n";
    out << "Coefficients of linear discriminants:\n\n" << coefficients_frame().to_md(digits) << "\n";

    dstruct::DataFrame trace;
    for (std::size_t ld = 0; ld < proportion_of_trace_.size(); ++ld)
        trace.add_column("LD" + std::to_string(ld + 1), std::vector<double>{proportion_of_trace_[ld]});
    out << "Proportion of trace:\n\n" << trace.to_md(digits);
    return out.str();
}

void LDA::save_discriminant_plot(const std::string& path) const { plot_discriminants().save(path); }

} // namespace datamunge::stats

namespace datamunge {

graphics::Figure plot(const stats::LDA& model) {
    auto discriminants = model.plot_discriminants();
    if (model.predictor_names().size() != 2) return discriminants;
    discriminants.title("Discriminant space");
    auto regions = model.plot_decision_regions();
    regions.title("Predictor space");
    return graphics::RLayout::create(1, 2)
        .add(discriminants)
        .add(regions)
        .size(1200, 520)
        .fit_panels();
}

} // namespace datamunge
