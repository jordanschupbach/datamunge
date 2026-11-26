/// @file survey.cpp
/// @brief Weighted totals, linearized ratio residuals, with-replacement, Sen–Yates–Grundy and Horvitz–Thompson variances, stratified ultimate-cluster variances, and replicate weights (Hadamard-balanced BRR, Fay, JK1, JKn).

#include <datamunge/stats/survey.hpp>

#include <datamunge/random/distributions.hpp>

#include <algorithm>
#include <cmath>
#include <map>
#include <utility>
#include <stdexcept>

namespace datamunge::stats {

/// @brief Internal helpers for survey estimation.
namespace {

/// @brief With-replacement variance estimate \f$\frac{n}{n-1}\sum(z_i-\bar z)^2\f$ of a sum of z.
/// @param z Weighted values.
/// @return Variance of their sum.
double with_replacement_variance(const std::vector<double>& z) {
    const double n = static_cast<double>(z.size());
    double mean = 0;
    for (const double v : z)
        mean += v/n;
    double ss = 0;
    for (const double v : z)
        ss += (v-mean)*(v-mean);
    return n/(n-1)*ss;
}

/// @brief Builds an estimate with a normal interval.
/// @param value Point estimate.
/// @param variance Variance estimate (a negative value gives a zero standard error).
/// @param z Normal quantile.
/// @return The estimate.
SurveyEstimate make_estimate(double value, double variance, double z) {
    const double se = std::sqrt(std::max(variance,0.0));
    return {value,se,value-z*se,value+z*se,variance};
}

/// @brief Exact design variance of the estimated total of u from joint inclusion probabilities.
/// @param u Values of the sampled units.
/// @param joint Joint inclusion probabilities (diagonal: first order).
/// @param method Sen–Yates–Grundy or Horvitz–Thompson form.
/// @return The variance estimate (possibly negative).
double joint_variance(const std::vector<double>& u, const linalg::DenseMatrix<double>& joint, SurveyVariance method) {
    const std::size_t n = u.size();
    std::vector<double> z(n);
    for (std::size_t i = 0; i < n; ++i)
        z[i] = u[i]/joint(i,i);
    double v = 0;
    if (method == SurveyVariance::sen_yates_grundy) {
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = i+1; j < n; ++j) {
                const double d = z[i]-z[j];
                v += (joint(i,i)*joint(j,j)-joint(i,j))/joint(i,j)*d*d;
            }
        return v;
    }
    for (std::size_t i = 0; i < n; ++i) {
        v += (1-joint(i,i))*z[i]*z[i];
        for (std::size_t j = i+1; j < n; ++j)
            v += 2*(joint(i,j)-joint(i,i)*joint(j,j))/joint(i,j)*z[i]*z[j];
    }
    return v;
}

/// @brief Validated design with dense stratum and PSU indices.
struct PreparedDesign {
    std::size_t n{};                          ///< Units.
    std::vector<double> weights;              ///< Weights.
    std::vector<std::size_t> psu;             ///< Dense PSU index per unit.
    std::vector<std::size_t> psu_stratum;     ///< Stratum index of each PSU.
    std::vector<double> stratum_fraction;     ///< First-stage sampling fraction per stratum (0 without correction).
    std::vector<std::size_t> stratum_psus;    ///< Sampled PSUs per stratum.
    LonelyPsu lonely{};                       ///< Lonely-PSU policy.
    double z{};                               ///< Normal quantile for intervals.
};

/// @brief Validates a design and assigns dense stratum and PSU indices.
/// @param d Design.
/// @param n Expected number of units.
/// @return The prepared design.
/// @throws std::invalid_argument For mismatched sizes, nonpositive weights, inconsistent or too small
/// population PSU counts, or invalid confidence.
PreparedDesign prepare(const SurveyDesign& d, std::size_t n) {
    if (n == 0 || d.weights.size() != n || (!d.strata.empty() && d.strata.size() != n) ||
        (!d.clusters.empty() && d.clusters.size() != n) || (!d.population_psus.empty() && d.population_psus.size() != n))
        throw std::invalid_argument("survey: design vectors must have one entry per unit");
    if (!(d.confidence > 0 && d.confidence < 1))
        throw std::invalid_argument("survey: confidence must lie in (0, 1)");
    if (d.lonely_psu != LonelyPsu::fail && d.lonely_psu != LonelyPsu::certainty)
        throw std::invalid_argument("survey: unknown lonely-PSU policy");
    PreparedDesign p;
    p.n = n;
    p.weights = d.weights;
    p.lonely = d.lonely_psu;
    p.z = random::normal_quantile(.5+d.confidence/2);
    for (const double w : d.weights)
        if (!std::isfinite(w) || w <= 0)
            throw std::invalid_argument("survey: weights must be finite and positive");
    std::map<int,std::size_t> stratum_index;
    std::map<std::pair<std::size_t,int>,std::size_t> psu_index;
    std::vector<double> population;
    for (std::size_t i = 0; i < n; ++i) {
        const int s_label = d.strata.empty() ? 0 : d.strata[i];
        const auto [s_it,s_new] = stratum_index.emplace(s_label,stratum_index.size());
        const std::size_t s = s_it->second;
        if (s_new) {
            p.stratum_psus.push_back(0);
            population.push_back(d.population_psus.empty() ? 0.0 : d.population_psus[i]);
        }
        if (!d.population_psus.empty() && d.population_psus[i] != population[s])
            throw std::invalid_argument("survey: population PSU counts must be constant within strata");
        const int c_label = d.clusters.empty() ? static_cast<int>(i) : d.clusters[i];
        const auto [c_it,c_new] = psu_index.emplace(std::pair{s,c_label},psu_index.size());
        if (c_new) {
            p.psu_stratum.push_back(s);
            ++p.stratum_psus[s];
        }
        p.psu.push_back(c_it->second);
    }
    for (std::size_t s = 0; s < population.size(); ++s) {
        if (d.population_psus.empty()) {
            p.stratum_fraction.push_back(0);
            continue;
        }
        if (!std::isfinite(population[s]) || population[s] < static_cast<double>(p.stratum_psus[s]))
            throw std::invalid_argument("survey: population PSU counts must be at least the sampled PSUs");
        p.stratum_fraction.push_back(static_cast<double>(p.stratum_psus[s])/population[s]);
    }
    return p;
}

/// @brief Stratified ultimate-cluster variance of the estimated total of linearized values.
/// @param p Prepared design.
/// @param z Linearized (already weighted) value per unit.
/// @return The variance.
/// @throws std::invalid_argument For a lonely PSU under LonelyPsu::fail.
double linearized_variance(const PreparedDesign& p, const std::vector<double>& z) {
    std::vector<double> psu_total(p.psu_stratum.size(),0.0);
    for (std::size_t i = 0; i < p.n; ++i)
        psu_total[p.psu[i]] += z[i];
    const std::size_t strata = p.stratum_psus.size();
    std::vector<double> mean(strata,0.0);
    for (std::size_t c = 0; c < psu_total.size(); ++c)
        mean[p.psu_stratum[c]] += psu_total[c]/static_cast<double>(p.stratum_psus[p.psu_stratum[c]]);
    std::vector<double> ss(strata,0.0);
    for (std::size_t c = 0; c < psu_total.size(); ++c) {
        const double dev = psu_total[c]-mean[p.psu_stratum[c]];
        ss[p.psu_stratum[c]] += dev*dev;
    }
    double v = 0;
    for (std::size_t s = 0; s < strata; ++s) {
        const double nh = static_cast<double>(p.stratum_psus[s]);
        if (p.stratum_psus[s] == 1) {
            if (p.stratum_fraction[s] == 1 || p.lonely == LonelyPsu::certainty)
                continue;
            throw std::invalid_argument("survey: stratum with a single PSU (set lonely_psu)");
        }
        v += (1-p.stratum_fraction[s])*nh/(nh-1)*ss[s];
    }
    return v;
}

/// @brief Multivariate ultimate-cluster covariance of the totals of several score columns.
/// @param p Prepared design.
/// @param z Scores, one row per unit.
/// @return Covariance matrix.
/// @throws std::invalid_argument For a single-PSU stratum under LonelyPsu::fail.
linalg::DenseMatrix<double> linearized_covariance(const PreparedDesign& p, const linalg::DenseMatrix<double>& z) {
    const std::size_t q = z.cols(), clusters = p.psu_stratum.size(), strata = p.stratum_psus.size();
    linalg::DenseMatrix<double> total(clusters,q), mean(strata,q), v(q,q);
    for (std::size_t i = 0; i < p.n; ++i)
        for (std::size_t a = 0; a < q; ++a)
            total(p.psu[i],a) += z(i,a);
    for (std::size_t c = 0; c < clusters; ++c)
        for (std::size_t a = 0; a < q; ++a)
            mean(p.psu_stratum[c],a) += total(c,a)/static_cast<double>(p.stratum_psus[p.psu_stratum[c]]);
    for (std::size_t s = 0; s < strata; ++s)
        if (p.stratum_psus[s] == 1 && p.stratum_fraction[s] != 1 && p.lonely != LonelyPsu::certainty)
            throw std::invalid_argument("survey: stratum with a single PSU (set lonely_psu)");
    for (std::size_t c = 0; c < clusters; ++c) {
        const std::size_t s = p.psu_stratum[c];
        const double nh = static_cast<double>(p.stratum_psus[s]);
        if (p.stratum_psus[s] == 1)
            continue;
        const double factor = (1-p.stratum_fraction[s])*nh/(nh-1);
        for (std::size_t a = 0; a < q; ++a)
            for (std::size_t b = 0; b < q; ++b)
                v(a,b) += factor*(total(c,a)-mean(s,a))*(total(c,b)-mean(s,b));
    }
    return v;
}

/// @brief Checks a variable and domain, returning domain indicators.
/// @param p Prepared design.
/// @param y Variable.
/// @param domain Domain mask (empty: all).
/// @return Indicator per unit.
/// @throws std::invalid_argument For mismatched sizes, non-finite values or an empty domain.
std::vector<double> indicators(const PreparedDesign& p, const std::vector<double>& y, const std::vector<bool>& domain) {
    if (y.size() != p.n || (!domain.empty() && domain.size() != p.n))
        throw std::invalid_argument("survey: variables and domain must have one entry per unit");
    std::vector<double> d(p.n,1.0);
    bool any = false;
    for (std::size_t i = 0; i < p.n; ++i) {
        if (!domain.empty() && !domain[i])
            d[i] = 0;
        else if (!std::isfinite(y[i]))
            throw std::invalid_argument("survey: values in the domain must be finite");
        any = any || d[i] > 0;
    }
    if (!any)
        throw std::invalid_argument("survey: the domain is empty");
    return d;
}

/// @brief Whether p is an odd prime.
/// @param p Candidate.
/// @return True for an odd prime.
bool odd_prime(std::size_t p) {
    if (p < 3 || p%2 == 0)
        return false;
    for (std::size_t d = 3; d*d <= p; d += 2)
        if (p%d == 0)
            return false;
    return true;
}

/// @brief Hadamard matrix of a given order by Sylvester doubling of 1, 2 or a Paley (type I) order.
/// @param m Order.
/// @return The matrix with entries +-1, or an empty matrix if the order is not constructible this way.
std::vector<std::vector<int>> hadamard(std::size_t m) {
    std::size_t base = m;
    while (base%2 == 0 && base > 2 && !(base%4 == 0 && odd_prime(base-1)))
        base /= 2;
    std::vector<std::vector<int>> h;
    if (base == 1 || base == 2) {
        h = base == 1 ? std::vector<std::vector<int>>{{1}} : std::vector<std::vector<int>>{{1,1},{1,-1}};
    } else if (base%4 == 0 && odd_prime(base-1)) {
        // Paley I: H = I + S with S = [[0, 1'], [-1, Q]] and Q_ij = chi(j - i) for p = 3 mod 4.
        const std::size_t p = base-1;
        std::vector<int> chi(p,-1);
        chi[0] = 0;
        for (std::size_t x = 1; x < p; ++x)
            chi[(x*x)%p] = 1;
        h.assign(base,std::vector<int>(base,0));
        for (std::size_t j = 1; j < base; ++j) {
            h[0][j] = 1;
            h[j][0] = -1;
        }
        for (std::size_t i = 0; i < p; ++i)
            for (std::size_t j = 0; j < p; ++j)
                h[i+1][j+1] = chi[(j+p-i)%p];
        for (std::size_t i = 0; i < base; ++i)
            h[i][i] += 1;
    } else {
        return {};
    }
    while (h.size() < m) {
        const std::size_t k = h.size();
        std::vector<std::vector<int>> d(2*k,std::vector<int>(2*k));
        for (std::size_t i = 0; i < k; ++i)
            for (std::size_t j = 0; j < k; ++j) {
                d[i][j] = d[i][j+k] = d[i+k][j] = h[i][j];
                d[i+k][j+k] = -h[i][j];
            }
        h = std::move(d);
    }
    // Normalize so the first column is all +1, making every other column sum to zero.
    for (auto& row : h)
        if (row[0] < 0)
            for (int& v : row)
                v = -v;
    return h;
}

/// @brief Weighted domain ratio estimate for one set of weights.
/// @param w Weights.
/// @param y Numerator values.
/// @param x Denominator values (empty: a total).
/// @param d Domain indicators.
/// @return The ratio (or total).
/// @throws std::invalid_argument For a zero denominator.
double weighted_ratio(const std::vector<double>& w, const std::vector<double>& y, const std::vector<double>* x,
                      const std::vector<double>& d) {
    double ty = 0, tx = 0;
    for (std::size_t i = 0; i < w.size(); ++i)
        if (d[i] > 0) {
            ty += w[i]*y[i];
            if (x)
                tx += w[i]*(*x)[i];
        }
    if (!x)
        return ty;
    if (tx == 0)
        throw std::invalid_argument("survey: the denominator total is zero in the sample or a replicate");
    return ty/tx;
}

/// @brief Replicate-variance estimate of a weighted ratio or total.
/// @param design Replicate design.
/// @param y Numerator values.
/// @param x Denominator values (null: a total).
/// @param domain Domain mask.
/// @return The estimate with standard error and interval.
/// @throws std::invalid_argument For an inconsistent design, mismatched sizes or an empty domain.
SurveyEstimate replicate_estimate(const ReplicateDesign& design, const std::vector<double>& y,
                                  const std::vector<double>* x, const std::vector<bool>& domain) {
    const std::size_t n = design.weights.size(), R = design.replicate_weights.cols();
    if (n == 0 || design.replicate_weights.rows() != n || R == 0 ||
        (!design.rscales.empty() && design.rscales.size() != R))
        throw std::invalid_argument("survey: replicate weights must have one row per unit and one scale per replicate");
    if (!(design.confidence > 0 && design.confidence < 1) || !std::isfinite(design.scale) || design.scale <= 0)
        throw std::invalid_argument("survey: invalid confidence or scale");
    PreparedDesign shape;
    shape.n = n;
    const auto d = indicators(shape,y,domain);
    if (x)
        (void)indicators(shape,*x,domain);
    for (const double w : design.weights)
        if (!std::isfinite(w) || w < 0)
            throw std::invalid_argument("survey: weights must be finite and nonnegative");
    const double full = weighted_ratio(design.weights,y,x,d);
    std::vector<double> rep(R), w(n);
    double mean = 0;
    for (std::size_t r = 0; r < R; ++r) {
        for (std::size_t i = 0; i < n; ++i) {
            w[i] = design.replicate_weights(i,r);
            if (!std::isfinite(w[i]) || w[i] < 0)
                throw std::invalid_argument("survey: replicate weights must be finite and nonnegative");
        }
        rep[r] = weighted_ratio(w,y,x,d);
        mean += rep[r]/static_cast<double>(R);
    }
    const double centre = design.mse ? full : mean;
    double v = 0;
    for (std::size_t r = 0; r < R; ++r)
        v += (design.rscales.empty() ? 1.0 : design.rscales[r])*(rep[r]-centre)*(rep[r]-centre);
    return make_estimate(full,design.scale*v,random::normal_quantile(.5+design.confidence/2));
}

} // namespace

ReplicateDesign make_replicate_design(const SurveyDesign& design, const ReplicateOptions& options) {
    const auto p = prepare(design,design.weights.size());
    if (options.method == ReplicateMethod::fay && !(options.fay_rho >= 0 && options.fay_rho < 1))
        throw std::invalid_argument("make_replicate_design: the Fay factor must lie in [0, 1)");
    const std::size_t strata = p.stratum_psus.size(), psus = p.psu_stratum.size();
    // PSU order within each stratum by first appearance.
    std::vector<std::vector<std::size_t>> members(strata);
    for (std::size_t c = 0; c < psus; ++c)
        members[p.psu_stratum[c]].push_back(c);
    ReplicateDesign out{.weights=design.weights,.mse=options.mse,.confidence=design.confidence};
    std::vector<std::vector<double>> factor; // per replicate, per PSU
    switch (options.method) {
    case ReplicateMethod::brr:
    case ReplicateMethod::fay: {
        for (const auto& m : members)
            if (m.size() != 2)
                throw std::invalid_argument("make_replicate_design: BRR and Fay need exactly two PSUs per stratum");
        std::vector<std::vector<int>> h;
        for (std::size_t m = strata+1; h.empty(); ++m)
            if (m <= 2 || m%4 == 0)
                h = hadamard(m);
        const double rho = options.method == ReplicateMethod::fay ? options.fay_rho : 0.0;
        for (const auto& row : h) {
            std::vector<double> f(psus);
            for (std::size_t s = 0; s < strata; ++s) {
                const bool first = row[s+1] > 0;
                f[members[s][0]] = first ? 2-rho : rho;
                f[members[s][1]] = first ? rho : 2-rho;
            }
            factor.push_back(std::move(f));
        }
        const double R = static_cast<double>(h.size());
        out.scale = 1/(R*(1-rho)*(1-rho));
        break;
    }
    case ReplicateMethod::jk1: {
        if (strata != 1)
            throw std::invalid_argument("make_replicate_design: JK1 needs an unstratified design");
        const double n = static_cast<double>(psus);
        if (psus < 2)
            throw std::invalid_argument("make_replicate_design: a jackknife needs two PSUs per stratum");
        for (std::size_t c = 0; c < psus; ++c) {
            std::vector<double> f(psus,n/(n-1));
            f[c] = 0;
            factor.push_back(std::move(f));
        }
        out.scale = (n-1)/n*(1-p.stratum_fraction[0]);
        break;
    }
    case ReplicateMethod::jkn: {
        out.scale = 1;
        for (std::size_t s = 0; s < strata; ++s) {
            const double nh = static_cast<double>(members[s].size());
            if (members[s].size() < 2)
                throw std::invalid_argument("make_replicate_design: a jackknife needs two PSUs per stratum");
            for (const std::size_t c : members[s]) {
                std::vector<double> f(psus,1.0);
                for (const std::size_t other : members[s])
                    f[other] = other == c ? 0.0 : nh/(nh-1);
                factor.push_back(std::move(f));
                out.rscales.push_back((nh-1)/nh*(1-p.stratum_fraction[s]));
            }
        }
        break;
    }
    default:
        throw std::invalid_argument("make_replicate_design: unknown method");
    }
    out.replicate_weights = linalg::DenseMatrix<double>(p.n,factor.size());
    for (std::size_t r = 0; r < factor.size(); ++r)
        for (std::size_t i = 0; i < p.n; ++i)
            out.replicate_weights(i,r) = design.weights[i]*factor[r][p.psu[i]];
    return out;
}

SurveyEstimate replicate_total(const ReplicateDesign& design, const std::vector<double>& y,
                               const std::vector<bool>& domain) {
    return replicate_estimate(design,y,nullptr,domain);
}

SurveyEstimate replicate_mean(const ReplicateDesign& design, const std::vector<double>& y,
                              const std::vector<bool>& domain) {
    const std::vector<double> ones(y.size(),1.0);
    return replicate_estimate(design,y,&ones,domain);
}

SurveyEstimate replicate_ratio(const ReplicateDesign& design, const std::vector<double>& numerator,
                               const std::vector<double>& denominator, const std::vector<bool>& domain) {
    if (denominator.size() != numerator.size())
        throw std::invalid_argument("survey: numerator and denominator must have one entry per unit");
    return replicate_estimate(design,numerator,&denominator,domain);
}

SurveyEstimate survey_total(const SurveyDesign& design, const std::vector<double>& y, const std::vector<bool>& domain) {
    const auto p = prepare(design,y.size());
    const auto d = indicators(p,y,domain);
    std::vector<double> z(p.n);
    double total = 0;
    for (std::size_t i = 0; i < p.n; ++i) {
        z[i] = d[i] > 0 ? p.weights[i]*y[i] : 0.0;
        total += z[i];
    }
    return make_estimate(total,linearized_variance(p,z),p.z);
}

SurveyEstimate survey_mean(const SurveyDesign& design, const std::vector<double>& y, const std::vector<bool>& domain) {
    return survey_ratio(design,y,std::vector<double>(y.size(),1.0),domain);
}

SurveyEstimate survey_ratio(const SurveyDesign& design, const std::vector<double>& numerator,
                            const std::vector<double>& denominator, const std::vector<bool>& domain) {
    const auto p = prepare(design,numerator.size());
    const auto d = indicators(p,numerator,domain);
    (void)indicators(p,denominator,domain);
    double ty = 0, tx = 0;
    for (std::size_t i = 0; i < p.n; ++i)
        if (d[i] > 0) {
            ty += p.weights[i]*numerator[i];
            tx += p.weights[i]*denominator[i];
        }
    if (tx == 0)
        throw std::invalid_argument("survey: the denominator total is zero");
    const double r = ty/tx;
    std::vector<double> z(p.n,0.0);
    for (std::size_t i = 0; i < p.n; ++i)
        if (d[i] > 0)
            z[i] = p.weights[i]*(numerator[i]-r*denominator[i])/tx;
    return make_estimate(r,linearized_variance(p,z),p.z);
}

HorvitzThompsonResult horvitz_thompson_joint(const std::vector<double>& y, const linalg::DenseMatrix<double>& joint_inclusion,
                                             const JointInclusionOptions& options) {
    const std::size_t n = y.size();
    if (n < 2 || joint_inclusion.rows() != n || joint_inclusion.cols() != n)
        throw std::invalid_argument("horvitz_thompson_joint: need at least two units and an n by n joint-probability matrix");
    if (options.method != SurveyVariance::sen_yates_grundy && options.method != SurveyVariance::horvitz_thompson)
        throw std::invalid_argument("horvitz_thompson_joint: unknown variance method");
    std::vector<double> pi(n);
    for (std::size_t i = 0; i < n; ++i)
        pi[i] = joint_inclusion(i,i);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = i+1; j < n; ++j) {
            const double a = joint_inclusion(i,j);
            if (!std::isfinite(a) || a != joint_inclusion(j,i))
                throw std::invalid_argument("horvitz_thompson_joint: joint probabilities must be finite and symmetric");
            if (!(a > 0 && a <= std::min(pi[i],pi[j])*(1+1e-12)))
                throw std::invalid_argument("horvitz_thompson_joint: joint probabilities must lie in (0, min(pi_i, pi_j)]");
        }
    // First-order checks, point estimates and the population-size option are shared.
    HorvitzThompsonResult r = horvitz_thompson(y,pi,{options.population_size,options.confidence});
    const double z = random::normal_quantile(.5+options.confidence/2);
    const auto exact = [&](SurveyEstimate& e, double variance) {
        r.negative_variance = r.negative_variance || variance < 0;
        e = make_estimate(e.estimate,variance,z);
    };
    const double total_variance = joint_variance(y,joint_inclusion,options.method);
    exact(r.total,total_variance);
    std::vector<double> residual(n);
    for (std::size_t i = 0; i < n; ++i)
        residual[i] = (y[i]-r.hajek_mean.estimate)/r.estimated_population_size;
    exact(r.hajek_mean,joint_variance(residual,joint_inclusion,options.method));
    if (r.ht_mean) {
        const double N = *options.population_size;
        exact(*r.ht_mean,total_variance/(N*N));
    }
    return r;
}

HorvitzThompsonResult horvitz_thompson(const std::vector<double>& y, const std::vector<double>& inclusion_probabilities,
                                       const HorvitzThompsonOptions& options) {
    const std::size_t n = y.size();
    if (inclusion_probabilities.size() != n || n < 2)
        throw std::invalid_argument("horvitz_thompson: need at least two units with one probability each");
    if (!(options.confidence > 0 && options.confidence < 1))
        throw std::invalid_argument("horvitz_thompson: confidence must lie in (0, 1)");
    if (options.population_size && !(std::isfinite(*options.population_size) && *options.population_size > 0))
        throw std::invalid_argument("horvitz_thompson: population size must be finite and positive");
    double total = 0, size = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const double p = inclusion_probabilities[i];
        if (!std::isfinite(y[i]) || !(p > 0 && p <= 1))
            throw std::invalid_argument("horvitz_thompson: values must be finite and probabilities in (0, 1]");
        total += y[i]/p;
        size += 1/p;
    }
    const double z = random::normal_quantile(.5+options.confidence/2);
    std::vector<double> weighted(n), residual(n);
    const double mean = total/size;
    for (std::size_t i = 0; i < n; ++i) {
        weighted[i] = y[i]/inclusion_probabilities[i];
        residual[i] = (y[i]-mean)/(inclusion_probabilities[i]*size);
    }
    HorvitzThompsonResult r;
    const double total_variance = with_replacement_variance(weighted);
    r.total = make_estimate(total,total_variance,z);
    r.hajek_mean = make_estimate(mean,with_replacement_variance(residual),z);
    if (options.population_size) {
        const double N = *options.population_size;
        r.ht_mean = make_estimate(total/N,total_variance/(N*N),z);
    }
    r.estimated_population_size = size;
    r.sample_size = n;
    return r;
}

linalg::DenseMatrix<double> survey_score_covariance(const SurveyDesign& design, const linalg::DenseMatrix<double>& scores) {
    const auto p = prepare(design,scores.rows());
    for (std::size_t i = 0; i < scores.rows(); ++i)
        for (std::size_t a = 0; a < scores.cols(); ++a)
            if (!std::isfinite(scores(i,a)))
                throw std::invalid_argument("survey_score_covariance: scores must be finite");
    return linearized_covariance(p,scores);
}

double survey_degrees_of_freedom(const SurveyDesign& design, const std::vector<bool>& domain) {
    const auto p = prepare(design,design.weights.size());
    if (!domain.empty() && domain.size() != p.n)
        throw std::invalid_argument("survey_degrees_of_freedom: domain must have one entry per unit");
    std::vector<bool> psu(p.psu_stratum.size(),false), stratum(p.stratum_psus.size(),false);
    for (std::size_t i = 0; i < p.n; ++i)
        if (domain.empty() || domain[i]) {
            psu[p.psu[i]] = true;
            stratum[p.psu_stratum[p.psu[i]]] = true;
        }
    return static_cast<double>(std::ranges::count(psu,true))-static_cast<double>(std::ranges::count(stratum,true));
}

} // namespace datamunge::stats
