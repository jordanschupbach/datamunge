/// @file multistate.cpp
/// @brief Product-integral of Nelson–Aalen intensity increments with per-subject influence propagation.

#include <datamunge/stats/multistate.hpp>

#include <datamunge/linalg/cholesky.hpp>
#include <datamunge/stats/detail/test_common.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <stdexcept>

namespace datamunge::stats {

/// @brief Internal helpers for multi-state estimation.
namespace {

/// @brief Dense double matrix.
using Matrix = linalg::DenseMatrix<double>;

} // namespace

std::vector<double> AalenJohansenEstimate::state_probabilities(double time) const {
    std::vector<double> p = initial;
    for (std::size_t k = 0; k < times.size() && times[k] <= time; ++k)
        for (std::size_t s = 0; s < states; ++s)
            p[s] = probabilities(k,s);
    return p;
}

Matrix AalenJohansenEstimate::transition_matrix(double s, double t) const {
    if (t < s)
        throw std::invalid_argument("AalenJohansenEstimate::transition_matrix: need s <= t");
    Matrix P = Matrix::identity(states);
    for (std::size_t k = 0; k < times.size(); ++k) {
        if (times[k] <= s || times[k] > t)
            continue;
        Matrix next(states,states);
        for (std::size_t a = 0; a < states; ++a)
            for (std::size_t b = 0; b < states; ++b) {
                double v = P(a,b);
                for (std::size_t c = 0; c < states; ++c)
                    v += P(a,c)*increments[k](c,b);
                next(a,b) = v;
            }
        P = next;
    }
    return P;
}

AalenJohansenEstimate aalen_johansen(const std::vector<MultiStateRecord>& records, std::size_t states) {
    const std::size_t n = records.size();
    if (n == 0)
        throw std::invalid_argument("aalen_johansen: no records");
    int largest = 0;
    for (const auto& r : records) {
        if (!std::isfinite(r.start) || !std::isfinite(r.stop) || !(r.start < r.stop))
            throw std::invalid_argument("aalen_johansen: intervals need finite start < stop");
        if (r.from < 0 || r.to < 0)
            throw std::invalid_argument("aalen_johansen: states must be nonnegative");
        largest = std::max({largest,r.from,r.to});
    }
    const std::size_t K = states == 0 ? static_cast<std::size_t>(largest)+1 : states;
    if (static_cast<std::size_t>(largest) >= K)
        throw std::invalid_argument("aalen_johansen: a state exceeds the state count");
    // Subjects: records with the same nonnegative id form one cluster.
    std::map<int,std::size_t> cluster_of;
    std::vector<std::size_t> cluster(n);
    std::size_t clusters = 0;
    for (std::size_t i = 0; i < n; ++i) {
        if (records[i].id < 0) {
            cluster[i] = clusters++;
        } else {
            const auto [it,inserted] = cluster_of.try_emplace(records[i].id,clusters);
            if (inserted)
                ++clusters;
            cluster[i] = it->second;
        }
    }
    AalenJohansenEstimate e;
    e.states = K;
    // Initial distribution: states of records entering at the earliest start time.
    double t0 = records[0].start;
    for (const auto& r : records)
        t0 = std::min(t0,r.start);
    e.initial.assign(K,0.0);
    double starters = 0;
    for (const auto& r : records)
        if (r.start == t0) {
            e.initial[static_cast<std::size_t>(r.from)] += 1;
            starters += 1;
        }
    for (double& v : e.initial)
        v /= starters;
    Matrix U(clusters,K);
    for (std::size_t i = 0; i < n; ++i)
        if (records[i].start == t0)
            for (std::size_t s = 0; s < K; ++s)
                U(cluster[i],s) += ((static_cast<std::size_t>(records[i].from) == s ? 1.0 : 0.0)-e.initial[s])/starters;

    // Records sorted by stop for the sweep; at risk at t means start < t <= stop.
    std::vector<std::size_t> by_stop(n);
    for (std::size_t i = 0; i < n; ++i)
        by_stop[i] = i;
    std::ranges::sort(by_stop,[&](std::size_t a, std::size_t b) { return records[a].stop < records[b].stop; });
    std::vector<double> p = e.initial;
    std::vector<std::vector<double>> rows_p, rows_se, rows_risk;
    for (std::size_t k = 0; k < n;) {
        const double t = records[by_stop[k]].stop;
        std::size_t end = k;
        while (end < n && records[by_stop[end]].stop == t)
            ++end;
        std::vector<double> Y(K,0.0);
        std::vector<std::size_t> active;
        for (std::size_t i = 0; i < n; ++i)
            if (records[i].start < t && t <= records[i].stop) {
                Y[static_cast<std::size_t>(records[i].from)] += 1;
                active.push_back(i);
            }
        Matrix dA(K,K);
        bool any = false;
        for (std::size_t j = k; j < end; ++j) {
            const auto& r = records[by_stop[j]];
            if (r.to != r.from) {
                dA(static_cast<std::size_t>(r.from),static_cast<std::size_t>(r.to)) += 1;
                any = true;
            }
        }
        for (std::size_t h = 0; h < K; ++h) {
            double out = 0;
            for (std::size_t c = 0; c < K; ++c)
                if (c != h && Y[h] > 0) {
                    dA(h,c) /= Y[h];
                    out += dA(h,c);
                }
            dA(h,h) = -out;
        }
        if (any) {
            // Influence: U(t) = U(t-)(I + dA) + p(t-) dA_i.
            Matrix next(clusters,K);
            for (std::size_t c = 0; c < clusters; ++c)
                for (std::size_t b = 0; b < K; ++b) {
                    double v = U(c,b);
                    for (std::size_t a = 0; a < K; ++a)
                        v += U(c,a)*dA(a,b);
                    next(c,b) = v;
                }
            for (const std::size_t i : active) {
                const auto h = static_cast<std::size_t>(records[i].from);
                const bool moved = records[i].stop == t && records[i].to != records[i].from;
                for (std::size_t b = 0; b < K; ++b) {
                    if (b == h)
                        continue;
                    const double jump = moved && static_cast<std::size_t>(records[i].to) == b ? 1.0 : 0.0;
                    const double influence = (jump-dA(h,b))/Y[h]; // on dA(h, b); dA(h, h) gets the negative
                    next(cluster[i],b) += p[h]*influence;
                    next(cluster[i],h) -= p[h]*influence;
                }
            }
            U = std::move(next);
            std::vector<double> q(K,0.0);
            for (std::size_t b = 0; b < K; ++b) {
                q[b] = p[b];
                for (std::size_t a = 0; a < K; ++a)
                    q[b] += p[a]*dA(a,b);
            }
            p = q;
        }
        std::vector<double> se(K,0.0);
        for (std::size_t c = 0; c < clusters; ++c)
            for (std::size_t b = 0; b < K; ++b)
                se[b] += U(c,b)*U(c,b);
        for (double& v : se)
            v = std::sqrt(v);
        e.times.push_back(t);
        e.increments.push_back(dA);
        rows_p.push_back(p);
        rows_se.push_back(se);
        rows_risk.push_back(Y);
        k = end;
    }
    const std::size_t T = e.times.size();
    e.probabilities = Matrix(T,K);
    e.standard_errors = Matrix(T,K);
    e.at_risk = Matrix(T,K);
    for (std::size_t r = 0; r < T; ++r)
        for (std::size_t s = 0; s < K; ++s) {
            e.probabilities(r,s) = rows_p[r][s];
            e.standard_errors(r,s) = rows_se[r][s];
            e.at_risk(r,s) = rows_risk[r][s];
        }
    return e;
}

AalenJohansenEstimate cumulative_incidence(const std::vector<CompetingRiskObservation>& observations) {
    if (observations.empty())
        throw std::invalid_argument("cumulative_incidence: no observations");
    std::vector<MultiStateRecord> records;
    for (const auto& o : observations) {
        if (o.cause < 0 || !std::isfinite(o.time) || !(o.time > 0))
            throw std::invalid_argument("cumulative_incidence: need positive finite times and causes >= 0");
        records.push_back({0,o.time,0,o.cause,-1});
    }
    return aalen_johansen(records);
}

/// @brief Internal helpers for Gray's test.
namespace {

/// @brief Score and packed variance contributions of one stratum (port of cmprsk's crst).
/// @param y Times sorted increasingly.
/// @param m 1 cause of interest, 2 other cause, 0 censored.
/// @param ig Group index 0..ng-1.
/// @param ng Number of groups.
/// @param rho Weight exponent.
/// @param s Receives the score (length ng-1).
/// @param v Receives the variance (ng-1 by ng-1).
void gray_stratum(const std::vector<double>& y, const std::vector<int>& m, const std::vector<std::size_t>& ig,
                  std::size_t ng, double rho, std::vector<double>& s, Matrix& v) {
    const std::size_t n = y.size(), ng1 = ng-1;
    std::vector<double> rs(ng,0.0), f1m(ng,0.0), f1(ng,0.0), skmm(ng,1.0), skm(ng,1.0), v3(ng,0.0);
    Matrix c(ng,ng), a(ng,ng), v2(ng1,ng);
    s.assign(ng1,0.0);
    v = Matrix(ng1,ng1);
    for (std::size_t i = 0; i < n; ++i)
        rs[ig[i]] += 1;
    double fm = 0, f = 0;
    for (std::size_t ll = 0; ll < n;) {
        std::size_t lu = ll;
        while (lu+1 < n && y[lu+1] <= y[ll])
            ++lu;
        std::vector<std::array<double,3>> d(ng,{0,0,0});
        for (std::size_t i = ll; i <= lu; ++i)
            d[ig[i]][static_cast<std::size_t>(m[i])] += 1;
        double nd1 = 0, nd2 = 0;
        for (std::size_t i = 0; i < ng; ++i) {
            nd1 += d[i][1];
            nd2 += d[i][2];
        }
        if (nd1 != 0 || nd2 != 0) {
            double tr = 0, tq = 0;
            for (std::size_t i = 0; i < ng; ++i) {
                if (rs[i] <= 0)
                    continue;
                const double td = d[i][1]+d[i][2];
                skm[i] = skmm[i]*(rs[i]-td)/rs[i];
                f1[i] = f1m[i]+skmm[i]*d[i][1]/rs[i];
                tr += rs[i]/skmm[i];
                tq += rs[i]*(1-f1m[i])/skmm[i];
            }
            f = fm+nd1/tr;
            const double fb = std::pow(1-fm,rho);
            for (std::size_t i = 0; i < ng; ++i) {
                for (std::size_t j = i; j < ng; ++j)
                    a(i,j) = 0;
                if (rs[i] <= 0)
                    continue;
                const double t1 = rs[i]/skmm[i];
                a(i,i) = fb*t1*(1-t1/tr);
                if (a(i,i) != 0)
                    c(i,i) += a(i,i)*nd1/(tr*(1-fm));
                for (std::size_t j = i+1; j < ng; ++j) {
                    if (rs[j] <= 0)
                        continue;
                    a(i,j) = -fb*t1*rs[j]/(skmm[j]*tr);
                    if (a(i,j) != 0)
                        c(i,j) += a(i,j)*nd1/(tr*(1-fm));
                }
            }
            for (std::size_t i = 1; i < ng; ++i)
                for (std::size_t j = 0; j < i; ++j) {
                    a(i,j) = a(j,i);
                    c(i,j) = c(j,i);
                }
            for (std::size_t i = 0; i < ng1; ++i)
                if (rs[i] > 0)
                    s[i] += fb*(d[i][1]-nd1*rs[i]*(1-f1m[i])/(skmm[i]*tq));
            if (nd1 > 0)
                for (std::size_t k = 0; k < ng; ++k) {
                    if (rs[k] <= 0)
                        continue;
                    const double t4 = skm[k] > 0 ? 1-(1-f)/skm[k] : 1.0;
                    const double t5 = nd1 > 1 ? 1-(nd1-1)/(tr*skmm[k]-1) : 1.0;
                    const double t3 = t5*skmm[k]*nd1/(tr*rs[k]);
                    v3[k] += t4*t4*t3;
                    for (std::size_t i = 0; i < ng1; ++i) {
                        const double t1 = a(i,k)-t4*c(i,k);
                        v2(i,k) += t1*t4*t3;
                        for (std::size_t j = 0; j <= i; ++j)
                            v(i,j) += t1*(a(j,k)-t4*c(j,k))*t3;
                    }
                }
            if (nd2 != 0)
                for (std::size_t k = 0; k < ng; ++k) {
                    if (skm[k] <= 0 || d[k][2] <= 0)
                        continue;
                    const double t4 = (1-f)/skm[k];
                    const double t5 = d[k][2] > 1 ? 1-(d[k][2]-1)/(rs[k]-1) : 1.0;
                    const double t3 = t5*skmm[k]*skmm[k]*d[k][2]/(rs[k]*rs[k]);
                    v3[k] += t4*t4*t3;
                    for (std::size_t i = 0; i < ng1; ++i) {
                        const double t1 = t4*c(i,k);
                        v2(i,k) -= t1*t4*t3;
                        for (std::size_t j = 0; j <= i; ++j)
                            v(i,j) += t1*t4*c(j,k)*t3;
                    }
                }
        }
        if (lu+1 >= n)
            break;
        for (std::size_t i = ll; i <= lu; ++i)
            rs[ig[i]] -= 1;
        fm = f;
        f1m = f1;
        skmm = skm;
        ll = lu+1;
    }
    for (std::size_t i = 0; i < ng1; ++i)
        for (std::size_t j = 0; j <= i; ++j) {
            for (std::size_t k = 0; k < ng; ++k)
                v(i,j) += c(i,k)*c(j,k)*v3[k]+c(i,k)*v2(j,k)+c(j,k)*v2(i,k);
            v(j,i) = v(i,j);
        }
}

} // namespace

std::vector<GrayTest> gray_test(const std::vector<CompetingRiskObservation>& observations, const std::vector<int>& groups,
                                const GrayTestOptions& options) {
    const std::size_t n = observations.size();
    if (n == 0 || groups.size() != n || (!options.strata.empty() && options.strata.size() != n))
        throw std::invalid_argument("gray_test: need one group (and stratum) per observation");
    if (!std::isfinite(options.rho) || options.rho < 0)
        throw std::invalid_argument("gray_test: rho must be finite and nonnegative");
    std::vector<int> labels = groups, causes;
    std::ranges::sort(labels);
    labels.erase(std::unique(labels.begin(),labels.end()),labels.end());
    const std::size_t ng = labels.size();
    if (ng < 2)
        throw std::invalid_argument("gray_test: need at least two groups");
    for (const auto& o : observations) {
        if (!std::isfinite(o.time) || o.time < 0 || o.cause < 0)
            throw std::invalid_argument("gray_test: invalid time or cause");
        if (o.cause > 0)
            causes.push_back(o.cause);
    }
    std::ranges::sort(causes);
    causes.erase(std::unique(causes.begin(),causes.end()),causes.end());
    if (causes.empty())
        throw std::invalid_argument("gray_test: no failures");
    std::vector<std::size_t> order(n);
    for (std::size_t i = 0; i < n; ++i)
        order[i] = i;
    std::ranges::stable_sort(order,[&](std::size_t a, std::size_t b) { return observations[a].time < observations[b].time; });
    std::vector<int> strata = options.strata;
    if (strata.empty())
        strata.assign(n,0);
    std::vector<int> stratum_labels = strata;
    std::ranges::sort(stratum_labels);
    stratum_labels.erase(std::unique(stratum_labels.begin(),stratum_labels.end()),stratum_labels.end());
    std::vector<GrayTest> out;
    for (const int cause : causes) {
        GrayTest test{cause,std::vector<double>(ng-1,0.0),Matrix(ng-1,ng-1),0,ng-1,1};
        for (const int label : stratum_labels) {
            std::vector<double> y;
            std::vector<int> m;
            std::vector<std::size_t> ig;
            for (const std::size_t i : order) {
                if (strata[i] != label)
                    continue;
                y.push_back(observations[i].time);
                m.push_back(observations[i].cause == 0 ? 0 : observations[i].cause == cause ? 1 : 2);
                ig.push_back(static_cast<std::size_t>(std::ranges::lower_bound(labels,groups[i])-labels.begin()));
            }
            std::vector<double> s;
            Matrix v;
            gray_stratum(y,m,ig,ng,options.rho,s,v);
            for (std::size_t a = 0; a+1 < ng; ++a) {
                test.score[a] += s[a];
                for (std::size_t b = 0; b+1 < ng; ++b)
                    test.variance(a,b) += v(a,b);
            }
        }
        const auto factor = linalg::cholesky(test.variance);
        if (!factor.ok)
            throw std::runtime_error("gray_test: score variance is singular");
        const auto solution = factor.solve(test.score);
        for (std::size_t a = 0; a+1 < ng; ++a)
            test.statistic += solution[a]*test.score[a];
        test.p_value = detail::chi_squared_test_survival(test.statistic,static_cast<double>(ng-1));
        out.push_back(std::move(test));
    }
    return out;
}

} // namespace datamunge::stats
