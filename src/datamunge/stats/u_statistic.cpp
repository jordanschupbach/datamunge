/// @file u_statistic.cpp
/// @brief Subset averaging, finite-sample projection diagnostics and exact binary-population U-statistic laws.

#include <datamunge/stats/u_statistic.hpp>
#include <datamunge/random/distributions.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <stdexcept>

namespace datamunge::stats {

/// @brief Checked combinatorics and probability helpers local to this implementation.
namespace {

/// @brief Exact binomial coefficient with integer cancellation before multiplication.
/// @param n Set size.
/// @param k Subset size, no larger than n.
/// @return C(n,k) as size_t.
/// @throws std::overflow_error If the coefficient cannot be represented.
std::size_t choose_count(std::size_t n, std::size_t k) {
    k=std::min(k,n-k);
    std::size_t count=1;
    for (std::size_t i=1;i<=k;++i) {
        std::size_t a=n-k+i,b=i;
        const auto g1=std::gcd(a,b); a/=g1; b/=g1;
        const auto g2=std::gcd(count,b); count/=g2; b/=g2;
        if (count>std::numeric_limits<std::size_t>::max()/a)
            throw std::overflow_error("UStatistic: complete term count exceeds size_t; use max_terms");
        count*=a;
    }
    return count;
}

/// @brief Log binomial coefficient for feasible indices, without materialising huge counts.
/// @param n Set size.
/// @param k Subset size no greater than n.
/// @return Natural log of C(n,k), in long double arithmetic.
long double log_choose(std::size_t n,std::size_t k) {
    return std::lgamma(static_cast<long double>(n)+1)-std::lgamma(static_cast<long double>(k)+1)
          -std::lgamma(static_cast<long double>(n-k)+1);
}

/// @brief Evaluate and validate a callback's scalar value; callback exceptions propagate.
/// @param kernel User-supplied symmetric kernel.
/// @param tuple Tuple of observations in enumeration or shuffle order.
/// @return Finite kernel value.
/// @throws std::invalid_argument If the result is not finite.
/// @throws std::bad_function_call For an empty callback.
double finite_kernel(const SymmetricKernel& kernel,const std::vector<double>& tuple) {
    const double value=kernel(tuple);
    if (!std::isfinite(value)) throw std::invalid_argument("UStatistic: kernel values must be finite");
    return value;
}

} // namespace

UStatistic::UStatistic(std::size_t degree,UStatisticOptions options):degree_(degree),options_(options) {
    if (!degree_) throw std::invalid_argument("UStatistic: degree must be >= 1");
}

UStatisticResult UStatistic::run(const std::vector<double>& sample,const SymmetricKernel& kernel) const {
    return evaluate(sample,kernel,nullptr).result;
}

UStatisticAnalysis UStatistic::analyze(const std::vector<double>& sample,const SymmetricKernel& kernel) const {
    return evaluate(sample,kernel,nullptr);
}

dstruct::DataFrame UStatistic::term_details(const std::vector<double>& sample,const SymmetricKernel& kernel) const {
    dstruct::DataFrame terms;
    (void)evaluate(sample,kernel,&terms);
    return terms;
}

UStatisticAnalysis UStatistic::evaluate(const std::vector<double>& sample,const SymmetricKernel& kernel,
                                       dstruct::DataFrame* terms) const {
    const std::size_t n=sample.size(),m=degree_;
    if (m>n) throw std::invalid_argument("UStatistic: degree must not exceed sample size");
    for (double x:sample) if (!std::isfinite(x))
        throw std::invalid_argument("UStatistic: sample values must be finite");
    const bool complete=options_.max_terms==0;
    const std::size_t total=complete ? choose_count(n,m) : options_.max_terms;
    UStatisticAnalysis out;
    auto& r=out.result;
    r.degree=m; r.complete=complete; r.num_terms=total;
    std::vector<long double> g_sum(n,0.0L);
    std::vector<std::size_t> g_count(n,0);
    std::vector<std::vector<double>> columns(terms ? 2*m+2 : 0);
    std::vector<double> tuple(m);
    long double mean=0.0L,m2=0.0L;
    std::size_t evaluated=0;
    /// @brief Accumulates one kernel value and its observation-inclusion sums.
    /// @param indices Tuple indices, each distinct within this evaluation.
    auto accumulate=[&](const std::vector<std::size_t>& indices) {
        for (std::size_t j=0;j<m;++j) tuple[j]=sample[indices[j]];
        const double h=finite_kernel(kernel,tuple);
        ++evaluated;
        const long double delta=h-mean;
        mean+=delta/evaluated; m2+=delta*(h-mean);
        for (std::size_t j=0;j<m;++j) {g_sum[indices[j]]+=h; ++g_count[indices[j]];}
        if (terms) {
            columns[0].push_back(evaluated);
            for (std::size_t j=0;j<m;++j) {
                columns[1+j].push_back(indices[j]); columns[1+m+j].push_back(tuple[j]);
            }
            columns.back().push_back(h);
        }
    };
    std::vector<std::size_t> combo(m);
    if (complete) {
        std::iota(combo.begin(),combo.end(),0);
        for (std::size_t term=0;term<total;++term) {
            accumulate(combo);
            if (term==total-1) break;
            std::size_t i=m;
            while (i>0 && combo[i-1]==n-m+i-1) --i;
            ++combo[i-1];
            for (std::size_t j=i;j<m;++j) combo[j]=combo[j-1]+1;
        }
    } else {
        std::mt19937_64 rng(options_.seed);
        std::vector<std::size_t> indices(n);
        std::iota(indices.begin(),indices.end(),0);
        for (std::size_t term=0;term<total;++term) {
            for (std::size_t j=0;j<m;++j) {
                std::uniform_int_distribution<std::size_t> pick(j,n-1);
                std::swap(indices[j],indices[pick(rng)]); combo[j]=indices[j];
            }
            accumulate(combo);
        }
    }
    r.estimate=static_cast<double>(mean);
    const double nan=std::numeric_limits<double>::quiet_NaN();
    std::vector<double> row,counts,conditional,loo,pseudo;
    long double ss=0.0L,jackknife_ss=0.0L;
    std::size_t covered=0;
    for (std::size_t i=0;i<n;++i) {
        row.push_back(i); counts.push_back(g_count[i]);
        const long double gi=g_count[i] ? g_sum[i]/g_count[i] : nan;
        conditional.push_back(static_cast<double>(gi));
        if (g_count[i]) {const auto delta=gi-mean; ss+=delta*delta; ++covered;}
        if (complete && n>m) {
            const long double deleted=mean+static_cast<long double>(m)/(n-m)*(mean-gi);
            loo.push_back(static_cast<double>(deleted));
            pseudo.push_back(static_cast<double>(mean+(n-1)*(mean-deleted)));
            jackknife_ss+=(deleted-mean)*(deleted-mean);
        } else {loo.push_back(nan); pseudo.push_back(nan);}
    }
    r.zeta1=covered>1 ? static_cast<double>(ss/(covered-1)) : 0.0;
    r.variance=static_cast<double>(m)*m/n*r.zeta1;
    r.standard_error=std::sqrt(std::max(0.0,r.variance));
    out.kernel_variance=complete ? static_cast<double>(m2/total)
                               : total>1 ? static_cast<double>(m2/(total-1)) : nan;
    out.monte_carlo_variance=complete ? 0.0 : out.kernel_variance/total;
    if (complete && n>m) out.jackknife_variance=static_cast<double>((n-1)*jackknife_ss/n);
    out.projections.add_column("row",row); out.projections.add_column("value",sample);
    out.projections.add_column("inclusions",counts); out.projections.add_column("conditional_mean",conditional);
    out.projections.add_column("leave_one_out",loo); out.projections.add_column("pseudo_value",pseudo);
    if (terms) {
        terms->add_column("term",columns[0]);
        for (std::size_t j=0;j<m;++j) terms->add_column("i"+std::to_string(j+1),columns[1+j]);
        for (std::size_t j=0;j<m;++j) terms->add_column("x"+std::to_string(j+1),columns[1+m+j]);
        terms->add_column("kernel",columns.back());
    }
    return out;
}

dstruct::DataFrame UStatisticAnalysis::summary() const {
    std::size_t covered=0;
    for (double count:projections.numeric_column("inclusions").values()) covered+=count>0;
    dstruct::DataFrame table;
    table.add_column("n",std::vector<double>{static_cast<double>(projections.nrows())});
    table.add_column("degree",std::vector<double>{static_cast<double>(result.degree)});
    table.add_column("terms",std::vector<double>{static_cast<double>(result.num_terms)});
    table.add_column("complete",std::vector<double>{result.complete ? 1.0 : 0.0});
    table.add_column("covered_rows",std::vector<double>{static_cast<double>(covered)});
    table.add_column("estimate",std::vector<double>{result.estimate});
    table.add_column("projection_se",std::vector<double>{result.standard_error});
    table.add_column("jackknife_se",std::vector<double>{std::sqrt(jackknife_variance)});
    table.add_column("conditional_mc_se",std::vector<double>{std::sqrt(monte_carlo_variance)});
    table.add_column("kernel_variance",std::vector<double>{kernel_variance});
    return table;
}

dstruct::DataFrame UStatistic::incomplete_path(const std::vector<double>& sample,const SymmetricKernel& kernel,
    const std::vector<std::size_t>& budgets,std::size_t repetitions) const {
    if (budgets.empty() || !repetitions || std::find(budgets.begin(),budgets.end(),0)!=budgets.end())
        throw std::invalid_argument("UStatistic::incomplete_path: positive budgets and repetitions required");
    const auto reference=UStatistic(degree_).analyze(sample,kernel);
    std::vector<double> terms,reps,complete,averages,bias,rmse,expected,estimated_variance;
    for (auto budget:budgets) {
        long double sum=0.0L,ss=0.0L,mc=0.0L;
        for (std::size_t r=0;r<repetitions;++r) {
            const auto fit=UStatistic(degree_,{budget,options_.seed+r}).analyze(sample,kernel);
            const long double error=static_cast<long double>(fit.result.estimate)-reference.result.estimate;
            sum+=fit.result.estimate; ss+=error*error; mc+=fit.monte_carlo_variance;
        }
        terms.push_back(budget); reps.push_back(repetitions); complete.push_back(reference.result.estimate);
        averages.push_back(static_cast<double>(sum/repetitions)); bias.push_back(averages.back()-reference.result.estimate);
        rmse.push_back(std::sqrt(ss/repetitions)); expected.push_back(std::sqrt(reference.kernel_variance/budget));
        estimated_variance.push_back(static_cast<double>(mc/repetitions));
    }
    dstruct::DataFrame table;
    table.add_column("terms",terms); table.add_column("repetitions",reps); table.add_column("complete_estimate",complete);
    table.add_column("mean_estimate",averages); table.add_column("bias_to_complete",bias);
    table.add_column("rmse_to_complete",rmse); table.add_column("expected_rmse",expected);
    table.add_column("mean_estimated_mc_variance",estimated_variance);
    return table;
}

UStatisticLaw UStatistic::two_point_law(std::size_t n,double low,double high,double p,
                                      const SymmetricKernel& kernel) const {
    if (n<degree_ || n==std::numeric_limits<std::size_t>::max() || !std::isfinite(low) ||
        !std::isfinite(high) || low==high || !std::isfinite(p) || p<0 || p>1)
        throw std::invalid_argument("UStatistic::two_point_law: invalid population or sample size");
    const std::size_t m=degree_;
    std::vector<double> h(m+1),tuple(m,low);
    for (std::size_t j=0;j<=m;++j) {
        if (j) tuple[j-1]=high;
        h[j]=finite_kernel(kernel,tuple);
    }
    UStatisticLaw law; law.sample_size=n;
    long double target=0.0L,g_low=0.0L,g_high=0.0L;
    for (std::size_t j=0;j<=m;++j) target+=random::binomial_pdf(j,m,p)*static_cast<long double>(h[j]);
    for (std::size_t j=0;j<m;++j) {
        const double w=random::binomial_pdf(j,m-1,p);
        g_low+=w*static_cast<long double>(h[j]); g_high+=w*static_cast<long double>(h[j+1]);
    }
    law.target=static_cast<double>(target);
    law.first_order_variance=static_cast<double>(static_cast<long double>(m)*m/n*p*(1-p)*(g_high-g_low)*(g_high-g_low));
    const long double denominator=log_choose(n,m);
    std::vector<double> counts,probabilities,estimates;
    long double mass=0.0L,mean=0.0L;
    for (std::size_t k=0;k<=n;++k) {
        const std::size_t lower=m>n-k ? m-(n-k) : 0,upper=std::min(m,k);
        long double weighted=0.0L,weight=0.0L;
        for (std::size_t j=lower;j<=upper;++j) {
            const long double w=std::exp(log_choose(k,j)+log_choose(n-k,m-j)-denominator);
            weighted+=w*h[j]; weight+=w;
        }
        const double probability=random::binomial_pdf(k,n,p),estimate=static_cast<double>(weighted/weight);
        counts.push_back(k); probabilities.push_back(probability); estimates.push_back(estimate);
        mass+=probability; mean+=static_cast<long double>(probability)*estimate;
    }
    mean/=mass; law.mean=static_cast<double>(mean);
    long double variance=0.0L;
    for (std::size_t k=0;k<=n;++k) {
        probabilities[k]=static_cast<double>(probabilities[k]/mass);
        const long double error=estimates[k]-mean;
        variance+=probabilities[k]*error*error;
    }
    law.variance=static_cast<double>(variance);
    law.distribution.add_column("high_count",counts); law.distribution.add_column("probability",probabilities);
    law.distribution.add_column("estimate",estimates);
    return law;
}

} // namespace datamunge::stats
