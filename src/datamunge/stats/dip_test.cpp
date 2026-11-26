/// @file dip_test.cpp
/// @brief AS 217 dip computation (with the diptest package's corrections) and uniform calibration.

#include <datamunge/stats/dip_test.hpp>

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>

namespace datamunge::stats {
/// @brief The dip kernel on sorted, one-based data.
namespace {

/// @brief Dip of sorted data, times 2n, with the modal interval's one-based indices.
/// @param x Sorted values with a dummy entry at index 0 (one-based indexing as in AS 217).
/// @param n Sample size.
/// @param low_out Set to the lower modal index.
/// @param high_out Set to the upper modal index.
/// @return Dip times \f$2n\f$.
double dip_kernel(const std::vector<double>& x, int n, int& low_out, int& high_out) {
    int low = 1, high = n;
    double dip = 1;
    low_out = low;
    high_out = high;
    if (n < 2 || x[n] == x[1])
        return dip;
    std::vector<int> mn(n+1), mj(n+1), gcm(n+2), lcm(n+2);
    // Indices for the greatest convex minorant.
    mn[1] = 1;
    for (int j = 2; j <= n; ++j) {
        mn[j] = j-1;
        while (true) {
            const int mnj = mn[j], mnmnj = mn[mnj];
            if (mnj == 1 || (x[j]-x[mnj])*(mnj-mnmnj) < (x[mnj]-x[mnmnj])*(j-mnj))
                break;
            mn[j] = mnmnj;
        }
    }
    // Indices for the least concave majorant.
    mj[n] = n;
    for (int k = n-1; k >= 1; --k) {
        mj[k] = k+1;
        while (true) {
            const int mjk = mj[k], mjmjk = mj[mjk];
            if (mjk == n || (x[k]-x[mjk])*(mjk-mjmjk) < (x[mjk]-x[mjmjk])*(k-mjk))
                break;
            mj[k] = mjmjk;
        }
    }
    while (true) {
        int i;
        gcm[1] = high;
        for (i = 1; gcm[i] > low; ++i)
            gcm[i+1] = mn[gcm[i]];
        const int l_gcm = i;
        int ig = l_gcm, ix = ig-1;
        lcm[1] = low;
        for (i = 1; lcm[i] < high; ++i)
            lcm[i+1] = mj[lcm[i]];
        const int l_lcm = i;
        int ih = l_lcm, iv = 2;
        long double d = 0;
        if (l_gcm != 2 || l_lcm != 2) {
            do {
                long double dx;
                const int gcmix = gcm[ix], lcmiv = lcm[iv];
                if (gcmix > lcmiv) {
                    const int gcmi1 = gcm[ix+1];
                    dx = (lcmiv-gcmi1+1)-
                         (static_cast<long double>(x[lcmiv])-x[gcmi1])*(gcmix-gcmi1)/(x[gcmix]-x[gcmi1]);
                    ++iv;
                    if (dx >= d) {
                        d = dx;
                        ig = ix+1;
                        ih = iv-1;
                    }
                } else {
                    const int lcmiv1 = lcm[iv-1];
                    dx = (static_cast<long double>(x[gcmix])-x[lcmiv1])*(lcmiv-lcmiv1)/(x[lcmiv]-x[lcmiv1])-
                         (gcmix-lcmiv1-1);
                    --ix;
                    if (dx >= d) {
                        d = dx;
                        ig = ix+1;
                        ih = iv;
                    }
                }
                ix = std::max(ix,1);
                iv = std::min(iv,l_lcm);
            } while (gcm[ix] != lcm[iv]);
        } else {
            d = 1;
        }
        if (d < dip)
            break;
        // Dips of the convex minorant and concave majorant on the current modal interval.
        double dip_l = 0, dip_u = 0;
        for (int j = ig; j < l_gcm; ++j) {
            double max_t = 1;
            const int jb = gcm[j+1], je = gcm[j];
            if (je-jb > 1 && x[je] != x[jb]) {
                const double c = (je-jb)/(x[je]-x[jb]);
                for (int jj = jb; jj <= je; ++jj)
                    max_t = std::max(max_t,(jj-jb+1)-(x[jj]-x[jb])*c);
            }
            dip_l = std::max(dip_l,max_t);
        }
        for (int j = ih; j < l_lcm; ++j) {
            double max_t = 1;
            const int jb = lcm[j], je = lcm[j+1];
            if (je-jb > 1 && x[je] != x[jb]) {
                const double c = (je-jb)/(x[je]-x[jb]);
                for (int jj = jb; jj <= je; ++jj)
                    max_t = std::max(max_t,(x[jj]-x[jb])*c-(jj-jb-1));
            }
            dip_u = std::max(dip_u,max_t);
        }
        dip = std::max(dip,std::max(dip_u,dip_l));
        low_out = low;
        high_out = high;
        if (low == gcm[ig] && high == lcm[ih])
            break;
        low = gcm[ig];
        high = lcm[ih];
        low_out = low;
        high_out = high;
    }
    return dip;
}

/// @brief Dip of an already sorted sample.
/// @param sorted Sorted finite values.
/// @return Statistic and modal interval.
DipStatistic dip_of_sorted(const std::vector<double>& sorted) {
    const int n = static_cast<int>(sorted.size());
    std::vector<double> one_based(n+1);
    std::copy(sorted.begin(),sorted.end(),one_based.begin()+1);
    int low, high;
    const double dip = dip_kernel(one_based,n,low,high);
    return {dip/(2.0*n),one_based[low],one_based[high]};
}

} // namespace

DipStatistic dip_statistic(const std::vector<double>& x) {
    if (x.empty() || !std::ranges::all_of(x,[](double v) { return std::isfinite(v); }))
        throw std::invalid_argument("dip_statistic: need nonempty finite values");
    std::vector<double> sorted(x);
    std::ranges::sort(sorted);
    return dip_of_sorted(sorted);
}

DipTestResult dip_test(const std::vector<double>& x, const DipTestOptions& options) {
    if (x.size() < 4)
        throw std::invalid_argument("dip_test: need at least four observations");
    if (options.replicates == 0)
        throw std::invalid_argument("dip_test: need at least one replicate");
    DipTestResult result;
    result.dip = dip_statistic(x);
    result.replicates = options.replicates;
    std::mt19937_64 random(options.seed);
    std::uniform_real_distribution<double> unit(0.0,1.0);
    std::vector<double> u(x.size());
    std::size_t exceed = 0;
    for (std::size_t b = 0; b < options.replicates; ++b) {
        for (auto& v : u)
            v = unit(random);
        std::ranges::sort(u);
        exceed += dip_of_sorted(u).statistic >= result.dip.statistic;
    }
    result.p_value = (1.0+static_cast<double>(exceed))/(1.0+static_cast<double>(options.replicates));
    return result;
}

} // namespace datamunge::stats
