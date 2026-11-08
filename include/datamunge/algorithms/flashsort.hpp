/// @file
/// @brief Checked arithmetic Flashsort, actual class/carry evidence and insertion-work counters.
#pragma once
#include <algorithm>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

/// @brief Canonical numerical and combinatorial algorithms.
namespace datamunge::algorithms {
/// @brief Actual work and class occupancy from one native Flashsort, without a history.
/// Min/max comparisons count two ordered tests per element after the first;
/// classifications counts every actual value-to-class call, including leader
/// searches. permutation_writes counts carried-value exchanges, including cycle
/// closure. Insertion comparisons count predecessor>key evaluations (including
/// false ones), shifts count successful predecessor moves, and placements count
/// final key stores. Loop-bound/finite-validation tests and allocations are excluded.
/// n<=INT_MAX keeps these whole-run uint64 counters representable. No timing claim.
/// @par Example
/// @code
/// std::vector<int> a{2,0,1};
/// const auto w=dm::algorithms::flashsort_work(a);
/// std::cout << w.class_count << ' ' << w.permutation_writes; // 2 3
/// @endcode
/// @see flashsort_work(), flashsort_trace()
struct FlashsortWork {
    std::size_t n=0; ///< Input length, including shortcut inputs.
    bool shortcut=true; ///< Empty/singleton/constant input, with no class vector or insertion tail.
    bool scaled_classification=false; ///< Whether safe scaled arithmetic replaced the historical double map.
    std::size_t class_count=0; ///< Actual allocated classes, zero on shortcut.
    std::size_t nonempty_classes=0; ///< Nonzero initial class counts, zero on shortcut.
    std::size_t max_occupancy=0; ///< Largest initial class count, zero on shortcut.
    std::uint64_t occupancy_square_sum=0; ///< Sum of squared initial class counts, zero on shortcut.
    std::uint64_t minmax_comparisons=0; ///< Actual scan comparisons, excluding finite validation.
    std::uint64_t classifications=0; ///< All actual classifier calls inside the sort.
    std::uint64_t permutation_writes=0; ///< Carried-value exchanges to decremented class boundaries.
    std::uint64_t insertion_comparisons=0; ///< Actual predecessor>key evaluations.
    std::uint64_t insertion_shifts=0; ///< Successful predecessor stores in the insertion tail.
    std::uint64_t insertion_placements=0; ///< Final key stores, one per active i=1..n-1.
};

/// @brief One actual carried-value exchange before the finishing insertion sort.
/// @tparam T Arithmetic value type other than bool.
/// @par Example
/// @code
/// const auto t=dm::algorithms::flashsort_trace(std::vector<int>{2,0,1});
/// std::cout << t.moves[0].destination; // 2
/// @endcode
/// @see FlashsortTrace, flashsort_trace()
template<class T> struct FlashsortMove {
    std::size_t cycle_start=0; ///< Current cycle leader position.
    std::size_t class_index=0; ///< Actual class of the carried value.
    std::size_t destination=0; ///< Decremented class boundary receiving that value.
    T carried{}; ///< Value written to destination by this exchange.
    T displaced{}; ///< Previous destination value, becoming the next carried value.
};

/// @brief Owned original, class-grouped and sorted states plus actual class/carry evidence.
/// A nonshortcut capture admits C=5n+2m entries before copying: input, output,
/// grouped values, original assignments, at most n carry records and two m-entry
/// class arrays. Shortcut C=2n stores only input/output; other arrays are empty.
/// Counts are entries, not bytes or peak RAM. No sorting/assembly is replayed.
/// @tparam T Arithmetic value type other than bool.
/// @par Example
/// @code
/// const auto t=dm::algorithms::flashsort_trace(std::vector<int>{2,0,1},19);
/// std::cout << t.output[0] << ' ' << t.output[2] << ' ' << t.moves.size(); // 0 2 3
/// @endcode
/// @see flashsort_trace(), FlashsortMove, FlashsortWork
template<class T> struct FlashsortTrace {
    std::vector<T> input; ///< Owned original values in encounter order.
    std::vector<T> classified; ///< Actual state after carry permutation, before insertion; empty on shortcut.
    std::vector<T> output; ///< Owned ascending result, including unchanged shortcut inputs.
    std::vector<std::size_t> class_of_input; ///< Actual original encounter assignments; empty on shortcut.
    std::vector<std::size_t> class_counts; ///< Original counts for all m classes, including empty ones.
    std::vector<std::size_t> class_ends; ///< Exclusive cumulative class ends before permutation changes them.
    std::vector<FlashsortMove<T>> moves; ///< Actual carry exchanges, including cycle closure.
    T minimum{}; ///< Actual minimum, zero placeholder for an empty input.
    T maximum{}; ///< Actual maximum, zero placeholder for an empty input.
    FlashsortWork work; ///< Actual counters and occupancy for this same execution.
};

/// @brief Consecutive class-index range and its total original occupancy.
/// @par Example
/// @code
/// const auto g=dm::algorithms::flashsort_class_groups({2,1,3,0},2);
/// std::cout << g[0].begin << ' ' << g[0].end << ' ' << g[0].count; // 0 2 3
/// @endcode
/// @see flashsort_class_groups()
struct FlashsortClassGroup {
    std::size_t begin=0; ///< Inclusive first class index.
    std::size_t end=0; ///< Exclusive class-index end; may equal begin for an empty group.
    std::uint64_t count=0; ///< Sum of original class occupancies in [begin,end).
};

/// @brief Return the historical nominal nonconstant-input class count without an array allocation.
/// Gives zero for n<=1 and max(2,int(0.43*n)) otherwise. A constant actual input
/// still takes the shortcut and allocates zero classes. O(1), no shared state.
/// @param n Virtual input length, at most INT_MAX.
/// @return Nominal active class count, or zero for an empty/singleton length.
/// @throws std::length_error If n exceeds the signed legacy index domain.
/// @par Example
/// @code
/// std::cout << dm::algorithms::flashsort_class_count(2000); // 860
/// @endcode
/// @par References
/// - K.-D. Neubert, Flashsort1 Algorithm, Dr. Dobb's Journal, February1998,p123,
///   https://www.neubert.net/FSOIntro.html ; this implementation uses 0.43*n.
/// @see flashsort_work(), flashsort_class_index()
std::size_t flashsort_class_count(std::size_t n);

/// @brief Aggregate actual original class counts into consecutive near-equal index ranges.
/// Group g contains [floor(g*m/groups),floor((g+1)*m/groups)). Empty input or
/// groups>m is allowed and can yield empty ranges. This sums actual class counts;
/// it does not reclassify values using a different number of value bins.
/// Validates the complete total and optional group cap before output allocation.
/// O(m+groups) time and O(groups) owned output, no input mutation/shared state.
/// @param class_counts Original occupancy vector, possibly empty.
/// @param groups Positive number of display groups.
/// @param max_groups Whole group cap, default1000000; zero disables only this optional cap.
/// @return Owned contiguous class ranges and exact uint64 totals, preserving the complete sum.
/// @throws std::invalid_argument If groups is zero.
/// @throws std::length_error If the complete cap or output capacity is exceeded.
/// @throws std::overflow_error If the whole occupancy total cannot fit uint64.
/// @throws std::bad_alloc If output storage cannot be allocated.
/// @par Example
/// @code
/// const auto g=dm::algorithms::flashsort_class_groups({2,1,3,0},2);
/// std::cout << g[0].count << ' ' << g[1].count; // 3 3
/// @endcode
/// @see FlashsortClassGroup, flashsort_trace()
std::vector<FlashsortClassGroup> flashsort_class_groups(const std::vector<std::size_t>& class_counts,
    std::size_t groups,std::size_t max_groups=1'000'000);

/// @brief Generate the original Flashsort report's local mt19937_64 integer-uniform sample.
/// Uses std::uniform_int_distribution<int> on inclusive [lower,upper]. Repeated
/// arguments reproduce on a given standard library; mappings across toolchains
/// are not promised identical. O(count) time/storage, no global random state.
/// @param count Sample length, default2000; empty input allowed.
/// @param seed Local mt19937_64 seed, default11.
/// @param lower Inclusive integer lower bound, default0.
/// @param upper Inclusive integer upper bound, default9999; at least lower.
/// @return Owned complete sample, without sorting it.
/// @throws std::invalid_argument If upper is less than lower.
/// @throws std::length_error If count exceeds vector capacity.
/// @throws std::bad_alloc If output storage cannot be allocated.
/// @par Example
/// @code
/// const auto a=dm::algorithms::flashsort_uniform_sample(3,11,4,4);
/// std::cout << a.size() << ' ' << a[0]; // 3 4
/// @endcode
/// @see flashsort_trace()
std::vector<int> flashsort_uniform_sample(std::size_t count=2000,std::uint64_t seed=11,int lower=0,int upper=9999);

/// @brief Internal checked mapping, preparation and shared actual transition kernels.
namespace flashsort_detail {
/// @brief Enforce the numeric template domain at compile time.
/// @tparam T Standard integral or floating type, excluding bool.
template<class T> constexpr void numeric_type() {
    static_assert(std::is_arithmetic_v<T> && !std::is_same_v<T,bool>,"Flashsort requires arithmetic values other than bool");
}
/// @brief Validate one floating scalar; integral values always satisfy this check.
/// @tparam T Supported arithmetic type.
/// @param value Candidate input value.
/// @throws std::invalid_argument If a floating value is not finite.
template<class T> void finite(T value) {
    numeric_type<T>();
    if constexpr(std::is_floating_point_v<T>)if(!std::isfinite(value))throw std::invalid_argument("Flashsort requires finite values");
}
/// @brief Compute a nonnegative integral difference in the corresponding unsigned domain.
/// @tparam T Integral type other than bool.
/// @param value Upper value, at least minimum in the original numeric order.
/// @param minimum Lower value.
/// @return Rounded double representation of the exact nonnegative integral difference.
template<class T> double integer_difference(T value,T minimum) {
    using U=std::make_unsigned_t<T>; ///< Unsigned domain with the same value width as T.
    const U difference=static_cast<U>(static_cast<U>(value)-static_cast<U>(minimum));
    return static_cast<double>(difference);
}
/// @brief Prepared monotone numeric classifier preserving the historical safe double expression.
/// @tparam T Supported arithmetic type.
template<class T> struct ClassMap {
    T minimum{}; ///< Finite lower observed value.
    T maximum{}; ///< Finite upper observed value.
    std::size_t classes=0; ///< Positive allocated/queried class count.
    double coefficient=0; ///< Historical (m-1)/double(max-min), only when safe.
    bool scaled=false; ///< Whether the historical expression was unsafe for this range.
    long double scale=1; ///< Positive finite magnitude scale for the fallback.
    long double scaled_minimum=0; ///< Lower endpoint divided by scale.
    long double scaled_span=1; ///< Difference of scaled endpoints, in (0,2].

    /// @brief Evaluate a validated in-range value, clamping before integer conversion.
    /// @param value Finite value inside the prepared endpoints.
    /// @return Rounded historical or scaled class index in [0,classes-1].
    std::size_t index(T value) const {
        if(minimum==maximum||classes==1)return 0;
        long double position;
        if(scaled)position=(classes-1)*((static_cast<long double>(value)/scale-scaled_minimum)/scaled_span);
        else {
            double difference;
            if constexpr(std::is_integral_v<T>)difference=integer_difference(value,minimum);
            else difference=static_cast<double>(value-minimum);
            position=coefficient*difference;
        }
        if(position<=0)return 0;
        if(position>=static_cast<long double>(classes-1))return classes-1;
        return static_cast<std::size_t>(position);
    }
};
/// @brief Prepare the exact historical safe expression or a finite scaled fallback.
/// @tparam T Supported arithmetic type.
/// @param minimum Finite lower endpoint.
/// @param maximum Finite upper endpoint, at least minimum.
/// @param classes Positive class count, at most INT_MAX.
/// @return Reusable validated mapping, with no heap allocation.
template<class T> ClassMap<T> make_map(T minimum,T maximum,std::size_t classes) {
    numeric_type<T>();ClassMap<T> map;map.minimum=minimum;map.maximum=maximum;map.classes=classes;
    if(minimum==maximum||classes==1)return map;
    double span=0;
    if constexpr(std::is_integral_v<T>)span=integer_difference(maximum,minimum);
    else {
        const bool overflow=minimum<0&&maximum>std::numeric_limits<T>::max()+minimum;
        if(!overflow){const T difference=maximum-minimum;if(static_cast<long double>(difference)<=std::numeric_limits<double>::max())span=static_cast<double>(difference);}
    }
    if(span>0&&std::isfinite(span)&&span>=static_cast<double>(classes-1)/std::numeric_limits<double>::max()) {
        map.coefficient=static_cast<double>(classes-1)/span;
        if(std::isfinite(map.coefficient)&&map.coefficient>0)return map;
    }
    map.scaled=true;
    map.scale=std::max(std::abs(static_cast<long double>(minimum)),std::abs(static_cast<long double>(maximum)));
    map.scaled_minimum=static_cast<long double>(minimum)/map.scale;
    map.scaled_span=static_cast<long double>(maximum)/map.scale-map.scaled_minimum;
    return map;
}
/// @brief Single min/max scan result, reused when admitting optional capture.
/// @tparam T Supported arithmetic type.
template<class T> struct Prepared {
    ClassMap<T> map; ///< Reusable classifier, unused on shortcut.
    FlashsortWork work; ///< Actual preparation counters and active class count.
    T minimum{}; ///< Actual minimum, zero placeholder for empty.
    T maximum{}; ///< Actual maximum, zero placeholder for empty.
};
/// @brief Validate the complete domain and scan extrema exactly once before array mutation.
/// @tparam T Supported arithmetic type.
/// @param a Caller input, never modified by preparation.
/// @return Actual preparation with finite data and safe signed index domain.
/// @throws std::length_error If the input length cannot fit the legacy index domain.
/// @throws std::invalid_argument If a floating input is nonfinite.
template<class T> Prepared<T> prepare(const std::vector<T>& a) {
    numeric_type<T>();(void)flashsort_class_count(a.size());
    for(T value:a)finite(value);
    Prepared<T> p;p.work.n=a.size();if(a.empty())return p;
    p.minimum=p.maximum=a[0];
    for(std::size_t i=1;i<a.size();++i){if(a[i]<p.minimum)p.minimum=a[i];if(a[i]>p.maximum)p.maximum=a[i];p.work.minmax_comparisons+=2;}
    if(!(p.minimum<p.maximum))return p;
    p.work.shortcut=false;p.work.class_count=flashsort_class_count(a.size());
    p.map=make_map(p.minimum,p.maximum,p.work.class_count);p.work.scaled_classification=p.map.scaled;return p;
}
/// @brief Run the original carry/insertion order on an already prepared valid vector.
/// @tparam T Supported arithmetic type.
/// @param a Mutable target, sorted in place.
/// @param prepared Actual once-only preparation, matching a.
/// @param capture Optional pre-admitted record destination.
/// @return Actual work including preparation, initial occupancies and the insertion tail.
/// @throws std::bad_alloc If class or optional record storage cannot be allocated.
/// @throws std::logic_error If an internal class-boundary invariant fails.
template<class T> FlashsortWork kernel(std::vector<T>& a,const Prepared<T>& prepared,FlashsortTrace<T>* capture) {
    auto work=prepared.work;if(work.shortcut)return work;
    const int n=static_cast<int>(a.size()),m=static_cast<int>(work.class_count);
    std::vector<int> L(work.class_count);
    /// @brief Count every actual shared classifier call made by this run.
    auto classify=[&](T value){++work.classifications;return static_cast<int>(prepared.map.index(value));};
    for(int i=0;i<n;++i){const int k=classify(a[i]);++L[k];if(capture)capture->class_of_input.push_back(k);}
    for(int k=0;k<m;++k){const auto count=static_cast<std::size_t>(L[k]);if(count)++work.nonempty_classes;work.max_occupancy=std::max(work.max_occupancy,count);work.occupancy_square_sum+=static_cast<std::uint64_t>(count)*count;if(capture)capture->class_counts.push_back(count);}
    for(int k=1;k<m;++k)L[k]+=L[k-1];
    if(capture)for(int end:L)capture->class_ends.push_back(static_cast<std::size_t>(end));
    int move=0,j=0,k=m-1;
    while(move<n-1) {
        while(j>L[k]-1){if(++j>=n)throw std::logic_error("Flashsort leader boundary invariant");k=classify(a[j]);}
        T flash=a[j];
        while(j!=L[k]) {
            k=classify(flash);
            if(L[k]<=0||move>=n)throw std::logic_error("Flashsort class placement invariant");
            const int idx=--L[k];
            if(capture)capture->moves.push_back({static_cast<std::size_t>(j),static_cast<std::size_t>(k),static_cast<std::size_t>(idx),flash,a[idx]});
            std::swap(flash,a[idx]);++move;++work.permutation_writes;
        }
    }
    if(capture)capture->classified=a;
    for(int i=1;i<n;++i) {
        T key=a[i];int p=i-1;
        while(p>=0){++work.insertion_comparisons;if(!(a[p]>key))break;a[p+1]=a[p];--p;++work.insertion_shifts;}
        a[p+1]=key;++work.insertion_placements;
    }
    return work;
}
} // namespace flashsort_detail

/// @brief Evaluate the sort's safe value classifier without allocating or sorting an array.
/// Preserves the historical rounded double coefficient/product when it is safe.
/// Integral differences use exact nonnegative unsigned subtraction. Floating
/// ranges that would overflow or make the coefficient nonfinite use a scaled
/// long-double ratio. Clamps before conversion; rounding near a class boundary
/// is part of the map and need not equal an arbitrary-precision floor formula.
/// A constant range maps its sole valid value to zero. O(1), no shared state.
/// @tparam T Arithmetic value type other than bool.
/// @param value Finite value in [minimum,maximum].
/// @param minimum Finite lower bound.
/// @param maximum Finite upper bound, at least minimum.
/// @param classes Positive class count, at most INT_MAX.
/// @return Index in [0,classes-1], with a monotone mapping in value.
/// @throws std::invalid_argument For nonfinite/reversed/out-of-range data or zero classes.
/// @throws std::length_error If classes exceeds the signed index domain.
/// @par Example
/// @code
/// std::cout << dm::algorithms::flashsort_class_index(5,0,10,3); // 1
/// @endcode
/// @par References
/// - K.-D. Neubert, Flashsort1 Algorithm, Dr. Dobb's Journal, February1998,p123,
///   https://www.neubert.net/FSOIntro.html ; checked finite arithmetic variant here.
/// @see flashsort_trace(), flashsort_class_count()
template<class T> std::size_t flashsort_class_index(T value,T minimum,T maximum,std::size_t classes) {
    flashsort_detail::finite(value);flashsort_detail::finite(minimum);flashsort_detail::finite(maximum);
    if(maximum<minimum||value<minimum||value>maximum||!classes)throw std::invalid_argument("Flashsort classifier domain");
    if(classes>static_cast<std::size_t>(INT_MAX))throw std::length_error("Flashsort class index domain");
    return flashsort_detail::make_map(minimum,maximum,classes).index(value);
}

/// @brief Sort an arithmetic vector in place by value classification, cycle carries and insertion.
/// Uses max(2,int(0.43*n)) classes for nonconstant n>1 input; empty/singleton/
/// constant inputs return without classes. Requires finite floating values and
/// n<=INT_MAX, checked before mutation/allocation. Integral spans are safe across
/// the full type range; unsafe floating ranges use scaled classification.
/// Historical safe classifications and carry/insertion order are retained; this
/// variant does not first swap the maximum to index zero. It is not stable.
/// O(n+m+sum(class_size²)) worst-case work, O(m) auxiliary metadata; m is Theta(n),
/// so auxiliary memory is not constant. Expected linear work needs a suitable
/// balanced-input model; all-equal data takes the shortcut. No shared state.
/// @tparam T Arithmetic value type other than bool.
/// @param a Values to sort into ascending order, preserving all duplicates.
/// @throws std::invalid_argument If any floating input is nonfinite.
/// @throws std::length_error If length exceeds the signed legacy index domain.
/// @throws std::bad_alloc If class storage cannot be allocated.
/// @throws std::logic_error If an internal class-boundary invariant fails; mutation may be partial.
/// @par Example
/// @code
/// std::vector<int> a{3,1,2};dm::algorithms::flashsort(a);
/// std::cout << a[0] << ' ' << a[2]; // 1 3
/// @endcode
/// @par References
/// - K.-D. Neubert, Flashsort1 Algorithm, Dr. Dobb's Journal, February1998,p123,
///   https://www.neubert.net/FSOIntro.html ; one whole-array insertion tail here.
/// @see flashsort_work(), flashsort_trace(), flashsort_class_index()
template<class T> void flashsort(std::vector<T>& a) {
    const auto p=flashsort_detail::prepare(a);(void)flashsort_detail::kernel(a,p,static_cast<FlashsortTrace<T>*>(nullptr));
}

/// @brief Sort in place with the same shared kernel and return actual class/primitive work.
/// No snapshots or carry history are stored. Complexity, validation, mutation
/// guarantees and distribution conditions are those of flashsort(). Counters
/// describe this execution; they do not measure time or certify expected work.
/// @tparam T Arithmetic value type other than bool.
/// @param a Values to sort in place, with finite floating entries and n<=INT_MAX.
/// @return Actual counters, active class count and original occupancy summaries.
/// @throws std::invalid_argument If any floating input is nonfinite.
/// @throws std::length_error If length exceeds the signed legacy index domain.
/// @throws std::bad_alloc If class storage cannot be allocated.
/// @throws std::logic_error If an internal invariant fails; mutation may be partial.
/// @par Example
/// @code
/// std::vector<int> a{2,0,1};const auto w=dm::algorithms::flashsort_work(a);
/// std::cout << w.insertion_shifts << ' ' << w.insertion_placements; // 0 2
/// @endcode
/// @par References
/// - K.-D. Neubert, Flashsort1 Algorithm, https://www.neubert.net/FSOIntro.html .
/// @see flashsort(), FlashsortWork, flashsort_trace()
template<class T> FlashsortWork flashsort_work(std::vector<T>& a) {
    const auto p=flashsort_detail::prepare(a);return flashsort_detail::kernel(a,p,static_cast<FlashsortTrace<T>*>(nullptr));
}

/// @brief Sort one owned copy and retain actual classes, carried exchanges and insertion work.
/// Validates/scans the input once, then admits complete C=5n+2m (shortcut2n)
/// before copying or sorting. Shared preparation and kernel avoid replay. Input
/// remains unchanged; all snapshots/records own their values and copies are
/// independent. O(n+m) output entries plus the ordinary sort's work; capture
/// limits count entries, not bytes, timing or a peak-memory bound.
/// @tparam T Arithmetic value type other than bool.
/// @param a Original values, with finite floating entries and n<=INT_MAX.
/// @param max_entries Complete entry cap, default1000000; zero disables only this optional cap.
/// @return Owned actual trace and ascending output; no partial trace on failure.
/// @throws std::invalid_argument If any floating input is nonfinite.
/// @throws std::length_error If signed length, accounting/capacity or complete cap is exceeded.
/// @throws std::bad_alloc If owned values/class arrays/records cannot be allocated.
/// @throws std::logic_error If an internal class-boundary invariant fails.
/// @par Example
/// @code
/// const auto t=dm::algorithms::flashsort_trace(std::vector<int>{2,0,1});
/// std::cout << t.class_counts[0] << ' ' << t.class_ends[0] << ' ' << t.classified[0]; // 2 2 0
/// @endcode
/// @par References
/// - K.-D. Neubert, Flashsort1 Algorithm, https://www.neubert.net/FSOIntro.html .
/// @see FlashsortTrace, flashsort_work(), flashsort_class_groups()
template<class T> FlashsortTrace<T> flashsort_trace(const std::vector<T>& a,std::size_t max_entries=1'000'000) {
    const auto p=flashsort_detail::prepare(a);
    const auto count=static_cast<unsigned __int128>(a.size())*(p.work.shortcut?2:5)+static_cast<unsigned __int128>(p.work.class_count)*2;
    if(count>std::numeric_limits<std::size_t>::max()||(max_entries&&count>max_entries))throw std::length_error("Flashsort complete capture budget");
    if(!p.work.shortcut&&a.size()>std::vector<FlashsortMove<T>>{}.max_size())throw std::length_error("Flashsort record capacity");
    FlashsortTrace<T> out;out.input=a;out.output=a;out.minimum=p.minimum;out.maximum=p.maximum;
    if(!p.work.shortcut){out.class_of_input.reserve(a.size());out.class_counts.reserve(p.work.class_count);out.class_ends.reserve(p.work.class_count);out.moves.reserve(a.size());}
    out.work=flashsort_detail::kernel(out.output,p,&out);return out;
}
} // namespace datamunge::algorithms
