/// @file factorial_design.cpp
/// @brief Bit-mask effects, generated columns and defining-relation cosets for two-level designs.

#include <datamunge/stats/factorial_design.hpp>

#include <algorithm>
#include <bit>
#include <cstdint>
#include <map>
#include <stdexcept>

namespace datamunge::stats {
/// @brief Effect masks and their letter names.
namespace {

/// @brief Factor letters A..Z without I, as in FrF2.
constexpr char letters[] = "ABCDEFGHJKLMNOPQRSTUVWXYZ";

/// @brief Name of an effect mask, letters in factor order.
/// @param mask Bit j set when factor j is in the effect.
/// @return Word such as "ABD".
std::string word(std::uint32_t mask) {
    std::string out;
    for (std::size_t j = 0; mask >> j; ++j)
        if (mask >> j & 1U)
            out += letters[j];
    return out;
}

/// @brief Mask of a factor name.
/// @param name Single letter.
/// @return Factor index, or -1 when unknown.
int letter_index(char name) {
    for (int j = 0; letters[j]; ++j)
        if (letters[j] == name)
            return j;
    return -1;
}

/// @brief Sort key: length, then alphabetical.
/// @param a First word.
/// @param b Second word.
/// @return Whether a orders before b.
bool shorter_first(const std::string& a, const std::string& b) {
    return a.size() != b.size() ? a.size() < b.size() : a < b;
}

/// @brief Builds a design from base factor count and generator masks over the base factors.
/// @param factors Total factors.
/// @param generator_masks Masks over base factors, one per generated factor.
/// @return Design with runs, names, generators, defining relation and resolution.
TwoLevelDesign build(std::size_t factors, const std::vector<std::uint32_t>& generator_masks) {
    const std::size_t q = generator_masks.size(), p = factors-q, n = std::size_t{1} << p;
    TwoLevelDesign design;
    design.runs = linalg::DenseMatrix<double>(n,factors);
    for (std::size_t j = 0; j < factors; ++j)
        design.factors.emplace_back(1,letters[j]);
    for (std::size_t r = 0; r < n; ++r) {
        for (std::size_t j = 0; j < p; ++j)
            design.runs(r,j) = (r >> j & 1U) ? 1.0 : -1.0;
        for (std::size_t g = 0; g < q; ++g) {
            double product = 1;
            for (std::size_t j = 0; j < p; ++j)
                if (generator_masks[g] >> j & 1U)
                    product *= design.runs(r,j);
            design.runs(r,p+g) = product;
        }
    }
    std::vector<std::uint32_t> words;
    for (std::size_t g = 0; g < q; ++g) {
        design.generators.push_back(std::string(1,letters[p+g])+"="+word(generator_masks[g]));
        words.push_back(generator_masks[g] | (std::uint32_t{1} << (p+g)));
    }
    for (std::uint32_t subset = 1; subset < (std::uint32_t{1} << q); ++subset) {
        std::uint32_t product = 0;
        for (std::size_t g = 0; g < q; ++g)
            if (subset >> g & 1U)
                product ^= words[g];
        design.defining_relation.push_back(word(product));
    }
    std::ranges::sort(design.defining_relation,shorter_first);
    design.word_length_pattern.assign(factors >= 3 ? factors-2 : 0,0);
    for (const auto& w : design.defining_relation)
        if (w.size() >= 3)
            ++design.word_length_pattern[w.size()-3];
    design.resolution = design.defining_relation.empty() ? 0 : design.defining_relation.front().size();
    return design;
}

} // namespace

TwoLevelDesign full_factorial_design(std::size_t factors) {
    if (factors < 1 || factors > 20)
        throw std::invalid_argument("full_factorial_design: need 1 to 20 factors");
    return build(factors,{});
}

TwoLevelDesign fractional_factorial_design(std::size_t factors, const std::vector<std::string>& generators) {
    if (factors < 2 || factors > 20)
        throw std::invalid_argument("fractional_factorial_design: need 2 to 20 factors");
    const std::size_t q = generators.size();
    if (q == 0 || q >= factors)
        throw std::invalid_argument("fractional_factorial_design: need between 1 and k-1 generators");
    const std::size_t p = factors-q;
    std::vector<std::uint32_t> masks;
    for (const auto& g : generators) {
        std::uint32_t mask = 0;
        for (const char c : g) {
            const int j = letter_index(c);
            if (j < 0 || static_cast<std::size_t>(j) >= p)
                throw std::invalid_argument("fractional_factorial_design: generator letters must name base factors");
            if (mask >> j & 1U)
                throw std::invalid_argument("fractional_factorial_design: repeated letter in a generator");
            mask |= std::uint32_t{1} << j;
        }
        if (std::popcount(mask) < 2)
            throw std::invalid_argument("fractional_factorial_design: generator words need at least two letters");
        if (std::ranges::find(masks,mask) != masks.end())
            throw std::invalid_argument("fractional_factorial_design: generator words must be distinct");
        masks.push_back(mask);
    }
    return build(factors,masks);
}

std::vector<std::vector<std::string>> alias_sets(const TwoLevelDesign& design, std::size_t max_order) {
    const std::size_t k = design.factors.size();
    if (max_order < 1 || max_order > k)
        throw std::invalid_argument("alias_sets: max_order must be between 1 and the number of factors");
    std::vector<std::uint32_t> group{0};
    for (const auto& w : design.defining_relation) {
        std::uint32_t mask = 0;
        for (const char c : w)
            mask |= std::uint32_t{1} << letter_index(c);
        group.push_back(mask);
    }
    // Coset representative: the smallest mask in effect * group.
    std::map<std::uint32_t,std::vector<std::string>> cosets;
    for (std::uint32_t effect = 1; effect < (std::uint32_t{1} << k); ++effect) {
        if (static_cast<std::size_t>(std::popcount(effect)) > max_order)
            continue;
        std::uint32_t key = effect;
        for (const auto g : group)
            key = std::min(key,effect ^ g);
        if (key == 0)
            continue; // aliased with the mean
        cosets[key].push_back(word(effect));
    }
    std::vector<std::vector<std::string>> out;
    for (auto& [key,members] : cosets)
        if (members.size() > 1) {
            std::ranges::sort(members,shorter_first);
            out.push_back(std::move(members));
        }
    std::ranges::sort(out,[](const auto& a, const auto& b) { return shorter_first(a.front(),b.front()); });
    return out;
}

} // namespace datamunge::stats
