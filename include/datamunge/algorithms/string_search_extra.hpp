#pragma once

// Two Boyer-Moore-family substring searchers, complementing boyer_moore.hpp:
//   - Boyer-Moore-Horspool: the bad-character-only simplification
//   - Zhu-Takaoka: a Boyer-Moore variant with a two-character bad-character
//     shift plus the good-suffix rule
// Each returns every start index at which `pattern` occurs in `text`.

#include <cstddef>
#include <string>
#include <vector>

namespace datamunge::algorithms {

std::vector<std::size_t> boyer_moore_horspool_search(const std::string& text, const std::string& pattern);

std::vector<std::size_t> zhu_takaoka_search(const std::string& text, const std::string& pattern);

} // namespace datamunge::algorithms
