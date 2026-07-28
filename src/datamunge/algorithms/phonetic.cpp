#include <datamunge/algorithms/phonetic.hpp>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <string>

namespace datamunge::algorithms {

namespace {

char up(char c) { return static_cast<char>(std::toupper(static_cast<unsigned char>(c))); }
char lo(char c) { return static_cast<char>(std::tolower(static_cast<unsigned char>(c))); }

std::string upper(const std::string& s) {
    std::string r(s.size(), '\0');
    std::transform(s.begin(), s.end(), r.begin(), up);
    return r;
}

// Membership of ch in a small set; a sentinel '\0' (out of range) matches nothing.
bool in(const char* set, char ch) { return ch != '\0' && std::strchr(set, ch) != nullptr; }

bool is_vowel(char c) { return in("AEIOU", c); }

} // namespace

std::string soundex(const std::string& name) {
    const std::string s = upper(name);
    if (s.empty()) return "";

    auto digit = [](char c) -> char {
        if (in("BFPV", c)) return '1';
        if (in("CGJKQSXZ", c)) return '2';
        if (in("DT", c)) return '3';
        if (c == 'L') return '4';
        if (in("MN", c)) return '5';
        if (c == 'R') return '6';
        return '0'; // vowels, Y, H, W: not coded
    };

    std::string result(1, s[0]);
    int         count = 1;
    char        last  = digit(s[0]); // '0' if the first letter is itself uncoded

    for (std::size_t i = 1; i < s.size() && count < 4; ++i) {
        const char c = s[i];
        const char d = digit(c);
        if (d != '0') {
            if (d != last) {
                result.push_back(d);
                ++count;
            }
            last = d;
        } else if (c != 'H' && c != 'W') {
            // vowels/Y reset the running code (H and W are transparent: leave it alone)
            last = '0';
        }
    }
    result.append(static_cast<std::size_t>(4 - count), '0');
    return result;
}

std::string nysiis(const std::string& name) {
    if (name.empty()) return "";
    std::string s = upper(name);

    // step 1 - prefixes
    if (s.rfind("MAC", 0) == 0)
        s = "MCC" + s.substr(3);
    else if (s.rfind("KN", 0) == 0)
        s = s.substr(1);
    else if (s.rfind("K", 0) == 0)
        s = "C" + s.substr(1);
    else if (s.rfind("PH", 0) == 0 || s.rfind("PF", 0) == 0)
        s = "FF" + s.substr(2);
    else if (s.rfind("SCH", 0) == 0)
        s = "SSS" + s.substr(3);

    // step 2 - suffixes
    auto ends = [&](const char* suf) {
        const std::string t(suf);
        return s.size() >= t.size() && s.compare(s.size() - t.size(), t.size(), t) == 0;
    };
    if (ends("IE") || ends("EE"))
        s = s.substr(0, s.size() - 2) + "Y";
    else if (ends("DT") || ends("RT") || ends("RD") || ends("NT") || ends("ND"))
        s = s.substr(0, s.size() - 2) + "D";

    // step 3 - first character of the key is taken verbatim
    std::string  key(1, s[0]);
    const std::size_t n = s.size();

    // step 4 - translate the remaining characters
    for (std::size_t i = 1; i < n; ++i) {
        const char  c   = s[i];
        const char  nx  = (i + 1 < n) ? s[i + 1] : '\0';
        std::string ch(1, c);

        if (c == 'E' && nx == 'V') {
            ch = "AF";
            ++i;
        } else if (is_vowel(c)) {
            ch = "A";
        } else if (c == 'Q') {
            ch = "G";
        } else if (c == 'Z') {
            ch = "S";
        } else if (c == 'M') {
            ch = "N";
        } else if (c == 'K') {
            ch = (nx == 'N') ? "N" : "C";
        } else if (c == 'S' && nx == 'C' && i + 2 < n && s[i + 2] == 'H') {
            ch = "SS";
            i += 2;
        } else if (c == 'P' && nx == 'H') {
            ch = "F";
            ++i;
        } else if (c == 'H' &&
                   (!is_vowel(s[i - 1]) || (i + 1 < n && !is_vowel(s[i + 1])) || (i + 1 == n))) {
            ch = is_vowel(s[i - 1]) ? std::string("A") : std::string(1, s[i - 1]);
        } else if (c == 'W' && is_vowel(s[i - 1])) {
            ch = std::string(1, s[i - 1]);
        }

        // append unless this token's last letter repeats the key's last letter
        if (ch.back() != key.back()) key += ch;
    }

    // step 5 - drop a trailing S
    if (key.size() > 1 && key.back() == 'S') key.pop_back();
    // step 6 - AY -> Y
    if (key.size() >= 2 && key.compare(key.size() - 2, 2, "AY") == 0) key = key.substr(0, key.size() - 2) + "Y";
    // step 7 - drop a trailing A
    if (key.size() > 1 && key.back() == 'A') key.pop_back();

    return key;
}

std::string metaphone(const std::string& name) {
    std::string s(name.size(), '\0');
    std::transform(name.begin(), name.end(), s.begin(), lo);

    // drop the first letter of a silent-onset cluster
    static const char* onsets[] = {"kn", "gn", "pn", "wr", "ae"};
    for (const char* o : onsets) {
        if (s.rfind(o, 0) == 0) {
            s = s.substr(1);
            break;
        }
    }

    std::string  out;
    const std::size_t n = s.size();
    auto lower_in = [](const char* set, char ch) { return ch != '\0' && std::strchr(set, ch) != nullptr; };

    for (std::size_t i = 0; i < n; ++i) {
        const char c   = s[i];
        const char nx  = (i + 1 < n) ? s[i + 1] : '\0';
        const char nnx = (i + 2 < n) ? s[i + 2] : '\0';

        // skip doubled letters, except "cc"
        if (c == nx && c != 'c') continue;

        if (lower_in("aeiou", c)) {
            if (i == 0 || s[i - 1] == ' ') out.push_back(c);
        } else if (c == 'b') {
            // silent b in a word-final "mb" (comb, dumb) -- but kept mid-word (amber, climber)
            if (!(i != 0 && s[i - 1] == 'm' && nx == '\0')) out.push_back('b');
        } else if (c == 'c') {
            if ((nx == 'i' && nnx == 'a') || nx == 'h') {
                out.push_back('x');
                ++i;
            } else if (lower_in("iey", nx)) {
                out.push_back('s');
                ++i;
            } else {
                out.push_back('k');
            }
        } else if (c == 'd') {
            if (nx == 'g' && lower_in("iey", nnx)) {
                out.push_back('j');
                i += 2;
            } else {
                out.push_back('t');
            }
        } else if (lower_in("fjlmnr", c)) {
            out.push_back(c);
        } else if (c == 'g') {
            if (lower_in("iey", nx)) {
                out.push_back('j');
            } else if (nx == 'h' && nnx != '\0' && !lower_in("aeiou", nnx)) {
                ++i; // silent gh before a consonant (light, night, eight)
            } else if (nx == 'n' && nnx == '\0') {
                ++i; // silent gn ending a word (sign, align, reign) -- consumes the n
            } else {
                out.push_back('k');
            }
        } else if (c == 'h') {
            if (i == 0 || lower_in("aeiou", nx) || !lower_in("aeiou", s[i - 1])) out.push_back('h');
        } else if (c == 'k') {
            if (i == 0 || s[i - 1] != 'c') out.push_back('k');
        } else if (c == 'p') {
            if (nx == 'h') {
                out.push_back('f');
                ++i;
            } else {
                out.push_back('p');
            }
        } else if (c == 'q') {
            out.push_back('k');
        } else if (c == 's') {
            if (nx == 'h') {
                out.push_back('x');
                ++i;
            } else if (nx == 'i' && (nnx == 'o' || nnx == 'a')) {
                out.push_back('x');
                i += 2;
            } else {
                out.push_back('s');
            }
        } else if (c == 't') {
            if (nx == 'i' && (nnx == 'o' || nnx == 'a')) {
                out.push_back('x');
            } else if (nx == 'h') {
                out.push_back('0');
                ++i;
            } else if (nx != 'c' || nnx != 'h') {
                out.push_back('t');
            }
        } else if (c == 'v') {
            out.push_back('f');
        } else if (c == 'w') {
            if (i == 0 && nx == 'h') {
                ++i;
                out.push_back('w');
            } else if (lower_in("aeiou", nx)) {
                out.push_back('w');
            }
        } else if (c == 'x') {
            if (i == 0) {
                if (nx == 'h' || (nx == 'i' && (nnx == 'o' || nnx == 'a')))
                    out.push_back('x');
                else
                    out.push_back('s');
            } else {
                out.push_back('k');
                out.push_back('s');
            }
        } else if (c == 'y') {
            if (lower_in("aeiou", nx)) out.push_back('y');
        } else if (c == 'z') {
            out.push_back('s');
        } else if (c == ' ') {
            if (!out.empty() && out.back() != ' ') out.push_back(' ');
        }
    }

    return upper(out);
}

std::string match_rating_codex(const std::string& name) {
    // uppercase, drop spaces; non-alphabetic characters are simply ignored
    std::string codex;
    char        prev  = '\0';
    bool        first = true;
    for (char raw : name) {
        if (raw == ' ') continue;
        if (!std::isalpha(static_cast<unsigned char>(raw))) continue;
        const char c = up(raw);
        // keep the first letter, and any consonant not equal to the letter before it
        if (first || (!is_vowel(c) && c != prev)) codex.push_back(c);
        prev  = c;
        first = false;
    }
    if (codex.size() > 6) return codex.substr(0, 3) + codex.substr(codex.size() - 3);
    return codex;
}

MatchRatingResult match_rating_comparison(const std::string& a, const std::string& b) {
    const std::string c1 = match_rating_codex(a);
    const std::string c2 = match_rating_codex(b);
    const int         l1 = static_cast<int>(c1.size());
    const int         l2 = static_cast<int>(c2.size());

    MatchRatingResult r;
    if (std::abs(l1 - l2) >= 3) return r; // not comparable
    r.comparable = true;

    const int sum = l1 + l2;
    r.min_rating  = (sum <= 4) ? 5 : (sum <= 7) ? 4 : (sum <= 11) ? 3 : 2;

    // strip letters common to both, scanning left to right
    std::string res1, res2;
    const std::size_t m = std::max(c1.size(), c2.size());
    for (std::size_t i = 0; i < m; ++i) {
        const char x = i < c1.size() ? c1[i] : '\0';
        const char y = i < c2.size() ? c2[i] : '\0';
        if (x != y) {
            if (x != '\0') res1.push_back(x);
            if (y != '\0') res2.push_back(y);
        }
    }

    // count remaining mismatches scanning right to left
    int unmatched1 = 0, unmatched2 = 0;
    const std::size_t m2 = std::max(res1.size(), res2.size());
    for (std::size_t i = 0; i < m2; ++i) {
        const char x = i < res1.size() ? res1[res1.size() - 1 - i] : '\0';
        const char y = i < res2.size() ? res2[res2.size() - 1 - i] : '\0';
        if (x != y) {
            if (x != '\0') ++unmatched1;
            if (y != '\0') ++unmatched2;
        }
    }

    r.similarity = 6 - std::max(unmatched1, unmatched2);
    r.match      = r.similarity >= r.min_rating;
    return r;
}

} // namespace datamunge::algorithms
