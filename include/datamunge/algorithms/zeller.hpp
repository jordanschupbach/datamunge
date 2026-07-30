#pragma once

/// \file zeller.hpp
/// \brief Zeller's congruence: day of the week for any Gregorian (or Julian) date.
///
/// Zeller's congruence (Christian Zeller, 1882) computes the weekday of a calendar date with
/// a single modular formula. Treating January and February as months 13 and 14 of the
/// *previous* year makes the leap day the last day of the year, so the month-length pattern
/// becomes regular enough to capture with \f$\lfloor 13(m+1)/5\rfloor\f$. The result is exact
/// for the proleptic Gregorian calendar over any year range.

#include <string>

namespace datamunge::algorithms {

/// \brief Day of the week for a Gregorian date via Zeller's congruence.
///
/// \return \c h in 0..6 with the Zeller convention 0=Saturday, 1=Sunday, ..., 6=Friday.
inline int zeller_gregorian(int year, int month, int day) {
    if (month < 3) {          // Jan, Feb counted as months 13, 14 of the prior year
        month += 12;
        year  -= 1;
    }
    int K = year % 100;       // year within the century
    int J = year / 100;       // zero-based century
    int h = (day + (13 * (month + 1)) / 5 + K + K / 4 + J / 4 + 5 * J) % 7;
    return h;
}

/// \brief Convert a Zeller \c h (0=Saturday..6=Friday) to a weekday name.
inline std::string zeller_day_name(int h) {
    static const char* names[] = {"Saturday", "Sunday",   "Monday", "Tuesday",
                                  "Wednesday", "Thursday", "Friday"};
    return names[((h % 7) + 7) % 7];
}

}  // namespace datamunge::algorithms
