#pragma once

/// \file doomsday.hpp
/// \brief Conway's Doomsday rule: mental-arithmetic day of the week.
///
/// John Conway's *Doomsday rule* (1973) is designed for mental calculation. Every year has a
/// *doomsday* -- the weekday shared by a set of easily remembered dates (4/4, 6/6, 8/8, 10/10,
/// 12/12, the last day of February, "pi day" 3/14, and "9-to-5 at 7-11": 9/5, 5/9, 7/11,
/// 11/7). Knowing the year's doomsday, any date's weekday follows by counting days from the
/// nearest anchor. This module implements the arithmetic exactly.

#include <string>

namespace datamunge::algorithms {

/// \brief Whether \c year is a Gregorian leap year.
inline bool doomsday_is_leap(int year) {
    return (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0);
}

/// \brief The weekday of the given year's doomsday (0=Sunday..6=Saturday).
inline int doomsday_of_year(int year) {
    int c      = year / 100;
    int anchor = (5 * (c % 4) + 2) % 7;   // century anchor day
    int y      = year % 100;
    return (anchor + y + y / 4) % 7;
}

/// \brief Day of the week for a Gregorian date via the Doomsday rule.
///
/// \return weekday in 0..6 with 0=Sunday, 1=Monday, ..., 6=Saturday.
inline int doomsday_weekday(int year, int month, int day) {
    // Doomsday reference date within each month (1-indexed months).
    static const int ref_common[] = {0, 3, 28, 14, 4, 9, 6, 11, 8, 5, 10, 7, 12};
    int ref = ref_common[month];
    if (doomsday_is_leap(year)) {
        if (month == 1) ref = 4;   // January doomsday is the 4th in a leap year
        if (month == 2) ref = 29;  // February doomsday is the 29th
    }
    int d = doomsday_of_year(year);
    return (((d + (day - ref)) % 7) + 7) % 7;
}

/// \brief Convert a Doomsday weekday (0=Sunday..6=Saturday) to a name.
inline std::string doomsday_day_name(int w) {
    static const char* names[] = {"Sunday",    "Monday",   "Tuesday", "Wednesday",
                                  "Thursday", "Friday",   "Saturday"};
    return names[((w % 7) + 7) % 7];
}

}  // namespace datamunge::algorithms
