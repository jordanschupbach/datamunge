#pragma once

/// \file computus.hpp
/// \brief Computus: the date of Easter Sunday (Anonymous Gregorian algorithm).
///
/// *Computus* is the calculation of Easter -- the first Sunday after the first ecclesiastical
/// full moon on or after 21 March. The "Anonymous Gregorian" / Meeus-Jones-Butcher algorithm
/// computes it with pure integer arithmetic: it tracks the Metonic 19-year lunar cycle (the
/// golden number), the solar and lunar Gregorian corrections that keep the ecclesiastical
/// moon aligned over centuries, and the weekday of the paschal full moon. This module returns
/// the month and day of Easter for any Gregorian year.

namespace datamunge::algorithms {

/// Month/day of Easter Sunday.
struct EasterDate {
    int month;  ///< 3 (March) or 4 (April).
    int day;    ///< Day of the month.
};

/// \brief Date of Easter Sunday in the Gregorian calendar (Meeus/Jones/Butcher algorithm).
inline EasterDate computus_gregorian(int year) {
    int a = year % 19;                    // position in the 19-year Metonic cycle
    int b = year / 100, c = year % 100;   // century and year-of-century
    int d = b / 4, e = b % 4;             // leap-century corrections
    int f = (b + 8) / 25;
    int g = (b - f + 1) / 3;
    int h = (19 * a + b - d - g + 15) % 30;   // days from 21 March to the paschal full moon
    int i = c / 4, k = c % 4;
    int l = (32 + 2 * e + 2 * i - h - k) % 7;  // weekday offset to the following Sunday
    int m = (a + 11 * h + 22 * l) / 451;       // lunar-cycle correction
    int month = (h + l - 7 * m + 114) / 31;
    int day   = ((h + l - 7 * m + 114) % 31) + 1;
    return {month, day};
}

}  // namespace datamunge::algorithms
