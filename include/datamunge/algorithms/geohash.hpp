#pragma once

/// \file geohash.hpp
/// \brief Geohash: encode a latitude/longitude pair as a short base-32 string.
///
/// A *geohash* (Gustavo Niemeyer, 2008) encodes a point on Earth as a string by recursively
/// bisecting the longitude and latitude ranges. Each bit answers "upper or lower half?",
/// alternating longitude, latitude, longitude, ...; the bit stream is grouped into 5-bit
/// symbols and written in a custom base-32 alphabet. The result has the *prefix property*:
/// nearby points share a leading prefix, so string-prefix comparison gives a cheap proximity
/// filter and geohashes index naturally in ordinary B-trees.

#include <string>

namespace datamunge::algorithms {

/// \brief Encode (lat, lon) as a geohash of the given length.
///
/// \param lat        latitude in [-90, 90].
/// \param lon        longitude in [-180, 180].
/// \param precision  number of base-32 characters to emit.
inline std::string geohash_encode(double lat, double lon, int precision) {
    static const char* base32 = "0123456789bcdefghjkmnpqrstuvwxyz";  // no a, i, l, o
    double lat_lo = -90.0,  lat_hi = 90.0;
    double lon_lo = -180.0, lon_hi = 180.0;

    std::string hash;
    bool even  = true;   // start by bisecting longitude
    int  bit   = 0;      // bit index within the current 5-bit symbol
    int  ch    = 0;      // accumulating symbol value

    while (static_cast<int>(hash.size()) < precision) {
        if (even) {  // longitude bit
            double mid = (lon_lo + lon_hi) / 2.0;
            if (lon >= mid) { ch = (ch << 1) | 1; lon_lo = mid; }
            else            { ch = (ch << 1);     lon_hi = mid; }
        } else {     // latitude bit
            double mid = (lat_lo + lat_hi) / 2.0;
            if (lat >= mid) { ch = (ch << 1) | 1; lat_lo = mid; }
            else            { ch = (ch << 1);     lat_hi = mid; }
        }
        even = !even;
        if (++bit == 5) {          // one base-32 symbol complete
            hash += base32[ch];
            bit = 0;
            ch  = 0;
        }
    }
    return hash;
}

}  // namespace datamunge::algorithms
