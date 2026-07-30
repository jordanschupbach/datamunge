#pragma once

/// \file nagle.hpp
/// \brief Nagle's algorithm: coalescing small TCP segments.
///
/// Nagle's algorithm (1984) improves TCP efficiency by *not* sending a small segment while an
/// earlier small segment is still unacknowledged. Instead the sender buffers the small data;
/// it transmits when either a full maximum segment (MSS) has accumulated *or* the outstanding
/// data is acknowledged. This collapses a burst of tiny writes (the "many small packets"
/// problem of interactive protocols) into fewer, fuller packets, cutting the per-packet header
/// overhead -- at the cost of a little latency. This module simulates the sender's packetization
/// under a stream of writes and acknowledgements.

#include <algorithm>
#include <vector>

namespace datamunge::algorithms {

/// A sender-side event: an application write, or an acknowledgement of sent bytes.
struct NagleEvent {
    enum Type { Write, Ack } type;
    int                      bytes;   ///< Bytes written (Write) or acknowledged (Ack).
};

/// \brief Simulate Nagle's algorithm over a sequence of events; returns the packet sizes sent.
inline std::vector<int> nagle_simulate(const std::vector<NagleEvent>& events, int mss) {
    std::vector<int> packets;
    int              buffer = 0, unacked = 0;

    auto emit = [&](int n) { packets.push_back(n); unacked += n; };

    for (const auto& e : events) {
        if (e.type == NagleEvent::Write) {
            buffer += e.bytes;
            while (buffer >= mss) { emit(mss); buffer -= mss; }   // full segments always go
            if (buffer > 0 && unacked == 0) { emit(buffer); buffer = 0; }  // small: only if idle
        } else {                                                  // Ack
            unacked = std::max(0, unacked - e.bytes);
            if (unacked == 0 && buffer > 0) { emit(buffer); buffer = 0; }  // ack releases held data
        }
    }
    return packets;
}

/// \brief Baseline without Nagle: every write is sent at once (split into MSS-sized packets).
inline std::vector<int> no_nagle_simulate(const std::vector<NagleEvent>& events, int mss) {
    std::vector<int> packets;
    for (const auto& e : events) {
        if (e.type != NagleEvent::Write) continue;
        int b = e.bytes;
        while (b > mss) { packets.push_back(mss); b -= mss; }
        if (b > 0) packets.push_back(b);
    }
    return packets;
}

}  // namespace datamunge::algorithms
