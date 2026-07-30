#pragma once

/// \file karn.hpp
/// \brief Karn's algorithm: robust TCP round-trip-time estimation.
///
/// TCP sets its retransmission timeout (RTO) from a running estimate of the round-trip time,
/// smoothed by the Jacobson/Karels filter. *Karn's algorithm* (1987) fixes a subtle bug: when a
/// segment is retransmitted, an arriving ACK is *ambiguous* -- it might acknowledge the original
/// or the retransmission -- so the measured RTT is unreliable. Karn's rule is to *discard* RTT
/// samples from retransmitted segments, and to hold the RTO at its *backed-off* (doubled) value
/// until a fresh, unambiguous sample arrives. This module implements the estimator with Karn's
/// two rules.

#include <cmath>

namespace datamunge::algorithms {

/// A TCP RTT/RTO estimator using the Jacobson/Karels filter plus Karn's algorithm.
class KarnEstimator {
  public:
    /// \param initial_rto  the starting timeout before any sample (seconds).
    explicit KarnEstimator(double initial_rto = 1.0) : rto_(initial_rto) {}

    /// \brief Incorporate an ACK measuring RTT \c rtt_sample.
    ///
    /// If \c was_retransmitted, the sample is *ignored* (Karn's rule) and the RTO is unchanged.
    void on_ack(double rtt_sample, bool was_retransmitted) {
        if (was_retransmitted) return;                 // ambiguous -> discard
        if (!initialized_) {
            srtt_        = rtt_sample;
            rttvar_      = rtt_sample / 2.0;
            initialized_ = true;
        } else {
            rttvar_ = 0.75 * rttvar_ + 0.25 * std::fabs(srtt_ - rtt_sample);  // beta = 1/4
            srtt_   = 0.875 * srtt_ + 0.125 * rtt_sample;                     // alpha = 1/8
        }
        rto_ = srtt_ + 4.0 * rttvar_;
    }

    /// \brief On a retransmission timeout, back the RTO off exponentially (Karn's rule).
    void on_timeout() { rto_ *= 2.0; }

    double rto() const { return rto_; }
    double srtt() const { return srtt_; }
    double rttvar() const { return rttvar_; }
    bool   initialized() const { return initialized_; }

  private:
    double srtt_ = 0.0, rttvar_ = 0.0, rto_;
    bool   initialized_ = false;
};

}  // namespace datamunge::algorithms
