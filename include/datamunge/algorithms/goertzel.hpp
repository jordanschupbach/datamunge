#pragma once
/// @file goertzel.hpp
/// @brief Checked ordered Goertzel power, actual recurrence records and synthetic tone experiments.
#include <array>
#include <cstddef>
#include <vector>

namespace datamunge::algorithms {
/// @brief Select the historical offset-truncated bin or evaluate the supplied frequency directly.
/// legacy_bin uses trunc(0.5+double(N)*f/fs), with an int representability check.
/// For nonnegative ordinary requests this chooses the nearest bin with upward
/// half ties; negative requests are asymmetric. unrounded uses omega=2*pi*f/fs
/// without integer quantization. Neither mode guarantees exact floating accuracy.
/// @par Example
/// @code
/// const auto m=dm::algorithms::GoertzelFrequencyMode::unrounded;
/// std::cout << (m==dm::algorithms::GoertzelFrequencyMode::unrounded); // 1
/// @endcode
/// @see goertzel_run(), goertzel_power_at_frequency()
enum class GoertzelFrequencyMode {
    legacy_bin, ///< Historical truncation of the offset bin expression, including its negative-frequency bias.
    unrounded ///< Supplied frequency directly; no integer-bin selection or automatic angle reduction.
};
/// @brief Owned actual selection, terminal states and unnormalized raw squared-magnitude estimate.
/// For bin_indexed calls bin holds the supplied size_t index, never reduced mod N;
/// frequency_mode/selected_bin are then unused placeholders. For legacy frequency
/// calls selected_bin is the actual signed int; for unrounded calls it is unused
/// zero. Empty input has has_coefficient=false, zero omega/coefficient/states/power
/// and zero updates; no frequency grid is evaluated. The supplied bin is still
/// retained for empty bin-indexed calls. All unused fields are placeholders.
/// power is not normalized by sample count, a PSD, calibrated watts or mean-square
/// signal power. Cancellation can produce a FINITE NEGATIVE value, retained
/// without clamping. Very long blocks, large unreduced angles and cancellation
/// can lose accuracy even when all states are finite. Copies own every scalar.
/// @par Example
/// @code
/// const auto r=dm::algorithms::goertzel_bin_run({1,2,3,4},0);
/// std::cout << r.power << " " << r.sample_updates; // 100 4
/// @endcode
/// @see goertzel_run(), goertzel_bin_run()
struct GoertzelRun {
    std::size_t sample_count{0}; ///< Number of supplied real samples.
    bool bin_indexed{false}; ///< True for the size_t-bin interface.
    GoertzelFrequencyMode frequency_mode{GoertzelFrequencyMode::legacy_bin}; ///< Frequency selector, unused for bin-indexed calls.
    int selected_bin{0}; ///< Actual offset-truncated signed bin, unused zero for empty/direct/bin-indexed calls.
    std::size_t bin{0}; ///< Supplied unsigned bin, unused zero for frequency calls.
    bool has_coefficient{false}; ///< Whether nonempty input caused omega/coefficient evaluation.
    double omega{0}; ///< Actual unreduced angular frequency in radians per sample.
    double coefficient{0}; ///< Actual rounded 2*cos(omega), zero placeholder for empty input.
    double s1{0}; ///< Actual final newest recurrence state.
    double s2{0}; ///< Actual final older recurrence state.
    double power{0}; ///< Actual ordered raw power expression; finite negatives are not clamped.
    std::size_t sample_updates{0}; ///< Actual recurrence updates, excluding validation, selection, trig, allocation and timing.
};
/// @brief Emission point in an online actual Goertzel execution.
/// Initial and stop records count even for empty input; a completed N-sample
/// history therefore contains N+2 complete records, not just N sample rows.
/// @par Example
/// @code
/// const auto k=dm::algorithms::GoertzelEventKind::sample;
/// std::cout << (k==dm::algorithms::GoertzelEventKind::sample); // 1
/// @endcode
/// @see GoertzelEvent, goertzel_trace()
enum class GoertzelEventKind {
    initial, ///< Before recurrence: zero states and value equal to actual coefficient, or empty placeholder zero.
    sample, ///< Actual input sample, before states and evaluated new state.
    stop ///< Final states and raw power; no recurrence update or coefficient reevaluation.
};
/// @brief One owned fixed-size actual recurrence record, without numerical replay.
/// For sample records index is zero based, input is x[index], before_s1/s2 are
/// the actual prior states and value is the evaluated s0. State assignments then
/// make s1=value and s2=before_s1. For stop, index=N, before fields are final states
/// and value is raw power. For initial index/before/input are zero and value is
/// the actual coefficient, or zero for empty input. Metadata input is unused zero.
/// @par Example
/// @code
/// const dm::algorithms::GoertzelEvent e{};
/// std::cout << e.index << " " << e.value; // 0 0
/// @endcode
/// @see GoertzelTrace, GoertzelEventKind
struct GoertzelEvent {
    GoertzelEventKind kind{GoertzelEventKind::initial}; ///< Emission point defining payload semantics.
    std::size_t index{0}; ///< Sample index, zero for initial, N for stop.
    double input{0}; ///< Actual input sample, unused zero for metadata.
    double before_s1{0}; ///< Actual newest prior state, or terminal newest state for stop.
    double before_s2{0}; ///< Actual older prior state, or terminal older state for stop.
    double value{0}; ///< Actual coefficient, evaluated s0 or terminal raw power according to kind.
};
/// @brief Owned ordinary result plus all fixed-size records captured online.
/// Complete initial/sample/stop records are charged, not numeric entries, bytes
/// or peak RAM. Input, root, ordinary result/selection metadata, constant working
/// states and allocator overhead are excluded. No input aliases, replay,
/// truncation or partial return survive. Copies own their history vectors.
/// @par Example
/// @code
/// const auto t=dm::algorithms::goertzel_bin_trace({1,2,3,4},0,6);
/// std::cout << t.run.power << " " << t.events.size(); // 100 6
/// @endcode
/// @see goertzel_trace(), goertzel_bin_trace()
struct GoertzelTrace {
    GoertzelRun run{}; ///< Ordinary result from this same execution.
    std::vector<GoertzelEvent> events; ///< Complete actual initial/sample/stop records in order.
};
/// @brief Return the original ordered raw Goertzel power after historical frequency-to-bin selection.
/// Nonempty input selects k=trunc(0.5+double(N)*target_freq/sample_rate), then uses
/// omega=2*pi*k/N. This is NOT the response at the exact requested frequency.
/// Negative-frequency selection is biased by offset-then-truncation; the old rule
/// is retained. Bin conversion is checked after truncation, including fractional
/// values just outside int bounds whose truncated integer still fits. No Nyquist
/// range restriction or automatic modulo reduction is imposed.
/// All samples and finite positive rate/finite target are validated BEFORE empty
/// input handling. Empty returns zero without selecting a bin/coefficient.
/// The shared recurrence keeps the original x+c*s1-s2 and terminal
/// s1*s1+s2*s2-c*s1*s2 expressions. Evaluated nonfinite selection/state/output is
/// rejected; finite underflow/cancellation and NEGATIVE raw power remain possible.
/// Input is borrowed without mutation or retained aliases. O(N) validation plus
/// O(N) recurrence, O(1) auxiliary state; no shared mutable state.
/// @param x Finite real samples; empty is accepted after complete parameter checks.
/// @param target_freq Finite signed frequency in Hz; no Nyquist clipping.
/// @param sample_rate Finite strictly positive sampling rate in Hz.
/// @return Finite unnormalized squared-magnitude estimate, possibly negative from cancellation.
/// @throws std::invalid_argument Any sample/target/rate is nonfinite or rate is nonpositive.
/// @throws std::overflow_error Evaluated selection/state/raw power is nonfinite, or selected integer cannot fit int.
/// @par Example
/// @code
/// std::cout << dm::algorithms::goertzel_power({1,2,3,4},0,4); // 100
/// @endcode
/// @par References
/// - G. Goertzel (1958), "An Algorithm for the Evaluation of Finite Trigonometric
///   Series", American Mathematical Monthly65(1),34–35, doi:10.2307/2310304.
/// - C. J. Chen (1996), Modified Goertzel Algorithm in DTMF Detection Using the
///   TMS320C80 DSP, Texas Instruments SPRA066, equations1–4.
///   https://www.ti.com/lit/an/spra066/spra066.pdf. This is the retained real
///   double-valued block variant, not the report's fixed-point DSP implementation.
/// @see goertzel_power_at_frequency(), goertzel_run(), goertzel_power_bin()
double goertzel_power(const std::vector<double>& x,double target_freq,double sample_rate);
/// @brief Return raw Goertzel power at the supplied unsigned DFT bin using its original unreduced angle.
/// All size_t indices are accepted, including k>=N. The actual angle is computed
/// as 2*pi*double(k)/double(N), WITHOUT k modulo N. Large indices can lose integer
/// bits and angle accuracy; ideal DFT alias periodicity is not a bitwise numerical
/// guarantee. Finite sample validation precedes empty return zero. Recurrence,
/// raw negative-power limitations, O(N) time/O(1) auxiliary space, borrowing and
/// independent-call safety are those of goertzel_power().
/// @param x Finite real samples, without mutation; empty returns zero after validation.
/// @param k Unsigned bin index, not narrowed or reduced modulo the sample count.
/// @return Finite unnormalized raw power, possibly negative from cancellation.
/// @throws std::invalid_argument Any input sample is nonfinite.
/// @throws std::overflow_error Evaluated angle/coefficient/state/raw power is nonfinite.
/// @par Example
/// @code
/// std::cout << dm::algorithms::goertzel_power_bin({1,2,3,4},0); // 100
/// @endcode
/// @par References
/// - Goertzel (1958), doi:10.2307/2310304; Chen (1996), TI SPRA066,
///   https://www.ti.com/lit/an/spra066/spra066.pdf, retained real recurrence.
/// @see goertzel_bin_run(), goertzel_bin_trace()
double goertzel_power_bin(const std::vector<double>& x,std::size_t k);
/// @brief Return raw power at the supplied frequency without integer-bin quantization.
/// Uses omega=2*pi*target_freq/sample_rate and the same ordered real recurrence.
/// This means direct frequency selection, NOT exact floating arithmetic. No
/// modulo-angle reduction is added. Full finite sample/parameter validation
/// precedes empty zero. Raw negative/cancellation/range limitations, constant
/// auxiliary state, O(N) time, borrowing and thread safety match goertzel_power().
/// @param x Finite real samples; empty accepted after complete parameter validation.
/// @param target_freq Finite signed target frequency in Hz.
/// @param sample_rate Finite positive sampling rate in Hz.
/// @return Finite unnormalized raw frequency-response power, possibly negative.
/// @throws std::invalid_argument Any sample/target/rate is nonfinite or rate nonpositive.
/// @throws std::overflow_error Actual angle/coefficient/state/raw power is nonfinite.
/// @par Example
/// @code
/// std::cout << dm::algorithms::goertzel_power_at_frequency({1,2,3,4},0,4); // 100
/// @endcode
/// @par References
/// - Goertzel (1958), doi:10.2307/2310304; real-filter formulation as in
///   Chen (1996), https://www.ti.com/lit/an/spra066/spra066.pdf.
/// @see goertzel_run(), GoertzelFrequencyMode
double goertzel_power_at_frequency(const std::vector<double>& x,double target_freq,double sample_rate);
/// @brief Execute the shared frequency-selected kernel and own its actual metadata/states.
/// Selection and numeric contracts are those of goertzel_power() for legacy_bin,
/// or goertzel_power_at_frequency() for unrounded. All parameters/samples are
/// validated before empty handling; an unknown mode is rejected. No history or
/// optional sample cap. sample_updates counts actual recurrence updates only,
/// excluding validation, trig, selection, allocation, timing and hardware FLOPs.
/// O(N) time, O(1) auxiliary state and owned scalar result; input is unchanged.
/// @param x Finite borrowed samples, with no retained aliases.
/// @param target_freq Finite signed frequency in Hz.
/// @param sample_rate Finite positive sampling rate in Hz.
/// @param mode Selector, default legacy_bin; unrounded avoids integer quantization.
/// @return Owned selection, final states, raw power and actual update count.
/// @throws std::invalid_argument Bad sample/target/rate or unknown mode.
/// @throws std::overflow_error Actual selection/state/output is nonfinite or legacy bin cannot fit int.
/// @par Example
/// @code
/// const auto r=dm::algorithms::goertzel_run({1,2,3,4},-1,4);
/// std::cout << r.selected_bin << " " << r.power; // 0 100
/// @endcode
/// @par References
/// - Goertzel (1958), doi:10.2307/2310304; Chen (1996),
///   https://www.ti.com/lit/an/spra066/spra066.pdf; retained real block variant.
/// @see GoertzelRun, goertzel_trace()
GoertzelRun goertzel_run(const std::vector<double>& x,double target_freq,double sample_rate,GoertzelFrequencyMode mode=GoertzelFrequencyMode::legacy_bin);
/// @brief Execute the shared unsigned-bin kernel with owned actual metadata/states.
/// Uses the unreduced-bin semantics and numerical limitations of goertzel_power_bin().
/// No history or optional input-size cap. O(N) validation/recurrence, O(1) working
/// state, no caller mutation/retained aliases/shared mutable state.
/// @param x Finite borrowed real samples, empty accepted after validation.
/// @param k Unsigned requested bin; stored unchanged even for empty input.
/// @return Owned actual bin, angle/coefficient, final states, raw power and update count.
/// @throws std::invalid_argument A sample is nonfinite.
/// @throws std::overflow_error Actual angle/coefficient/state/raw power is nonfinite.
/// @par Example
/// @code
/// const auto r=dm::algorithms::goertzel_bin_run({},7);
/// std::cout << r.bin << " " << r.has_coefficient; // 7 0
/// @endcode
/// @par References
/// - Goertzel (1958), doi:10.2307/2310304; Chen (1996),
///   https://www.ti.com/lit/an/spra066/spra066.pdf.
/// @see goertzel_power_bin(), GoertzelRun
GoertzelRun goertzel_bin_run(const std::vector<double>& x,std::size_t k);
/// @brief Capture a complete actual frequency-selected recurrence online, without replay.
/// Before EACH complete append, max_records and vector capacity are checked.
/// Default one million records, zero disables only the optional quota. Initial
/// and stop metadata count: N4 needs6 (reject5/admit6), empty needs2 (reject1/admit2).
/// Full sample/parameter validation precedes selection and history admission.
/// No partial history, truncation, input mutation or ordinary result on failure.
/// O(N+H) time and O(H) extra complete records, excluding input/root/ordinary
/// metadata and result/constant working states/allocator overhead. Not numeric
/// entries, bytes or peak RAM. Records are from this same execution.
/// @param x Finite borrowed samples, no retained aliases.
/// @param target_freq Finite signed frequency in Hz.
/// @param sample_rate Finite positive sampling rate in Hz.
/// @param mode Selector, default legacy_bin; unrounded avoids integer quantization.
/// @param max_records Optional complete-record quota, default1000000; zero disables it.
/// @return Owned run and every actual initial/sample/stop record.
/// @throws std::invalid_argument Bad sample/target/rate or unknown mode, before quota handling.
/// @throws std::overflow_error Evaluated selection/state/output is nonfinite or selected legacy integer is unrepresentable.
/// @throws std::length_error The next complete record exceeds quota/container capacity.
/// @throws std::bad_alloc Retained history allocation fails.
/// @par Example
/// @code
/// const auto t=dm::algorithms::goertzel_trace({},0,4,dm::algorithms::GoertzelFrequencyMode::legacy_bin,2);
/// std::cout << t.events.size() << " " << t.run.power; // 2 0
/// @endcode
/// @par References
/// - Goertzel (1958), doi:10.2307/2310304; Chen (1996),
///   https://www.ti.com/lit/an/spra066/spra066.pdf; actual states, not replay.
/// @see GoertzelTrace, goertzel_run()
GoertzelTrace goertzel_trace(const std::vector<double>& x,double target_freq,double sample_rate,GoertzelFrequencyMode mode=GoertzelFrequencyMode::legacy_bin,std::size_t max_records=1000000);
/// @brief Capture every actual unsigned-bin recurrence state with the same online quota.
/// Raw signal processing semantics are those of goertzel_bin_run().
/// History, validation-before-quota, metadata count, failure, ownership and memory
/// exclusions are those of goertzel_trace(). Empty retains two records, with no
/// coefficient evaluation. O(N+H) time/O(H) extra fixed-size history; no replay.
/// @param x Finite borrowed real samples, with no retained aliases or mutation.
/// @param k Unsigned unreduced bin index.
/// @param max_records Optional complete-record quota, default1000000; zero disables it.
/// @return Owned run and complete actual initial/sample/stop records.
/// @throws std::invalid_argument Any sample is nonfinite, before quota handling.
/// @throws std::overflow_error An actual angle/coefficient/state/raw power is nonfinite.
/// @throws std::length_error The next complete record exceeds quota/container capacity.
/// @throws std::bad_alloc History allocation fails.
/// @par Example
/// @code
/// const auto t=dm::algorithms::goertzel_bin_trace({1,2},0,4);
/// std::cout << t.run.power << " " << t.events.size(); // 9 4
/// @endcode
/// @par References
/// - Goertzel (1958), doi:10.2307/2310304; Chen (1996), TI SPRA066,
///   https://www.ti.com/lit/an/spra066/spra066.pdf.
/// @see goertzel_bin_run(), goertzel_trace()
GoertzelTrace goertzel_bin_trace(const std::vector<double>& x,std::size_t k,std::size_t max_records=1000000);
/// @brief One finite sinusoid descriptor in Hz, sample-amplitude units and radians.
/// Descriptors do not validate construction; sinusoid_signal() checks all fields.
/// Negative amplitudes/frequencies and arbitrary finite phases are allowed. Copies
/// own scalars and share no mutable state.
/// @par Example
/// @code
/// const dm::algorithms::Sinusoid s{770,1,0};
/// std::cout << s.frequency_hz << " " << s.amplitude; // 770 1
/// @endcode
/// @see sinusoid_signal(), dtmf_signal()
struct Sinusoid {
    double frequency_hz{0}; ///< Finite signed frequency in Hz, default0.
    double amplitude{1}; ///< Finite signed amplitude in sample units, default1.
    double phase_radians{0}; ///< Finite phase in radians, default0.
};
/// @brief Generate an owned sum of sinusoids in descriptor order with whole output admission.
/// Sample n sums amplitude*sin(2*pi*frequency_hz*double(n)/sample_rate+phase).
/// All descriptors and the rate are checked before output quota/capacity/allocation,
/// including empty output. Optional max_samples counts the WHOLE output samples,
/// default one million; zero disables it. Computed phases and accumulated sample
/// values are checked before retention. Empty descriptors produce zeros. Frequencies
/// are not Nyquist-clipped or reduced. O(n*terms.size()) time plus O(n) output
/// initialization, O(n) owned output. Inputs/root/allocator/unused capacity are not
/// quota units; it is not bytes or peak RAM. No input mutation/retained aliases.
/// @param n Number of generated samples, zero accepted.
/// @param sample_rate Finite positive sampling rate in Hz.
/// @param terms Finite descriptors evaluated in supplied order, without mutation.
/// @param max_samples Optional whole output-sample quota, default1000000; zero disables it.
/// @return Independent finite sample vector of length n.
/// @throws std::invalid_argument Nonfinite descriptor/rate, or nonpositive rate, before quota checks.
/// @throws std::overflow_error Computed phase or accumulated sample is nonfinite.
/// @throws std::length_error Whole sample count exceeds quota/container capacity.
/// @throws std::bad_alloc Output allocation fails after admission.
/// @par Example
/// @code
/// const auto x=dm::algorithms::sinusoid_signal(4,8,{{0,2,0}},4);
/// std::cout << x.size() << " " << x[0]; // 4 0
/// @endcode
/// @see Sinusoid, dtmf_signal()
std::vector<double> sinusoid_signal(std::size_t n,double sample_rate,const std::vector<Sinusoid>& terms,std::size_t max_samples=1000000);
/// @brief Generate a unit-amplitude zero-phase DTMF row-plus-column illustration.
/// Uses the Q.23 frequency grid for keys123A/456B/789C/*0#D. It is a synthetic
/// finite block, not a telephone transmitter conformance claim. The key and rate
/// are checked before whole output admission, including zero samples. Ordered
/// two-sinusoid generation and sample/range/ownership costs are those of
/// sinusoid_signal(). O(n) time and output, no shared mutable state.
/// @param key One of 0123456789*#ABCD, with uppercase letters only.
/// @param n Number of samples, zero accepted; no prescribed standard block length.
/// @param sample_rate Finite positive sampling rate in Hz, default8000.
/// @param max_samples Optional whole output-sample quota, default1000000; zero disables it.
/// @return Owned finite row-plus-column sample vector, preserving descriptor order.
/// @throws std::invalid_argument Unsupported key or invalid sampling rate.
/// @throws std::overflow_error Computed phase or accumulated sample is nonfinite.
/// @throws std::length_error Whole sample count exceeds quota/container capacity.
/// @throws std::bad_alloc Temporary descriptors or output allocation fails.
/// @par Example
/// @code
/// const auto x=dm::algorithms::dtmf_signal('5',410);
/// std::cout << x.size() << " " << x[0]; // 410 0
/// @endcode
/// @par References
/// - ITU-T Recommendation Q.23 (11/1988), Section6 and Figure1/Q.23.
///   https://www.itu.int/rec/T-REC-Q.23-198811-I/en
/// @see dtmf_peaks(), sinusoid_signal()
std::vector<double> dtmf_signal(char key,std::size_t n,double sample_rate=8000,std::size_t max_samples=1000000);
/// @brief Owned eight-frequency results and first-maximum keypad candidate, not a validated detection.
/// In results from dtmf_peaks(), the first four targets are rows697/770/852/941Hz,
/// then columns1209/1336/1477/1633Hz. Default construction leaves the target array
/// zero-filled; dtmf_peaks() fills the grid and all responses.
/// row/column index their own four-element groups. Equal powers retain the first
/// index. Empty input and silence choose candidate '1' by ties. No energy, duration,
/// frequency tolerance, level ratio, speech or noise validation is performed.
/// Finite negative raw powers remain visible. All fixed-size results own scalars;
/// no input aliases or mutable shared state survive.
/// @par Example
/// @code
/// const auto p=dm::algorithms::dtmf_peaks(dm::algorithms::dtmf_signal('5',410));
/// std::cout << p.candidate << " " << p.sample_updates; // 5 3280
/// @endcode
/// @par References
/// - ITU-T Q.23 (11/1988), frequency grid; Q.24 (11/1988), reception parameters.
///   https://www.itu.int/rec/T-REC-Q.24-198811-I/en
/// @see dtmf_peaks()
struct DtmfPeaks {
    std::array<double,8> target_hz{}; ///< Requested row frequencies followed by requested column frequencies, in Hz.
    std::array<GoertzelRun,8> responses{}; ///< Actual selected frequency/bin metadata, states and raw powers.
    std::size_t row{0}; ///< First maximum in the four row powers.
    std::size_t column{0}; ///< First maximum in the four column powers, indexed zero through three.
    char candidate{'1'}; ///< Key at the peak intersection, not a detection or receiver certificate.
    std::size_t sample_updates{0}; ///< Checked sum of actual updates from the eight executions, excluding validation/trig/time.
};
/// @brief Compute eight native Goertzel responses and return the first row/column power maxima.
/// The fixed Q.23 grid and key mapping are shared with dtmf_signal(). Each response
/// uses goertzel_run() in the requested mode. No receiver validation is added;
/// empty/silent input gives candidate '1'. Complete finite sample/rate/mode checks
/// precede evaluation. O(8N) work, including repeated validation, O(1) auxiliary
/// storage and fixed-size owned output; input is neither modified nor referenced
/// after return. No history is retained. Aggregate update counts are checked.
/// @param x Finite real samples; empty accepted after rate/mode validation.
/// @param sample_rate Finite positive sampling rate in Hz, default8000.
/// @param mode Frequency selector, default legacy_bin; unrounded targets actual grid frequencies.
/// @return Eight owned raw responses and a first-maximum keypad candidate.
/// @throws std::invalid_argument Any sample/rate/mode is invalid.
/// @throws std::overflow_error A selected bin/state/output is unrepresentable/nonfinite.
/// @throws std::length_error Aggregate sample-update count overflows size_t.
/// @par Example
/// @code
/// const auto p=dm::algorithms::dtmf_peaks(std::vector<double>(4,0));
/// std::cout << p.candidate << " " << p.sample_updates; // 1 32
/// @endcode
/// @par References
/// - ITU-T Q.23/Q.24 (11/1988); Chen (1996), TI SPRA066, which also explicitly
///   assumes prior tone/timing validation rather than providing a complete receiver.
///   https://www.ti.com/lit/an/spra066/spra066.pdf
/// @see DtmfPeaks, dtmf_signal(), goertzel_run()
DtmfPeaks dtmf_peaks(const std::vector<double>& x,double sample_rate=8000,GoertzelFrequencyMode mode=GoertzelFrequencyMode::legacy_bin);
} // namespace datamunge::algorithms
