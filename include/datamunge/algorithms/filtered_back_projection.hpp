#pragma once
/// @file filtered_back_projection.hpp
/// @brief Checked unit-grid pixel splatting, finite Ram-Lak filtering and parallel-beam backprojection.
/// Images are row-major square pixel arrays; sinograms are [view][detector].
/// These finite discrete operators are educational approximations, not an exact
/// matrix inverse or a physical scanner/calibration/attenuation model.
#include <cstddef>
#include <vector>
namespace datamunge::algorithms {
/// @brief Create an owned centered disk on a square unit-spaced pixel grid.
/// Pixel(r,c) uses x=c-(N-1)/2,y=r-(N-1)/2. Includes pixels with hypot(x,y)<=radius;
/// others are zero. Radius zero selects the center pixel only for odd N. Uses
/// hypot rather than squared-radius arithmetic, accepting every finite nonnegative
/// radius and finite signed/zero intensity. O(N²) work/output, constant scratch;
/// no input aliases or shared state. This is a numerical phantom, not medical data.
/// @param N Positive square image side length.
/// @param radius Finite nonnegative radius in pixel-spacing units.
/// @param intensity Finite value inside the disk, default1.
/// @return Owned N*N row-major pixels.
/// @throws std::invalid_argument If N<=0, radius is negative/nonfinite, or intensity nonfinite.
/// @throws std::length_error If pixel count cannot fit size_t or vector limits.
/// @throws std::bad_alloc If pixel allocation fails.
/// @par Example
/// @code
/// const auto p=dm::algorithms::disk_phantom(3,0);
/// std::print("{} {}\n",p.size(),p[4]); // 9 1
/// @endcode
/// @see radon_transform(), parallel_projection_angles()
std::vector<double> disk_phantom(int N,double radius,double intensity=1.0);

/// @brief Generate rounded uniform half-turn projection angles in radians.
/// Computes pi*i/count in the original report's arithmetic order, for i=0..count-1.
/// Values are clamped below pi if final rounding reaches it; extremely large
/// counts need not produce distinct doubles. Count1 gives only zero. O(count)
/// work/output, constant scratch, no shared state. Supplies ordinary pi/count
/// angular quadrature; arbitrary caller angle lists do not gain adaptive weights.
/// @param count Positive number of projection views.
/// @return Owned rounded angles in [0,pi), starting at zero.
/// @throws std::invalid_argument If count==0.
/// @throws std::length_error If output length exceeds vector limits.
/// @throws std::bad_alloc If angle allocation fails.
/// @par Example
/// @code
/// const auto a=dm::algorithms::parallel_projection_angles(4);
/// std::print("{} {:.6f}\n",a.size(),a[2]); // 4 1.570796
/// @endcode
/// @par References
/// - A. C. Kak and M. Slaney (1988), Principles of Computerized Tomographic Imaging,
///   chapter3, https://engineering.purdue.edu/~malcolm/pct/CTI_Ch03.pdf
/// @see radon_transform(), back_projection()
std::vector<double> parallel_projection_angles(std::size_t count);

/// @brief Splat square-image pixel values linearly into a parallel-beam sinogram.
/// Retains the original callable signature and valid arithmetic order. Pixel
/// coordinates are centered with unit spacing. For each angle, detector coordinate
/// is s=x*cos(theta)+y*sin(theta)+(num_detectors-1)/2. Two adjacent bins receive
/// weights1-frac(s) and frac(s); out-of-range bins receive nothing. This finite
/// pixel-splat operator is not exact continuous line integration. Clipping can
/// lose projected mass. Empty angles, shape mismatch and nonfinite data reject
/// before output allocation. Floor is cast only inside supported detector bounds.
/// O(MN²+MD) work, O(MD) output and constant scratch, M=angles.size(). Owned
/// output/input unchanged; concurrent reads require unchanged inputs; no shared state.
/// @param img Borrowed finite N*N row-major pixel values, including signed/zero values.
/// @param N Positive image side length.
/// @param angles Nonempty borrowed finite angles in radians; arbitrary order is accepted.
/// @param num_detectors Positive number of centered unit-spaced bins per view.
/// @return Owned rectangular sinogram [angle][detector].
/// @throws std::invalid_argument On nonpositive dimensions, wrong shape, empty/nonfinite angles or pixels.
/// @throws std::length_error On checked pixel/sinogram container or aggregate element limits.
/// @throws std::overflow_error If finite inputs produce a nonfinite accumulation/coordinate.
/// @throws std::bad_alloc If sinogram allocation fails.
/// @par Example
/// @code
/// const auto s=dm::algorithms::radon_transform({2},1,{0},3);
/// std::print("{} {} {}\n",s[0][0],s[0][1],s[0][2]); // 0 2 0
/// @endcode
/// @par References
/// - Kak and Slaney (1988), chapter3.1-3.3, https://engineering.purdue.edu/~malcolm/pct/CTI_Ch03.pdf
///   Uses pixel splatting and zero outside detector support rather than analytic line integrals.
/// @see back_projection(), ram_lak_filter(), filtered_back_projection()
std::vector<std::vector<double>> radon_transform(const std::vector<double>& img,int N,const std::vector<double>& angles,int num_detectors);

/// @brief Filter each finite projection row by the original unit-spacing finite Ram-Lak convolution.
/// Uses h[0]=1/4,h[k]=-1/(pi²k²) for odd k,zero for nonzero even k. For row width D,
/// retains lags-(D-1)..D-1 with zero extension and direct convolution, preserving
/// the original summation order. This is not circular FFT convolution; the finite
/// kernel does not annihilate all finite constant rows. Empty sinogram returns
/// empty; nonempty rows must have equal positive width and finite entries.
/// O(MD²) work, O(MD) output plus O(D) kernel storage; no input mutation or aliases.
/// @param sinogram Borrowed rectangular finite [view][detector] data, possibly empty.
/// @return Owned filtered rows with the same shape.
/// @throws std::invalid_argument On ragged/zero-width rows or nonfinite entries.
/// @throws std::length_error On checked kernel/sinogram container or aggregate element limits.
/// @throws std::overflow_error If filtering finite data produces a nonfinite sum.
/// @throws std::bad_alloc If filtered/kernel allocation fails.
/// @par Example
/// @code
/// const auto f=dm::algorithms::ram_lak_filter({{0,1,0}});
/// std::print("{:.2f}\n",f[0][1]); // 0.25
/// @endcode
/// @par References
/// - Kak and Slaney (1988), chapter3.3.3, https://engineering.purdue.edu/~malcolm/pct/CTI_Ch03.pdf
///   Direct finite convolution with unit detector spacing and the standard ramp samples.
/// @see filtered_back_projection(), back_projection()
std::vector<std::vector<double>> ram_lak_filter(const std::vector<std::vector<double>>& sinogram);

/// @brief Gather projection rows linearly onto a centered image with the original pi/M scale.
/// Uses the same detector weights as radon_transform, so in real arithmetic this
/// is(pi/M) times that finite splat operator's transpose. Floating sums round.
/// No filter is applied here. Fixed pi/M angular weights are appropriate to the
/// ordinary uniform full-half-turn grid; arbitrary finite angle lists are accepted
/// but not reweighted or certified as valid quadrature. Outside detector support
/// contributes zero. Validates all shapes/finite data before allocation/gather.
/// O(MN²+MD) work, O(N²) output and constant scratch, no shared state or input alias.
/// @param sinogram Borrowed finite rectangle with angles.size() rows of num_detectors entries.
/// @param N Positive square output side length.
/// @param angles Nonempty finite view angles in radians, matched in order to rows.
/// @param num_detectors Positive unit-spaced detector count.
/// @return Owned row-major N*N backprojected values, not a general inverse.
/// @throws std::invalid_argument On dimensions, row counts/widths, empty/nonfinite angles or data.
/// @throws std::length_error On checked output/input geometry limits.
/// @throws std::overflow_error If finite inputs produce a nonfinite accumulation/coordinate.
/// @throws std::bad_alloc If output allocation fails.
/// @par Example
/// @code
/// const auto b=dm::algorithms::back_projection({{2}},1,{0},1);
/// std::print("{:.6f}\n",b[0]); // 6.283185
/// @endcode
/// @par References
/// - Kak and Slaney (1988), chapter3.3, https://engineering.purdue.edu/~malcolm/pct/CTI_Ch03.pdf
///   Linear detector interpolation and uniform angular quadrature, here paired to the finite splat.
/// @see radon_transform(), ram_lak_filter(), filtered_back_projection()
std::vector<double> back_projection(const std::vector<std::vector<double>>& sinogram,int N,const std::vector<double>& angles,int num_detectors);

/// @brief Reconstruct using the original finite Ram-Lak filter followed by shared backprojection.
/// Retains the original callable signature and valid arithmetic/summation order.
/// All dimensions/shapes/finite values are validated before filtering/allocation.
/// Delegates to the same kernels exposed by ram_lak_filter and back_projection;
/// in real arithmetic output is(pi/M)A-transpose H sinogram. This finite operator
/// is not the inverse of its pixel-splat A. Accuracy depends on pixel/detector/view
/// sampling, support and object; correlation alone does not certify amplitude or
/// pointwise accuracy. Unit spacing and fixed pi/M weights, as back_projection.
/// O(MD²+MN²) work; output O(N²), temporary filtered O(MD) and kernel O(D).
/// No history quota, input mutation, retained references or shared mutable state.
/// @param sinogram Borrowed finite rectangle with one row per angle and D columns.
/// @param N Positive output side length.
/// @param angles Nonempty finite angles in radians; ordinary quadrature assumes uniform[0,pi).
/// @param num_detectors Positive detector count D matching every row.
/// @return Owned N*N row-major finite reconstruction if all arithmetic succeeds.
/// @throws std::invalid_argument On dimensions, shapes, empty/nonfinite angles or data.
/// @throws std::length_error On checked pixel/kernel/sinogram geometry limits.
/// @throws std::overflow_error If filtering or gathering finite data produces nonfinite arithmetic.
/// @throws std::bad_alloc If output/intermediate allocation fails.
/// @par Example
/// @code
/// const auto r=dm::algorithms::filtered_back_projection({{4}},1,{0},1);
/// std::print("{:.6f}\n",r[0]); // 3.141593
/// @endcode
/// @par References
/// - Kak and Slaney (1988), Principles of Computerized Tomographic Imaging, chapter3.3,
///   https://engineering.purdue.edu/~malcolm/pct/CTI_Ch03.pdf ; finite unit-grid implementation.
/// @see radon_transform(), ram_lak_filter(), back_projection(), disk_phantom()
std::vector<double> filtered_back_projection(const std::vector<std::vector<double>>& sinogram,int N,const std::vector<double>& angles,int num_detectors);
} // namespace datamunge::algorithms
