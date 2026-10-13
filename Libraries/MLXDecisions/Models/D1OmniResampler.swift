// Copyright © 2026 Apple Inc.
//
// mlx-audio's `resample_audio`, which d1-omni's Python `load_audio` runs on files that are not
// at 16 kHz. It ports scipy's `signal.firwin` and `signal.resample_poly`, the Cephes I0 series
// and `chbevl` of scipy's `special.i0`, and numpy's pairwise sum, so that its float32 output is
// the same as Python's. scipy and numpy distribute them under these licences, from their
// LICENSE files:
//
// SciPy:
//
// Copyright (c) 2001-2002 Enthought, Inc. 2003, SciPy Developers.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//
// 1. Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above
//    copyright notice, this list of conditions and the following
//    disclaimer in the documentation and/or other materials provided
//    with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// NumPy:
//
// Copyright (c) 2005-2025, NumPy Developers.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
//        notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above
//        copyright notice, this list of conditions and the following
//        disclaimer in the documentation and/or other materials provided
//        with the distribution.
//
//     * Neither the name of the NumPy Developers nor the names of any
//        contributors may be used to endorse or promote products derived
//        from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import Accelerate
import Foundation

/// `mlx_audio.utils.resample_audio`, which Python's `load_audio` runs on files that are not at
/// 16 kHz: scipy's `resample_poly` with resampy's `kaiser_best` filter and edge padding.
///
/// It builds scipy's filter bit for bit and sums each output in scipy's order with fused
/// multiply-adds, as scipy's arm64 build does, so its float32 output is the same as Python's
/// on Apple silicon.
struct D1OmniResampler: Sendable {
    static let zeroCrossings = 64
    static let rolloff = 0.9475937167399596
    static let kaiserBeta = 14.769656459379492
    /// Largest term of the reduced rate ratio. It keeps the filter under 132,000 taps and
    /// covers the standard rates up to 384 kHz.
    static let maximumRatioTerm = 1024
    /// Outputs per parallel block.
    static let blockSize = 1024

    /// The reduced `targetRate / sourceRate`.
    let up: Int
    let down: Int
    /// Taps per phase.
    let width: Int
    /// `[up * width]`: the taps of each phase, oldest input sample first.
    let taps: [Double]
    /// Leading outputs that `resample_poly` drops to centre the filter.
    let delay: Int

    /// Nil when the rates are equal or a term of their reduced ratio is above
    /// `maximumRatioTerm`.
    init?(sourceRate: Int, targetRate: Int = D1OmniMelFrontend.sampleRate) {
        guard sourceRate > 0, sourceRate != targetRate else { return nil }
        var (divisor, remainder) = (sourceRate, targetRate)
        while remainder != 0 { (divisor, remainder) = (remainder, divisor % remainder) }
        let (up, down) = (targetRate / divisor, sourceRate / divisor)
        let maxRate = Swift.max(up, down)
        guard maxRate <= Self.maximumRatioTerm else { return nil }

        // scipy.signal.firwin(2 * half + 1, rolloff / maxRate, window=("kaiser", beta)) * up.
        // The filter is symmetric, so the second half mirrors the first.
        let half = Self.zeroCrossings * maxRate
        let cutoff = Self.rolloff / Double(maxRate)
        let scale = scipyI0(Self.kaiserBeta)
        let rising = (0 ... half).map { (n: Int) -> Double in
            let m = Double(n - half)
            let x = Double.pi * (cutoff * m)
            let sinc = n == half ? 1 : Foundation.sin(x) / x
            let r = m / Double(half)
            let window = scipyI0(Self.kaiserBeta * (1 - r * r).squareRoot()) / scale
            return cutoff * sinc * window
        }
        var filter = rising + rising.dropLast().reversed()
        let sum = filter.withUnsafeBufferPointer(pairwiseSum)
        filter = filter.map { (tap: Double) -> Double in tap / sum * Double(up) }

        // resample_poly pads the filter with zeros so that output 0 is centred on input 0.
        let prePad = down - half % down
        let width = (prePad + filter.count + up - 1) / up
        var taps = [Double](repeating: 0, count: up * width)
        for (n, tap) in filter.enumerated() {
            let k = n + prePad
            taps[(k % up) * width + (width - 1 - k / up)] = tap
        }
        self.up = up
        self.down = down
        self.width = width
        self.taps = taps
        self.delay = (half + prePad) / down
    }

    /// How many leading input samples the first `count` outputs read.
    func inputCount(forOutput count: Int) -> Int {
        ((count - 1 + delay) * down) / up + 1
    }

    /// `samples` at the target rate, at most `limit` samples.
    func callAsFunction(_ samples: [Float], limit: Int = .max) -> [Float] {
        let count = Swift.min(limit, (samples.count * up + down - 1) / down)
        guard count > 0, let first = samples.first, let last = samples.last else { return [] }
        var output = [Float](repeating: 0, count: count)
        let blocks = (count + Self.blockSize - 1) / Self.blockSize
        samples.withUnsafeBufferPointer { samples in
            taps.withUnsafeBufferPointer { taps in
                output.withUnsafeMutableBufferPointer { output in
                    // Blocks write disjoint outputs, and `concurrentPerform` returns only
                    // when all are done.
                    nonisolated(unsafe) let samples = samples
                    nonisolated(unsafe) let taps = taps.baseAddress!
                    nonisolated(unsafe) let output = output.baseAddress!
                    DispatchQueue.concurrentPerform(iterations: blocks) { (block: Int) -> Void in
                        let start = block * Self.blockSize
                        let end = Swift.min(count, start + Self.blockSize)
                        // The block's input in Double. Edge padding: the first and last
                        // samples repeat beyond the ends.
                        let lower = (start + delay) * down / up - (width - 1)
                        let upper = (end - 1 + delay) * down / up
                        let from = Swift.max(lower, 0)
                        let to = Swift.min(upper, samples.count - 1)
                        var input = [Double](repeating: Double(first), count: from - lower)
                        input += vDSP.floatToDouble(
                            UnsafeBufferPointer(rebasing: samples[from ... to]))
                        input += [Double](repeating: Double(last), count: upper - to)
                        input.withUnsafeBufferPointer { input in
                            // Output n reads `width` inputs from x(n) with the taps at t(n).
                            func x(_ n: Int) -> UnsafePointer<Double> {
                                input.baseAddress! + ((n + delay) * down / up - (width - 1) - lower)
                            }
                            func t(_ n: Int) -> UnsafePointer<Double> {
                                taps + ((n + delay) * down % up) * width
                            }
                            // Eight outputs per pass keep the FPU busy; each lane sums in order.
                            var n = start
                            while n + 8 <= end {
                                let (x0, x1, x2, x3) = (x(n), x(n + 1), x(n + 2), x(n + 3))
                                let (x4, x5, x6, x7) = (x(n + 4), x(n + 5), x(n + 6), x(n + 7))
                                let (t0, t1, t2, t3) = (t(n), t(n + 1), t(n + 2), t(n + 3))
                                let (t4, t5, t6, t7) = (t(n + 4), t(n + 5), t(n + 6), t(n + 7))
                                var sum = SIMD8<Double>()
                                for k in 0 ..< width {
                                    sum.addProduct(
                                        SIMD8(
                                            x0[k], x1[k], x2[k], x3[k], x4[k], x5[k], x6[k], x7[k]),
                                        SIMD8(
                                            t0[k], t1[k], t2[k], t3[k], t4[k], t5[k], t6[k], t7[k]))
                                }
                                for lane in 0 ..< 8 { output[n + lane] = Float(sum[lane]) }
                                n += 8
                            }
                            for n in n ..< end {
                                let (xn, tn) = (x(n), t(n))
                                var sum = 0.0
                                for k in 0 ..< width { sum.addProduct(xn[k], tn[k]) }
                                output[n] = Float(sum)
                            }
                        }
                    }
                }
            }
        }
        return output
    }
}

/// Resamplers by source rate, so that each filter is built once: the 44.1 kHz one takes about
/// 1 ms. It keeps at most `capacity` rates and empties when a new rate would exceed it.
final class D1OmniResamplerCache: @unchecked Sendable {
    static let shared = D1OmniResamplerCache(capacity: 8)

    let capacity: Int
    private let lock = NSLock()
    private var resamplers: [Int: D1OmniResampler] = [:]

    init(capacity: Int) {
        self.capacity = capacity
    }

    /// The number of rates held.
    var count: Int { lock.withLock { resamplers.count } }

    /// `D1OmniResampler(sourceRate:)`, built on first use.
    func resampler(sourceRate: Int) -> D1OmniResampler? {
        if let cached = lock.withLock({ resamplers[sourceRate] }) { return cached }
        guard let resampler = D1OmniResampler(sourceRate: sourceRate) else { return nil }
        lock.withLock {
            if resamplers.count >= capacity { resamplers.removeAll() }
            resamplers[sourceRate] = resampler
        }
        return resampler
    }
}

/// `scipy.special.i0`: Cephes's Chebyshev series for the modified Bessel function I0, with the
/// fused multiply-adds of scipy's arm64 build.
private func scipyI0(_ x: Double) -> Double {
    let x = abs(x)
    if x <= 8 {
        return Foundation.exp(x) * chebyshev(x / 2 - 2, besselI0Coefficients.low)
    }
    return Foundation.exp(x) * chebyshev(32 / x - 2, besselI0Coefficients.high) / x.squareRoot()
}

/// Cephes `chbevl`.
private func chebyshev(_ x: Double, _ coefficients: [Double]) -> Double {
    var (b0, b1, b2) = (coefficients[0], 0.0, 0.0)
    for c in coefficients.dropFirst() {
        (b2, b1) = (b1, b0)
        b0 = (-b2).addingProduct(x, b1) + c
    }
    return 0.5 * (b0 - b2)
}

/// Cephes `i0` coefficients for `exp(-x) I0(x)` on [0, 8] and `exp(-x) sqrt(x) I0(x)` above 8.
private let besselI0Coefficients: (low: [Double], high: [Double]) = (
    [
        -4.41534164647933937950E-18, 3.33079451882223809783E-17, -2.43127984654795469359E-16,
        1.71539128555513303061E-15, -1.16853328779934516808E-14, 7.67618549860493561688E-14,
        -4.85644678311192946090E-13, 2.95505266312963983461E-12, -1.72682629144155570723E-11,
        9.67580903537323691224E-11, -5.18979560163526290666E-10, 2.65982372468238665035E-9,
        -1.30002500998624804212E-8, 6.04699502254191894932E-8, -2.67079385394061173391E-7,
        1.11738753912010371815E-6, -4.41673835845875056359E-6, 1.64484480707288970893E-5,
        -5.75419501008210370398E-5, 1.88502885095841655729E-4, -5.76375574538582365885E-4,
        1.63947561694133579842E-3, -4.32430999505057594430E-3, 1.05464603945949983183E-2,
        -2.37374148058994688156E-2, 4.93052842396707084878E-2, -9.49010970480476444210E-2,
        1.71620901522208775349E-1, -3.04682672343198398683E-1, 6.76795274409476084995E-1,
    ],
    [
        -7.23318048787475395456E-18, -4.83050448594418207126E-18, 4.46562142029675999901E-17,
        3.46122286769746109310E-17, -2.82762398051658348494E-16, -3.42548561967721913462E-16,
        1.77256013305652638360E-15, 3.81168066935262242075E-15, -9.55484669882830764870E-15,
        -4.15056934728722208663E-14, 1.54008621752140982691E-14, 3.85277838274214270114E-13,
        7.18012445138366623367E-13, -1.79417853150680611778E-12, -1.32158118404477131188E-11,
        -3.14991652796324136454E-11, 1.18891471078464383424E-11, 4.94060238822496958910E-10,
        3.39623202570838634515E-9, 2.26666899049817806459E-8, 2.04891858946906374183E-7,
        2.89137052083475648297E-6, 6.88975834691682398426E-5, 3.36911647825569408990E-3,
        8.04490411014108831608E-1,
    ]
)

/// numpy's pairwise sum, which `np.sum` runs on float64 and firwin uses to normalise the taps.
private func pairwiseSum(_ values: UnsafeBufferPointer<Double>) -> Double {
    let n = values.count
    if n < 8 {
        return values.reduce(0, +)
    }
    if n <= 128 {
        var partial = SIMD8(values[0 ..< 8])
        var i = 8
        while i + 8 <= n {
            partial += SIMD8(values[i ..< i + 8])
            i += 8
        }
        var sum =
            ((partial[0] + partial[1]) + (partial[2] + partial[3]))
            + ((partial[4] + partial[5]) + (partial[6] + partial[7]))
        for value in values[i...] { sum += value }
        return sum
    }
    let half = n / 2 - n / 2 % 8
    return pairwiseSum(UnsafeBufferPointer(rebasing: values[..<half]))
        + pairwiseSum(UnsafeBufferPointer(rebasing: values[half...]))
}
