// Copyright © 2026 Apple Inc.

import AVFoundation
import CryptoKit
import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXDecisions
@testable import MLXVLM

// Unit tests run on the CPU backend (see LFM2BidirectionalTests). The parity suite at the end
// needs the real checkpoint and the fixtures of tools/d1_fixtures.py.

/// `audio_config` of the tiny d1_omni case in mlx-vlm's `extraction_cases.json`.
private let tinyAudioConfigJSON = """
    {
      "feat_in": 16, "n_layers": 1, "d_model": 16, "subsampling_conv_channels": 4,
      "ff_expansion_factor": 2, "n_heads": 2, "conv_kernel_size": 3, "residual_width": 8
    }
    """

private func tinyAudioConfig() throws -> D1OmniAudioConfiguration {
    try JSONDecoder().decode(D1OmniAudioConfiguration.self, from: Data(tinyAudioConfigJSON.utf8))
}

/// A 32-bit integer hash of `0 ..< count`, the same as `hashed` in
/// `tools/d1_fixtures.py`, so Python can build the same weights and clips.
private func hashed(count: Int, seed: Int) -> [UInt32] {
    (0 ..< count).map { k in
        var h = UInt32(k) &* 0x9E37_79B1 &+ UInt32(seed) &* 0x85EB_CA77
        h ^= h >> 16
        h = h &* 0x7FEB_352D
        h ^= h >> 15
        h = h &* 0x846C_A68B
        h ^= h >> 16
        return h
    }
}

/// int16 noise in [-8192, 8191] (`pcm` in the fixture script).
private func hashedSamples(_ count: Int) -> [Int16] {
    hashed(count: count, seed: count).map { Int16(bitPattern: UInt16($0 >> 16)) >> 2 }
}

private func hashedPCM(_ count: Int) -> MLXArray {
    MLXArray(hashedSamples(count))
}

private func floats(_ pcm: [Int16]) -> [Float] {
    pcm.map { (sample: Int16) -> Float in Float(sample) / 32768 }
}

/// A triangle wave whose zero crossings fall on 16 kHz output samples from 22.05, 44.1 and
/// 48 kHz. The exact output there is zero, so the computed one is rounding error and changes
/// with any change to the taps or the order of the sum.
private func triangle(_ count: Int) -> [Int16] {
    (0 ..< count).map { (n: Int) -> Int16 in
        let s = n % 1764
        return Int16(16 * (s < 441 ? s : s < 1323 ? 882 - s : s - 1764))
    }
}

/// The first 16 hex digits of the SHA-256 of the float32 samples.
private func digest(_ samples: [Float]) -> String {
    samples.withUnsafeBytes { (bytes: UnsafeRawBufferPointer) -> String in
        SHA256.hash(data: bytes).prefix(8).map { (byte: UInt8) -> String in
            String(format: "%02x", byte)
        }.joined()
    }
}

/// A 16-bit PCM WAV file of interleaved samples.
private func wavData(_ samples: [Int16], channels: Int = 1, sampleRate: Int = 16000) -> Data {
    func le<T: FixedWidthInteger>(_ value: T) -> Data {
        withUnsafeBytes(of: value.littleEndian) { Data($0) }
    }
    let payload = samples.withUnsafeBytes { Data($0) }
    var data = Data("RIFF".utf8)
    data.append(le(UInt32(36 + payload.count)))
    data.append(Data("WAVEfmt ".utf8))
    data.append(le(UInt32(16)))
    data.append(le(UInt16(1)))
    data.append(le(UInt16(channels)))
    data.append(le(UInt32(sampleRate)))
    data.append(le(UInt32(sampleRate * 2 * channels)))
    data.append(le(UInt16(2 * channels)))
    data.append(le(UInt16(16)))
    data.append(Data("data".utf8))
    data.append(le(UInt32(payload.count)))
    data.append(payload)
    return data
}

/// Runs `body` with a WAV file of `samples`, then removes it.
private func withWAV<R>(
    _ samples: [Int16], channels: Int = 1, sampleRate: Int = 16000,
    _ body: (URL) async throws -> R
) async throws -> R {
    try await withDirectory { directory in
        let file = directory.appending(component: "clip.wav")
        try wavData(samples, channels: channels, sampleRate: sampleRate).write(to: file)
        return try await body(file)
    }
}

/// Runs `body` in a new temporary directory, then removes it.
private func withDirectory<R>(_ body: (URL) async throws -> R) async throws -> R {
    let directory = FileManager.default.temporaryDirectory
        .appending(component: "d1-omni-audio-tests-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }
    return try await body(directory)
}

/// A 10-byte ID3v2 header for a 20-byte tag, then 20 bytes of zero padding, as some taggers write.
private let id3TagAndPadding = Data(
    Array("ID3".utf8) + [4, 0, 0, 0, 0, 0, 20] + [UInt8](repeating: 0, count: 40))

/// Writes `samples` with AVAudioFile as `settings` say; the extension picks the file type.
private func writeAudio(
    _ samples: [Float], to file: URL, sampleRate: Int = 16_000, settings: [String: Any]
) throws {
    let format = try #require(
        AVAudioFormat(
            commonFormat: .pcmFormatFloat32, sampleRate: Double(sampleRate), channels: 1,
            interleaved: false))
    let buffer = try #require(
        AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(samples.count)))
    buffer.frameLength = AVAudioFrameCount(samples.count)
    let channel = try #require(buffer.floatChannelData?[0])
    channel.update(from: samples, count: samples.count)
    var settings = settings
    settings[AVSampleRateKey] = sampleRate
    settings[AVNumberOfChannelsKey] = 1
    let writer = try AVAudioFile(
        forWriting: file, settings: settings, commonFormat: .pcmFormatFloat32, interleaved: false)
    try writer.write(from: buffer)
}

/// Linear PCM settings for `writeAudio`.
private func pcmSettings(bits: Int = 16, float: Bool = false, bigEndian: Bool) -> [String: Any] {
    [
        AVFormatIDKey: kAudioFormatLinearPCM, AVLinearPCMBitDepthKey: bits,
        AVLinearPCMIsFloatKey: float, AVLinearPCMIsBigEndianKey: bigEndian,
        AVLinearPCMIsNonInterleaved: false,
    ]
}

/// The decode before files were cut while reading: the whole file at 16 kHz through
/// `asMLXArray`, then MLX's mean over the channels.
private func wholeFileWaveform(_ file: URL, channels: Int) async throws -> [Float] {
    var processing = UserInput.AudioProcessing()
    processing.sampleRate = 16_000
    processing.channels = channels
    var samples = try await UserInput.Audio.url(file).asMLXArray(processing: processing)
    if channels > 1 {
        samples = samples.reshaped(-1, channels).mean(axis: 1)
    }
    return try d1OmniWaveform(samples).asArray(Float.self)
}

private func fileWaveform(_ file: URL) async throws -> [Float] {
    try await d1OmniWaveform(.url(file)).asArray(Float.self)
}

/// The tiny audio model with hashed weights in [-0.5, 0.5); running variances are in [0.5, 1].
private func tinyAudio(outputDimensions: Int = 64) throws -> D1OmniAudio {
    let model = D1OmniAudio(try tinyAudioConfig(), outputDimensions: outputDimensions)
    var weights = [String: MLXArray]()
    let parameters = model.parameters().flattened().sorted { $0.0 < $1.0 }
    for (index, (key, value)) in parameters.enumerated() {
        let uniform = hashed(count: value.size, seed: index).map { Float($0 >> 8) / 0x1p24 - 0.5 }
        let w = MLXArray(uniform, value.shape)
        weights[key] = key.hasSuffix("running_var") ? abs(w) + 0.5 : w
    }
    try model.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
    return model
}

private func noise(_ count: Int, seed: UInt64 = 7) -> MLXArray {
    MLXRandom.normal([count], key: MLXRandom.key(seed)) * 0.1
}

private func maxAbsDifference(_ a: MLXArray, _ b: MLXArray) -> Float {
    abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
}

@Suite("d1-omni audio")
struct D1OmniAudioTests {

    // MARK: Configuration

    @Test("audio_config decodes snake_case keys and falls back to the Python defaults")
    func configuration() throws {
        let tiny = try tinyAudioConfig()
        #expect(
            tiny
                == D1OmniAudioConfiguration(
                    featIn: 16, layerCount: 1, dModel: 16, subsamplingConvChannels: 4,
                    ffExpansionFactor: 2, headCount: 2, convKernelSize: 3, residualWidth: 8))
        let defaults = try JSONDecoder().decode(
            D1OmniAudioConfiguration.self, from: Data("{}".utf8))
        #expect(defaults == D1OmniAudioConfiguration())
        #expect(defaults.featIn == 128 && defaults.layerCount == 17 && defaults.dModel == 512)
        #expect(defaults.subsamplingConvChannels == 256 && defaults.ffExpansionFactor == 4)
        #expect(defaults.headCount == 8 && defaults.convKernelSize == 9)
        #expect(defaults.residualWidth == 512)
    }

    // MARK: Waveform

    @Test("int16 samples are scaled, other dtypes are cast, and short clips are padded")
    func waveformInputs() throws {
        let pcm: [Int16] = [-32768, -1, 0, 1, 32767]
        let samples = pcm.map { Float($0) / 32768 }
        try Device.withDefaultDevice(.cpu) {
            for input in [MLXArray(pcm), MLXArray(samples)] {
                let x = try d1OmniWaveform(input)
                #expect(x.dtype == .float32)
                #expect(x.shape == [8000])
                #expect(x[..<5].asArray(Float.self) == samples)
                #expect(!x[5...].any().item(Bool.self))
            }
            // Python casts every other dtype to float32 without scaling.
            let int32 = try d1OmniWaveform(MLXArray([Int32(3), -2]))
            #expect(int32[..<2].asArray(Float.self) == [3, -2])
            let half = try d1OmniWaveform(MLXArray([Float(0.25), -0.5]).asType(.float16))
            #expect(half.dtype == .float32)
            #expect(half[..<2].asArray(Float.self) == [0.25, -0.5])
        }
    }

    @Test("Clips are cut to 30 s, empty clips become 0.5 s of silence, stereo is rejected")
    func waveformLimits() throws {
        try Device.withDefaultDevice(.cpu) {
            #expect(try d1OmniWaveform(MLXArray.ones([31 * 16000])).shape == [480000])
            let full = try d1OmniWaveform(MLXArray.ones([480000]))
            #expect(full.shape == [480000])
            let empty = try d1OmniWaveform(MLXArray.zeros([0], type: Int16.self))
            #expect(empty.shape == [8000])
            #expect(empty.dtype == .float32)
            #expect(!empty.any().item(Bool.self))
        }
        let error = #expect(throws: DecisionError.self) {
            try d1OmniWaveform(MLXArray.zeros([8000, 2], type: Int16.self))
        }
        guard case .invalidAudio(let message) = error else {
            Issue.record("expected invalidAudio, got \(String(describing: error))")
            return
        }
        #expect(message.contains("mono"))
    }

    // MARK: Resampling

    @Test("The resampler takes the standard rates and leaves the others to AVFoundation")
    func resamplerRates() throws {
        let standard = [
            6_000, 7_350, 8_000, 11_025, 12_000, 22_050, 24_000, 32_000, 44_100, 48_000, 88_200,
            96_000, 176_400, 192_000, 352_800, 384_000,
        ]
        for rate in standard {
            #expect(D1OmniResampler(sourceRate: rate) != nil, "\(rate)")
        }
        for rate in [16_000, 44_101, 0, -44_100] {
            #expect(D1OmniResampler(sourceRate: rate) == nil, "\(rate)")
        }
        let resampler = try #require(D1OmniResampler(sourceRate: 44_100))
        #expect(resampler.up == 160 && resampler.down == 441)
        // 30 s of input and 177 samples of filter tail.
        #expect(resampler.inputCount(forOutput: 480_000) == 1_323_177)
    }

    @Test("The resampler gives the float32 samples of mlx-audio's resample_audio")
    func resamplerMatchesPython() throws {
        // `resample_audio(pcm(n, n) / 32768, rate, 16000)`, with `pcm` of tools/d1_fixtures.py
        // (mlx-audio 0.5.8, scipy 1.18.1 on arm64): the output count, the first 16 hex digits of
        // the SHA-256 of its float32 bytes, and the first two, middle and last two samples.
        // The short clips are mostly edge padding.
        let golden: [(rate: Int, input: Int, output: Int, digest: String, samples: [Float])] = [
            (
                8000, 4000, 8000, "ae9e9dd19dc2d66c",
                [0.120827898, 0.193322688, -0.156758815, -0.247600019, -0.238811642]
            ),
            (
                11025, 5513, 8001, "d655416e310a6b61",
                [0.15216355, 0.0924091265, -0.18712008, -0.125743955, -0.244883627]
            ),
            (
                22050, 11025, 8000, "c4a83d6a8981ce65",
                [-0.0754025802, -0.0834252834, -0.228442594, 0.130204275, 0.0266897026]
            ),
            (
                44100, 22050, 8000, "3c35020e62c1bac7",
                [-0.179809183, -0.0512967445, 0.0173770338, 0.0606641509, 0.0331048295]
            ),
            (
                48000, 24000, 8000, "ef6f1f0c92fde779",
                [-0.00642745709, 0.224188462, -0.0590780005, -0.0738903359, -0.0910097286]
            ),
            (
                96000, 48001, 8001, "045e74ecc071ece5",
                [0.138052553, -0.0318708234, -0.0293612331, -0.0376928449, -0.120367244]
            ),
            (
                6000, 3000, 8000, "ea6b9f271b5b7ec1",
                [0.0167946089, -0.124388881, 0.0543216467, 0.250459045, 0.239677265]
            ),
            (
                384_000, 192_000, 8000, "5d60c54535f2bf53",
                [-0.0568940118, 0.0260040443, 0.0230491851, 0.0569103323, -0.00534410123]
            ),
            (44100, 7, 3, "fd7aa4d1eb53ee4b", [0.0878667086, 0.106446072, 0.140213639]),
            (44100, 1, 1, "150c0a1dd1e2aff1", [-0.115966797]),
        ]
        for row in golden {
            let resampler = try #require(D1OmniResampler(sourceRate: row.rate))
            let output = resampler(floats(hashedSamples(row.input)))
            #expect(output.count == row.output, "\(row.rate)")
            guard output.count == row.output else { continue }
            let last = output.count - 1
            let indices = Set([0, 1, output.count / 2, last - 1, last]).sorted()
                .filter { (index: Int) -> Bool in output.indices.contains(index) }
            #expect(
                indices.map { (index: Int) -> Float in output[index] } == row.samples,
                "\(row.rate)")
            #expect(digest(output) == row.digest, "\(row.rate)")
        }
    }

    @Test("Where the output is rounding error, the resampler still gives mlx-audio's samples")
    func resamplerRoundsLikePython() throws {
        // `resample_audio(triangle(2 * rate) / 32768, rate, 16000)`, digested as above. The
        // Kaiser window, the tap normalisation and the order of each sum must all match scipy.
        let golden = [
            (44100, "edcf4903aff83122"), (48000, "3fafa7c7161a56fb"),
            (22050, "132236ac76c6aa3a"),
        ]
        for (rate, expected) in golden {
            let resampler = try #require(D1OmniResampler(sourceRate: rate))
            let output = resampler(floats(triangle(2 * rate)))
            #expect(output.count == 32000, "\(rate)")
            #expect(digest(output) == expected, "\(rate)")
        }
    }

    @Test("The cache builds each rate's filter once and holds at most its capacity")
    func resamplerCache() throws {
        let cache = D1OmniResamplerCache(capacity: 2)
        let first = try #require(cache.resampler(sourceRate: 44_100))
        let again = try #require(cache.resampler(sourceRate: 44_100))
        let address = { (resampler: D1OmniResampler) -> UnsafeRawPointer? in
            resampler.taps.withUnsafeBytes { (taps: UnsafeRawBufferPointer) -> UnsafeRawPointer? in
                taps.baseAddress
            }
        }
        #expect(address(first) == address(again))
        #expect(first.taps == D1OmniResampler(sourceRate: 44_100)?.taps)
        #expect(cache.resampler(sourceRate: 16_000) == nil)
        #expect(cache.resampler(sourceRate: 44_101) == nil)
        #expect(cache.count == 1)
        #expect(cache.resampler(sourceRate: 48_000) != nil)
        #expect(cache.count == 2)
        // A third rate empties the cache first.
        #expect(cache.resampler(sourceRate: 8_000) != nil)
        #expect(cache.count == 1)
        let rebuilt = try #require(cache.resampler(sourceRate: 44_100))
        #expect(rebuilt.taps == first.taps)
        #expect(cache.count == 2)
    }

    @Test("A cut resample reads inputCount(forOutput:) samples and equals the full one")
    func resamplerLimit() throws {
        for rate in [8_000, 44_100, 48_000] {
            let resampler = try #require(D1OmniResampler(sourceRate: rate))
            let input = floats(hashedSamples(rate / 2))
            let full = resampler(input)
            #expect(full.count == 8000)
            let limit = 5000
            let needed = resampler.inputCount(forOutput: limit)
            #expect(needed < input.count)
            #expect(resampler(input, limit: limit) == Array(full[..<limit]), "\(rate)")
            #expect(
                resampler(Array(input[..<needed]), limit: limit) == Array(full[..<limit]),
                "\(rate)")
            #expect(resampler(input, limit: 0).isEmpty)
            #expect(resampler([]).isEmpty)
        }
    }

    // MARK: Files

    @Test("A 10 s 44.1 kHz file decodes to the same 160,000 samples every time, as in Python")
    func resampledFileRepeats() async throws {
        let pcm = hashedSamples(441_000)
        try await Device.withDefaultDevice(.cpu) {
            let mono = try await withWAV(pcm, sampleRate: 44_100) { file in
                let first = try await fileWaveform(file)
                for _ in 1 ..< 5 {
                    let again = try await fileWaveform(file)
                    #expect(again == first)
                }
                return first
            }
            // `load_audio(file, sr=16000)` of mlx-vlm 370f4ccf (mlx-audio 0.5.8) on this file.
            #expect(mono.count == 160_000)
            guard mono.count == 160_000 else { return }
            let samples = [0, 1, 80_000, 159_998, 159_999].map { (index: Int) -> Float in
                mono[index]
            }
            #expect(
                samples == [
                    0.0402998105, -0.0992421508, 0.0424116999, -0.0360000767, -0.066407524,
                ])
            let sum = mono.reduce(0.0) { (sum: Double, sample: Float) -> Double in
                sum + Double(sample)
            }
            #expect(abs(sum - 27.3142860277) < 1e-9)

            // The channels are averaged before resampling, so this stereo file is the same clip.
            let stereo = pcm.enumerated().flatMap { (index: Int, sample: Int16) -> [Int16] in
                let offset = Int16(index % 17 - 8) * 500
                return [sample + offset, sample - offset]
            }
            let mixed = try await withWAV(stereo, channels: 2, sampleRate: 44_100, fileWaveform)
            #expect(mixed == mono)
        }
    }

    @Test("A long file gives the same waveform as its first 30 s")
    func longFilesAreCut() async throws {
        let frames = 40 * 16_000
        let cut = D1OmniMelFrontend.maximumSamples
        try await Device.withDefaultDevice(.cpu) {
            let mono = hashedSamples(frames)
            let long = try await withWAV(mono, fileWaveform)
            #expect(long.count == cut)
            let first = try await withWAV(Array(mono[..<cut]), fileWaveform)
            #expect(first == long)
            #expect(long == floats(Array(mono[..<cut])))

            let stereo = hashedSamples(2 * frames)
            let longStereo = try await withWAV(stereo, channels: 2, fileWaveform)
            #expect(longStereo.count == cut)
            let start = try await withWAV(Array(stereo[..<(2 * cut)]), channels: 2, fileWaveform)
            #expect(start == longStereo)
        }
    }

    @Test("Only the samples behind the first 30 s are decoded")
    func decodeStopsAtTheWindow() async throws {
        try await Device.withDefaultDevice(.cpu) {
            let atModelRate = try await withWAV(hashedSamples(40 * 16_000)) { file in
                try await d1OmniDecodedAudio(file)
            }
            #expect(atModelRate.resampler == nil)
            #expect(atModelRate.samples.count == 480_000)

            // Resampling the cut samples equals cutting the whole file resampled.
            let pcm = hashedSamples(40 * 44_100)
            let (decoded, waveform) = try await withWAV(pcm, sampleRate: 44_100) { file in
                (try await d1OmniDecodedAudio(file), try await fileWaveform(file))
            }
            let resampler = try #require(decoded.resampler)
            #expect(decoded.samples.count == 1_323_177)
            #expect(decoded.samples == floats(Array(pcm[..<1_323_177])))
            #expect(waveform == Array(resampler(floats(pcm))[..<480_000]))
        }
    }

    @Test("Short 16 kHz mono and stereo files decode to the samples the whole-file decode gave")
    func shortFilesAreUnchanged() async throws {
        try await Device.withDefaultDevice(.cpu) {
            for (channels, frames) in [(1, 12_000), (1, 4_000), (2, 12_000), (2, 4_001)] {
                let pcm = hashedSamples(frames * channels)
                let (waveform, previous) = try await withWAV(pcm, channels: channels) { file in
                    (
                        try await fileWaveform(file),
                        try await wholeFileWaveform(file, channels: channels)
                    )
                }
                #expect(waveform.count == Swift.max(frames, 8000))
                #expect(
                    waveform.map(\.bitPattern) == previous.map(\.bitPattern),
                    "\(channels) channels, \(frames) frames")
            }
        }
    }

    @Test("Channels are averaged in order as numpy does, within a few ulps of MLX's mean")
    func channelsAreAveragedInOrder() async throws {
        let (channels, frames) = (6, 12_000)
        let pcm = floats(hashedSamples(frames * channels))
        let expected = (0 ..< frames).map { (frame: Int) -> Float in
            var sum = pcm[frame * channels]
            for channel in 1 ..< channels { sum += pcm[frame * channels + channel] }
            return sum / Float(channels)
        }
        try await Device.withDefaultDevice(.cpu) {
            let (waveform, previous) = try await withWAV(
                hashedSamples(frames * channels), channels: channels
            ) { file in
                (
                    try await fileWaveform(file),
                    try await wholeFileWaveform(file, channels: channels)
                )
            }
            #expect(waveform == expected)
            // MLX multiplies the sum by 1/6 and sums in an order of its own, which can move a
            // mean by a few ulps. On this file it is one: the means are below 0.25, where one
            // ulp is 2^-26.
            let difference = zip(waveform, previous).map { (a: Float, b: Float) -> Float in
                abs(a - b)
            }.max()
            #expect((difference ?? 1) <= 0x1p-26)
        }
    }

    @Test("Rates outside 8 to 192 kHz that the resampler takes are read as stored")
    func unusualRatesAreReadAsStored() async throws {
        try await Device.withDefaultDevice(.cpu) {
            let rates = [(6_000, 2, 8, 3), (7_350, 2, 320, 147), (384_000, 1, 1, 24)]
            for (rate, seconds, up, down) in rates {
                let pcm = hashedSamples(seconds * rate)
                let (decoded, waveform) = try await withWAV(pcm, sampleRate: rate) { file in
                    (try await d1OmniDecodedAudio(file), try await fileWaveform(file))
                }
                let resampler = try #require(decoded.resampler, "\(rate)")
                #expect(resampler.up == up && resampler.down == down, "\(rate)")
                #expect(decoded.samples == floats(pcm), "\(rate)")
                // The buffer holds about the clip, not 30 s.
                #expect(decoded.samples.capacity < 2 * pcm.count, "\(rate)")
                #expect(waveform.count == seconds * 16_000, "\(rate)")
                #expect(waveform == resampler(floats(pcm)), "\(rate)")
            }
        }
    }

    @Test("AudioToolbox converts the rates the resampler does not take, and those above 384 kHz")
    func otherRatesAreConverted() async throws {
        #expect(d1OmniReadRate(storedRate: 400_000) == 16_000)
        #expect(d1OmniReadRate(storedRate: 384_000) == nil)
        #expect(d1OmniReadRate(storedRate: 48_000) == 48_000)
        try await Device.withDefaultDevice(.cpu) {
            for rate in [44_101, 96_001, 400_000] {
                #expect(d1OmniReadRate(storedRate: rate) == 16_000)
                try await withWAV(hashedSamples(2 * rate), sampleRate: rate) { file in
                    let decoded = try await d1OmniDecodedAudio(file)
                    #expect(decoded.resampler == nil, "\(rate)")
                    #expect(decoded.samples.count == 32_000, "\(rate)")
                    #expect(try await d1OmniDecodedAudio(file).samples == decoded.samples)
                }
            }
        }
    }

    // MARK: Formats

    @Test("Big-endian AIFF, AIFC, AU and CAF files decode to the samples of the WAV")
    func bigEndianFilesDecode() async throws {
        let pcm = hashedSamples(12_000)
        let formats: [(name: String, settings: [String: Any])] = [
            ("clip.aiff", pcmSettings(bigEndian: true)),
            ("clip.aifc", pcmSettings(bigEndian: true)),
            ("float.aifc", pcmSettings(bits: 32, float: true, bigEndian: true)),
            ("clip.au", pcmSettings(bigEndian: true)),
            ("clip.caf", pcmSettings(bigEndian: true)),
        ]
        try await Device.withDefaultDevice(.cpu) {
            try await withDirectory { directory in
                for (name, settings) in formats {
                    let file = directory.appending(component: name)
                    try writeAudio(floats(pcm), to: file, settings: settings)
                    #expect(try await fileWaveform(file) == floats(pcm), "\(name)")
                }
            }
        }
    }

    @Test("G.711 and AAC files decode to their full length, the same on every read")
    func compressedFilesRepeat() async throws {
        // AVAssetReader often dropped the last 0.1 s of these 5 s G.711 clips.
        let formats: [(name: String, rate: Int, frames: Int, settings: [String: Any])] = [
            ("ulaw.wav", 8_000, 40_000, [AVFormatIDKey: kAudioFormatULaw]),
            ("alaw.wav", 8_000, 40_000, [AVFormatIDKey: kAudioFormatALaw]),
            ("ulaw.caf", 8_000, 40_000, [AVFormatIDKey: kAudioFormatULaw]),
            ("aac.m4a", 16_000, 160_000, [AVFormatIDKey: kAudioFormatMPEG4AAC]),
        ]
        try await Device.withDefaultDevice(.cpu) {
            try await withDirectory { directory in
                for (name, rate, frames, settings) in formats {
                    let file = directory.appending(component: name)
                    try writeAudio(
                        floats(hashedSamples(frames)), to: file, sampleRate: rate,
                        settings: settings)
                    let first = try await d1OmniDecodedAudio(file)
                    #expect(first.samples.count == frames, "\(name)")
                    for _ in 1 ..< 20 {
                        let again = try await d1OmniDecodedAudio(file)
                        #expect(again.samples == first.samples, "\(name)")
                    }
                }
            }
        }
    }

    @Test("FLAC files decode to their exact samples, short ones included")
    func flacFilesAreExact() async throws {
        let settings: [String: Any] = [AVFormatIDKey: kAudioFormatFLAC]
        try await Device.withDefaultDevice(.cpu) {
            try await withDirectory { directory in
                // Quiet noise keeps the file under 64 KiB, where AVAssetReader added zeros.
                let short = hashedSamples(51_300).map { $0 >> 6 }
                let file = directory.appending(component: "short.flac")
                try writeAudio(floats(short), to: file, settings: settings)
                let size = try #require(
                    try FileManager.default.attributesOfItem(atPath: file.path)[.size] as? Int)
                #expect(size < 65_536)
                #expect(try await fileWaveform(file) == floats(short))

                // A 40 s file gives its first 30 s.
                let long = hashedSamples(40 * 16_000)
                let longFile = directory.appending(component: "long.flac")
                try writeAudio(floats(long), to: longFile, settings: settings)
                #expect(
                    try await fileWaveform(longFile)
                        == floats(Array(long[..<D1OmniMelFrontend.maximumSamples])))
            }
        }
    }

    @Test("Files with no extension or a wrong one decode by their content")
    func filesDecodeByContent() async throws {
        let pcm = hashedSamples(12_000)
        try await Device.withDefaultDevice(.cpu) {
            try await withDirectory { directory in
                let wav = directory.appending(component: "clip.wav")
                try wavData(pcm).write(to: wav)
                let aiff = directory.appending(component: "clip.aiff")
                try writeAudio(floats(pcm), to: aiff, settings: pcmSettings(bigEndian: true))
                let m4a = directory.appending(component: "clip.m4a")
                try writeAudio(
                    floats(pcm), to: m4a, settings: [AVFormatIDKey: kAudioFormatMPEG4AAC])
                let aac = try await fileWaveform(m4a)
                #expect(aac.count == pcm.count)

                let renamed: [(source: URL, name: String, expected: [Float])] = [
                    (wav, "wav", floats(pcm)), (wav, "wav.m4a", floats(pcm)),
                    (wav, "wav.mp3", floats(pcm)), (aiff, "aiff.bin", floats(pcm)),
                    (m4a, "m4a", aac), (m4a, "m4a.mp3", aac),
                ]
                for (source, name, expected) in renamed {
                    let file = directory.appending(component: name)
                    try FileManager.default.copyItem(at: source, to: file)
                    #expect(try await fileWaveform(file) == expected, "\(name)")
                }
            }
        }
    }

    @Test("Ogg Vorbis files start at their first sample, as in mlx-audio")
    func oggVorbisStartsOnTime() async throws {
        // Resources/chirp.ogg is `chirp` below as 16-bit WAV, encoded with
        // `ffmpeg -i chirp.wav -c:a libvorbis -q:a 4 -page_duration 50000 chirp.ogg`.
        func chirp(_ i: Int) -> Double {
            let t = Double(i) / 16_000
            let envelope = Foundation.exp(-Foundation.pow(Double(i - 4_000) / 1_200, 2))
            return 0.5 * envelope * Foundation.sin(2 * Double.pi * (300 * t + 2_700 * t * t))
        }
        let url = try #require(Bundle.module.url(forResource: "chirp", withExtension: "ogg"))
        let decoded = try await d1OmniDecodedAudio(url)
        #expect(decoded.resampler == nil)
        #expect(decoded.samples.count == 8_000)
        // The lag that best lines the decode up with the source; AudioToolbox gives -128.
        let source = (0 ..< 8_000).map(chirp)
        func correlation(_ lag: Int) -> Double {
            var (dot, energy) = (0.0, 0.0)
            for i in 700 ..< 7_300 {
                let x = Double(decoded.samples[i + lag])
                dot += x * source[i]
                energy += x * x
            }
            return dot / energy.squareRoot()
        }
        #expect((-300 ... 300).max { correlation($0) < correlation($1) } == 0)

        // The same with an ID3v2 tag and zero padding in front.
        try await withDirectory { directory in
            let tagged = directory.appending(component: "tagged.ogg")
            try (id3TagAndPadding + Data(contentsOf: url)).write(to: tagged)
            #expect(try await d1OmniDecodedAudio(tagged).samples == decoded.samples)
        }
    }

    @Test("Ogg Vorbis files with no extension or an unknown one decode like the .ogg file")
    func oggVorbisDecodesByContent() async throws {
        let url = try #require(Bundle.module.url(forResource: "chirp", withExtension: "ogg"))
        let expected = try await d1OmniDecodedAudio(url).samples
        let ogg = try Data(contentsOf: url)
        let tagged = id3TagAndPadding + ogg
        try await withDirectory { directory in
            let files: [(String, Data)] = [
                ("chirp", ogg), ("chirp.vorbis", ogg), ("chirp.bin", ogg),
                ("tagged", tagged), ("tagged.bin", tagged),
            ]
            for (name, data) in files {
                let file = directory.appending(component: name)
                try data.write(to: file)
                let samples = try await d1OmniDecodedAudio(file).samples
                // AudioToolbox gives 7,936 samples here, starting 128 late.
                #expect(samples.count == expected.count, "\(name)")
                #expect(samples == expected, "\(name)")
            }
        }
    }

    @Test("Ogg FLAC throws invalidAudio that names the format, whatever the file is called")
    func oggFLACIsNamed() async throws {
        // The first page of an Ogg FLAC file: page header, one segment, the FLAC mapping packet.
        var bytes = Array("OggS".utf8) + [0, 2] + [UInt8](repeating: 0, count: 20) + [1, 51]
        bytes += [0x7F] + Array("FLAC".utf8) + [1, 0, 0, 1] + Array("fLaC".utf8)
        bytes += [UInt8](repeating: 0, count: 38)
        try await withDirectory { directory in
            for name in ["clip.ogg", "clip.wav"] {
                let file = directory.appending(component: name)
                try Data(bytes).write(to: file)
                await #expect(
                    throws: DecisionError.invalidAudio("\(name): Ogg FLAC is not supported")
                ) {
                    try await d1OmniWaveform(.url(file))
                }
            }
        }
    }

    @Test("NaN or infinite samples throw invalidAudio, in arrays and in files")
    func nonFiniteSamplesThrow() async throws {
        let error = DecisionError.invalidAudio("the clip has NaN or infinite samples")
        for value in [Float.nan, .infinity, -.infinity] {
            #expect(throws: error) {
                try d1OmniWaveform(MLXArray([0, value, 0.5]))
            }
        }
        try await withDirectory { directory in
            let file = directory.appending(component: "nan.wav")
            try writeAudio(
                [0, .nan, 0.25, 0], to: file,
                settings: pcmSettings(bits: 32, float: true, bigEndian: false))
            await #expect(throws: error) {
                try await d1OmniWaveform(.url(file))
            }
        }
    }

    // MARK: Lengths

    @Test("Frame counts match the reference table")
    func lengthsTable() throws {
        let table: [(samples: Int, mel: Int, valid: Int, encoder: Int, prefix: Int)] = [
            (8000, 51, 50, 7, 7),
            (16640, 105, 104, 14, 13),
            (176000, 1101, 1100, 138, 138),
            (480000, 3001, 3000, 376, 375),
        ]
        try Device.withDefaultDevice(.cpu) {
            let model = try tinyAudio()
            // Shapes and lengths only; nothing is evaluated.
            for row in table {
                let (features, valid) = model.frontend(MLXArray.zeros([row.samples]))
                let (encoded, length) = model.encoder(features, validFrames: valid)
                #expect(features.dim(1) == row.mel)
                #expect(valid == row.valid)
                #expect(encoded.dim(1) == row.encoder)
                #expect(length == row.prefix)
            }
        }
    }

    // MARK: Front end

    @Test("The Slaney filterbank matches librosa's checksums")
    func filterbank() {
        let bins = 257
        let fb = D1OmniMelFrontend.slaneyFilterbank(melBins: 128)
        #expect(fb.count == 128 * bins)
        let sum = fb.reduce(0.0) { $0 + Double($1) }
        #expect(abs(sum - 4.09048748) < 1e-6)
        #expect(abs(fb[1] - 0.0283775423) < 1e-9)
        let peak = fb.indices.max { fb[$0] < fb[$1] }!
        #expect(peak == 3 * bins + 3)
        #expect(abs(fb[peak] - 0.04236608) < 1e-8)
        let lastRow = (0 ..< bins).filter { fb[127 * bins + $0] != 0 }
        #expect(lastRow == Array(244 ... 255))
        // Each filter is one contiguous triangle.
        for row in 0 ..< 128 {
            let columns = (0 ..< bins).filter { fb[row * bins + $0] != 0 }
            #expect(!columns.isEmpty)
            #expect(columns.last! - columns.first! + 1 == columns.count)
        }
    }

    @Test("The 400-sample Hann window is centred in 512 samples")
    func window() {
        let window = D1OmniMelFrontend.window()
        #expect(window.count == 512)
        #expect(window[..<56].allSatisfy { $0 == 0 })
        #expect(window[456...].allSatisfy { $0 == 0 })
        #expect(window[56] == 0)
        #expect(window[55 + 400] < 1e-7)
        for n in 0 ..< 400 {
            #expect(abs(window[56 + n] - window[455 - n]) < 2e-7)
        }
        #expect(abs(window.reduce(0.0) { $0 + Double($1) } - 199.5) < 1e-4)
        #expect(window[56 + 199] > 0.99998 && window[56 + 200] > 0.99998)
    }

    @Test("Features match a direct DFT log-mel and are normalised over the valid frames")
    func featuresMatchDirectComputation() {
        let count = 8000
        let x = Device.withDefaultDevice(.cpu) { noise(count).asArray(Float.self) }
        let (features, valid) = Device.withDefaultDevice(.cpu) {
            D1OmniMelFrontend(melBins: 128)(MLXArray(x))
        }
        #expect(valid == 50)
        #expect(features.shape == [1, 51, 128])
        let got = features.asArray(Float.self)

        // Direct computation in Double.
        let n = 512
        let bins = n / 2 + 1
        let window = D1OmniMelFrontend.window().map(Double.init)
        let fb = D1OmniMelFrontend.slaneyFilterbank(melBins: 128).map(Double.init)
        var emphasized = [Double](repeating: 0, count: count + n)
        for t in 0 ..< count {
            let value = Double(x[t]) - (t > 0 ? Double(Float(0.97)) * Double(x[t - 1]) : 0)
            emphasized[256 + t] = value
        }
        var cosTable = [Double](repeating: 0, count: n * bins)
        var sinTable = [Double](repeating: 0, count: n * bins)
        for k in 0 ..< bins {
            for j in 0 ..< n {
                let angle = 2 * Double.pi * Double(k * j % n) / Double(n)
                cosTable[k * n + j] = Foundation.cos(angle)
                sinTable[k * n + j] = Foundation.sin(angle)
            }
        }
        var mel = [[Double]](repeating: [Double](repeating: 0, count: 128), count: 51)
        for frame in 0 ..< 51 {
            var power = [Double](repeating: 0, count: bins)
            for k in 0 ..< bins {
                var re = 0.0
                var im = 0.0
                for j in 0 ..< n {
                    let s = emphasized[frame * 160 + j] * window[j]
                    re += s * cosTable[k * n + j]
                    im -= s * sinTable[k * n + j]
                }
                power[k] = re * re + im * im
            }
            for m in 0 ..< 128 {
                var sum = 0.0
                for k in 0 ..< bins { sum += fb[m * bins + k] * power[k] }
                mel[frame][m] = Foundation.log(sum + 0x1p-24)
            }
        }
        var worst = 0.0
        for m in 0 ..< 128 {
            let column = (0 ..< 50).map { mel[$0][m] }
            let mean = column.reduce(0, +) / 50
            let std = (column.map { ($0 - mean) * ($0 - mean) }.reduce(0, +) / 49).squareRoot()
            for frame in 0 ..< 50 {
                let expected = (mel[frame][m] - mean) / (std + 1e-5)
                worst = Swift.max(worst, abs(expected - Double(got[frame * 128 + m])))
            }
            #expect(got[50 * 128 + m] == 0)
        }
        #expect(worst < 1e-3)
    }

    // MARK: Encoder

    @Test("rel_shift moves relative position T-1-i+j to column j")
    func relShift() {
        Device.withDefaultDevice(.cpu) {
            let (h, t) = (2, 5)
            let p = 2 * t - 1
            let scores = MLXRandom.normal([1, h, t, p], key: MLXRandom.key(3))
            let shifted = d1OmniRelShift(scores)
            #expect(shifted.shape == [1, h, t, p])
            let a = scores.asArray(Float.self)
            let b = shifted.asArray(Float.self)
            for head in 0 ..< h {
                for i in 0 ..< t {
                    for j in 0 ..< t {
                        let base = (head * t + i) * p
                        #expect(b[base + j] == a[base + t - 1 - i + j])
                    }
                }
            }
        }
    }

    @Test("Relative positions are interleaved sin/cos from T-1 down to -(T-1)")
    func relativePositions() {
        Device.withDefaultDevice(.cpu) {
            let table = D1OmniConformer.relativePositions(count: 3, dimensions: 4)
            #expect(table.shape == [5, 4])
            let values = table.asArray(Float.self)
            let inverse = [1.0, Foundation.exp(2 * -(Foundation.log(10_000.0) / 4))]
            for row in 0 ..< 5 {
                let position = Double(2 - row)
                for (i, frequency) in inverse.enumerated() {
                    let angle = position * frequency
                    #expect(abs(Double(values[row * 4 + 2 * i]) - Foundation.sin(angle)) < 1e-6)
                    #expect(
                        abs(Double(values[row * 4 + 2 * i + 1]) - Foundation.cos(angle)) < 1e-6)
                }
            }
        }
    }

    @Test(
        "The tiny model matches mlx-vlm on the same weights and clips",
        arguments: [12000, 12800])
    func tinyModelMatchesPython(sampleCount: Int) throws {
        // prefix[0, :, ::16] from `python tools/d1_fixtures.py tiny-audio` (mlx-vlm
        // 370f4ccf on the CPU).
        // 12800 samples give 80 valid mel frames, so the encoder has one padded frame.
        let golden: [Int: [Float]] = [
            12000: [
                0.553351, 0.654037, 0.782454, -1.19793,
                0.677955, 0.445838, 0.473903, -1.16988,
                0.636717, 0.47333, 0.561729, -1.16299,
                0.636602, 0.44792, 0.514891, -1.1337,
                0.649937, 0.465429, 0.521778, -1.16622,
                0.627132, 0.442336, 0.537478, -1.12583,
                0.678099, 0.434619, 0.457297, -1.15538,
                0.630927, 0.451946, 0.530861, -1.13406,
                0.691353, 0.42383, 0.431957, -1.15132,
                0.622487, 0.434195, 0.514618, -1.10298,
            ],
            12800: [
                0.495225, 0.775375, 0.93665, -1.18523,
                0.65333, 0.442835, 0.495558, -1.14395,
                0.684001, 0.425781, 0.455638, -1.14952,
                0.637645, 0.478433, 0.545303, -1.15697,
                0.615154, 0.493524, 0.607929, -1.1455,
                0.663478, 0.45473, 0.498319, -1.16056,
                0.624137, 0.488419, 0.57085, -1.15786,
                0.635638, 0.422992, 0.484012, -1.10441,
                0.645903, 0.446191, 0.516226, -1.13784,
                0.67236, 0.431182, 0.482538, -1.14544,
            ],
        ]
        try Device.withDefaultDevice(.cpu) {
            let model = try tinyAudio()
            #expect(model.parameters().flattened().count == 63)
            let out = model(try d1OmniWaveform(hashedPCM(sampleCount)))
            #expect(out.shape == [1, 10, 64])
            #expect(out.dtype == .float32)
            let expected = MLXArray(golden[sampleCount]!, [10, 4])
            let error = maxAbsDifference(out[0, 0..., .stride(by: 16)], expected)
            #expect(error < 1e-4)
        }
    }

    @Test("The padding mel frame never reaches the encoder")
    func subsamplingMasksPadding() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = try tinyAudio()
            // 56 valid mel frames: the encoder has 8 frames, the last one padding.
            let (features, valid) = model.frontend(noise(8960))
            #expect(valid == 56)
            let (clean, length) = model.encoder(features, validFrames: valid)
            #expect(clean.shape == [1, 8, 16])
            #expect(length == 7)
            let garbage = MLXRandom.normal([1, 1, 16], key: MLXRandom.key(11)) * 100
            let noisy = concatenated([features[0..., ..<valid], garbage], axis: 1)
            let (dirty, _) = model.encoder(noisy, validFrames: valid)
            #expect(maxAbsDifference(clean, dirty) == 0)
        }
    }

    @Test("Attention and the conv module ignore padded encoder frames")
    func blockMasksPadding() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = try tinyAudio()
            let block = model.encoder.layers[0]
            let positions = D1OmniConformer.relativePositions(count: 8, dimensions: 16)
            let valid = MLXArray(0 ..< 8) .< 7
            let h = MLXRandom.normal([1, 8, 16], key: MLXRandom.key(12))
            let garbage = MLXRandom.normal([1, 1, 16], key: MLXRandom.key(13)) * 100
            let changed = concatenated([h[0..., ..<7], garbage], axis: 1)

            let clean = block(h, positions: positions, valid: valid)
            let dirty = block(changed, positions: positions, valid: valid)
            #expect(maxAbsDifference(clean[0..., ..<7], dirty[0..., ..<7]) < 1e-5)
            // Without the masks the padded frame leaks into the real ones.
            let unmasked = block(h, positions: positions, valid: nil)
            let leaked = block(changed, positions: positions, valid: nil)
            #expect(maxAbsDifference(unmasked[0..., ..<7], leaked[0..., ..<7]) > 1e-2)
        }
    }

    @Test("BatchNorm reads its running statistics, and training mode would change the prefix")
    func batchNormEvalMode() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = try tinyAudio()
            #expect(model.modules().allSatisfy { !$0.training })

            let norm = model.encoder.layers[0].conv.norm
            let stats = Dictionary(uniqueKeysWithValues: norm.parameters().flattened())
            let x = MLXRandom.normal([1, 9, 16], key: MLXRandom.key(5))
            let expected =
                (x - stats["running_mean"]!) * rsqrt(stats["running_var"]! + 1e-5)
                * stats["weight"]! + stats["bias"]!
            #expect(maxAbsDifference(norm(x), expected) < 1e-5)

            // A forward pass in training mode also overwrites the running statistics.
            let waveform = try d1OmniWaveform(noise(12000))
            let trained = try tinyAudio()
            trained.train(true)
            #expect(maxAbsDifference(model(waveform), trained(waveform)) > 1e-2)
        }
    }

    // MARK: Weights

    @Test("The full-size module tree has the checkpoint's 687 audio parameters")
    func fullSizeParameters() {
        Device.withDefaultDevice(.cpu) {
            let model = D1OmniAudio(D1OmniAudioConfiguration(), outputDimensions: 1024)
            let shapes = Dictionary(
                uniqueKeysWithValues: model.parameters().flattened().map { ($0.0, $0.1.shape) })
            #expect(shapes.count == 687)
            #expect(shapes["encoder.subsampling.layers.0.weight"] == [256, 3, 3, 1])
            #expect(shapes["encoder.subsampling.layers.2.weight"] == [256, 3, 3, 1])
            #expect(shapes["encoder.subsampling.layers.3.weight"] == [256, 1, 1, 256])
            #expect(shapes["encoder.subsampling.linear.weight"] == [512, 4096])
            #expect(shapes["encoder.layers.16.self_attn.relative_k_proj.weight"] == [512, 512])
            #expect(shapes["encoder.layers.16.self_attn.relative_k_proj.bias"] == nil)
            #expect(shapes["encoder.layers.0.self_attn.bias_u"] == [8, 64])
            #expect(shapes["encoder.layers.0.conv.pointwise_conv1.weight"] == [1024, 1, 512])
            #expect(shapes["encoder.layers.0.conv.depthwise_conv.weight"] == [512, 9, 1])
            #expect(shapes["encoder.layers.0.conv.norm.running_var"] == [512])
            #expect(shapes["encoder.layers.0.feed_forward1.linear1.weight"] == [2048, 512])
            #expect(shapes["adapter.linear_1.weight"] == [1024, 512])
            #expect(shapes["residual.down.weight"] == [512, 1024])
            #expect(shapes["residual.up.weight"] == [1024, 512])
        }
    }

    @Test("sanitize maps the reference checkpoint's NeMo names and torch layout, idempotently")
    func sanitize() throws {
        let renames = [
            ("encoder.pre_encode.conv.", "encoder.subsampling.layers."),
            ("encoder.pre_encode.out.", "encoder.subsampling.linear."),
            ("self_attn.linear_q.", "self_attn.q_proj."),
            ("self_attn.linear_k.", "self_attn.k_proj."),
            ("self_attn.linear_v.", "self_attn.v_proj."),
            ("self_attn.linear_out.", "self_attn.o_proj."),
            ("self_attn.linear_pos.", "self_attn.relative_k_proj."),
            ("self_attn.pos_bias_u", "self_attn.bias_u"),
            ("self_attn.pos_bias_v", "self_attn.bias_v"),
            ("conv.batch_norm.", "conv.norm."),
        ]
        try Device.withDefaultDevice(.cpu) {
            let model = D1OmniAudio(try tinyAudioConfig(), outputDimensions: 64)
            var expected = [String: MLXArray]()
            var source = [String: MLXArray]()
            var start = 0
            for (key, value) in model.parameters().flattened() {
                let numbered = MLXArray(start ..< start + value.size).asType(value.dtype)
                    .reshaped(value.shape)
                start += value.size
                expected["audio." + key] = numbered
                var torchKey = "audio." + key
                for (old, new) in renames {
                    torchKey = torchKey.replacingOccurrences(of: new, with: old)
                }
                switch value.ndim {
                case 4: source[torchKey] = numbered.transposed(0, 3, 1, 2)
                case 3: source[torchKey] = numbered.transposed(0, 2, 1)
                default: source[torchKey] = numbered
                }
            }
            for key in [
                "audio.encoder.pre_encode.conv.3.weight",
                "audio.encoder.pre_encode.out.weight",
                "audio.encoder.layers.0.self_attn.linear_pos.weight",
                "audio.encoder.layers.0.self_attn.pos_bias_u",
                "audio.encoder.layers.0.conv.depthwise_conv.weight",
                "audio.encoder.layers.0.conv.batch_norm.running_var",
                "audio.residual.down.weight",
            ] {
                #expect(source[key] != nil, "\(key)")
            }
            source["audio.encoder.layers.0.conv.batch_norm.num_batches_tracked"] = MLXArray(3)
            let other = MLXArray(0 ..< 6).reshaped(2, 1, 3)
            source["encoder.layers.0.conv.conv.weight"] = other
            expected["encoder.layers.0.conv.conv.weight"] = other

            func assertEqual(_ actual: [String: MLXArray]) {
                #expect(Set(actual.keys) == Set(expected.keys))
                for (key, value) in expected {
                    guard let got = actual[key] else { continue }
                    #expect(got.shape == value.shape, "\(key)")
                    #expect(got.shape == value.shape && arrayEqual(got, value).item(Bool.self))
                }
            }
            let converted = d1OmniSanitizeAudio(source)
            assertEqual(converted)
            assertEqual(d1OmniSanitizeAudio(converted))

            var audio = [String: MLXArray]()
            for (key, value) in converted where key.hasPrefix("audio.") {
                audio[String(key.dropFirst("audio.".count))] = value
            }
            try model.update(parameters: ModuleParameters.unflattened(audio), verify: .all)
        }
    }
}

// MARK: - Parity with mlx-vlm (real weights)

/// The audio prefix against mlx-vlm 370f4ccf on clips cut from `jfk.wav`, around the front
/// end's edges (0.5 s padding, the 30 s cut, int16 input). Needs `MLX_D1_OMNI_MODEL` and
/// `MLX_D1_FIXTURES` (see README.md).
@Suite(
    "d1-omni audio parity",
    .enabled(if: D1Fixtures.omniModel != nil && D1Fixtures.directory != nil))
struct D1OmniAudioParityTests {

    private func loadModel() throws -> D1OmniAudio {
        let directory = try #require(D1Fixtures.omniModel)
        let all = try loadArrays(url: directory.appendingPathComponent("model.safetensors"))
        var weights = [String: MLXArray]()
        for (key, value) in d1OmniSanitizeAudio(all) where key.hasPrefix("audio.") {
            weights[String(key.dropFirst("audio.".count))] = value
        }
        #expect(weights.count == 687)
        let model = D1OmniAudio(D1OmniAudioConfiguration(), outputDimensions: 1024)
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
        eval(model)
        return model
    }

    /// Max abs difference and worst-row cosine of two `[rows, D]` arrays.
    private func report(_ label: String, _ a: MLXArray, _ b: MLXArray) -> (
        maxAbs: Float, cosine: Float
    ) {
        let a = a.asType(.float32)
        let b = b.asType(.float32)
        let cosine =
            (a * b).sum(axis: -1)
            / (sqrt(a.square().sum(axis: -1)) * sqrt(b.square().sum(axis: -1)))
        let result = (maxAbsDifference(a, b), cosine.min().item(Float.self))
        print(
            "d1-omni audio parity \(label): maxAbs \(String(format: "%.3e", result.0))"
                + " worstRowCos \(String(format: "%.8f", result.1))")
        return result
    }

    @Test("Mel, encoder and prefix match mlx-vlm")
    func prefixes() async throws {
        let mlxvlm = try loadArrays(
            url: try D1Fixtures.url("d1-omni-600M-float32-audio.safetensors"))
        let model = try loadModel()
        let names = mlxvlm.keys.filter { $0.hasSuffix(".input") }
            .map { String($0.dropLast(".input".count)) }.sorted()
        try #require(!names.isEmpty)

        for name in names {
            let waveform = try d1OmniWaveform(mlxvlm["\(name).input"]!)
            let (mel, valid) = model.frontend(waveform)
            let (encoded, length) = model.encoder(mel, validFrames: valid)
            let prefix = model(waveform)
            #expect(valid == mlxvlm["\(name).mel_valid"]!.item(Int.self))
            #expect(length == mlxvlm["\(name).enc_valid"]!.item(Int.self))
            #expect(encoded.dim(1) == mlxvlm["\(name).enc"]!.dim(0))
            #expect(prefix.shape == [1, mlxvlm["\(name).prefix"]!.dim(0), 1024])
            #expect(!mel[0, valid...].any().item(Bool.self))

            _ = report(
                "\(name) mel vs mlx-vlm", mel[0, ..<valid], mlxvlm["\(name).mel"]![..<valid])
            _ = report(
                "\(name) enc[:P] vs mlx-vlm", encoded[0, ..<length],
                mlxvlm["\(name).enc"]![..<length])
            // mlx-vlm's own GPU and CPU prefixes differ by up to 1e-2 on the sweep.
            let python = report("\(name) prefix vs mlx-vlm", prefix[0], mlxvlm["\(name).prefix"]!)
            #expect(python.maxAbs <= 5e-4, "\(name)")
            #expect(python.cosine >= 0.99999, "\(name)")
        }

        // The file decodes to exactly the samples mlx-vlm read.
        let decoded = try await d1OmniWaveform(.url(try D1Fixtures.url("jfk.wav")))
        #expect(decoded.shape == [176000])
        #expect(maxAbsDifference(decoded, mlxvlm["jfk11s.input"]!) == 0)
    }
}
