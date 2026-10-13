// Copyright © 2026 Apple Inc.
// Port of mlx-vlm's d1-omni model (Blaizzy/mlx-vlm#2467), which follows LiquidAI's reference code.
// The Parakeet encoder and feature extractor come from mlx-vlm's nemotron_h_nano_omni/audio.py.

import AVFoundation
import AudioToolbox
import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Configuration

/// The `audio_config` of a d1-omni checkpoint. Missing keys take the Python defaults.
struct D1OmniAudioConfiguration: Codable, Sendable, Equatable {
    var featIn: Int
    var layerCount: Int
    var dModel: Int
    var subsamplingConvChannels: Int
    var ffExpansionFactor: Int
    var headCount: Int
    var convKernelSize: Int
    var residualWidth: Int

    enum CodingKeys: String, CodingKey {
        case featIn = "feat_in"
        case layerCount = "n_layers"
        case dModel = "d_model"
        case subsamplingConvChannels = "subsampling_conv_channels"
        case ffExpansionFactor = "ff_expansion_factor"
        case headCount = "n_heads"
        case convKernelSize = "conv_kernel_size"
        case residualWidth = "residual_width"
    }

    init(
        featIn: Int = 128, layerCount: Int = 17, dModel: Int = 512,
        subsamplingConvChannels: Int = 256, ffExpansionFactor: Int = 4, headCount: Int = 8,
        convKernelSize: Int = 9, residualWidth: Int = 512
    ) {
        self.featIn = featIn
        self.layerCount = layerCount
        self.dModel = dModel
        self.subsamplingConvChannels = subsamplingConvChannels
        self.ffExpansionFactor = ffExpansionFactor
        self.headCount = headCount
        self.convKernelSize = convKernelSize
        self.residualWidth = residualWidth
    }

    init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let defaults = Self()
        func value(_ key: CodingKeys, _ fallback: Int) throws -> Int {
            try container.decodeIfPresent(Int.self, forKey: key) ?? fallback
        }
        self.init(
            featIn: try value(.featIn, defaults.featIn),
            layerCount: try value(.layerCount, defaults.layerCount),
            dModel: try value(.dModel, defaults.dModel),
            subsamplingConvChannels: try value(
                .subsamplingConvChannels, defaults.subsamplingConvChannels),
            ffExpansionFactor: try value(.ffExpansionFactor, defaults.ffExpansionFactor),
            headCount: try value(.headCount, defaults.headCount),
            convKernelSize: try value(.convKernelSize, defaults.convKernelSize),
            residualWidth: try value(.residualWidth, defaults.residualWidth))
    }
}

// MARK: - Waveform

/// Mono 16 kHz samples -> float32 `[N]`, cut to 30 s and zero padded to 0.5 s (Python `waveform`).
///
/// `int16` samples are divided by 32768; every other dtype is cast to float32 unscaled. NaN or
/// infinite samples throw.
func d1OmniWaveform(_ samples: MLXArray) throws -> MLXArray {
    guard samples.ndim == 1 else {
        throw DecisionError.invalidAudio("audio must be mono: a 1-D array of 16 kHz samples")
    }
    let minimum = D1OmniMelFrontend.minimumSamples
    let count = Swift.min(samples.dim(0), D1OmniMelFrontend.maximumSamples)
    guard count > 0 else {
        return MLXArray.zeros([minimum], type: Float.self)
    }
    // On the CPU, which also handles float64.
    return try Device.withDefaultDevice(.cpu) {
        var x = samples[..<count]
        x = samples.dtype == .int16 ? x.asType(.float32) / Float(32768) : x.asType(.float32)
        guard all(isFinite(x)).item(Bool.self) else {
            throw DecisionError.invalidAudio("the clip has NaN or infinite samples")
        }
        if count < minimum {
            x = padded(x, widths: [IntOrPair((0, minimum - count))])
        }
        return x
    }
}

/// A request's clip as a waveform. Arrays are samples as given. A file's first 30 s are decoded,
/// the channels averaged and other sample rates resampled to 16 kHz, as Python's `load_audio`
/// does. A file that fails to load or decode, or has no samples, throws.
func d1OmniWaveform(_ audio: UserInput.Audio) async throws -> MLXArray {
    switch audio.source {
    case .array(let samples):
        try d1OmniWaveform(samples)
    case .url(let url):
        try await d1OmniWaveform(file: url)
    }
}

private func d1OmniWaveform(file: URL) async throws -> MLXArray {
    var samples: [Float] = []
    var failure: (any Error)?
    do {
        let decoded = try await d1OmniDecodedAudio(file)
        samples =
            decoded.resampler?(decoded.samples, limit: D1OmniMelFrontend.maximumSamples)
            ?? decoded.samples
    } catch let error as DecisionError {
        throw error
    } catch {
        failure = error
    }
    guard !samples.isEmpty else {
        // AVFoundation can fail on an empty track instead of returning no samples.
        if await hasEmptyAudioTrack(file) {
            throw DecisionError.invalidAudio("the file has no samples")
        }
        let reason = failure?.localizedDescription ?? "no samples could be decoded"
        throw DecisionError.invalidAudio("\(file.lastPathComponent): \(reason)")
    }
    return try d1OmniWaveform(MLXArray(samples))
}

/// A decode failure that AVFoundation gives no error for.
private struct AudioFileError: LocalizedError {
    let errorDescription: String?
}

/// The start of a file's first audio track with its channels averaged: only the samples
/// behind the 30 s the model reads. `resampler` brings them to 16 kHz; nil means they are
/// at 16 kHz already.
///
/// AudioToolbox reads the file and finds its format from its bytes. Only if that fails does it
/// go by the file extension, because with that hint its MP3 parser accepts other formats too.
/// AVFoundation reads Ogg files first, whatever their name, because AudioToolbox starts Ogg
/// Vorbis 128 frames late, and the files AudioToolbox cannot open. Ogg FLAC throws.
func d1OmniDecodedAudio(_ url: URL) async throws -> (
    samples: [Float], resampler: D1OmniResampler?
) {
    switch OggStream(url) {
    case .flac:
        // AVFoundation fails on it, and AudioToolbox reads only part of it.
        throw DecisionError.invalidAudio("\(url.lastPathComponent): Ogg FLAC is not supported")
    case .other:
        if let decoded = try? await d1OmniOggAudio(url), !decoded.samples.isEmpty {
            return decoded
        }
    case nil:
        break
    }
    if let contents = AudioFileContents(url), let file = AudioFileDecoder(wrapping: contents),
        let decoded = try file.read()
    {
        return decoded
    }
    if let file = AudioFileDecoder(url), let decoded = try file.read() {
        return decoded
    }
    return try await d1OmniAssetAudio(url)
}

/// The codec of an Ogg file, from the first packet of its first page.
private enum OggStream {
    case flac
    case other

    /// Nil when the file is not Ogg or cannot be read. ID3v2 tags and zero padding before the
    /// first page are skipped, as AVFoundation and ffmpeg skip them.
    init?(_ url: URL) {
        guard let handle = try? FileHandle(forReadingFrom: url) else { return nil }
        var offset: UInt64 = 0
        while offset < 1 << 20 {
            guard (try? handle.seek(toOffset: offset)) != nil,
                let data = try? handle.read(upToCount: 4096), !data.isEmpty
            else { return nil }
            let bytes = [UInt8](data)
            if bytes.starts(with: Array("ID3".utf8)), bytes.count >= 10 {
                // The size is 4 bytes of 7 bits each; a footer adds 10 bytes.
                let size = bytes[6 ..< 10].reduce(UInt64(0)) { $0 << 7 | UInt64($1 & 0x7F) }
                offset += 10 + size + (bytes[5] & 0x10 != 0 ? 10 : 0)
                continue
            }
            if offset > 0, bytes[0] == 0 {
                offset += UInt64(bytes.firstIndex { $0 != 0 } ?? bytes.count)
                continue
            }
            guard bytes.starts(with: Array("OggS".utf8)), bytes.count > 27 else { return nil }
            // The packet follows the 27-byte page header and its segment table.
            let packet = bytes.dropFirst(27 + Int(bytes[26]))
            self = packet.starts(with: [0x7F] + Array("FLAC".utf8)) ? .flac : .other
            return
        }
        return nil
    }
}

/// An AudioToolbox audio file that reads through callbacks, so that its name gives no type hint.
private final class AudioFileContents {
    private let handle: FileHandle
    private let length: Int64
    private(set) var id: AudioFileID?

    init?(_ url: URL) {
        guard let handle = try? FileHandle(forReadingFrom: url),
            let end = try? handle.seekToEnd()
        else { return nil }
        self.handle = handle
        self.length = Int64(end)
        let status = AudioFileOpenWithCallbacks(
            Unmanaged.passUnretained(self).toOpaque(),
            { context, position, count, buffer, read in
                let contents = Unmanaged<AudioFileContents>.fromOpaque(context)
                    .takeUnretainedValue()
                let n = pread(contents.handle.fileDescriptor, buffer, Int(count), position)
                read.pointee = UInt32(Swift.max(n, 0))
                return n < 0 ? kAudioFileUnspecifiedError : noErr
            }, nil,
            { context in
                Unmanaged<AudioFileContents>.fromOpaque(context).takeUnretainedValue().length
            }, nil, 0, &id)
        guard status == noErr, id != nil else { return nil }
    }

    deinit {
        if let id { AudioFileClose(id) }
    }
}

/// An AudioToolbox ExtAudioFile, which decodes to float32 and converts the sample rate.
private final class AudioFileDecoder {
    private let file: ExtAudioFileRef
    /// Kept open while `file` reads from it.
    private let contents: AudioFileContents?

    /// Opens `url` with its extension as a type hint.
    init?(_ url: URL) {
        var file: ExtAudioFileRef?
        guard ExtAudioFileOpenURL(url as CFURL, &file) == noErr, let file else { return nil }
        self.file = file
        self.contents = nil
    }

    init?(wrapping contents: AudioFileContents) {
        var file: ExtAudioFileRef?
        guard let id = contents.id, ExtAudioFileWrapAudioFileID(id, false, &file) == noErr,
            let file
        else { return nil }
        self.file = file
        self.contents = contents
    }

    deinit {
        ExtAudioFileDispose(file)
    }

    /// The samples behind the model's 30 s, or nil when the file reports no frames or none
    /// decode. Integer rates up to 384 kHz that `D1OmniResampler` takes are read as stored;
    /// AudioToolbox converts the other rates to 16 kHz.
    func read(cache: D1OmniResamplerCache = .shared) throws -> (
        samples: [Float], resampler: D1OmniResampler?
    )? {
        var format = AudioStreamBasicDescription()
        var formatSize = UInt32(MemoryLayout<AudioStreamBasicDescription>.size)
        var frames: Int64 = 0
        var framesSize = UInt32(MemoryLayout<Int64>.size)
        // Ogg FLAC files report no frames, then decode only some of them.
        guard
            ExtAudioFileGetProperty(
                file, kExtAudioFileProperty_FileDataFormat, &formatSize, &format) == noErr,
            ExtAudioFileGetProperty(
                file, kExtAudioFileProperty_FileLengthFrames, &framesSize, &frames) == noErr,
            format.mChannelsPerFrame > 0, frames > 0
        else { return nil }
        let storedRate = Int(exactly: format.mSampleRate) ?? 0
        let rate = d1OmniReadRate(storedRate: storedRate, cache: cache) ?? storedRate
        let resampler = cache.resampler(sourceRate: rate)
        let channels = Int(format.mChannelsPerFrame)
        var client = AudioStreamBasicDescription(
            mSampleRate: Double(rate), mFormatID: kAudioFormatLinearPCM,
            mFormatFlags: kAudioFormatFlagsNativeFloatPacked,
            mBytesPerPacket: UInt32(4 * channels), mFramesPerPacket: 1,
            mBytesPerFrame: UInt32(4 * channels), mChannelsPerFrame: UInt32(channels),
            mBitsPerChannel: 32, mReserved: 0)
        guard
            ExtAudioFileSetProperty(
                file, kExtAudioFileProperty_ClientDataFormat, formatSize, &client) == noErr
        else { return nil }

        let limit =
            resampler?.inputCount(forOutput: D1OmniMelFrontend.maximumSamples)
            ?? D1OmniMelFrontend.maximumSamples
        var samples: [Float] = []
        samples.reserveCapacity(Int(Swift.min(frames, Int64(limit))))
        let chunk = 32_768
        var buffer = [Float](repeating: 0, count: chunk * channels)
        while samples.count < limit {
            var count = UInt32(Swift.min(chunk, limit - samples.count))
            let status = buffer.withUnsafeMutableBytes { bytes in
                var list = AudioBufferList(
                    mNumberBuffers: 1,
                    mBuffers: AudioBuffer(
                        mNumberChannels: UInt32(channels), mDataByteSize: UInt32(bytes.count),
                        mData: bytes.baseAddress))
                return ExtAudioFileRead(file, &count, &list)
            }
            guard status == noErr else {
                throw NSError(domain: NSOSStatusErrorDomain, code: Int(status))
            }
            if count == 0 { break }
            // Python averages the channels.
            for frame in stride(from: 0, to: Int(count) * channels, by: channels) {
                var sum = buffer[frame]
                for index in frame + 1 ..< frame + channels { sum += buffer[index] }
                samples.append(sum / Float(channels))
            }
        }
        return samples.isEmpty ? nil : (samples, resampler)
    }
}

/// `d1OmniAssetAudio(_:)` for an Ogg file, through a temporary `.ogg` link, because AVFoundation
/// finds the format from the file extension. With an `audio/ogg` MIME type instead, it reads Ogg
/// behind an ID3 tag and zero padding as MPEG audio.
private func d1OmniOggAudio(_ url: URL) async throws -> (
    samples: [Float], resampler: D1OmniResampler?
) {
    let link = FileManager.default.temporaryDirectory
        .appending(component: "d1-omni-\(UUID().uuidString).ogg")
    do {
        try FileManager.default.createSymbolicLink(at: link, withDestinationURL: url.absoluteURL)
    } catch {
        // Without a link, a file named .ogg still opens directly.
        return try await d1OmniAssetAudio(url)
    }
    defer { try? FileManager.default.removeItem(at: link) }
    return try await d1OmniAssetAudio(link)
}

/// `d1OmniDecodedAudio(_:)` with AVFoundation, for files AudioToolbox cannot open.
///
/// Integer rates up to 384 kHz that `D1OmniResampler` takes are read as stored. AVFoundation
/// converts the other rates to 16 kHz itself, which does not repeat exactly between runs.
/// HE-AAC is read at the core rate its format reports, half its output rate.
private func d1OmniAssetAudio(_ url: URL) async throws -> (
    samples: [Float], resampler: D1OmniResampler?
) {
    let asset = AVURLAsset(url: url)
    guard let track = try await asset.loadTracks(withMediaType: .audio).first else {
        throw AudioFileError(errorDescription: "No audio data in file: \(url.path)")
    }
    let format = (try? await track.load(.formatDescriptions).first).flatMap {
        CMAudioFormatDescriptionGetStreamBasicDescription($0)?.pointee
    }
    // One channel when the format cannot be read.
    let channels = Swift.max(1, Int(format?.mChannelsPerFrame ?? 1))
    let duration = (try? await track.load(.timeRange))?.duration.seconds ?? .nan
    var settings: [String: Any] = [
        AVFormatIDKey: kAudioFormatLinearPCM,
        AVNumberOfChannelsKey: channels,
        AVLinearPCMBitDepthKey: 32,
        AVLinearPCMIsFloatKey: true,
        AVLinearPCMIsBigEndianKey: false,
        AVLinearPCMIsNonInterleaved: false,
    ]
    // The filter is built before reading starts, so that the reader does not decode ahead.
    let storedRate = format.flatMap { Int(exactly: $0.mSampleRate) } ?? 0
    let cache = D1OmniResamplerCache.shared
    if let rate = d1OmniReadRate(storedRate: storedRate, cache: cache) {
        settings[AVSampleRateKey] = rate
    }
    let output = AVAssetReaderTrackOutput(track: track, outputSettings: settings)
    let reader = try AVAssetReader(asset: asset)
    let unreadable = AudioFileError(errorDescription: "Unable to read audio from \(url.path)")
    guard reader.canAdd(output) else { throw unreadable }
    reader.add(output)
    guard reader.startReading() else { throw reader.error ?? unreadable }

    // The rate of the decoded samples, from the first buffer.
    var pending = output.copyNextSampleBuffer()
    let rate =
        pending.flatMap(CMSampleBufferGetFormatDescription).flatMap {
            CMAudioFormatDescriptionGetStreamBasicDescription($0)?.pointee.mSampleRate
        } ?? Double(D1OmniMelFrontend.sampleRate)
    var resampler: D1OmniResampler?
    if rate != Double(D1OmniMelFrontend.sampleRate) {
        resampler = Int(exactly: rate).flatMap(cache.resampler)
        guard resampler != nil else {
            throw AudioFileError(
                errorDescription: "Unsupported sample rate \(rate) Hz in \(url.path)")
        }
    }

    let limit =
        resampler?.inputCount(forOutput: D1OmniMelFrontend.maximumSamples)
        ?? D1OmniMelFrontend.maximumSamples
    var samples: [Float] = []
    // A short clip reserves about its own length, not the whole 30 s.
    let frames = (duration * rate * 17 / 16).rounded(.up)
    samples.reserveCapacity(frames >= 0 && frames < Double(limit) ? Int(frames) : limit)
    while samples.count < limit, let buffer = pending ?? output.copyNextSampleBuffer() {
        pending = nil
        guard let block = CMSampleBufferGetDataBuffer(buffer) else { continue }
        let byteCount = CMBlockBufferGetDataLength(block)
        var chunk = [Float](repeating: 0, count: byteCount / MemoryLayout<Float>.size)
        CMBlockBufferCopyDataBytes(
            block, atOffset: 0, dataLength: byteCount, destination: &chunk)
        // AVFoundation mixes to mono as (L + R) / sqrt(2); Python averages the channels.
        let frames = Swift.min(chunk.count / channels, limit - samples.count)
        for frame in stride(from: 0, to: frames * channels, by: channels) {
            var sum = chunk[frame]
            for index in frame + 1 ..< frame + channels { sum += chunk[index] }
            samples.append(sum / Float(channels))
        }
    }
    if samples.count >= limit {
        reader.cancelReading()
    } else if reader.status != .completed {
        throw reader.error ?? unreadable
    }
    return (samples, resampler)
}

/// The rate to decode at, or nil to read the file as stored. Rates that `D1OmniResampler` does
/// not take, and those above 384 kHz, are converted to 16 kHz.
func d1OmniReadRate(storedRate: Int, cache: D1OmniResamplerCache = .shared) -> Int? {
    if storedRate > 384_000 || cache.resampler(sourceRate: storedRate) == nil {
        return D1OmniMelFrontend.sampleRate
    }
    // AVFoundation outputs only these rates. Without the key it reads the file as stored.
    return (8_000 ... 192_000).contains(storedRate) ? storedRate : nil
}

private func hasEmptyAudioTrack(_ url: URL) async -> Bool {
    let asset = AVURLAsset(url: url)
    guard let track = try? await asset.loadTracks(withMediaType: .audio).first,
        let range = try? await track.load(.timeRange)
    else { return false }
    return range.duration == .zero
}

// MARK: - Mel front end

/// NeMo `FilterbankFeatures` in eval mode: preemphasis, centred STFT, power, Slaney mel, log and
/// per-bin normalisation over the valid frames. Always float32.
struct D1OmniMelFrontend {
    static let sampleRate = 16_000
    static let minimumSamples = 8_000
    static let maximumSamples = 480_000
    static let fftSize = 512
    static let windowLength = 400
    static let hopLength = 160
    static let preemphasis: Float = 0.97
    static let logGuard: Float = 0x1p-24
    static let normalizationEpsilon: Float = 1e-5

    let melBins: Int
    /// `[fftSize]`: a symmetric 400-sample Hann window, centred.
    let window: MLXArray
    /// `[fftSize / 2 + 1, melBins]`
    let filterbank: MLXArray

    init(melBins: Int) {
        self.melBins = melBins
        self.window = MLXArray(Self.window())
        self.filterbank = MLXArray(
            Self.slaneyFilterbank(melBins: melBins), [melBins, Self.fftSize / 2 + 1]
        ).transposed()
    }

    /// `torch.hann_window(400, periodic=False)` zero padded to 512 on both sides (56 + 400 + 56).
    static func window() -> [Float] {
        let left = (fftSize - windowLength) / 2
        var window = [Float](repeating: 0, count: fftSize)
        let denominator = Double(windowLength - 1)
        for n in 0 ..< windowLength {
            window[left + n] = Float(
                0.5 * (1 - Foundation.cos(2 * Double.pi * Double(n) / denominator)))
        }
        return window
    }

    /// librosa `filters.mel(sr, n_fft, n_mels, norm="slaney")`, row-major `[melBins, fftSize / 2 + 1]`.
    ///
    /// Computed in Double with librosa's float32 roundings, which reproduces it bit for bit.
    static func slaneyFilterbank(
        melBins: Int, sampleRate: Int = sampleRate, fftSize: Int = fftSize
    ) -> [Float] {
        let fSp = 200.0 / 3
        let minLogHz = 1000.0
        let minLogMel = minLogHz / fSp
        let logStep = Foundation.log(6.4) / 27.0
        func hzToMel(_ f: Double) -> Double {
            f >= minLogHz ? minLogMel + Foundation.log(f / minLogHz) / logStep : f / fSp
        }
        func melToHz(_ m: Double) -> Double {
            m >= minLogMel ? minLogHz * Foundation.exp(logStep * (m - minLogMel)) : fSp * m
        }

        // np.linspace: i * step + start, with the last point set to stop.
        let low = hzToMel(0)
        let high = hzToMel(Double(sampleRate) / 2)
        let points = melBins + 2
        let step = (high - low) / Double(points - 1)
        var mels = (0 ..< points).map { Double($0) * step + low }
        mels[points - 1] = high
        let hz = mels.map(melToHz)

        // np.fft.rfftfreq
        let binWidth = 1.0 / (Double(fftSize) * (1.0 / Double(sampleRate)))
        let bins = fftSize / 2 + 1
        var weights = [Float](repeating: 0, count: melBins * bins)
        for i in 0 ..< melBins {
            let norm = 2.0 / (hz[i + 2] - hz[i])
            for k in 0 ..< bins {
                let f = Double(k) * binWidth
                let lower = -(hz[i] - f) / (hz[i + 1] - hz[i])
                let upper = (hz[i + 2] - f) / (hz[i + 2] - hz[i + 1])
                let triangle = Float(Swift.max(0, Swift.min(lower, upper)))
                weights[i * bins + k] = Float(Double(triangle) * norm)
            }
        }
        return weights
    }

    /// A waveform `[N]` -> features `[1, N / 160 + 1, melBins]` and the number of valid frames.
    func callAsFunction(_ waveform: MLXArray) -> (features: MLXArray, validFrames: Int) {
        let count = waveform.dim(0)
        let validFrames = count / Self.hopLength
        let frameCount = validFrames + 1

        let x = waveform.asType(.float32)
        let emphasized = concatenated([x[..<1], x[1...] - Self.preemphasis * x[..<(count - 1)]])
        let half = Self.fftSize / 2
        let signal = padded(emphasized, widths: [IntOrPair((half, half))])
        let frames = asStrided(signal, [frameCount, Self.fftSize], strides: [Self.hopLength, 1])
        // The default stream, as in mlx-vlm. On a sine sweep, mlx-vlm's CPU and GPU runs give
        // prefixes 1e-2 apart.
        let power = abs(MLXFFT.rfft(frames * window, axis: -1)).square()
        let mel = log(matmul(power, filterbank) + Self.logGuard)

        let valid = mel[..<validFrames]
        let mean = valid.sum(axis: 0) / Float(validFrames)
        let variance =
            (valid - mean).square().sum(axis: 0) / Float(Swift.max(validFrames - 1, 1))
        let keep = (MLXArray(0 ..< frameCount) .< validFrames).asType(.float32)
        let features =
            (mel - mean) / (sqrt(variance) + Self.normalizationEpsilon)
            * keep[0..., .newAxis]
        return (features.expandedDimensions(axis: 0), validFrames)
    }
}

// MARK: - FastConformer

/// `rel_shift`: pads one zero column on the left and re-reads the scores so that column `j` of
/// query `i` holds relative position `T - 1 - i + j`. `[B, H, T, 2T - 1]` in and out.
func d1OmniRelShift(_ scores: MLXArray) -> MLXArray {
    let (b, h, t, p) = (scores.dim(0), scores.dim(1), scores.dim(2), scores.dim(3))
    let shifted = padded(
        scores, widths: [IntOrPair(0), IntOrPair(0), IntOrPair(0), IntOrPair((1, 0))])
    return shifted.reshaped(b, h, p + 1, t)[0..., 0..., 1...].reshaped(b, h, t, p)
}

enum D1OmniConformer {
    /// Three stride-2 stages: 8x subsampling.
    static let subsamplingStages = 3

    /// Output length of a kernel-3, stride-2, padding-1 convolution: `ceil(n / 2)`.
    static func halved(_ n: Int) -> Int { (n + 1) / 2 }

    /// Interleaved sin/cos embeddings for relative positions `T - 1` down to `-(T - 1)`: `[2T - 1, D]`.
    static func relativePositions(count: Int, dimensions: Int) -> MLXArray {
        let positions = MLXArray((0 ..< (2 * count - 1)).map { Float(count - 1 - $0) })
        let inverse = MLXArray(
            stride(from: 0, to: dimensions, by: 2).map {
                Float(Foundation.exp(Double($0) * -(Foundation.log(10_000.0) / Double(dimensions))))
            })
        let angles = positions[0..., .newAxis] * inverse[.newAxis, 0...]
        return stacked([sin(angles), cos(angles)], axis: -1).reshaped(2 * count - 1, dimensions)
    }

    /// dw_striding subsampling: a conv, then depthwise-separable convs, all with time masks.
    final class Subsampling: Module {
        @ModuleInfo var layers: [UnaryLayer]
        @ModuleInfo var linear: Linear
        /// Stride of each layer, 0 for activations.
        let strides: [Int]

        init(_ config: D1OmniAudioConfiguration) {
            let channels = config.subsamplingConvChannels
            var layers: [UnaryLayer] = [
                Conv2d(
                    inputChannels: 1, outputChannels: channels, kernelSize: 3, stride: 2,
                    padding: 1),
                ReLU(),
            ]
            var strides = [2, 0]
            for _ in 1 ..< D1OmniConformer.subsamplingStages {
                layers += [
                    Conv2d(
                        inputChannels: channels, outputChannels: channels, kernelSize: 3,
                        stride: 2, padding: 1, groups: channels),
                    Conv2d(inputChannels: channels, outputChannels: channels, kernelSize: 1),
                    ReLU(),
                ]
                strides += [2, 1, 0]
            }
            var frequencies = config.featIn
            for _ in 0 ..< D1OmniConformer.subsamplingStages {
                frequencies = D1OmniConformer.halved(frequencies)
            }
            self._layers.wrappedValue = layers
            self._linear.wrappedValue = Linear(channels * frequencies, config.dModel)
            self.strides = strides
        }

        /// `[1, T, F]` features -> `[1, T', D]` and the valid length after subsampling.
        func callAsFunction(_ features: MLXArray, validFrames: Int) -> (MLXArray, Int) {
            var h = features[.ellipsis, .newAxis]
            var length = validFrames
            for (layer, stride) in zip(layers, strides) {
                h = layer(h)
                guard stride > 0 else { continue }
                if stride == 2 {
                    length = D1OmniConformer.halved(length)
                }
                let frames = h.dim(1)
                if length < frames {
                    let keep = (MLXArray(0 ..< frames) .< length).asType(h.dtype)
                    h = h * keep.reshaped(1, frames, 1, 1)
                }
            }
            let (b, t, f, c) = (h.dim(0), h.dim(1), h.dim(2), h.dim(3))
            h = h.transposed(0, 1, 3, 2).reshaped(b, t, c * f)
            return (linear(h), length)
        }
    }

    final class FeedForward: Module, UnaryLayer {
        @ModuleInfo var linear1: Linear
        @ModuleInfo var linear2: Linear

        init(dimensions: Int, hiddenDimensions: Int) {
            self._linear1.wrappedValue = Linear(dimensions, hiddenDimensions)
            self._linear2.wrappedValue = Linear(hiddenDimensions, dimensions)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            linear2(silu(linear1(x)))
        }
    }

    /// Multi-head self-attention with Transformer-XL relative positions (`bias_u`, `bias_v`).
    final class Attention: Module {
        let headCount: Int
        let headDimensions: Int
        let scale: Float

        @ModuleInfo(key: "q_proj") var qProj: Linear
        @ModuleInfo(key: "k_proj") var kProj: Linear
        @ModuleInfo(key: "v_proj") var vProj: Linear
        @ModuleInfo(key: "o_proj") var oProj: Linear
        @ModuleInfo(key: "relative_k_proj") var relativeKProj: Linear
        @ParameterInfo(key: "bias_u") var biasU: MLXArray
        @ParameterInfo(key: "bias_v") var biasV: MLXArray

        init(dimensions: Int, headCount: Int) {
            self.headCount = headCount
            self.headDimensions = dimensions / headCount
            self.scale = 1 / Float(headDimensions).squareRoot()
            self._qProj.wrappedValue = Linear(dimensions, dimensions)
            self._kProj.wrappedValue = Linear(dimensions, dimensions)
            self._vProj.wrappedValue = Linear(dimensions, dimensions)
            self._oProj.wrappedValue = Linear(dimensions, dimensions)
            self._relativeKProj.wrappedValue = Linear(dimensions, dimensions, bias: false)
            self._biasU.wrappedValue = MLXArray.zeros([headCount, headDimensions])
            self._biasV.wrappedValue = MLXArray.zeros([headCount, headDimensions])
        }

        /// `valid` marks the real frames `[T]`; nil when every frame is real.
        func callAsFunction(_ x: MLXArray, positions: MLXArray, valid: MLXArray?) -> MLXArray {
            let (b, t) = (x.dim(0), x.dim(1))
            func heads(_ y: MLXArray) -> MLXArray {
                y.reshaped(b, t, headCount, headDimensions).transposed(0, 2, 1, 3)
            }
            let q = heads(qProj(x))
            let k = heads(kProj(x))
            let v = heads(vProj(x))
            let p = relativeKProj(positions).reshaped(1, -1, headCount, headDimensions)
                .transposed(0, 2, 3, 1)

            let u = biasU.reshaped(1, headCount, 1, headDimensions)
            let w = biasV.reshaped(1, headCount, 1, headDimensions)
            var bd = d1OmniRelShift(matmul(q + w, p))[.ellipsis, ..<t] * scale
            if let valid {
                // Mask keys only; padded query rows are zeroed below.
                bd = MLX.where(
                    valid.reshaped(1, 1, 1, t), bd, -bd.dtype.greatestFiniteMagnitudeArray)
            }
            var out = MLXFast.scaledDotProductAttention(
                queries: q + u, keys: k, values: v, scale: scale, mask: .array(bd))
            if let valid {
                out = out * valid.reshaped(1, 1, t, 1).asType(out.dtype)
            }
            return oProj(out.transposed(0, 2, 1, 3).reshaped(b, t, -1))
        }
    }

    /// Pointwise conv, GLU, depthwise conv, BatchNorm (running statistics in eval mode), SiLU, pointwise conv.
    final class ConvolutionModule: Module {
        @ModuleInfo(key: "pointwise_conv1") var pointwiseConv1: Conv1d
        @ModuleInfo(key: "depthwise_conv") var depthwiseConv: Conv1d
        @ModuleInfo var norm: BatchNorm
        @ModuleInfo(key: "pointwise_conv2") var pointwiseConv2: Conv1d

        init(dimensions: Int, kernelSize: Int) {
            self._pointwiseConv1.wrappedValue = Conv1d(
                inputChannels: dimensions, outputChannels: 2 * dimensions, kernelSize: 1)
            self._depthwiseConv.wrappedValue = Conv1d(
                inputChannels: dimensions, outputChannels: dimensions, kernelSize: kernelSize,
                padding: (kernelSize - 1) / 2, groups: dimensions)
            self._norm.wrappedValue = BatchNorm(featureCount: dimensions)
            self._pointwiseConv2.wrappedValue = Conv1d(
                inputChannels: dimensions, outputChannels: dimensions, kernelSize: 1)
        }

        func callAsFunction(_ x: MLXArray, valid: MLXArray?) -> MLXArray {
            var h = glu(pointwiseConv1(x), axis: -1)
            if let valid {
                h = h * valid.reshaped(1, -1, 1).asType(h.dtype)
            }
            return pointwiseConv2(silu(norm(depthwiseConv(h))))
        }
    }

    final class Block: Module {
        @ModuleInfo(key: "feed_forward1") var feedForward1: FeedForward
        @ModuleInfo(key: "self_attn") var selfAttention: Attention
        @ModuleInfo var conv: ConvolutionModule
        @ModuleInfo(key: "feed_forward2") var feedForward2: FeedForward
        @ModuleInfo(key: "norm_feed_forward1") var normFeedForward1: LayerNorm
        @ModuleInfo(key: "norm_self_att") var normSelfAttention: LayerNorm
        @ModuleInfo(key: "norm_conv") var normConv: LayerNorm
        @ModuleInfo(key: "norm_feed_forward2") var normFeedForward2: LayerNorm
        @ModuleInfo(key: "norm_out") var normOut: LayerNorm

        init(_ config: D1OmniAudioConfiguration) {
            let d = config.dModel
            let hidden = d * config.ffExpansionFactor
            self._feedForward1.wrappedValue = FeedForward(dimensions: d, hiddenDimensions: hidden)
            self._selfAttention.wrappedValue = Attention(dimensions: d, headCount: config.headCount)
            self._conv.wrappedValue = ConvolutionModule(
                dimensions: d, kernelSize: config.convKernelSize)
            self._feedForward2.wrappedValue = FeedForward(dimensions: d, hiddenDimensions: hidden)
            self._normFeedForward1.wrappedValue = LayerNorm(dimensions: d)
            self._normSelfAttention.wrappedValue = LayerNorm(dimensions: d)
            self._normConv.wrappedValue = LayerNorm(dimensions: d)
            self._normFeedForward2.wrappedValue = LayerNorm(dimensions: d)
            self._normOut.wrappedValue = LayerNorm(dimensions: d)
        }

        func callAsFunction(_ x: MLXArray, positions: MLXArray, valid: MLXArray?) -> MLXArray {
            var h = x + 0.5 * feedForward1(normFeedForward1(x))
            h = h + selfAttention(normSelfAttention(h), positions: positions, valid: valid)
            h = h + conv(normConv(h), valid: valid)
            h = h + 0.5 * feedForward2(normFeedForward2(h))
            return normOut(h)
        }
    }

    final class Encoder: Module {
        let dModel: Int
        @ModuleInfo var subsampling: Subsampling
        @ModuleInfo var layers: [Block]

        init(_ config: D1OmniAudioConfiguration) {
            self.dModel = config.dModel
            self._subsampling.wrappedValue = Subsampling(config)
            self._layers.wrappedValue = (0 ..< config.layerCount).map { _ in Block(config) }
        }

        /// `[1, T, F]` features -> `[1, T_enc, D]` and the number of valid frames `P`.
        func callAsFunction(_ features: MLXArray, validFrames: Int) -> (MLXArray, Int) {
            var (h, length) = subsampling(features, validFrames: validFrames)
            let frames = h.dim(1)
            let positions = D1OmniConformer.relativePositions(count: frames, dimensions: dModel)
                .asType(h.dtype)
            let valid: MLXArray? = length < frames ? MLXArray(0 ..< frames) .< length : nil
            for layer in layers {
                h = layer(h, positions: positions, valid: valid)
            }
            return (h, length)
        }
    }
}

// MARK: - Adapter, residual, audio prefix

final class D1OmniAudioAdapter: Module, UnaryLayer {
    @ModuleInfo var norm: LayerNorm
    @ModuleInfo(key: "linear_1") var linear1: Linear
    @ModuleInfo(key: "linear_2") var linear2: Linear

    init(inputDimensions: Int, outputDimensions: Int) {
        self._norm.wrappedValue = LayerNorm(dimensions: inputDimensions)
        self._linear1.wrappedValue = Linear(inputDimensions, outputDimensions)
        self._linear2.wrappedValue = Linear(outputDimensions, outputDimensions)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        linear2(gelu(linear1(norm(x))))
    }
}

final class D1OmniAudioResidual: Module, UnaryLayer {
    @ModuleInfo var ln: LayerNorm
    @ModuleInfo var down: Linear
    @ModuleInfo var up: Linear

    init(dimensions: Int, width: Int) {
        self._ln.wrappedValue = LayerNorm(dimensions: dimensions)
        self._down.wrappedValue = Linear(dimensions, width)
        self._up.wrappedValue = Linear(width, dimensions)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        x + up(gelu(down(ln(x))))
    }
}

/// Speech -> prefix embeddings: log-mel front end, FastConformer, adapter and residual.
/// One embedding per 80 ms of audio.
final class D1OmniAudio: Module {
    let frontend: D1OmniMelFrontend
    @ModuleInfo var encoder: D1OmniConformer.Encoder
    @ModuleInfo var adapter: D1OmniAudioAdapter
    @ModuleInfo var residual: D1OmniAudioResidual

    init(_ config: D1OmniAudioConfiguration, outputDimensions: Int) {
        self.frontend = D1OmniMelFrontend(melBins: config.featIn)
        self._encoder.wrappedValue = D1OmniConformer.Encoder(config)
        self._adapter.wrappedValue = D1OmniAudioAdapter(
            inputDimensions: config.dModel, outputDimensions: outputDimensions)
        self._residual.wrappedValue = D1OmniAudioResidual(
            dimensions: outputDimensions, width: config.residualWidth)
        super.init()
        // Inference only: BatchNorm must read its running statistics.
        train(false)
    }

    /// A waveform from ``d1OmniWaveform(_:)`` -> prefix embeddings `[1, P, outputDimensions]`.
    func callAsFunction(_ waveform: MLXArray) -> MLXArray {
        precondition(
            waveform.ndim == 1 && waveform.dim(0) >= D1OmniMelFrontend.minimumSamples,
            "expected a waveform from d1OmniWaveform")
        let (features, validFrames) = frontend(waveform)
        let dtype = adapter.norm.weight?.dtype ?? .float32
        let (hidden, length) = encoder(features.asType(dtype), validFrames: validFrames)
        return residual(adapter(hidden[0..., ..<length]))
    }
}

// MARK: - Weights

private let d1OmniAudioRenames = [
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

/// Maps the checkpoint's NeMo `audio.*` names and torch conv layout to ``D1OmniAudio`` (Python
/// `sanitize_audio`). Other keys pass through. Idempotent, so converted checkpoints load too.
func d1OmniSanitizeAudio(_ weights: [String: MLXArray]) -> [String: MLXArray] {
    var result = [String: MLXArray]()
    for (key, value) in weights {
        guard key.hasPrefix("audio.") else {
            result[key] = value
            continue
        }
        if key.hasSuffix("num_batches_tracked") {
            continue
        }
        var key = key
        for (old, new) in d1OmniAudioRenames {
            key = key.replacingOccurrences(of: old, with: new)
        }
        // torch (out, in/groups, *kernel) -> MLX (out, *kernel, in/groups). 2-D kernels are
        // square; 1-D convs are pointwise (kernel 1) or depthwise (in/groups 1).
        var value = value
        if value.ndim == 4, value.dim(2) == value.dim(3) {
            value = value.transposed(0, 2, 3, 1)
        } else if value.ndim == 3, value.dim(key.contains("pointwise") ? -1 : 1) == 1 {
            value = value.transposed(0, 2, 1)
        }
        result[key] = value
    }
    return result
}
