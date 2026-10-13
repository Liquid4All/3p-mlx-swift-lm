// Copyright © 2026 Apple Inc.

import Foundation
import MLXDecisions
import MLXLMCommon
import Testing

/// Checkpoints and fixtures for the gated real-weight d1 suites, named by environment
/// variables. `tools/d1_fixtures.py` writes the fixture directory; README.md explains how
/// to run the suites.
enum D1Fixtures {
    private static let environment = ProcessInfo.processInfo.environment

    /// The `LiquidAI/d1-3B` snapshot, `MLX_D1_3B_MODEL`.
    static let d1Model = environment["MLX_D1_3B_MODEL"].map { URL(fileURLWithPath: $0) }

    /// The directory `tools/d1_fixtures.py` writes, `MLX_D1_FIXTURES`.
    static let directory = environment["MLX_D1_FIXTURES"].map { URL(fileURLWithPath: $0) }

    /// A fixture or input file in ``directory``. Fails when it is missing.
    static func url(_ name: String) throws -> URL {
        let url = try #require(directory, "Set MLX_D1_FIXTURES").appending(component: name)
        try #require(
            FileManager.default.fileExists(atPath: url.path),
            "\(url.path) is missing; write it with tools/d1_fixtures.py")
        return url
    }

    /// One question of a fixture case, in the order Python asked it.
    struct Question: Decodable {
        var name: String
        var type: String
        var instructions: String
        /// `[label, description or null]`
        var options: [[String?]]?
        var levels: [String]?
        var whenTrue: String?
        var whenFalse: String?

        enum CodingKeys: String, CodingKey {
            case name, type, instructions, options, levels
            case whenTrue = "when_true"
            case whenFalse = "when_false"
        }

        var named: DecisionRequest.NamedQuestion {
            let question: DecisionQuestion =
                switch type {
                case "choice":
                    .choice(
                        instructions,
                        options: (options ?? []).map {
                            DecisionOption($0[0] ?? "", description: $0.count > 1 ? $0[1] : nil)
                        })
                case "score": .score(instructions, levels: levels ?? [])
                default: .bool(instructions, whenTrue: whenTrue, whenFalse: whenFalse)
                }
            return .init(name: name, question: question)
        }
    }

    /// A fixture state: a JSON string is text, as Python reads a `str`.
    static func state(_ value: JSONValue?) -> DecisionState? {
        switch value {
        case nil, .null?: nil
        case .string(let text)?: .text(text)
        case let value?: .json(DecisionJSON(value))
        }
    }
}
