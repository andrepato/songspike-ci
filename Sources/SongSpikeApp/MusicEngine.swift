import Foundation

/// Planner quantisation tier — the only quality/size knob the spike
/// exercises. The raw values double as the picker labels' backing and as
/// the tier string written to the run log (SPIKE.md measurement sheet).
enum PlannerTier: String, Sendable, CaseIterable, Identifiable {
    case int8
    case int4

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .int8: return "int8 planner (default)"
        case .int4: return "int4 planner (smaller)"
        }
    }

    /// Bits for the engine's at-load planner quantisation
    /// (ENGINE-NOTES.md: `plannerBits`).
    var plannerBits: Int {
        switch self {
        case .int8: return 8
        case .int4: return 4
        }
    }

    /// Pre-quantised planner bundle for this tier on Hugging Face.
    var plannerRepoID: String {
        switch self {
        case .int8: return "mlx-community/ACE-Step-1.5-planner-1.7B-8bit"
        case .int4: return "mlx-community/ACE-Step-1.5-planner-1.7B-4bit"
        }
    }

    /// Local folder name for this tier's planner = the repo ID's last
    /// component. Verified 2026-11-16 against the engine source: these
    /// are exactly `ACEStepBundles.plannerInt8Name` /
    /// `plannerInt4Name` in xocialize/mlx-ace-step-swift.
    var plannerFolderName: String {
        switch self {
        case .int8: return "ACE-Step-1.5-planner-1.7B-8bit"
        case .int4: return "ACE-Step-1.5-planner-1.7B-4bit"
        }
    }
}

/// One generation request, engine-agnostic.
struct GenerationRequest: Sendable, Equatable {
    var caption: String
    var lyrics: String
    var language: String = "en"
    var durationSeconds: Int
    var seed: Int = 7
}

/// A progress beat from inside the engine. `stage` is the engine's own
/// phase name (plan / codes / denoise / decode on the direct API);
/// `fraction` is set when the engine reports done/total for the phase.
struct EngineProgress: Sendable {
    var stage: String
    var fraction: Double?
}

struct GenerationOutput: Sendable {
    var wavURL: URL
    /// Planner-chosen bpm / key / caption, as a printable summary.
    var metadataSummary: String
}

enum MusicEngineError: Error, CustomStringConvertible {
    /// The engine library is not linked into this build, or the adapter
    /// reached a call site whose exact signature is still unverified.
    case notIntegrated(String)
    case bundlesMissing(URL)

    var description: String {
        switch self {
        case .notIntegrated(let why): return "Engine not integrated: \(why)"
        case .bundlesMissing(let dir): return "Model bundles not found at \(dir.path)"
        }
    }
}

/// Cooperative cancellation flag. The direct engine API polls a throwing
/// `checkpoint` closure before every planner token, DiT step and VAE
/// window (ENGINE-NOTES.md), so a shared flag is the cleanest bridge
/// between the UI's Cancel button and that closure.
final class CancellationFlag: @unchecked Sendable {
    private let lock = NSLock()
    private var value = false

    func cancel() {
        lock.lock()
        value = true
        lock.unlock()
    }

    func reset() {
        lock.lock()
        value = false
        lock.unlock()
    }

    var isCancelled: Bool {
        lock.lock()
        defer { lock.unlock() }
        return value
    }
}

/// The seam between the app and any song engine. The MLX implementation
/// is `MLXACEStepEngine`; everything else in the app talks only to this.
protocol MusicEngine: Sendable {
    /// Generates one song and writes it as a WAV at `outputURL`.
    /// `bundlesDirectory` is the directory the downloader filled with the
    /// four bundle folders for `tier`. `progress` is called from the
    /// generation thread; implementations must treat it as @Sendable.
    func generate(
        _ request: GenerationRequest,
        bundlesDirectory: URL,
        tier: PlannerTier,
        outputURL: URL,
        cancellation: CancellationFlag,
        progress: @escaping @Sendable (EngineProgress) -> Void
    ) async throws -> GenerationOutput
}
