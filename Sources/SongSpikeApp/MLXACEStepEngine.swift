import Foundation

#if canImport(ACEStepCore)
import ACEStepCore
#endif

/// Adapter over xocialize/mlx-ace-step-swift's direct `ACEStepCore` API.
///
/// RECONCILED 2026-11-16 against the cloned engine source
/// (xocialize/mlx-ace-step-swift @ fc9e4be, v0.3.0 —
/// Sources/ACEStepCore/Pipeline/ACEStepPipeline.swift, Support/WAV.swift).
/// Every call below now matches the real signatures:
///
/// - `ACEStepPipeline(bundles: ACEStepBundles, dtype: DType?,
///   plannerBits: Int?, …)` (designated init). The bundles value is built
///   explicitly because the downloader stores the text-encoder repo under
///   its HF name (`ACE-Step-1.5-text-encoder-bf16`) while the engine's
///   converter layout calls that folder `Qwen3-Embedding-0.6B-bf16` —
///   explicit URLs sidestep the naming question entirely.
/// - `ACEStepRequest(caption:lyrics:language:duration:seed:)` with
///   `duration: Double?` and `seed: UInt64` (the scaffold had Int for
///   both — would not have compiled).
/// - `pipe.generate(req, progress:checkpoint:)` with
///   `progress: (String, Int, Int) -> Void` and a throwing checkpoint.
/// - `song.audio` is `[Float]` interleaved stereo; `song.channels` /
///   `song.sampleRate` carry the format; `WAV.write16(_:channels:
///   sampleRate:to:)` matches verbatim.
/// - `song.metadata` is `PlannerMetadata` (bpm / keyscale / duration /
///   language / caption, all optional) — formatted field by field below
///   instead of the scaffold's struct dump.
///
/// `req.timestamps` is set to false: LRC timestamps cost an extra DiT
/// cross-attention pass and the spike UI does not display them.
///
/// When the engine package is not linked (build without
/// SONGSPIKE_WITH_ACE=1), this adapter compiles to a stub that throws
/// `notIntegrated`, so the shell, logging and measurement flow can be
/// exercised end to end before the engine is wired in.
struct MLXACEStepEngine: MusicEngine {

    func generate(
        _ request: GenerationRequest,
        bundlesDirectory: URL,
        tier: PlannerTier,
        outputURL: URL,
        cancellation: CancellationFlag,
        progress: @escaping @Sendable (EngineProgress) -> Void
    ) async throws -> GenerationOutput {
        guard FileManager.default.fileExists(atPath: bundlesDirectory.path) else {
            throw MusicEngineError.bundlesMissing(bundlesDirectory)
        }

        #if canImport(ACEStepCore)
        // The engine API is synchronous and throwing, and a generation
        // runs for minutes — keep it off every actor's executor on a
        // detached, user-initiated task. The pipeline is created and
        // used entirely inside that task, so no engine type crosses an
        // isolation boundary.
        return try await Task.detached(priority: .userInitiated) {
            // Explicit bundle URLs into the downloader's folders
            // (folder names = HF repo last components; see
            // ACEStepBundles' own *Name constants for the engine-side
            // equivalents — turbo / planner-8bit / planner-4bit / vae
            // agree, the text encoder does not, hence explicit URLs).
            let bundles = ACEStepBundles(
                turbo: bundlesDirectory.appendingPathComponent("ACE-Step-1.5-turbo-bf16"),
                planner: bundlesDirectory.appendingPathComponent(tier.plannerFolderName),
                textEncoder: bundlesDirectory.appendingPathComponent("ACE-Step-1.5-text-encoder-bf16"),
                vae: bundlesDirectory.appendingPathComponent("ACE-Step-1.5-vae-fp32")
            )

            // plannerBits matches the pre-quantised planner bundle for
            // the tier; the engine validates the match and throws on a
            // mismatch (ACEStepPipeline.init).
            let pipe = try ACEStepPipeline(
                bundles: bundles,
                dtype: .bfloat16,
                plannerBits: tier.plannerBits
            )

            var req = ACEStepRequest(
                caption: request.caption,
                lyrics: request.lyrics,
                language: request.language,
                duration: Double(request.durationSeconds),
                seed: UInt64(request.seed)
            )
            req.timestamps = false

            let song = try pipe.generate(
                req,
                progress: { phase, done, total in
                    progress(EngineProgress(
                        stage: phase,
                        fraction: total > 0 ? Double(done) / Double(total) : nil
                    ))
                },
                checkpoint: {
                    // The engine's own packages pass
                    // Task.checkCancellation here; a detached task would
                    // not observe the UI's cancellation, so the
                    // checkpoint polls the shared flag instead. It runs
                    // before every planner token, DiT step and VAE
                    // window, and the engine rethrows its error
                    // unchanged.
                    if cancellation.isCancelled {
                        throw CancellationError()
                    }
                }
            )

            try WAV.write16(song.audio, channels: song.channels, sampleRate: song.sampleRate, to: outputURL)

            let meta = song.metadata
            let summary = [
                meta.bpm.map { "bpm \($0)" },
                meta.keyscale.map { "key \($0)" },
                meta.duration.map { "\($0)s" },
                "lang \(song.language)",
                meta.caption.map { "caption: \($0)" },
            ].compactMap { $0 }.joined(separator: " · ")
            return GenerationOutput(
                wavURL: outputURL,
                metadataSummary: summary
            )
        }.value
        #else
        throw MusicEngineError.notIntegrated(
            "ACEStepCore is not linked in this build. Build with SONGSPIKE_WITH_ACE=1 "
                + "to resolve xocialize/mlx-ace-step-swift (see Package.swift and ENGINE-NOTES.md)."
        )
        #endif
    }
}
