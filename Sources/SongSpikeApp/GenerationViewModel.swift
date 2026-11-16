import Foundation
import UIKit

/// Drives the single spike screen: model download, generation with live
/// stage / elapsed / peak-memory readouts, and run logging.
///
/// Visible progress is a core feature of this app (André's rule: enough
/// visibility to tell a slow load from a stuck one), so the ticker below
/// samples the clock and the memory footprint for the whole run, not
/// just at the end.
@MainActor
final class GenerationViewModel: ObservableObject {

    // Inputs
    @Published var caption: String =
        "Female-fronted symphonic heavy metal, powerful gritty alto lead vocal, "
        + "slight rasp, huge layered chorus harmonies, double-kick drums, "
        + "distorted rhythm guitars, orchestral strings, anthemic, 150 BPM, D minor"
    @Published var lyrics: String =
        "[Verse]\nI was raised where the pine trees bend,\n"
        + "Where the winter writes its name in stone,\n"
        + "Every scar is a map, every storm is a friend,\n"
        + "And I learned how to sing alone.\n\n"
        + "[Chorus]\nRaise the hammer, raise the storm,\n"
        + "I was born where the cold winds form,\n"
        + "Sing for the fallen, sing for the free,\n"
        + "Valhalla is calling me."
    @Published var durationSeconds: Int = 60
    @Published var tier: PlannerTier = .int8 {
        didSet { refreshModelsReady() }
    }

    // Live run state
    @Published private(set) var isGenerating = false
    @Published private(set) var stageLabel = "Idle"
    @Published private(set) var stageFraction: Double?
    @Published private(set) var elapsedSeconds: Double = 0
    @Published private(set) var liveMemoryMB: Double = 0
    @Published private(set) var peakMemoryMB: Double = 0
    @Published private(set) var availableMemoryMB: Double = 0

    // Download state
    @Published private(set) var modelsReady = false
    @Published private(set) var isDownloading = false
    @Published private(set) var downloadFraction: Double = 0

    // Results
    @Published private(set) var lastOutputURL: URL?
    @Published private(set) var lastMetadataSummary: String?
    @Published private(set) var lastMeasurementRow: String?
    @Published var errorMessage: String?

    let durationChoices = [30, 60, 180]

    private let engine: MusicEngine = MLXACEStepEngine()
    private let downloader = ModelDownloader()
    private let cancellation = CancellationFlag()
    private var logger: RunLogger?
    private var hasRunSinceLaunch = false
    private var ticker: Task<Void, Never>?

    init() {
        UIDevice.current.isBatteryMonitoringEnabled = true
        refreshModelsReady()
        sampleMemory()
    }

    // MARK: - Paths

    private var documentsDirectory: URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
    }

    private var outputsDirectory: URL {
        let directory = documentsDirectory.appendingPathComponent("Outputs", isDirectory: true)
        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return directory
    }

    private func runLogger() -> RunLogger {
        if let logger { return logger }
        let created = RunLogger(documentsDirectory: documentsDirectory)
        logger = created
        return created
    }

    // MARK: - Download

    private func refreshModelsReady() {
        modelsReady = ModelBundles.isTierReady(tier)
    }

    func downloadModels() {
        guard !isDownloading else { return }
        isDownloading = true
        downloadFraction = 0
        errorMessage = nil
        let tier = tier
        Task {
            do {
                try await downloader.download(tier: tier) { [weak self] fraction in
                    Task { @MainActor in
                        self?.downloadFraction = fraction
                    }
                }
                refreshModelsReady()
            } catch {
                errorMessage = "\(error)"
            }
            isDownloading = false
        }
    }

    // MARK: - Generation

    func generate() {
        guard !isGenerating else { return }
        guard modelsReady else {
            errorMessage = "Model bundles for the \(tier.rawValue) tier are not downloaded yet."
            return
        }
        isGenerating = true
        errorMessage = nil
        lastOutputURL = nil
        lastMetadataSummary = nil
        stageLabel = "Starting"
        stageFraction = nil
        elapsedSeconds = 0
        peakMemoryMB = 0
        cancellation.reset()

        let request = GenerationRequest(
            caption: caption,
            lyrics: lyrics,
            durationSeconds: durationSeconds
        )
        let tier = tier
        let coldOrWarm = hasRunSinceLaunch ? "warm" : "cold"
        hasRunSinceLaunch = true
        let thermalBefore = ProcessInfo.processInfo.thermalState
        let batteryBefore = Self.batteryPercent()
        let started = Date()
        let outputURL = outputsDirectory.appendingPathComponent(
            "song-\(Self.fileTimestamp(started))-\(tier.rawValue)-\(durationSeconds)s.wav"
        )

        startTicker(started: started)

        Task {
            var outcome = "completed"
            var notes = ""
            var output: GenerationOutput?
            do {
                output = try await engine.generate(
                    request,
                    bundlesDirectory: ModelBundles.modelsDirectory,
                    tier: tier,
                    outputURL: outputURL,
                    cancellation: cancellation,
                    progress: { [weak self] beat in
                        Task { @MainActor in
                            guard let self else { return }
                            self.stageLabel = beat.stage
                            self.stageFraction = beat.fraction
                        }
                    }
                )
            } catch is CancellationError {
                outcome = "cancelled"
                notes = "Cancelled from the UI."
            } catch {
                outcome = "failed"
                notes = "\(error)"
                errorMessage = "\(error)"
            }

            ticker?.cancel()
            sampleMemory()
            let wall = Date().timeIntervalSince(started)
            elapsedSeconds = wall
            stageLabel = outcome == "completed" ? "Done" : outcome.capitalized

            if let output {
                lastOutputURL = output.wavURL
                lastMetadataSummary = output.metadataSummary
            }

            // Log every run — completed, failed or cancelled.
            let logger = runLogger()
            let record = RunRecord(
                id: await logger.nextRunNumber,
                date: started,
                deviceModel: DeviceInfo.modelIdentifier,
                osVersion: DeviceInfo.osVersion,
                tier: tier.rawValue,
                caption: request.caption,
                seed: request.seed,
                durationSeconds: request.durationSeconds,
                coldOrWarm: coldOrWarm,
                wallSeconds: wall,
                peakMemoryMB: peakMemoryMB,
                thermalBefore: DeviceInfo.thermalStateName(thermalBefore),
                thermalAfter: DeviceInfo.thermalStateName(ProcessInfo.processInfo.thermalState),
                batteryBeforePercent: batteryBefore,
                batteryAfterPercent: Self.batteryPercent(),
                outcome: outcome,
                outputFileName: output?.wavURL.lastPathComponent,
                notes: notes
            )
            do {
                try await logger.append(record)
                lastMeasurementRow = RunLogger.measurementRow(record)
            } catch {
                errorMessage = "Run finished but logging failed: \(error)"
            }

            isGenerating = false
        }
    }

    func cancelGeneration() {
        cancellation.cancel()
        stageLabel = "Cancelling…"
    }

    func copyMeasurementRow() {
        guard let row = lastMeasurementRow else { return }
        UIPasteboard.general.string = row
    }

    // MARK: - Ticker / sampling

    private func startTicker(started: Date) {
        ticker?.cancel()
        ticker = Task { [weak self] in
            while !Task.isCancelled {
                guard let self else { return }
                self.elapsedSeconds = Date().timeIntervalSince(started)
                self.sampleMemory()
                try? await Task.sleep(for: .milliseconds(250))
            }
        }
    }

    private func sampleMemory() {
        if let footprint = MemorySampler.footprintBytes() {
            let mb = MemorySampler.megabytes(footprint)
            liveMemoryMB = mb
            peakMemoryMB = max(peakMemoryMB, mb)
        }
        availableMemoryMB = MemorySampler.megabytes(MemorySampler.availableBytes())
    }

    // MARK: - Helpers

    private static func batteryPercent() -> Int? {
        let level = UIDevice.current.batteryLevel
        guard level >= 0 else { return nil }
        return Int((level * 100).rounded())
    }

    private static func fileTimestamp(_ date: Date) -> String {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyyMMdd-HHmmss"
        return formatter.string(from: date)
    }
}
