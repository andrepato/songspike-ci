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
        didSet {
            refreshModelsReady()
            downloader.repairScan(tier: tier)
        }
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
    @Published private(set) var downloadSnapshot = DownloadSnapshot.empty
    /// A model file the integrity checks found corrupt (Build 4):
    /// the models section names it and offers Repair, which
    /// re-downloads just that file.
    @Published private(set) var corruptFileName: String?

    /// True while the background download session has a live run.
    var isDownloading: Bool { downloadSnapshot.isActive }

    /// Show the detailed download UI when a download is running, or
    /// when the selected tier has a partial download on disk.
    var showDownloadProgress: Bool {
        let snapshot = downloadSnapshot
        guard snapshot.filesTotal > 0, snapshot.overallFraction < 1 else { return false }
        return snapshot.isActive || snapshot.tierRawValue == tier.rawValue
    }

    // Results
    @Published private(set) var lastOutputURL: URL?
    @Published private(set) var lastMetadataSummary: String?
    @Published private(set) var lastMeasurementRow: String?
    @Published var errorMessage: String?

    let durationChoices = [30, 60, 180]

    private let engine: MusicEngine = MLXACEStepEngine()
    private let downloader = ModelDownloader.shared
    private let cancellation = CancellationFlag()
    private var logger: RunLogger?
    private var hasRunSinceLaunch = false
    private var ticker: Task<Void, Never>?

    init() {
        UIDevice.current.isBatteryMonitoringEnabled = true
        refreshModelsReady()
        sampleMemory()
        wireDownloader()
        // Re-attach to the background session and auto-resume an
        // interrupted download from the persisted state.
        downloader.restore()
        // Build 4: validate what earlier builds called complete —
        // a corrupt file (André's turbo model) is repaired here,
        // before it can poison a generation.
        downloader.repairScan(tier: tier)
    }

    private func wireDownloader() {
        downloader.onSnapshot = { [weak self] snapshot in
            Task { @MainActor in
                guard let self else { return }
                self.downloadSnapshot = snapshot
                // Refresh on every snapshot (Build 4): a repair scan
                // can REMOVE a repo's .complete marker, and readiness
                // must drop the moment it does.
                self.refreshModelsReady()
                if snapshot.status == .complete {
                    self.corruptFileName = nil
                }
            }
        }
        downloader.onFinished = { [weak self] result in
            Task { @MainActor in
                guard let self else { return }
                if case .failure(let error) = result {
                    self.errorMessage = "\(error)"
                }
                self.refreshModelsReady()
            }
        }
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
        errorMessage = nil
        downloader.start(tier: tier)
    }

    /// Re-downloads just the file the integrity checks named as
    /// corrupt (Build 4), through the downloader's normal
    /// storage-gated flow.
    func repairCorruptFile() {
        guard let name = corruptFileName else { return }
        corruptFileName = nil
        errorMessage = nil
        downloader.repairFile(tier: tier, fileName: name)
    }

    /// Cheap pre-generation integrity check (Build 4): structurally
    /// validate every safetensors on disk for the tier and return
    /// the first corrupt file's name. Reads only file headers, so it
    /// costs milliseconds — and it catches exactly the failure that
    /// reached the engine on André's device ("Invalid json header
    /// length") before the engine ever sees the file.
    private static func firstCorruptSafetensors(tier: PlannerTier) -> String? {
        for repo in ModelBundles.repos(for: tier) {
            let directory = ModelBundles.repoDirectory(repo)
            guard let entries = try? FileManager.default.contentsOfDirectory(
                at: directory, includingPropertiesForKeys: nil) else { continue }
            for url in entries.sorted(by: { $0.lastPathComponent < $1.lastPathComponent })
            where url.lastPathComponent.hasSuffix(".safetensors") {
                if !ModelIntegrity.validateSafetensorsHeader(at: url) {
                    return url.lastPathComponent
                }
            }
        }
        return nil
    }

    // MARK: - Generation

    func generate() {
        guard !isGenerating else { return }
        guard modelsReady else {
            errorMessage = "Model bundles for the \(tier.rawValue) tier are not downloaded yet."
            return
        }
        // Build 4 pre-flight: a corrupt model file gets a clear
        // message and a one-file Repair here, not a cryptic engine
        // error after the load starts.
        if let corrupt = Self.firstCorruptSafetensors(tier: tier) {
            corruptFileName = corrupt
            errorMessage = "A model file looks corrupted: \(corrupt). "
                + "Tap Repair to re-download just that file."
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
                // Build 4 safety net: a load failure that names the
                // safetensors layer gets translated into the
                // corrupted-file message + Repair, with the corrupt
                // file identified by the same header check the
                // pre-flight uses.
                if "\(error)".lowercased().contains("safetensors"),
                   let corrupt = Self.firstCorruptSafetensors(tier: tier) {
                    corruptFileName = corrupt
                    errorMessage = "A model file looks corrupted: \(corrupt). "
                        + "Tap Repair to re-download just that file."
                }
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
