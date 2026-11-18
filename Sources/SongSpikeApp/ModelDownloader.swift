import Foundation

/// One Hugging Face bundle repo the engine loads. Every tier needs four:
/// the turbo DiT, one planner, the text encoder and the VAE
/// (ENGINE-NOTES.md, "Bundles").
struct BundleRepo: Sendable, Equatable {
    var repoID: String
    /// Approximate download size from the HF collection page. Kept for
    /// reference only — progress display uses the real byte sizes from
    /// the HF blobs API, never this estimate.
    var approxMegabytes: Double

    /// Local folder name = last path component of the repo ID.
    /// Verified 2026-11-16 against the engine source: the pipeline
    /// never looks bundles up by folder name — the adapter passes
    /// explicit per-bundle URLs (see MLXACEStepEngine), so the
    /// downloader's repo-name folders are fine as-is, including the
    /// text encoder, whose engine-side converter name
    /// (`Qwen3-Embedding-0.6B-bf16`) differs from its repo name.
    var folderName: String {
        repoID.split(separator: "/").last.map(String.init) ?? repoID
    }

    /// Plain-words label for the download UI ("what is downloading").
    var displayName: String {
        if repoID.contains("turbo") { return "Turbo model" }
        if repoID.contains("planner") { return "Planner" }
        if repoID.contains("text-encoder") { return "Text encoder" }
        if repoID.contains("vae") { return "VAE" }
        return folderName
    }
}

enum ModelBundles {
    static func repos(for tier: PlannerTier) -> [BundleRepo] {
        [
            BundleRepo(repoID: "mlx-community/ACE-Step-1.5-turbo-bf16", approxMegabytes: 4_500),
            BundleRepo(repoID: tier.plannerRepoID, approxMegabytes: tier == .int8 ? 2_400 : 1_700),
            BundleRepo(repoID: "mlx-community/ACE-Step-1.5-text-encoder-bf16", approxMegabytes: 1_100),
            BundleRepo(repoID: "mlx-community/ACE-Step-1.5-vae-fp32", approxMegabytes: 600),
        ]
    }

    /// Application Support/Models — the `bundlesDirectory` handed to the
    /// engine adapter. One subfolder per repo; both tiers share the
    /// turbo / text-encoder / VAE folders, planners sit side by side.
    /// The adapter resolves the four per-bundle URLs from this root
    /// (verified against the engine source 2026-11-16).
    static var modelsDirectory: URL {
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        return base.appendingPathComponent("Models", isDirectory: true)
    }

    static func repoDirectory(_ repo: BundleRepo) -> URL {
        modelsDirectory.appendingPathComponent(repo.folderName, isDirectory: true)
    }

    static func isDownloaded(_ repo: BundleRepo) -> Bool {
        FileManager.default.fileExists(
            atPath: repoDirectory(repo).appendingPathComponent(".complete").path
        )
    }

    static func isTierReady(_ tier: PlannerTier) -> Bool {
        repos(for: tier).allSatisfy(isDownloaded)
    }
}

enum ModelDownloaderError: Error, CustomStringConvertible {
    case fileListFailed(repoID: String, underlying: String)
    case downloadFailed(file: String, underlying: String)
    /// The device cannot hold the rest of the download. A first-class,
    /// user-actionable state — never a silent stall (Build 3).
    case insufficientSpace(neededBytes: Int64, availableBytes: Int64)
    /// No byte progress for over a minute while a file was supposedly
    /// downloading (Build 3 watchdog).
    case stalled(file: String)

    var description: String {
        switch self {
        case .fileListFailed(let repoID, let why):
            return "Could not list files for \(repoID): \(why)"
        case .downloadFailed(let file, let why):
            return "Download failed for \(file): \(why)"
        case .insufficientSpace(let needed, let available):
            let format = { (bytes: Int64) in
                ByteCountFormatter.string(fromByteCount: bytes, countStyle: .file)
            }
            return "Not enough free space — this download needs \(format(needed)) more, "
                + "but only \(format(available)) is free. Free up space and try again."
        case .stalled(let file):
            return "Download stalled on \(file) — no progress for over a minute. "
                + "Check the connection and tap Resume to continue."
        }
    }
}

// MARK: - Persistent download state

/// On-disk record of tier downloads, kept in Application Support next to
/// the bundles (`Models/download-state.json`) so a download survives
/// the app being backgrounded, terminated or relaunched. Written when
/// the plan is built, as files complete, and whenever resume data is
/// captured. The per-repo `.complete` markers keep their Build 1
/// meaning and are written from this state.
///
/// Build 3 keeps this format byte-compatible with Build 2 — André's
/// on-device state (turbo repo complete, planner part-done) is read
/// as-is, nothing already downloaded is fetched again.
struct DownloadStateFile: Codable, Sendable {
    /// Tier of the most recently started download — the one `restore()`
    /// auto-resumes on launch.
    var activeTier: String?
    var tiers: [String: TierDownloadState] = [:]
}

struct TierDownloadState: Codable, Sendable {
    var repos: [RepoDownloadState] = []
}

struct RepoDownloadState: Codable, Sendable {
    var repoID: String
    var files: [FileDownloadState] = []
}

struct FileDownloadState: Codable, Sendable {
    var name: String
    /// Real byte size from the HF blobs API; -1 while unknown.
    var size: Int64 = -1
    var completed: Bool = false
    var bytesDownloaded: Int64 = 0
    /// Resume data captured when the file's task was cancelled or
    /// failed (base64 in the JSON via Data's default Codable). A file
    /// with resume data resumes mid-file; without it, it restarts.
    var resumeData: Data?
}

// MARK: - UI snapshot

/// Truthful run state for the UI (Build 3). The models section renders
/// its headline from this alone, so it can never say "Downloading"
/// while nothing is actually running — the Build 2 screenshot showed
/// "Downloading bundles…" and a Resume button at the same time because
/// the headline ignored whether the run was alive.
enum DownloadStatus: Sendable, Equatable {
    case idle
    case downloading
    /// Stopped with a partial download on disk; Resume continues it.
    case paused
    /// Stopped on an error; the message says why.
    case failed(String)
    /// Stopped before (or during) the download because the device
    /// cannot hold the remaining bytes.
    case insufficientSpace(neededBytes: Int64, availableBytes: Int64)
    case complete
}

/// Immutable progress snapshot for the UI, rebuilt from the persisted
/// state plus the live task counters on every delegate callback.
struct DownloadSnapshot: Sendable, Equatable {
    var tierRawValue: String = ""
    var isActive: Bool = false
    var status: DownloadStatus = .idle
    /// Friendly name of the bundle currently downloading.
    var repoLabel: String?
    var fileName: String?
    var fileFraction: Double = 0
    var overallFraction: Double = 0
    var bytesDownloaded: Int64 = 0
    var bytesTotal: Int64 = 0
    var filesCompleted: Int = 0
    var filesTotal: Int = 0
    /// Free space on the models volume, refreshed while a run is
    /// active (Build 3); -1 while unknown.
    var availableCapacityBytes: Int64 = -1
    /// A partial download exists for this tier and no task is running
    /// (e.g. the last attempt stopped on an error).
    var hasResumableState: Bool = false

    /// True when the run stopped on an error (status carries why).
    var isFailed: Bool {
        if case .failed = status { return true }
        return false
    }

    static let empty = DownloadSnapshot()
}

// MARK: - Downloader

/// Downloads run on a BACKGROUND URLSession
/// (`com.andrepato.songspike.downloads`), so transfers continue while
/// the app is backgrounded or terminated and their events are delivered
/// on the next launch. Files download one at a time in plan order —
/// every sibling of the four repos — with real byte sizes from the HF
/// blobs API, resume data captured into the persisted state on every
/// interruption, and `restore()` re-attaching / auto-resuming on
/// launch.
///
/// Build 3 adds, after André's iPad ran OUT OF STORAGE mid-download
/// and the stall looked like a hang:
/// - a storage pre-flight gate (before a run starts, before the
///   launch auto-resume proceeds, and before every file): remaining
///   bytes for the tier vs the volume's available capacity for
///   important usage, with a safety margin. Insufficient space stops
///   the run in a dedicated, specific state — it never starts.
/// - out-of-space write/completion errors are classified and routed
///   to that same visible state instead of a generic failure.
/// - a watchdog: no byte progress for ~60 s while a file is in flight
///   cancels the task and stops the run with a visible "stalled"
///   reason instead of hanging forever.
/// - truthful UI state via `DownloadSnapshot.status`, and free-space
///   reporting in the snapshot while downloading.
///
/// Concurrency: all mutable state is guarded by `lock`; delegate
/// callbacks arrive on a serial queue; the UI handlers are @Sendable
/// closures invoked outside the lock. Delegate callbacks are matched to
/// files state-first (current task, then URL, then — for finishes —
/// the active tier's first incomplete file, which is airtight because
/// the plan is strictly one task at a time), so completions delivered
/// after a relaunch still land their files.
final class ModelDownloader: NSObject, URLSessionDownloadDelegate, @unchecked Sendable {

    static let shared = ModelDownloader()
    static let backgroundSessionIdentifier = "com.andrepato.songspike.downloads"

    /// Headroom kept free beyond the bytes still to download, so a
    /// run that passes the gate can actually finish (temp-file churn,
    /// other apps writing, rounding).
    static let spaceSafetyMargin: Int64 = 512 * 1024 * 1024

    /// Progress updates, invoked whenever anything changes.
    var onSnapshot: (@Sendable (DownloadSnapshot) -> Void)? {
        get { lock.lock(); defer { lock.unlock() }; return snapshotHandler }
        set { lock.lock(); snapshotHandler = newValue; lock.unlock() }
    }

    /// Invoked once when a download run finishes or stops on an error.
    var onFinished: (@Sendable (Result<Void, Error>) -> Void)? {
        get { lock.lock(); defer { lock.unlock() }; return finishedHandler }
        set { lock.lock(); finishedHandler = newValue; lock.unlock() }
    }

    private let lock = NSLock()
    private var snapshotHandler: (@Sendable (DownloadSnapshot) -> Void)?
    private var finishedHandler: (@Sendable (Result<Void, Error>) -> Void)?
    private var session: URLSession?
    private var persisted: DownloadStateFile
    private var running = false
    private var currentTier: PlannerTier?
    private var currentRepoID: String?
    private var currentFileName: String?
    private var currentTaskIdentifier: Int?
    private var currentDownloadTask: URLSessionDownloadTask?
    private var backgroundCompletionHandler: (() -> Void)?
    private var lastPersistAt = Date.distantPast
    // Build 3 state
    private var lastFailureMessage: String?
    private var spaceBlock: (needed: Int64, available: Int64)?
    private var lastProgressAt = Date.distantPast
    private var hasBegunFile = false
    private var watchdogCancelPending = false
    private var watchdogTimer: DispatchSourceTimer?
    private var cachedAvailableCapacity: Int64 = -1
    private var lastCapacityCheckAt = Date.distantPast

    private let watchdogQueue = DispatchQueue(label: "com.andrepato.songspike.watchdog")

    override init() {
        persisted = Self.loadPersistedState() ?? DownloadStateFile()
        super.init()
    }

    // MARK: - Public entry points

    /// Starts (or resumes) the download for `tier`. Safe to call while
    /// a download is already running — it just re-emits the snapshot.
    func start(tier: PlannerTier) {
        let session = ensureSession()
        lock.lock()
        if running {
            lock.unlock()
            emitSnapshot()
            return
        }
        running = true
        currentTier = tier
        lastFailureMessage = nil
        spaceBlock = nil
        lastProgressAt = Date()
        persisted.activeTier = tier.rawValue
        persistLocked()
        lock.unlock()
        emitSnapshot()
        Task {
            do {
                try await self.buildPlan(tier: tier)
            } catch {
                self.finishRun(with: .failure(error))
                return
            }
            self.proceedWithRun(session: session, notify: true)
        }
    }

    /// Launch-time restore: re-attaches to the background session and,
    /// if the persisted state shows an incomplete download for the
    /// active tier, resumes it without the user re-tapping Download —
    /// adopting a still-running background task when one exists. The
    /// storage gate applies here too: if the device cannot hold the
    /// remaining bytes, the resume does not start and the snapshot
    /// carries the specific not-enough-space state instead.
    func restore() {
        let session = ensureSession()
        lock.lock()
        guard let activeRaw = persisted.activeTier,
              let tier = PlannerTier(rawValue: activeRaw),
              !running,
              let tierState = persisted.tiers[activeRaw],
              tierState.repos.contains(where: { $0.files.contains { !$0.completed } })
        else {
            lock.unlock()
            emitSnapshot()
            return
        }
        running = true
        currentTier = tier
        lastProgressAt = Date()
        lock.unlock()
        emitSnapshot()
        Task {
            // Refresh the plan if the app died mid-plan; buildPlan is
            // idempotent and keeps stored per-file state. Offline is
            // fine — the persisted plan is enough to resume its files.
            try? await self.buildPlan(tier: tier)
            self.proceedWithRun(session: session, notify: false)
        }
    }

    /// Stores the handler iOS passes to
    /// `application(_:handleEventsForBackgroundURLSession:completionHandler:)`;
    /// `urlSessionDidFinishEvents` invokes it once the system has
    /// delivered every pending event for the background session.
    func setBackgroundCompletionHandler(_ handler: @escaping () -> Void, for identifier: String) {
        guard identifier == Self.backgroundSessionIdentifier else {
            handler()
            return
        }
        _ = ensureSession()
        lock.lock()
        backgroundCompletionHandler = handler
        lock.unlock()
    }

    // MARK: - Session

    private func ensureSession() -> URLSession {
        lock.lock()
        defer { lock.unlock() }
        if let session { return session }
        let configuration: URLSessionConfiguration
        #if os(iOS)
        configuration = URLSessionConfiguration.background(
            withIdentifier: Self.backgroundSessionIdentifier)
        configuration.sessionSendsLaunchEvents = true
        configuration.isDiscretionary = false
        #else
        configuration = URLSessionConfiguration.default
        #endif
        #if os(iOS)
        // Read-write in the Apple SDK; swift-corelibs-foundation exposes
        // it get-only, and this session only means anything on iOS.
        configuration.waitsForConnectivity = true
        #endif
        let queue = OperationQueue()
        queue.maxConcurrentOperationCount = 1
        queue.name = "com.andrepato.songspike.downloads.delegate"
        let created = URLSession(configuration: configuration, delegate: self, delegateQueue: queue)
        session = created
        return created
    }

    // MARK: - Storage gate (Build 3)

    /// Free space on the models volume as iOS reports it for important
    /// usage — this counts purgeable space the system can reclaim,
    /// which is the number that decides whether writes will succeed.
    static func availableCapacity() -> Int64? {
        let directory = ModelBundles.modelsDirectory
        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        #if os(iOS)
        if let values = try? directory.resourceValues(
            forKeys: [.volumeAvailableCapacityForImportantUsageKey]),
           let capacity = values.volumeAvailableCapacityForImportantUsage {
            return capacity
        }
        #endif
        if let values = try? directory.resourceValues(
            forKeys: [.volumeAvailableCapacityKey]),
           let capacity = values.volumeAvailableCapacity {
            return Int64(capacity)
        }
        return nil
    }

    /// Bytes still to fetch for the tier: sum over incomplete files of
    /// (size − already downloaded). Files with unknown size count as 0
    /// (in practice only tiny metadata lacks a blobs-API size).
    /// Call with lock held.
    private func remainingBytesLocked(tier: PlannerTier) -> Int64 {
        guard let tierState = persisted.tiers[tier.rawValue] else { return 0 }
        var remaining: Int64 = 0
        for repoState in tierState.repos {
            for file in repoState.files where !file.completed && file.size > 0 {
                remaining += max(0, file.size - file.bytesDownloaded)
            }
        }
        return remaining
    }

    /// Self-locking variant for callers that don't hold the lock.
    private func remainingBytes(tier: PlannerTier) -> Int64 {
        lock.lock()
        defer { lock.unlock() }
        return remainingBytesLocked(tier: tier)
    }

    /// The pre-flight check. Returns true when the run may proceed.
    /// On insufficient space the run is stopped in the dedicated
    /// space-blocked state (visible, specific, resumable after the
    /// user frees space) and false is returned. When capacity cannot
    /// be measured at all, the run is allowed through — the write
    /// errors, if any, are classified when they happen.
    private func checkSpaceGate(tier: PlannerTier, notify: Bool) -> Bool {
        guard let available = Self.availableCapacity() else { return true }
        let remaining = remainingBytes(tier: tier)
        if available >= remaining + Self.spaceSafetyMargin {
            lock.lock()
            spaceBlock = nil
            lock.unlock()
            return true
        }
        blockRunForInsufficientSpace(neededBytes: remaining, availableBytes: available, notify: notify)
        return false
    }

    /// Stops the run in the insufficient-space state. `notify` also
    /// fires `onFinished` with the specific error (user-initiated
    /// attempts); the launch auto-resume passes false so the message
    /// lives in the models section without an extra error banner.
    private func blockRunForInsufficientSpace(neededBytes: Int64, availableBytes: Int64, notify: Bool) {
        lock.lock()
        let wasRunning = running
        running = false
        currentTaskIdentifier = nil
        currentDownloadTask = nil
        currentRepoID = nil
        currentFileName = nil
        hasBegunFile = false
        spaceBlock = (needed: neededBytes, available: availableBytes)
        lastFailureMessage = nil
        cachedAvailableCapacity = availableBytes
        lastCapacityCheckAt = Date()
        let handler = finishedHandler
        lock.unlock()
        stopWatchdog()
        emitSnapshot()
        if notify && wasRunning {
            handler?(.failure(ModelDownloaderError.insufficientSpace(
                neededBytes: neededBytes, availableBytes: availableBytes)))
        }
    }

    /// After the plan is built: storage gate first, then advance.
    private func proceedWithRun(session: URLSession, notify: Bool) {
        lock.lock()
        guard running, let tier = currentTier else {
            lock.unlock()
            return
        }
        lock.unlock()
        guard checkSpaceGate(tier: tier, notify: notify) else { return }
        startWatchdog()
        reconcileAndAdvance(session: session)
    }

    /// True when an error is a write that failed for lack of space —
    /// POSIX ENOSPC under Cocoa/URL error wrappers (Build 3 routing:
    /// these become the insufficient-space state, not a generic
    /// failure and never a silent stall).
    static func isOutOfSpaceError(_ error: Error) -> Bool {
        let nsError = error as NSError
        if nsError.domain == NSPOSIXErrorDomain && nsError.code == 28 { // ENOSPC
            return true
        }
        if nsError.domain == NSCocoaErrorDomain
            && nsError.code == CocoaError.Code.fileWriteOutOfSpace.rawValue {
            return true
        }
        if let underlying = nsError.userInfo[NSUnderlyingErrorKey] as? NSError {
            return isOutOfSpaceError(underlying)
        }
        return false
    }

    // MARK: - Watchdog (Build 3)

    private func startWatchdog() {
        lock.lock()
        if watchdogTimer != nil {
            lock.unlock()
            return
        }
        let timer = DispatchSource.makeTimerSource(queue: watchdogQueue)
        timer.schedule(deadline: .now() + 5, repeating: .seconds(5))
        timer.setEventHandler { [weak self] in
            self?.watchdogTick()
        }
        watchdogTimer = timer
        lock.unlock()
        timer.resume()
    }

    private func stopWatchdog() {
        lock.lock()
        let timer = watchdogTimer
        watchdogTimer = nil
        watchdogCancelPending = false
        lock.unlock()
        timer?.cancel()
    }

    /// Every 5 s while a run is active: if a file has been in flight
    /// for over a minute with zero byte progress, the transfer is
    /// dead — cancel it and stop the run with a visible "stalled"
    /// reason (the cancellation's error callback completes the stop).
    /// If no file is in flight but the run has made no progress for
    /// over a minute after files were begun (a dropped callback),
    /// stop the same way. Either way the UI never sits on a fake
    /// "Downloading" forever.
    private func watchdogTick() {
        lock.lock()
        guard running else {
            lock.unlock()
            return
        }
        let idle = Date().timeIntervalSince(lastProgressAt)
        guard idle > 60, hasBegunFile else {
            lock.unlock()
            return
        }
        if let task = currentDownloadTask {
            watchdogCancelPending = true
            lock.unlock()
            task.cancel()
            return
        }
        let fileName = currentFileName ?? "the current file"
        lock.unlock()
        finishRun(with: .failure(ModelDownloaderError.stalled(file: fileName)))
    }

    // MARK: - Plan

    /// Builds or refreshes the per-file plan for a tier from the HF
    /// blobs API (real byte sizes), merging with stored state so
    /// completed flags and resume data survive, and reconciling with
    /// what is actually on disk (this is also the Build 1 migration:
    /// files Build 1 completed are recognised by size and not fetched
    /// again).
    private func buildPlan(tier: PlannerTier) async throws {
        for repo in ModelBundles.repos(for: tier) {
            let fetched: [FileDownloadState]
            do {
                fetched = try await fetchFileList(repoID: repo.repoID)
            } catch {
                if hasStoredRepo(tier: tier, repoID: repo.repoID) { continue }
                throw error
            }

            let stored = storedFiles(tier: tier, repoID: repo.repoID)

            var merged: [FileDownloadState] = fetched.map { fresh in
                if var existing = stored.first(where: { $0.name == fresh.name }) {
                    existing.size = fresh.size
                    return existing
                }
                return fresh
            }

            let directory = ModelBundles.repoDirectory(repo)
            let repoComplete = ModelBundles.isDownloaded(repo)
            for index in merged.indices {
                let destination = directory.appendingPathComponent(merged[index].name)
                if repoComplete {
                    merged[index].completed = true
                    merged[index].resumeData = nil
                    if merged[index].size > 0 {
                        merged[index].bytesDownloaded = merged[index].size
                    }
                } else if !merged[index].completed,
                          let diskSize = Self.fileSize(at: destination),
                          diskSize > 0,
                          merged[index].size <= 0 || diskSize == merged[index].size {
                    merged[index].completed = true
                    merged[index].bytesDownloaded = diskSize
                    merged[index].resumeData = nil
                }
            }

            storePlan(tier: tier, repoID: repo.repoID, files: merged)
            emitSnapshot()
        }
        refreshMarkers(tier: tier)
    }

    /// Synchronous locked accessors — `buildPlan` is async, and
    /// NSLock's lock/unlock are unavailable from async contexts.

    private func hasStoredRepo(tier: PlannerTier, repoID: String) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        return persisted.tiers[tier.rawValue]?.repos
            .contains { $0.repoID == repoID } ?? false
    }

    private func storedFiles(tier: PlannerTier, repoID: String) -> [FileDownloadState] {
        lock.lock()
        defer { lock.unlock() }
        return persisted.tiers[tier.rawValue]?.repos
            .first { $0.repoID == repoID }?.files ?? []
    }

    private func storePlan(tier: PlannerTier, repoID: String, files: [FileDownloadState]) {
        lock.lock()
        var tierState = persisted.tiers[tier.rawValue] ?? TierDownloadState()
        if let existing = tierState.repos.firstIndex(where: { $0.repoID == repoID }) {
            tierState.repos[existing].files = files
        } else {
            tierState.repos.append(RepoDownloadState(repoID: repoID, files: files))
        }
        persisted.tiers[tier.rawValue] = tierState
        // Plan building is run progress too — don't let the watchdog
        // mistake a slow plan fetch for a stalled transfer.
        lastProgressAt = Date()
        persistLocked()
        lock.unlock()
    }

    private struct HFModelInfo: Decodable {
        struct Sibling: Decodable {
            let rfilename: String
            let size: Int64?
        }
        let siblings: [Sibling]
    }

    /// File list WITH byte sizes (`?blobs=true` adds `size` to every
    /// sibling). Same file set the Build 1 downloader fetched: all
    /// siblings — turbo: model.safetensors, config.json,
    /// silence_latent.safetensors; planners + text encoder:
    /// + tokenizer.json; VAE: model + config; plus each repo's small
    /// metadata files.
    private func fetchFileList(repoID: String) async throws -> [FileDownloadState] {
        guard let url = URL(string: "https://huggingface.co/api/models/\(repoID)?blobs=true") else {
            throw ModelDownloaderError.fileListFailed(repoID: repoID, underlying: "bad URL")
        }
        do {
            let (data, _) = try await URLSession.shared.data(from: url)
            let info = try JSONDecoder().decode(HFModelInfo.self, from: data)
            return info.siblings.map {
                FileDownloadState(name: $0.rfilename, size: $0.size ?? -1)
            }
        } catch {
            throw ModelDownloaderError.fileListFailed(repoID: repoID, underlying: "\(error)")
        }
    }

    // MARK: - Run control

    /// Adopts a still-running background task for the next file when
    /// one exists (the relaunch case); otherwise creates the next task.
    private func reconcileAndAdvance(session: URLSession) {
        session.getAllTasks { [weak self] tasks in
            guard let self else { return }
            self.lock.lock()
            guard self.running, let tier = self.currentTier else {
                self.lock.unlock()
                return
            }
            guard let next = self.nextIncompleteLocked(tier: tier) else {
                self.lock.unlock()
                self.finishRun(with: .success(()))
                return
            }
            let expectedURL = Self.downloadURL(repoID: next.repo.repoID, file: next.file.name)
            let downloads = tasks.compactMap { $0 as? URLSessionDownloadTask }
            if let live = downloads.first(where: {
                $0.originalRequest?.url == expectedURL || $0.currentRequest?.url == expectedURL
            }) {
                self.currentRepoID = next.repo.repoID
                self.currentFileName = next.file.name
                self.currentTaskIdentifier = live.taskIdentifier
                self.currentDownloadTask = live
                self.hasBegunFile = true
                self.lastProgressAt = Date()
                // The plan is strictly one file at a time — cancel any
                // stray tasks from an earlier interrupted state.
                for other in downloads where other.taskIdentifier != live.taskIdentifier {
                    other.cancel()
                }
                self.lock.unlock()
                self.emitSnapshot()
                return
            }
            self.lock.unlock()
            self.beginFile(session: session, tier: tier, repo: next.repo, file: next.file)
        }
    }

    private func beginFile(session: URLSession, tier: PlannerTier, repo: BundleRepo, file: FileDownloadState) {
        // Per-file storage re-check (Build 3): space can shrink while
        // earlier files download; never start a file the device
        // cannot hold the rest of.
        guard checkSpaceGate(tier: tier, notify: true) else { return }
        let task: URLSessionDownloadTask
        if let resumeData = file.resumeData, !resumeData.isEmpty {
            task = session.downloadTask(withResumeData: resumeData)
        } else if let url = Self.downloadURL(repoID: repo.repoID, file: file.name) {
            task = session.downloadTask(with: url)
        } else {
            finishRun(with: .failure(
                ModelDownloaderError.downloadFailed(file: file.name, underlying: "bad URL")))
            return
        }
        lock.lock()
        currentRepoID = repo.repoID
        currentFileName = file.name
        currentTaskIdentifier = task.taskIdentifier
        currentDownloadTask = task
        hasBegunFile = true
        lastProgressAt = Date()
        lock.unlock()
        task.resume()
        emitSnapshot()
    }

    /// Advances after the current file landed: next file, or finish.
    private func advanceAfterCurrentFile() {
        lock.lock()
        guard running, let tier = currentTier, let session else {
            lock.unlock()
            return
        }
        currentTaskIdentifier = nil
        currentDownloadTask = nil
        currentRepoID = nil
        currentFileName = nil
        guard let next = nextIncompleteLocked(tier: tier) else {
            lock.unlock()
            finishRun(with: .success(()))
            return
        }
        lock.unlock()
        beginFile(session: session, tier: tier, repo: next.repo, file: next.file)
    }

    private func finishRun(with result: Result<Void, Error>) {
        lock.lock()
        let wasRunning = running
        running = false
        currentTaskIdentifier = nil
        currentDownloadTask = nil
        currentRepoID = nil
        currentFileName = nil
        hasBegunFile = false
        switch result {
        case .success:
            lastFailureMessage = nil
            spaceBlock = nil
            persistLocked()
        case .failure(let error):
            if let modelError = error as? ModelDownloaderError {
                lastFailureMessage = modelError.description
            } else {
                lastFailureMessage = error.localizedDescription
            }
        }
        let handler = finishedHandler
        let tier = currentTier
        lock.unlock()
        stopWatchdog()
        if case .success = result, let tier {
            refreshMarkers(tier: tier)
        }
        emitSnapshot()
        if wasRunning {
            handler?(result)
        }
    }

    // MARK: - Delegate

    func urlSession(
        _ session: URLSession,
        downloadTask: URLSessionDownloadTask,
        didWriteData bytesWritten: Int64,
        totalBytesWritten: Int64,
        totalBytesExpectedToWrite: Int64
    ) {
        guard let match = matchFile(for: downloadTask) else { return }
        lock.lock()
        lastProgressAt = Date()
        updateFileLocked(tier: match.tier, repoID: match.repoID, fileName: match.fileName) { state in
            state.bytesDownloaded = totalBytesWritten
            if state.size <= 0, totalBytesExpectedToWrite > 0 {
                state.size = max(totalBytesExpectedToWrite, totalBytesWritten)
            }
        }
        if Date().timeIntervalSince(lastPersistAt) > 2 {
            persistLocked()
        }
        lock.unlock()
        emitSnapshot()
    }

    func urlSession(
        _ session: URLSession,
        downloadTask: URLSessionDownloadTask,
        didFinishDownloadingTo location: URL
    ) {
        guard let match = matchFile(for: downloadTask) else {
            // Never fail silently (Build 3): an unmatched completion
            // means a downloaded file could not be placed. Record it
            // where the UI can show it; the temp file is gone when
            // this callback returns, so the file will be re-fetched
            // on the next run.
            lock.lock()
            if lastFailureMessage == nil {
                lastFailureMessage = "A finished download could not be matched to a model file; it will be fetched again on Resume."
            }
            lock.unlock()
            emitSnapshot()
            return
        }
        let repo = BundleRepo(repoID: match.repoID, approxMegabytes: 0)
        let destination = ModelBundles.repoDirectory(repo).appendingPathComponent(match.fileName)
        var moveError: Error?
        do {
            // The temp file is deleted when this callback returns, so
            // the move must happen here, synchronously.
            try FileManager.default.createDirectory(
                at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)
            if FileManager.default.fileExists(atPath: destination.path) {
                try FileManager.default.removeItem(at: destination)
            }
            try FileManager.default.moveItem(at: location, to: destination)
        } catch {
            moveError = error
        }

        if let moveError {
            // Out of space is a first-class state, not a generic
            // failure (Build 3) — and a failed move is never swallowed
            // just because the file wasn't the "current" one.
            if Self.isOutOfSpaceError(moveError) {
                blockRunForInsufficientSpace(
                    neededBytes: remainingBytes(tier: match.tier),
                    availableBytes: Self.availableCapacity() ?? 0,
                    notify: true)
            } else {
                lock.lock()
                let isCurrent = running && currentTaskIdentifier == downloadTask.taskIdentifier
                if !isCurrent && lastFailureMessage == nil {
                    lastFailureMessage = "Could not save \(match.fileName): \(moveError.localizedDescription)"
                }
                lock.unlock()
                if isCurrent {
                    finishRun(with: .failure(ModelDownloaderError.downloadFailed(
                        file: match.fileName, underlying: "\(moveError)")))
                } else {
                    emitSnapshot()
                }
            }
            return
        }

        // Verify what landed before calling it complete (Build 3):
        // a short or wrong-sized file must not poison the bundle.
        let landedSize = Self.fileSize(at: destination)
        let expectedSize = storedSize(tier: match.tier, repoID: match.repoID, fileName: match.fileName)
        if let expectedSize, expectedSize > 0, let landedSize, landedSize != expectedSize {
            try? FileManager.default.removeItem(at: destination)
            lock.lock()
            let isCurrent = running && currentTaskIdentifier == downloadTask.taskIdentifier
            lock.unlock()
            if isCurrent {
                finishRun(with: .failure(ModelDownloaderError.downloadFailed(
                    file: match.fileName,
                    underlying: "saved file is \(landedSize) bytes, expected \(expectedSize)")))
            } else {
                lock.lock()
                if lastFailureMessage == nil {
                    lastFailureMessage = "Saved copy of \(match.fileName) had the wrong size and was discarded; it will be fetched again on Resume."
                }
                lock.unlock()
                emitSnapshot()
            }
            return
        }

        lock.lock()
        let isCurrent = running && currentTaskIdentifier == downloadTask.taskIdentifier
        lastProgressAt = Date()
        updateFileLocked(tier: match.tier, repoID: match.repoID, fileName: match.fileName) { state in
            state.completed = true
            state.resumeData = nil
            if let landedSize {
                state.bytesDownloaded = landedSize
                if state.size <= 0 { state.size = landedSize }
            } else if state.size > 0 {
                state.bytesDownloaded = state.size
            }
        }
        persistLocked()
        lock.unlock()

        writeMarkerIfRepoComplete(tier: match.tier, repo: repo)
        emitSnapshot()
        if isCurrent {
            advanceAfterCurrentFile()
        }
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        // Success is handled in didFinishDownloadingTo; this fires
        // after it (error == nil, ignored) or alone on failure.
        guard let error else { return }
        guard let match = matchFile(for: task) else { return }
        lock.lock()
        let isCurrent = running && currentTaskIdentifier == task.taskIdentifier
        // A stray duplicate task for the same file (cancelled during
        // reconcile) must not clobber the live task's progress.
        let isStrayDuplicate = running && !isCurrent
            && currentTaskIdentifier != nil
            && currentTier == match.tier && currentFileName == match.fileName
        if isStrayDuplicate {
            lock.unlock()
            return
        }
        // The watchdog cancelled this task for stalling: the stop is
        // completed below with the stalled reason, not the raw
        // cancellation error (Build 3).
        let watchdogCancel = watchdogCancelPending && isCurrent
        if watchdogCancel {
            watchdogCancelPending = false
        }
        // The resume-data userInfo key, by its documented string value
        // (the NSURLSessionDownloadTaskResumeData constant).
        let resumeData = (error as NSError)
            .userInfo["NSURLSessionDownloadTaskResumeData"] as? Data
        let received = (task as? URLSessionDownloadTask)?.countOfBytesReceived ?? 0
        updateFileLocked(tier: match.tier, repoID: match.repoID, fileName: match.fileName) { state in
            if received > state.bytesDownloaded {
                state.bytesDownloaded = received
            }
            if let resumeData {
                state.resumeData = resumeData
            } else {
                // No resume data from the system: any stored data is
                // stale, so the next attempt restarts this file clean.
                state.resumeData = nil
            }
        }
        if isCurrent {
            currentTaskIdentifier = nil
            currentDownloadTask = nil
        }
        persistLocked()
        lock.unlock()
        emitSnapshot()
        if isCurrent {
            if watchdogCancel {
                finishRun(with: .failure(ModelDownloaderError.stalled(file: match.fileName)))
            } else if Self.isOutOfSpaceError(error) {
                // A transfer that died because the disk filled is the
                // insufficient-space state, not a generic failure.
                blockRunForInsufficientSpace(
                    neededBytes: remainingBytes(tier: match.tier),
                    availableBytes: Self.availableCapacity() ?? 0,
                    notify: true)
            } else {
                finishRun(with: .failure(ModelDownloaderError.downloadFailed(
                    file: match.fileName, underlying: error.localizedDescription)))
            }
        }
    }

    func urlSessionDidFinishEvents(forBackgroundURLSession session: URLSession) {
        lock.lock()
        let handler = backgroundCompletionHandler
        backgroundCompletionHandler = nil
        lock.unlock()
        // Safe to invoke from the delegate queue: the handler only
        // signals the system that event delivery is complete.
        handler?()
    }

    // MARK: - State helpers (lock held unless noted)

    private struct FileMatch {
        var tier: PlannerTier
        var repoID: String
        var fileName: String
    }

    /// Matches a delegate callback's task to a planned file: current
    /// task first, then by request URL, then — the relaunch case, where
    /// a task created from resume data has no usable request URL — the
    /// active tier's first incomplete file. That last fallback is exact
    /// because downloads are strictly one task at a time: any live or
    /// just-finished background task belongs to that file.
    private func matchFile(for task: URLSessionTask) -> FileMatch? {
        lock.lock()
        defer { lock.unlock() }
        if running, let tier = currentTier,
           currentTaskIdentifier == task.taskIdentifier,
           let repoID = currentRepoID, let fileName = currentFileName {
            return FileMatch(tier: tier, repoID: repoID, fileName: fileName)
        }
        if let url = task.originalRequest?.url ?? task.currentRequest?.url {
            for (tierRaw, tierState) in persisted.tiers {
                guard let tier = PlannerTier(rawValue: tierRaw) else { continue }
                for repoState in tierState.repos {
                    for file in repoState.files where !file.completed {
                        if Self.downloadURL(repoID: repoState.repoID, file: file.name) == url {
                            return FileMatch(tier: tier, repoID: repoState.repoID, fileName: file.name)
                        }
                    }
                }
            }
        }
        if let activeRaw = persisted.activeTier,
           let tier = PlannerTier(rawValue: activeRaw),
           let next = nextIncompleteLocked(tier: tier) {
            return FileMatch(tier: tier, repoID: next.repo.repoID, fileName: next.file.name)
        }
        return nil
    }

    /// The next incomplete (repo, file) in plan order. Call with lock held.
    private func nextIncompleteLocked(tier: PlannerTier) -> (repo: BundleRepo, file: FileDownloadState)? {
        guard let tierState = persisted.tiers[tier.rawValue] else { return nil }
        for repo in ModelBundles.repos(for: tier) {
            guard let repoState = tierState.repos.first(where: { $0.repoID == repo.repoID })
            else { continue }
            if let file = repoState.files.first(where: { !$0.completed }) {
                return (repo, file)
            }
        }
        return nil
    }

    /// Call with lock held.
    private func updateFileLocked(
        tier: PlannerTier,
        repoID: String,
        fileName: String,
        _ mutate: (inout FileDownloadState) -> Void
    ) {
        guard var tierState = persisted.tiers[tier.rawValue],
              let repoIndex = tierState.repos.firstIndex(where: { $0.repoID == repoID }),
              let fileIndex = tierState.repos[repoIndex].files
                .firstIndex(where: { $0.name == fileName })
        else { return }
        mutate(&tierState.repos[repoIndex].files[fileIndex])
        persisted.tiers[tier.rawValue] = tierState
    }

    /// Self-locking read of a file's planned size (-1 when unknown).
    private func storedSize(tier: PlannerTier, repoID: String, fileName: String) -> Int64? {
        lock.lock()
        defer { lock.unlock() }
        return persisted.tiers[tier.rawValue]?
            .repos.first { $0.repoID == repoID }?
            .files.first { $0.name == fileName }?
            .size
    }

    private func refreshMarkers(tier: PlannerTier) {
        for repo in ModelBundles.repos(for: tier) {
            writeMarkerIfRepoComplete(tier: tier, repo: repo)
        }
    }

    private func writeMarkerIfRepoComplete(tier: PlannerTier, repo: BundleRepo) {
        lock.lock()
        let complete = persisted.tiers[tier.rawValue]?
            .repos.first { $0.repoID == repo.repoID }
            .map { !$0.files.isEmpty && $0.files.allSatisfy { $0.completed } } ?? false
        lock.unlock()
        guard complete else { return }
        let directory = ModelBundles.repoDirectory(repo)
        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try? Data().write(to: directory.appendingPathComponent(".complete"), options: [.atomic])
    }

    // MARK: - Snapshot

    private func emitSnapshot() {
        updateCachedCapacity()
        lock.lock()
        let handler = snapshotHandler
        let snapshot = snapshotLocked()
        lock.unlock()
        handler?(snapshot)
    }

    /// Keeps the snapshot's free-space figure fresh (checked at most
    /// every 5 s) without a syscall on every progress callback.
    private func updateCachedCapacity() {
        lock.lock()
        let stale = Date().timeIntervalSince(lastCapacityCheckAt) > 5
        lock.unlock()
        guard stale, let capacity = Self.availableCapacity() else { return }
        lock.lock()
        cachedAvailableCapacity = capacity
        lastCapacityCheckAt = Date()
        lock.unlock()
    }

    /// Call with lock held.
    private func snapshotLocked() -> DownloadSnapshot {
        var snapshot = DownloadSnapshot()
        let tierRaw = currentTier?.rawValue ?? persisted.activeTier ?? ""
        snapshot.tierRawValue = tierRaw
        snapshot.isActive = running
        snapshot.availableCapacityBytes = cachedAvailableCapacity
        guard let tierState = persisted.tiers[tierRaw] else {
            snapshot.status = running ? .downloading : .idle
            return snapshot
        }

        var total: Int64 = 0
        var downloaded: Int64 = 0
        var filesTotal = 0
        var filesCompleted = 0
        for repoState in tierState.repos {
            for file in repoState.files {
                filesTotal += 1
                if file.completed { filesCompleted += 1 }
                if file.size > 0 {
                    total += file.size
                    downloaded += file.completed ? file.size : min(file.bytesDownloaded, file.size)
                } else if file.completed {
                    total += file.bytesDownloaded
                    downloaded += file.bytesDownloaded
                } else {
                    downloaded += file.bytesDownloaded
                }
            }
        }
        snapshot.bytesTotal = total
        snapshot.bytesDownloaded = downloaded
        snapshot.filesTotal = filesTotal
        snapshot.filesCompleted = filesCompleted
        snapshot.overallFraction = total > 0 ? Double(downloaded) / Double(total) : 0
        snapshot.hasResumableState = !running && filesTotal > 0 && filesCompleted < filesTotal

        // Truthful status (Build 3): derived from the run's actual
        // state, in priority order — a blocked or failed run never
        // reports as downloading.
        if filesTotal > 0 && filesCompleted == filesTotal {
            snapshot.status = .complete
        } else if let block = spaceBlock {
            snapshot.status = .insufficientSpace(neededBytes: block.needed, availableBytes: block.available)
        } else if running {
            snapshot.status = .downloading
        } else if let message = lastFailureMessage {
            snapshot.status = .failed(message)
        } else if filesTotal > 0 && filesCompleted < filesTotal {
            snapshot.status = .paused
        } else {
            snapshot.status = .idle
        }

        if let repoID = currentRepoID, let fileName = currentFileName,
           let tier = PlannerTier(rawValue: tierRaw) {
            snapshot.repoLabel = ModelBundles.repos(for: tier)
                .first { $0.repoID == repoID }?.displayName
            snapshot.fileName = fileName
            if let file = tierState.repos.first(where: { $0.repoID == repoID })?
                .files.first(where: { $0.name == fileName }), file.size > 0 {
                snapshot.fileFraction = min(Double(file.bytesDownloaded) / Double(file.size), 1)
            }
        }
        return snapshot
    }

    // MARK: - Files / persistence

    private static func downloadURL(repoID: String, file: String) -> URL? {
        URL(string: "https://huggingface.co/\(repoID)/resolve/main/\(file)")
    }

    private static func fileSize(at url: URL) -> Int64? {
        guard let attributes = try? FileManager.default.attributesOfItem(atPath: url.path),
              let size = attributes[.size] as? NSNumber else {
            return nil
        }
        return size.int64Value
    }

    private static var stateFileURL: URL {
        ModelBundles.modelsDirectory.appendingPathComponent("download-state.json")
    }

    private static func loadPersistedState() -> DownloadStateFile? {
        guard let data = try? Data(contentsOf: stateFileURL),
              let decoded = try? JSONDecoder().decode(DownloadStateFile.self, from: data) else {
            return nil
        }
        return decoded
    }

    /// Call with `lock` held.
    private func persistLocked() {
        lastPersistAt = Date()
        guard let data = try? JSONEncoder().encode(persisted) else { return }
        try? FileManager.default.createDirectory(
            at: ModelBundles.modelsDirectory, withIntermediateDirectories: true)
        try? data.write(to: Self.stateFileURL, options: [.atomic])
    }
}
