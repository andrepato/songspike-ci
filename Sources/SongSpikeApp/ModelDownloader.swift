import Foundation

/// One Hugging Face bundle repo the engine loads. Every tier needs four:
/// the turbo DiT, one planner, the text encoder and the VAE
/// (ENGINE-NOTES.md, "Bundles").
struct BundleRepo: Sendable, Equatable {
    var repoID: String
    /// Approximate download size from the HF collection page. Kept for
    /// reference only — Build 2's progress display uses the real byte
    /// sizes from the HF blobs API, never this estimate.
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

    var description: String {
        switch self {
        case .fileListFailed(let repoID, let why):
            return "Could not list files for \(repoID): \(why)"
        case .downloadFailed(let file, let why):
            return "Download failed for \(file): \(why)"
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

/// Immutable progress snapshot for the UI, rebuilt from the persisted
/// state plus the live task counters on every delegate callback.
struct DownloadSnapshot: Sendable, Equatable {
    var tierRawValue: String = ""
    var isActive: Bool = false
    /// Friendly name of the bundle currently downloading.
    var repoLabel: String?
    var fileName: String?
    var fileFraction: Double = 0
    var overallFraction: Double = 0
    var bytesDownloaded: Int64 = 0
    var bytesTotal: Int64 = 0
    var filesCompleted: Int = 0
    var filesTotal: Int = 0
    /// A partial download exists for this tier and no task is running
    /// (e.g. the last attempt stopped on an error).
    var hasResumableState: Bool = false

    static let empty = DownloadSnapshot()
}

// MARK: - Downloader

/// Build 2 downloader. Downloads run on a BACKGROUND URLSession
/// (`com.andrepato.songspike.downloads`), so transfers continue while
/// the app is backgrounded or terminated and their events are delivered
/// on the next launch. Files download one at a time in plan order —
/// the same file set Build 1 fetched (every sibling of the four repos)
/// — but the plan now carries real byte sizes from the HF blobs API,
/// every interruption captures resume data into the persisted state,
/// and `restore()` re-attaches to live tasks / auto-resumes on launch.
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
    private var backgroundCompletionHandler: (() -> Void)?
    private var lastPersistAt = Date.distantPast

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
            self.reconcileAndAdvance(session: session)
        }
    }

    /// Launch-time restore: re-attaches to the background session and,
    /// if the persisted state shows an incomplete download for the
    /// active tier, resumes it without the user re-tapping Download —
    /// adopting a still-running background task when one exists.
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
        lock.unlock()
        emitSnapshot()
        Task {
            // Refresh the plan if the app died mid-plan; buildPlan is
            // idempotent and keeps stored per-file state. Offline is
            // fine — the persisted plan is enough to resume its files.
            try? await self.buildPlan(tier: tier)
            self.reconcileAndAdvance(session: session)
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
        currentRepoID = nil
        currentFileName = nil
        if case .success = result {
            persistLocked()
        }
        let handler = finishedHandler
        let tier = currentTier
        lock.unlock()
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
        guard let match = matchFile(for: downloadTask) else { return }
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

        lock.lock()
        let isCurrent = running && currentTaskIdentifier == downloadTask.taskIdentifier
        if moveError == nil {
            let landedSize = Self.fileSize(at: destination)
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
        }
        lock.unlock()

        if let moveError {
            if isCurrent {
                finishRun(with: .failure(ModelDownloaderError.downloadFailed(
                    file: match.fileName, underlying: "\(moveError)")))
            }
            return
        }
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
        }
        persistLocked()
        lock.unlock()
        emitSnapshot()
        if isCurrent {
            finishRun(with: .failure(ModelDownloaderError.downloadFailed(
                file: match.fileName, underlying: error.localizedDescription)))
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
        lock.lock()
        let handler = snapshotHandler
        let snapshot = snapshotLocked()
        lock.unlock()
        handler?(snapshot)
    }

    /// Call with lock held.
    private func snapshotLocked() -> DownloadSnapshot {
        var snapshot = DownloadSnapshot()
        let tierRaw = currentTier?.rawValue ?? persisted.activeTier ?? ""
        snapshot.tierRawValue = tierRaw
        snapshot.isActive = running
        guard let tierState = persisted.tiers[tierRaw] else { return snapshot }

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
