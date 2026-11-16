import Foundation

/// One Hugging Face bundle repo the engine loads. Every tier needs four:
/// the turbo DiT, one planner, the text encoder and the VAE
/// (ENGINE-NOTES.md, "Bundles").
struct BundleRepo: Sendable, Equatable {
    var repoID: String
    /// Approximate download size from the HF collection page, for the
    /// progress display only — never used for verification.
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

/// Downloads the four bundle repos for a tier from Hugging Face into
/// Application Support, file by file, reporting overall progress.
///
/// The per-repo file list is discovered at runtime from the HF API
/// instead of being hardcoded. Verified live 2026-11-16: all five
/// repos answer `/api/models/<repo>` with `siblings[].rfilename`, and
/// every file the engine opens is present — turbo: model.safetensors,
/// config.json, silence_latent.safetensors; planners + text encoder:
/// model.safetensors, config.json, tokenizer.json; VAE:
/// model.safetensors, config.json (the engine's per-bundle contract,
/// `ACEStepBundles.*Files` in the engine source).
struct ModelDownloader: Sendable {

    /// Overall progress 0...1 across all files of all repos in this call.
    typealias ProgressHandler = @Sendable (Double) -> Void

    func download(tier: PlannerTier, progress: @escaping ProgressHandler) async throws {
        let repos = ModelBundles.repos(for: tier)
        let pending = repos.filter { !ModelBundles.isDownloaded($0) }
        guard !pending.isEmpty else {
            progress(1)
            return
        }

        // Resolve every file list first so progress has a stable total.
        var plan: [(repo: BundleRepo, files: [String])] = []
        var totalFiles = 0
        for repo in pending {
            let files = try await fetchFileList(repoID: repo.repoID)
            plan.append((repo, files))
            totalFiles += files.count
        }

        var completedFiles = 0
        for (repo, files) in plan {
            let directory = ModelBundles.repoDirectory(repo)
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            for file in files {
                let destination = directory.appendingPathComponent(file)
                if !fileExistsAndNonEmpty(destination) {
                    try await downloadFile(repoID: repo.repoID, file: file, to: destination) { fileFraction in
                        progress((Double(completedFiles) + fileFraction) / Double(totalFiles))
                    }
                }
                completedFiles += 1
                progress(Double(completedFiles) / Double(totalFiles))
            }
            // Marker last: a repo counts as downloaded only when complete.
            try Data().write(to: directory.appendingPathComponent(".complete"), options: [.atomic])
        }
    }

    // MARK: - HF API

    private struct HFModelInfo: Decodable {
        struct Sibling: Decodable { let rfilename: String }
        let siblings: [Sibling]
    }

    private func fetchFileList(repoID: String) async throws -> [String] {
        guard let url = URL(string: "https://huggingface.co/api/models/\(repoID)") else {
            throw ModelDownloaderError.fileListFailed(repoID: repoID, underlying: "bad URL")
        }
        do {
            let (data, _) = try await URLSession.shared.data(from: url)
            let info = try JSONDecoder().decode(HFModelInfo.self, from: data)
            return info.siblings.map(\.rfilename)
        } catch {
            throw ModelDownloaderError.fileListFailed(repoID: repoID, underlying: "\(error)")
        }
    }

    // MARK: - One file

    private func fileExistsAndNonEmpty(_ url: URL) -> Bool {
        guard let attributes = try? FileManager.default.attributesOfItem(atPath: url.path),
              let size = attributes[.size] as? NSNumber else {
            return false
        }
        return size.intValue > 0
    }

    private func downloadFile(
        repoID: String,
        file: String,
        to destination: URL,
        progress: @escaping @Sendable (Double) -> Void
    ) async throws {
        guard let url = URL(string: "https://huggingface.co/\(repoID)/resolve/main/\(file)") else {
            throw ModelDownloaderError.downloadFailed(file: file, underlying: "bad URL")
        }
        try FileManager.default.createDirectory(
            at: destination.deletingLastPathComponent(), withIntermediateDirectories: true
        )
        let delegate = FileDownloadDelegate(destination: destination, progress: progress)
        let session = URLSession(configuration: .default, delegate: delegate, delegateQueue: nil)
        defer { session.finishTasksAndInvalidate() }
        do {
            try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
                delegate.continuation = continuation
                let task = session.downloadTask(with: url)
                delegate.taskIdentifier = task.taskIdentifier
                task.resume()
            }
        } catch {
            throw ModelDownloaderError.downloadFailed(file: file, underlying: "\(error)")
        }
    }
}

/// Per-file download delegate. The downloaded temp file must be moved
/// inside `didFinishDownloadingTo` — the system deletes it when that
/// callback returns.
private final class FileDownloadDelegate: NSObject, URLSessionDownloadDelegate, @unchecked Sendable {
    private let destination: URL
    private let progress: ModelDownloader.ProgressHandler
    private let lock = NSLock()
    private var resumed = false

    var continuation: CheckedContinuation<Void, Error>?
    var taskIdentifier: Int = -1

    init(destination: URL, progress: @escaping ModelDownloader.ProgressHandler) {
        self.destination = destination
        self.progress = progress
    }

    func urlSession(
        _ session: URLSession,
        downloadTask: URLSessionDownloadTask,
        didWriteData bytesWritten: Int64,
        totalBytesWritten: Int64,
        totalBytesExpectedToWrite: Int64
    ) {
        guard totalBytesExpectedToWrite > 0 else { return }
        progress(Double(totalBytesWritten) / Double(totalBytesExpectedToWrite))
    }

    func urlSession(
        _ session: URLSession,
        downloadTask: URLSessionDownloadTask,
        didFinishDownloadingTo location: URL
    ) {
        do {
            if FileManager.default.fileExists(atPath: destination.path) {
                try FileManager.default.removeItem(at: destination)
            }
            try FileManager.default.moveItem(at: location, to: destination)
            resume(with: .success(()))
        } catch {
            resume(with: .failure(error))
        }
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        // Success is resumed in didFinishDownloadingTo; this fires after
        // it (ignored via the resumed flag) or alone on failure.
        if let error {
            resume(with: .failure(error))
        }
    }

    private func resume(with result: Result<Void, Error>) {
        lock.lock()
        guard !resumed else {
            lock.unlock()
            return
        }
        resumed = true
        lock.unlock()
        switch result {
        case .success: continuation?.resume()
        case .failure(let error): continuation?.resume(throwing: error)
        }
    }
}
