import SwiftUI

/// The spike's one screen. Sections, top to bottom: model bundles,
/// song inputs, live run readouts, result (playback / share / log row).
struct ContentView: View {
    @StateObject private var viewModel = GenerationViewModel()
    @StateObject private var player = AudioPlayer()

    var body: some View {
        NavigationStack {
            Form {
                modelsSection
                songSection
                runSection
                if viewModel.lastOutputURL != nil {
                    resultSection
                }
                if let errorMessage = viewModel.errorMessage {
                    Section {
                        Text(errorMessage)
                            .foregroundStyle(.red)
                            .font(.footnote)
                    }
                }
            }
            .navigationTitle("Song Spike")
            .onChange(of: viewModel.lastOutputURL) { _, newURL in
                if let newURL {
                    player.load(url: newURL)
                }
            }
        }
    }

    // MARK: - Sections

    private var modelsSection: some View {
        Section("Model bundles") {
            if let corrupt = viewModel.corruptFileName {
                // Build 4: a corrupt model file is named, with a
                // one-file repair — never a cryptic load error.
                VStack(alignment: .leading, spacing: 6) {
                    Text("A model file looks corrupted: \(corrupt)")
                        .font(.headline)
                    Text("Tap Repair to re-download just that file.")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                    Button("Repair") {
                        viewModel.repairCorruptFile()
                    }
                }
            } else if viewModel.modelsReady {
                Label("Bundles ready (\(viewModel.tier.rawValue) tier)", systemImage: "checkmark.circle")
            } else if viewModel.showDownloadProgress {
                downloadProgress
            } else {
                Button("Download model bundles (≈ 8 GB)") {
                    viewModel.downloadModels()
                }
            }
            if viewModel.downloadSnapshot.isValidating {
                Text("Verifying model files…")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }
            Picker("Planner tier", selection: $viewModel.tier) {
                ForEach(PlannerTier.allCases) { tier in
                    Text(tier.displayName).tag(tier)
                }
            }
        }
    }

    /// Live download feedback: which bundle and file are downloading,
    /// per-file and overall progress, bytes downloaded of total, the
    /// file count, and the device's free space. The headline follows
    /// the snapshot's truthful status (Build 3): it says "Downloading"
    /// only while a transfer is actually running — a stopped run shows
    /// paused / failed / not-enough-space with its reason and a Resume
    /// or Check-again button, never both states at once.
    private var downloadProgress: some View {
        let snapshot = viewModel.downloadSnapshot
        return VStack(alignment: .leading, spacing: 6) {
            if case .insufficientSpace(let needed, let available) = snapshot.status {
                Text("Not enough free space")
                    .font(.headline)
                Text("This download needs \(Self.formatBytes(needed)) more, "
                    + "but only \(Self.formatBytes(available)) is free. "
                    + "Free up space, then try again.")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            } else if case .failed(let message) = snapshot.status {
                Text("Download stopped")
                    .font(.headline)
                Text(message)
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            } else if snapshot.status == .paused {
                Text("Download paused")
                    .font(.headline)
            } else if snapshot.status == .complete {
                Text("Download complete")
                    .font(.headline)
            } else if snapshot.status == .downloading, let repairFile = snapshot.repairFileName {
                // Build 4: a re-download triggered by the repair
                // scan is headlined as what it is.
                Text("Repairing: re-downloading \(repairFile)")
                    .font(.headline)
            } else if let repoLabel = snapshot.repoLabel {
                Text("Downloading: \(repoLabel)")
                    .font(.headline)
            } else {
                Text("Downloading bundles…")
                    .font(.headline)
            }
            if snapshot.status == .downloading, let fileName = snapshot.fileName {
                Text(fileName)
                    .font(.footnote)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                ProgressView(value: snapshot.fileFraction)
            }
            ProgressView(value: snapshot.overallFraction)
            Text("\(Int(snapshot.overallFraction * 100))% — "
                + "\(Self.formatBytes(snapshot.bytesDownloaded)) of \(Self.formatBytes(snapshot.bytesTotal))")
                .font(.footnote)
            Text("Files \(snapshot.filesCompleted) of \(snapshot.filesTotal)")
                .font(.footnote)
                .foregroundStyle(.secondary)
            if snapshot.availableCapacityBytes >= 0 {
                Text("Free space: \(Self.formatBytes(snapshot.availableCapacityBytes))")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }
            if snapshot.status == .downloading {
                Text("Continues in the background if you leave the app. "
                    + "If a download stops, the reason appears here and "
                    + "finished files are kept.")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            } else if case .insufficientSpace = snapshot.status {
                Button("Check again & resume") {
                    viewModel.downloadModels()
                }
            } else if snapshot.status == .paused || snapshot.isFailed {
                Button("Resume download") {
                    viewModel.downloadModels()
                }
            }
        }
    }

    private static func formatBytes(_ bytes: Int64) -> String {
        ByteCountFormatter.string(fromByteCount: bytes, countStyle: .file)
    }

    private var songSection: some View {
        Section("Song") {
            TextField("Style / caption", text: $viewModel.caption, axis: .vertical)
                .lineLimit(2...4)
            VStack(alignment: .leading, spacing: 4) {
                Text("Lyrics")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
                TextEditor(text: $viewModel.lyrics)
                    .frame(minHeight: 140)
                    .font(.body.monospaced())
            }
            Picker("Duration", selection: $viewModel.durationSeconds) {
                ForEach(viewModel.durationChoices, id: \.self) { seconds in
                    Text("\(seconds) s").tag(seconds)
                }
            }
            .pickerStyle(.segmented)
        }
    }

    private var runSection: some View {
        Section("Run") {
            if viewModel.isGenerating {
                Button("Cancel", role: .destructive) {
                    viewModel.cancelGeneration()
                }
            } else {
                Button("Generate") {
                    viewModel.generate()
                }
                .disabled(!viewModel.modelsReady)
            }

            // Live readouts — visible for the whole run, so a slow load
            // is distinguishable from a stuck one.
            LabeledContent("Stage", value: viewModel.stageLabel)
            if let fraction = viewModel.stageFraction {
                ProgressView(value: fraction)
            }
            LabeledContent("Elapsed", value: String(format: "%.1f s", viewModel.elapsedSeconds))
            LabeledContent("Memory (live)", value: String(format: "%.0f MB", viewModel.liveMemoryMB))
            LabeledContent("Memory (peak)", value: String(format: "%.0f MB", viewModel.peakMemoryMB))
            LabeledContent("Memory (available)", value: String(format: "%.0f MB", viewModel.availableMemoryMB))
            LabeledContent(
                "Thermal",
                value: DeviceInfo.thermalStateName(ProcessInfo.processInfo.thermalState)
            )
        }
    }

    private var resultSection: some View {
        Section("Result") {
            if let metadata = viewModel.lastMetadataSummary {
                Text(metadata)
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }
            HStack(spacing: 16) {
                Button(player.isPlaying ? "Pause" : "Play") {
                    player.togglePlayPause()
                }
                if let url = viewModel.lastOutputURL {
                    ShareLink(item: url) {
                        Label("Export WAV", systemImage: "square.and.arrow.up")
                    }
                }
            }
            if viewModel.lastMeasurementRow != nil {
                Button("Copy measurement row") {
                    viewModel.copyMeasurementRow()
                }
                Text(viewModel.lastMeasurementRow ?? "")
                    .font(.caption.monospaced())
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
            }
        }
    }
}
