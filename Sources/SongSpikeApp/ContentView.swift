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
            if viewModel.modelsReady {
                Label("Bundles ready (\(viewModel.tier.rawValue) tier)", systemImage: "checkmark.circle")
            } else if viewModel.showDownloadProgress {
                downloadProgress
            } else {
                Button("Download model bundles (≈ 8 GB)") {
                    viewModel.downloadModels()
                }
            }
            Picker("Planner tier", selection: $viewModel.tier) {
                ForEach(PlannerTier.allCases) { tier in
                    Text(tier.displayName).tag(tier)
                }
            }
        }
    }

    /// Live download feedback: which bundle and file are downloading,
    /// per-file and overall progress, bytes downloaded of total, and
    /// the file count. Downloads run on a background session, so they
    /// continue while the app is backgrounded or closed and resume
    /// automatically on the next launch.
    private var downloadProgress: some View {
        let snapshot = viewModel.downloadSnapshot
        return VStack(alignment: .leading, spacing: 6) {
            if let repoLabel = snapshot.repoLabel {
                Text("Downloading: \(repoLabel)")
                    .font(.headline)
            } else {
                Text("Downloading bundles…")
                    .font(.headline)
            }
            if let fileName = snapshot.fileName {
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
            if snapshot.isActive {
                Text("Continues in the background if you leave the app, "
                    + "and resumes automatically after a close or a failed file.")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            } else {
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
