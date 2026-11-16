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
            } else if viewModel.isDownloading {
                VStack(alignment: .leading, spacing: 6) {
                    Text("Downloading bundles… \(Int(viewModel.downloadFraction * 100))%")
                    ProgressView(value: viewModel.downloadFraction)
                    Text("≈ 8 GB on Wi-Fi. Stored in Application Support; "
                        + "the app binary stays small.")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }
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
