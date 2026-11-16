import Foundation

/// One generation run, exactly the SPIKE.md measurement-sheet columns
/// plus the context needed to reproduce it (caption, seed, OS version).
/// A row is written for failed and cancelled runs too — no cherry-picking.
struct RunRecord: Codable, Sendable, Identifiable {
    var id: Int
    var date: Date
    var deviceModel: String
    var osVersion: String
    var tier: String
    var caption: String
    var seed: Int
    var durationSeconds: Int
    /// "cold" = first generation after app launch (includes pipeline
    /// load); "warm" = every run after that in the same process.
    var coldOrWarm: String
    var wallSeconds: Double
    var peakMemoryMB: Double
    var thermalBefore: String
    var thermalAfter: String
    /// Battery level in percent at start/end; nil when the device
    /// reports -1 (monitoring unavailable, e.g. Simulator).
    var batteryBeforePercent: Int?
    var batteryAfterPercent: Int?
    /// True only when generation completed — a jetsam kill ends the
    /// process, so no row is ever written for one; survival is implied
    /// by the row existing with `outcome == "completed"`.
    var outcome: String // "completed" | "failed" | "cancelled"
    var outputFileName: String?
    var notes: String

    var batteryDeltaPercent: Int? {
        guard let before = batteryBeforePercent, let after = batteryAfterPercent else {
            return nil
        }
        return after - before
    }
}

/// Appends run rows to `runs.csv` and `runs.json` in the app's Documents
/// directory. An actor so the view model and any future callers can log
/// without coordinating writes.
actor RunLogger {
    private let csvURL: URL
    private let jsonURL: URL
    private var records: [RunRecord]

    init(documentsDirectory: URL) {
        csvURL = documentsDirectory.appendingPathComponent("runs.csv")
        jsonURL = documentsDirectory.appendingPathComponent("runs.json")
        // Reload history so run numbers keep increasing across launches.
        if let data = try? Data(contentsOf: jsonURL),
           let decoded = try? JSONDecoder.runLog.decode([RunRecord].self, from: data) {
            records = decoded
        } else {
            records = []
        }
    }

    var nextRunNumber: Int { (records.map(\.id).max() ?? 0) + 1 }

    func append(_ record: RunRecord) throws {
        records.append(record)
        try writeJSON()
        try appendCSV(record)
    }

    private func writeJSON() throws {
        let data = try JSONEncoder.runLog.encode(records)
        try data.write(to: jsonURL, options: [.atomic])
    }

    private func appendCSV(_ record: RunRecord) throws {
        let fileManager = FileManager.default
        if !fileManager.fileExists(atPath: csvURL.path) {
            try Data(Self.csvHeader.utf8).write(to: csvURL, options: [.atomic])
        }
        let handle = try FileHandle(forWritingTo: csvURL)
        defer { try? handle.close() }
        try handle.seekToEnd()
        try handle.write(contentsOf: Data(Self.csvRow(record).utf8))
    }

    // MARK: - Formatting

    static let csvHeader =
        "run,date,device,os,tier,duration_s,cold_warm,wall_s,peak_mem_mb,"
        + "thermal_before,thermal_after,battery_before_pct,battery_after_pct,"
        + "battery_delta_pct,outcome,output_file,seed,notes\n"

    static func csvRow(_ record: RunRecord) -> String {
        let fields: [String] = [
            String(record.id),
            ISO8601DateFormatter().string(from: record.date),
            record.deviceModel,
            record.osVersion,
            record.tier,
            String(record.durationSeconds),
            record.coldOrWarm,
            String(format: "%.1f", record.wallSeconds),
            String(format: "%.0f", record.peakMemoryMB),
            record.thermalBefore,
            record.thermalAfter,
            record.batteryBeforePercent.map(String.init) ?? "",
            record.batteryAfterPercent.map(String.init) ?? "",
            record.batteryDeltaPercent.map(String.init) ?? "",
            record.outcome,
            record.outputFileName ?? "",
            String(record.seed),
            record.notes,
        ]
        return fields.map(csvEscape).joined(separator: ",") + "\n"
    }

    private static func csvEscape(_ field: String) -> String {
        guard field.contains(",") || field.contains("\"") || field.contains("\n") else {
            return field
        }
        return "\"" + field.replacingOccurrences(of: "\"", with: "\"\"") + "\""
    }

    /// The row in SPIKE.md's measurement-sheet markdown format, ready to
    /// paste into the sheet (the UI's "Copy measurement row" button).
    static func measurementRow(_ record: RunRecord) -> String {
        let date = ISO8601DateFormatter().string(from: record.date)
        let wall = String(format: "%.0f s", record.wallSeconds)
        let peak = String(format: "%.0f MB", record.peakMemoryMB)
        let battery = record.batteryDeltaPercent.map { "\($0)%" } ?? "n/a"
        let jetsam = record.outcome == "completed" ? "No" : "n/a (\(record.outcome))"
        let kept = record.outputFileName != nil ? "Yes" : "No"
        let notes = record.notes.isEmpty ? (record.outputFileName ?? "") : record.notes
        return "| \(record.id) | \(date) | \(record.deviceModel) | \(record.tier) "
            + "| \(record.durationSeconds) s | \(record.coldOrWarm) | \(wall) | \(peak) "
            + "| \(record.thermalBefore)→\(record.thermalAfter) | \(battery) "
            + "| \(jetsam) | \(kept) | \(notes) |"
    }
}

private extension JSONEncoder {
    static var runLog: JSONEncoder {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        return encoder
    }
}

private extension JSONDecoder {
    static var runLog: JSONDecoder {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return decoder
    }
}
