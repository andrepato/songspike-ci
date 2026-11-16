import Darwin
import Foundation

/// Memory readouts for the live progress display and the run log.
///
/// Two numbers matter for the spike:
/// - `phys_footprint` (task_vm_info): what jetsam actually charges the
///   app. Its peak across a generation is the measurement that decides
///   whether a tier survives on an 8 GB device.
/// - `os_proc_available_memory()`: headroom left before jetsam kills us.
enum MemorySampler {

    /// Current physical footprint in bytes, or nil if task_info fails.
    static func footprintBytes() -> UInt64? {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(
            MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size
        )
        let kr = withUnsafeMutablePointer(to: &info) { pointer in
            pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        guard kr == KERN_SUCCESS else { return nil }
        return info.phys_footprint
    }

    /// Bytes still available to this process before the jetsam limit.
    static func availableBytes() -> UInt64 {
        let available = os_proc_available_memory()
        return available > 0 ? UInt64(available) : 0
    }

    static func megabytes(_ bytes: UInt64) -> Double {
        Double(bytes) / 1_000_000
    }
}

/// Static device facts written into every run-log row.
enum DeviceInfo {

    /// Hardware identifier, e.g. "iPad14,4" (André's M2 iPad Pro 11").
    static var modelIdentifier: String {
        var systemInfo = utsname()
        uname(&systemInfo)
        let mirror = Mirror(reflecting: systemInfo.machine)
        let identifier = mirror.children.reduce(into: "") { partial, child in
            guard let value = child.value as? Int8, value != 0 else { return }
            partial.append(Character(UnicodeScalar(UInt8(value))))
        }
        return identifier.isEmpty ? "unknown" : identifier
    }

    static var osVersion: String {
        let version = ProcessInfo.processInfo.operatingSystemVersion
        return "\(version.majorVersion).\(version.minorVersion).\(version.patchVersion)"
    }

    static func thermalStateName(_ state: ProcessInfo.ThermalState) -> String {
        switch state {
        case .nominal: return "nominal"
        case .fair: return "fair"
        case .serious: return "serious"
        case .critical: return "critical"
        @unknown default: return "unknown"
        }
    }
}
