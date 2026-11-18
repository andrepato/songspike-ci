import Foundation

/// Model-file integrity validation (Build 4).
///
/// Why this exists: André's turbo `model.safetensors` was damaged
/// during the Builds 1–2 disk-full chaos (resumed / rewritten across
/// ENOSPC failures) and then trusted forever, because every check the
/// downloader ran was size-only — the file had the right byte count
/// with bad bytes inside, and generation died at load with MLX's
/// "[load_safetensors] Invalid json header length". These checks are
/// the missing layer:
///
/// - `validateSafetensorsHeader(at:)` — cheap and local: the 8-byte
///   little-endian header length, the JSON header it prefixes, and
///   every tensor's dtype / shape / data_offsets are structurally
///   validated against the actual file size. Reads at most the header
///   (≤ 100 MB cap, in practice a few hundred KB), so it can run on
///   every launch and before every generation.
/// - `sha256Hex(ofFileAt:)` — the strong check: a streaming SHA-256
///   of the whole file, compared against the `lfs.sha256` the HF
///   blobs API publishes for every LFS file in these repos (all the
///   safetensors carry one — verified live 2026-11-18). Pure Swift
///   (FIPS 180-4) so the exact code that ships is exercised by the
///   local test harness; a few seconds for the 4.8 GB turbo file.
enum ModelIntegrity {

    // MARK: - Safetensors header validation

    /// Byte sizes for the dtypes whose tensor byte counts can be
    /// cross-checked against `data_offsets`.
    private static let dtypeByteSizes: [String: Int64] = [
        "F64": 8, "F32": 4, "F16": 2, "BF16": 2,
        "I64": 8, "I32": 4, "I16": 2, "I8": 1,
        "U8": 1, "U16": 2, "U32": 4, "U64": 8,
        "BOOL": 1, "F8E4M3": 1, "F8E5M2": 1, "F8E8M0": 1,
    ]

    /// Structural validation of a safetensors file's header. Returns
    /// false for anything that is not a well-formed safetensors file:
    /// missing/short file, header length 0, absurd (≥ 100 MB) or past
    /// end-of-file, header that is not a JSON object, a tensor entry
    /// without dtype/shape/data_offsets, offsets outside the payload,
    /// or an offsets span that disagrees with shape × dtype size.
    static func validateSafetensorsHeader(at url: URL) -> Bool {
        guard let fileSize = fileSize(at: url), fileSize > 8 else { return false }
        guard let handle = try? FileHandle(forReadingFrom: url) else { return false }
        defer { try? handle.close() }

        guard let lengthData = try? handle.read(upToCount: 8),
              lengthData.count == 8 else { return false }
        // Little-endian per the safetensors spec; assembled byte by
        // byte so the check is endian-explicit on any host.
        var headerLength: UInt64 = 0
        for (index, byte) in lengthData.enumerated() {
            headerLength |= UInt64(byte) << UInt64(index * 8)
        }
        guard headerLength > 0,
              headerLength < 100_000_000,
              8 + headerLength <= UInt64(fileSize) else { return false }

        guard let headerData = try? handle.read(upToCount: Int(headerLength)),
              headerData.count == Int(headerLength),
              let parsed = try? JSONSerialization.jsonObject(with: headerData),
              let header = parsed as? [String: Any] else { return false }

        let payloadSize = fileSize - 8 - Int64(headerLength)
        var tensorCount = 0
        for (name, value) in header {
            if name == "__metadata__" { continue }
            guard let tensor = value as? [String: Any],
                  let dtype = tensor["dtype"] as? String,
                  let shape = tensor["shape"] as? [Any],
                  let offsets = tensor["data_offsets"] as? [Any],
                  offsets.count == 2,
                  let begin = (offsets[0] as? NSNumber)?.int64Value,
                  let end = (offsets[1] as? NSNumber)?.int64Value,
                  begin >= 0, end >= begin, end <= payloadSize
            else { return false }

            // Cross-check the offsets span against shape × dtype
            // size when both are known — catches headers whose JSON
            // parses but whose numbers were scrambled.
            if let itemSize = dtypeByteSizes[dtype] {
                var elements: Int64 = 1
                var computable = true
                for dimension in shape {
                    guard let dimension = (dimension as? NSNumber)?.int64Value,
                          dimension >= 0 else {
                        computable = false
                        break
                    }
                    let product = elements.multipliedReportingOverflow(by: dimension)
                    if product.overflow {
                        computable = false
                        break
                    }
                    elements = product.partialValue
                }
                if computable {
                    let bytes = elements.multipliedReportingOverflow(by: itemSize)
                    if !bytes.overflow && bytes.partialValue != end - begin {
                        return false
                    }
                }
            }
            tensorCount += 1
        }
        return tensorCount > 0
    }

    // MARK: - SHA-256 (streaming, pure Swift)

    /// Hex SHA-256 of a whole file, read in 4 MB chunks. Nil when the
    /// file cannot be read.
    static func sha256Hex(ofFileAt url: URL) -> String? {
        guard let handle = try? FileHandle(forReadingFrom: url) else { return nil }
        defer { try? handle.close() }
        var hasher = SHA256Hasher()
        while true {
            let chunk: Data?
            do {
                chunk = try handle.read(upToCount: 4 * 1024 * 1024)
            } catch {
                return nil
            }
            // EOF is an empty Data on Darwin and nil in
            // swift-corelibs-foundation — both end the loop.
            guard let chunk, !chunk.isEmpty else { break }
            hasher.update(chunk)
        }
        return hasher.finalizeHex()
    }

    private static func fileSize(at url: URL) -> Int64? {
        guard let attributes = try? FileManager.default.attributesOfItem(atPath: url.path),
              let size = attributes[.size] as? NSNumber else {
            return nil
        }
        return size.int64Value
    }
}

/// Streaming SHA-256 (FIPS 180-4). Feed `Data` chunks with `update`,
/// read the digest with `finalizeHex` (consuming — call once).
struct SHA256Hasher {
    private var state: [UInt32] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
    ]
    private var pending: [UInt8] = []
    private var totalBytes: UInt64 = 0

    private static let k: [UInt32] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
    ]

    mutating func update(_ data: Data) {
        totalBytes &+= UInt64(data.count)
        data.withUnsafeBytes { raw in
            feed(raw)
        }
    }

    mutating func finalizeHex() -> String {
        let bitLength = totalBytes &* 8
        var tail: [UInt8] = [0x80]
        while (totalBytes + UInt64(tail.count)) % 64 != 56 {
            tail.append(0)
        }
        for shift in stride(from: 56, through: 0, by: -8) {
            tail.append(UInt8(truncatingIfNeeded: bitLength >> UInt64(shift)))
        }
        tail.withUnsafeBufferPointer { buffer in
            feed(UnsafeRawBufferPointer(buffer))
        }
        precondition(pending.isEmpty, "SHA-256 padding must end on a block boundary")

        let hexDigits: [UInt8] = Array("0123456789abcdef".utf8)
        var hex: [UInt8] = []
        hex.reserveCapacity(64)
        for word in state {
            for shift in stride(from: 24, through: 0, by: -8) {
                let byte = UInt8(truncatingIfNeeded: word >> UInt32(shift))
                hex.append(hexDigits[Int(byte >> 4)])
                hex.append(hexDigits[Int(byte & 0x0f)])
            }
        }
        return String(decoding: hex, as: UTF8.self)
    }

    /// Feeds raw bytes into the block pipeline (does not count them —
    /// `update` owns the byte count, padding feeds through here too).
    private mutating func feed(_ raw: UnsafeRawBufferPointer) {
        guard let base = raw.baseAddress, raw.count > 0 else { return }
        var index = 0
        if !pending.isEmpty {
            let take = min(64 - pending.count, raw.count)
            for offset in 0..<take {
                pending.append(base.load(fromByteOffset: offset, as: UInt8.self))
            }
            index += take
            if pending.count == 64 {
                processBlock(pending)
                pending.removeAll(keepingCapacity: true)
            }
        }
        while index + 64 <= raw.count {
            processBlock(base.advanced(by: index))
            index += 64
        }
        while index < raw.count {
            pending.append(base.load(fromByteOffset: index, as: UInt8.self))
            index += 1
        }
    }

    private mutating func processBlock(_ block: [UInt8]) {
        block.withUnsafeBufferPointer { buffer in
            processBlock(buffer.baseAddress!)
        }
    }

    private mutating func processBlock(_ base: UnsafeRawPointer) {
        var w: [UInt32] = [UInt32](repeating: 0, count: 64)
        for i in 0..<16 {
            w[i] = UInt32(bigEndian: base.loadUnaligned(fromByteOffset: i * 4, as: UInt32.self))
        }
        for i in 16..<64 {
            let s0 = w[i - 15].rotatedRight(7) ^ w[i - 15].rotatedRight(18) ^ (w[i - 15] >> 3)
            let s1 = w[i - 2].rotatedRight(17) ^ w[i - 2].rotatedRight(19) ^ (w[i - 2] >> 10)
            w[i] = w[i - 16] &+ s0 &+ w[i - 7] &+ s1
        }
        var a = state[0], b = state[1], c = state[2], d = state[3]
        var e = state[4], f = state[5], g = state[6], h = state[7]
        for i in 0..<64 {
            let s1 = e.rotatedRight(6) ^ e.rotatedRight(11) ^ e.rotatedRight(25)
            let ch = (e & f) ^ (~e & g)
            let temp1 = h &+ s1 &+ ch &+ Self.k[i] &+ w[i]
            let s0 = a.rotatedRight(2) ^ a.rotatedRight(13) ^ a.rotatedRight(22)
            let maj = (a & b) ^ (a & c) ^ (b & c)
            let temp2 = s0 &+ maj
            h = g; g = f; f = e; e = d &+ temp1
            d = c; c = b; b = a; a = temp1 &+ temp2
        }
        state[0] &+= a; state[1] &+= b; state[2] &+= c; state[3] &+= d
        state[4] &+= e; state[5] &+= f; state[6] &+= g; state[7] &+= h
    }
}

private extension UInt32 {
    func rotatedRight(_ amount: UInt32) -> UInt32 {
        (self >> amount) | (self << (32 - amount))
    }
}
