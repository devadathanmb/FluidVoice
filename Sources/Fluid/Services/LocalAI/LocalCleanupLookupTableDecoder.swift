import Foundation
import MLX

/// Turns a FluidDecode lookup-table checkpoint (`"format": "fluiddecode-lut"`)
/// into a standard 8-bit affine MLX checkpoint that mlx-swift-lm loads as-is.
///
/// FluidDecode stores each projection as `<layer>.lut_packed` (uint32 words holding
/// one k-bit index per weight, packed least-significant bit first as one bitstream
/// per row, so 3-bit indices can straddle words) and `<layer>.lut_table` (float32,
/// one table of 2^k values per group of consecutive rows). The layout was confirmed
/// by matching decoded layers against Fluid-1 Mini, a fine-tune of the same base.
/// The speculative drafter (`*.dflash.*`) is dropped; the runtime does not use it.
nonisolated enum LocalCleanupLookupTableDecoder {
    struct Quantization: Equatable {
        let groupSize: Int
        let bits: Int
    }

    static let output = Quantization(groupSize: 64, bits: 8)

    /// Rows decoded per step, bounding peak memory for the 98k-row embedding.
    static let rowsPerChunk = 8192

    /// Convert `weights` into `destination` and return the quantization to record in
    /// config.json. `progress` receives the converted fraction, in order.
    static func convert(
        weights: URL,
        to destination: URL,
        progress: (Double) async -> Void
    ) async throws -> Quantization {
        let arrays = try loadArrays(url: weights)
        let packedKeys = arrays.keys.filter { $0.hasSuffix(".lut_packed") && !self.isDrafter($0) }.sorted()
        guard !packedKeys.isEmpty else {
            throw LocalCleanupError.unsupportedCheckpoint("no lookup-table layers")
        }
        let totalBytes = packedKeys.reduce(0) { $0 + (arrays[$1]?.nbytes ?? 0) }

        var converted: [String: MLXArray] = [:]
        for (key, value) in arrays where !self.isDrafter(key) && !key.contains(".lut_") {
            converted[key] = value
        }

        var doneBytes = 0
        for key in packedKeys {
            try Task.checkCancellation()
            let layer = String(key.dropLast(".lut_packed".count))
            guard let packed = arrays[key], let table = arrays[layer + ".lut_table"] else {
                throw LocalCleanupError.unsupportedCheckpoint("\(layer) is missing its lookup table")
            }
            let (weight, scales, biases) = try self.decodeAndQuantize(packed: packed, table: table, layer: layer)
            converted[layer + ".weight"] = weight
            converted[layer + ".scales"] = scales
            converted[layer + ".biases"] = biases
            doneBytes += packed.nbytes
            await progress(Double(doneBytes) / Double(totalBytes))
        }

        let temporary = destination.deletingLastPathComponent()
            .appendingPathComponent("model.converting.safetensors")
        try? FileManager.default.removeItem(at: temporary)
        try save(arrays: converted, metadata: ["format": "mlx"], url: temporary)
        try? FileManager.default.removeItem(at: destination)
        try FileManager.default.moveItem(at: temporary, to: destination)
        return self.output
    }

    /// Copy config.json, replacing the lookup-table quantization block with ours.
    static func writeConfig(from source: URL, to destination: URL, quantization: Quantization) throws {
        guard var config = try JSONSerialization.jsonObject(with: Data(contentsOf: source)) as? [String: Any] else {
            throw LocalCleanupError.unsupportedCheckpoint("config.json is not an object")
        }
        let block: [String: Any] = ["group_size": quantization.groupSize, "bits": quantization.bits]
        config["quantization"] = block
        config["quantization_config"] = block
        let data = try JSONSerialization.data(withJSONObject: config, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: destination, options: .atomic)
    }

    /// Decode one layer to its dense weights. `rows` must be a multiple of the table groups.
    static func decode(packed: MLXArray, table: MLXArray, rows range: Range<Int>) throws -> MLXArray {
        let rows = packed.dim(0)
        let words = packed.dim(1)
        let entries = table.dim(1)
        let bits = entries.trailingZeroBitCount
        guard entries > 1, entries == 1 << bits, (words * 32) % bits == 0,
              table.dim(0) > 0, rows % table.dim(0) == 0
        else {
            throw LocalCleanupError.unsupportedCheckpoint(
                "lookup table \(table.shape) does not fit packed \(packed.shape)"
            )
        }
        let rowsPerTable = rows / table.dim(0)
        guard !range.isEmpty, range.lowerBound >= 0, range.upperBound <= rows else {
            throw LocalCleanupError.unsupportedCheckpoint("row chunk \(range) is outside 0..<\(rows)")
        }
        guard range.lowerBound % rowsPerTable == 0, range.upperBound % rowsPerTable == 0 else {
            throw LocalCleanupError.unsupportedCheckpoint("row chunk is not aligned to lookup-table groups")
        }

        // Index j occupies bits [j*bits, (j+1)*bits) of its row's bitstream. The high
        // word only contributes when the index straddles a word boundary; it is zeroed
        // where the shift is 0 so no lane shifts a uint32 by 32.
        let columns = words * 32 / bits
        var lowWord = [Int32](repeating: 0, count: columns)
        var highWord = [Int32](repeating: 0, count: columns)
        var lowShift = [UInt32](repeating: 0, count: columns)
        var highShift = [UInt32](repeating: 0, count: columns)
        var highKeep = [UInt32](repeating: 0, count: columns)
        for column in 0..<columns {
            let position = column * bits
            let shift = position % 32
            lowWord[column] = Int32(position / 32)
            highWord[column] = Int32(min(position / 32 + 1, words - 1))
            lowShift[column] = UInt32(shift)
            highShift[column] = shift == 0 ? 0 : UInt32(32 - shift)
            highKeep[column] = shift == 0 ? 0 : 1
        }

        let chunk = packed[range.lowerBound..<range.upperBound]
        let low = rightShift(chunk.take(MLXArray(lowWord), axis: 1), MLXArray(lowShift))
        let high = leftShift(chunk.take(MLXArray(highWord), axis: 1) * MLXArray(highKeep), MLXArray(highShift))
        let indices = ((low | high) & MLXArray(UInt32(entries - 1))).asType(.int32)

        let tables = table[(range.lowerBound / rowsPerTable)..<(range.upperBound / rowsPerTable)]
        let rowTables = repeated(tables, count: rowsPerTable, axis: 0)
        return takeAlong(rowTables, indices, axis: 1)
    }

    private static func decodeAndQuantize(
        packed: MLXArray,
        table: MLXArray,
        layer: String
    ) throws -> (MLXArray, MLXArray, MLXArray) {
        let rows = packed.dim(0)
        let rowsPerTable = rows / max(table.dim(0), 1)
        let step = max(rowsPerTable, self.rowsPerChunk / max(rowsPerTable, 1) * rowsPerTable)
        var weights: [MLXArray] = []
        var scales: [MLXArray] = []
        var biases: [MLXArray] = []
        for start in stride(from: 0, to: rows, by: step) {
            let dense = try self.decode(packed: packed, table: table, rows: start..<min(start + step, rows))
            guard dense.dim(1) % self.output.groupSize == 0 else {
                throw LocalCleanupError.unsupportedCheckpoint("\(layer) width \(dense.dim(1)) is not a multiple of 64")
            }
            let quantized = MLX.quantized(
                dense.asType(.bfloat16), groupSize: self.output.groupSize, bits: self.output.bits
            )
            guard let bias = quantized.biases else {
                throw LocalCleanupError.unsupportedCheckpoint("affine quantization returned no biases")
            }
            eval(quantized.wq, quantized.scales, bias)
            weights.append(quantized.wq)
            scales.append(quantized.scales)
            biases.append(bias)
        }
        let result = (concatenated(weights, axis: 0), concatenated(scales, axis: 0), concatenated(biases, axis: 0))
        eval(result.0, result.1, result.2)
        return result
    }

    private static func isDrafter(_ key: String) -> Bool {
        key.contains(".dflash.")
    }
}
