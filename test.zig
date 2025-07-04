const std = @import("std");
const Allocator = std.mem.Allocator;
const math = std.math;
const mem = std.mem;
const fs = std.fs;

// CSVResult is now a top-level struct, making it more accessible
pub const CSVResult = struct {
    columns: [][][]const u8,
    file_data: []const u8,

    // Cleanup method for CSVResult
    pub fn deinit(self: *CSVResult, allocator: Allocator) void {
        for (self.columns) |col| {
            allocator.free(col);
        }
        allocator.free(self.columns);
        allocator.free(self.file_data);
    }
};

// Compact event structure for better cache efficiency
const ParseEvent = packed struct(u64) {
    pos: u32, // 32-bit position (supports files up to 4GB)
    is_newline: u1, // 1-bit flag
    _padding: u31, // Padding to align to 64-bit
};

// Fast bit manipulation utilities
const BitUtils = struct {
    // Ultra-fast bit scanning using CPU intrinsics
    inline fn extractAllSetBits(mask: u64) [64]u6 {
        var positions: [64]u6 = undefined;
        var count: u6 = 0;
        var bits = mask;

        // Unrolled bit extraction for maximum speed
        comptime var i = 0;
        inline while (i < 8) : (i += 1) {
            if (bits != 0) {
                const pos = @ctz(bits);
                positions[count] = @intCast(pos);
                count += 1;
                bits &= bits - 1; // Clear lowest set bit
            }
        }

        return positions;
    }

    // SIMD-optimized byte comparison with early termination
    inline fn findBytesVectorized(data: []const u8, byte1: u8, byte2: u8, results: []ParseEvent) usize {
        var count: usize = 0;
        const vector_size = 32; // Use 32-byte vectors for better performance
        const chunks = data.len / vector_size;

        var chunk_idx: usize = 0;
        while (chunk_idx < chunks and count < results.len - 64) {
            const base = chunk_idx * vector_size;
            const vec: @Vector(vector_size, u8) = data[base..][0..vector_size].*;

            const pattern1: @Vector(vector_size, u8) = @splat(byte1);
            const pattern2: @Vector(vector_size, u8) = @splat(byte2);

            const mask1 = vec == pattern1;
            const mask2 = vec == pattern2;

            // Convert to integers for bit manipulation
            const bits1 = @as(u32, @bitCast(mask1));
            const bits2 = @as(u32, @bitCast(mask2));

            // Process both patterns simultaneously
            var combined_bits = bits1 | bits2;
            while (combined_bits != 0 and count < results.len) {
                const bit_pos = @ctz(combined_bits);
                const actual_pos = base + bit_pos;

                results[count] = ParseEvent{
                    .pos = @intCast(actual_pos),
                    .is_newline = if ((bits1 & (@as(u32, 1) << @intCast(bit_pos))) != 0) 1 else 0,
                    ._padding = 0,
                };
                count += 1;

                combined_bits &= combined_bits - 1; // Clear lowest bit
            }

            chunk_idx += 1;
        }

        // Handle remainder
        const remainder_start = chunks * vector_size;
        for (data[remainder_start..], remainder_start..) |byte, pos| {
            if (count >= results.len) break;
            if (byte == byte1) {
                results[count] = ParseEvent{ .pos = @intCast(pos), .is_newline = 1, ._padding = 0 };
                count += 1;
            } else if (byte == byte2) {
                results[count] = ParseEvent{ .pos = @intCast(pos), .is_newline = 0, ._padding = 0 };
                count += 1;
            }
        }

        return count;
    }
};

pub const CSVLoader = struct {
    alloc: Allocator,
    filename: []const u8,
    n_columns: usize,
    delimiter: u8,
    num_threads: usize,

    // Pre-allocated buffers for reuse
    event_buffer: []ParseEvent,
    field_buffer: [][]const u8,

    pub fn init(alloc: Allocator, filename: []const u8, delimiter: u8, num_threads: usize) !*CSVLoader {
        var self = try alloc.create(CSVLoader);
        self.alloc = alloc;
        self.filename = filename;
        self.delimiter = delimiter;
        self.num_threads = num_threads;
        self.n_columns = 0;

        // Pre-allocate buffers for better performance
        self.event_buffer = try alloc.alloc(ParseEvent, 1024 * 1024); // 1M events
        self.field_buffer = try alloc.alloc([]const u8, 1024); // 1K fields per row

        return self;
    }

    pub fn deinit(self: *CSVLoader) void {
        self.alloc.free(self.event_buffer);
        self.alloc.free(self.field_buffer);
        self.alloc.destroy(self);
    }

    pub fn load(self: *CSVLoader) !CSVResult {
        const file = try fs.cwd().openFile(self.filename, .{});
        defer file.close();
        const file_size = try file.getEndPos();
        const mapped_len = math.cast(usize, file_size) orelse return error.FileTooBig;

        // Read file and store in result for lifetime management
        const file_data = try self.alloc.alloc(u8, mapped_len);
        errdefer self.alloc.free(file_data);

        _ = try file.readAll(file_data);

        // Detect column count from first row
        self.n_columns = try self.detectColumnCount(file_data);

        // Process file data with ultra-optimized version
        const result = try self.processChunkUltraFast(file_data, true);
        return .{ .columns = result.columns, .file_data = file_data };
    }

    fn detectColumnCount(self: *CSVLoader, data: []const u8) !usize {
        if (mem.indexOfScalar(u8, data, '\n')) |first_newline| {
            const first_row = data[0..first_newline];
            var count: usize = 1;
            // Use SIMD for delimiter counting in first row
            const vector_size = 32;
            const chunks = first_row.len / vector_size;

            var chunk_idx: usize = 0;
            while (chunk_idx < chunks) {
                const base = chunk_idx * vector_size;
                const vec: @Vector(vector_size, u8) = first_row[base..][0..vector_size].*;
                const pattern: @Vector(vector_size, u8) = @splat(self.delimiter);
                const mask = vec == pattern;
                count += @popCount(@as(u32, @bitCast(mask)));
                chunk_idx += 1;
            }

            // Handle remainder
            for (first_row[chunks * vector_size ..]) |byte| {
                if (byte == self.delimiter) count += 1;
            }

            return count;
        }
        return error.NoNewlineFound;
    }

    // Ultra-fast column assignment using function templates
    fn assignFieldsTemplate(comptime col_count: usize, columns: [][][]const u8, fields: [][]const u8, row_index: usize) void {
        comptime var i = 0;
        inline while (i < col_count) : (i += 1) {
            if (i < fields.len and row_index < columns[i].len) {
                columns[i][row_index] = fields[i];
            } else if (row_index < columns[i].len) {
                columns[i][row_index] = "";
            }
        }
    }

    // Memory-efficient field assignment dispatcher
    fn assignFieldsDispatch(self: *CSVLoader, columns: [][][]const u8, fields: [][]const u8, row_index: usize) void {
        // Use compile-time specialization for common column counts
        switch (self.n_columns) {
            1 => assignFieldsTemplate(1, columns, fields, row_index),
            2 => assignFieldsTemplate(2, columns, fields, row_index),
            3 => assignFieldsTemplate(3, columns, fields, row_index),
            4 => assignFieldsTemplate(4, columns, fields, row_index),
            5 => assignFieldsTemplate(5, columns, fields, row_index),
            6 => assignFieldsTemplate(6, columns, fields, row_index),
            7 => assignFieldsTemplate(7, columns, fields, row_index),
            8 => assignFieldsTemplate(8, columns, fields, row_index),
            else => {
                // General case with 16-way unrolling for large column counts
                const field_count = @min(fields.len, self.n_columns);
                var col: usize = 0;

                // 16-way unrolled loop
                while (col + 16 <= field_count) {
                    comptime var j = 0;
                    inline while (j < 16) : (j += 1) {
                        if (row_index < columns[col + j].len) {
                            columns[col + j][row_index] = fields[col + j];
                        }
                    }
                    col += 16;
                }

                // Handle remainder
                while (col < field_count) : (col += 1) {
                    if (row_index < columns[col].len) {
                        columns[col][row_index] = fields[col];
                    }
                }

                // Fill empty columns
                while (col < self.n_columns) : (col += 1) {
                    if (row_index < columns[col].len) {
                        columns[col][row_index] = "";
                    }
                }
            },
        }
    }

    // Ultra-optimized chunk processing with minimal overhead
    fn processChunkUltraFast(self: *CSVLoader, chunk: []const u8, is_last_chunk: bool) !struct { columns: [][][]const u8 } {
        const newline_count = countNewlinesSIMDOptimized(chunk);
        const row_count = if (chunk.len == 0) 0 else if (is_last_chunk and chunk[chunk.len - 1] != '\n')
            newline_count + 1
        else
            newline_count;

        // Allocate column storage with better memory alignment
        var columns = try self.alloc.alloc([][]const u8, self.n_columns);
        errdefer self.alloc.free(columns);

        var allocated_columns: usize = 0;
        errdefer {
            for (columns[0..allocated_columns]) |col| {
                self.alloc.free(col);
            }
        }

        // Allocate all columns at once for better memory locality
        for (columns) |*col| {
            col.* = try self.alloc.alloc([]const u8, row_count);
            allocated_columns += 1;
        }

        if (chunk.len == 0) return .{ .columns = columns };

        // Ultra-fast event finding
        const event_count = BitUtils.findBytesVectorized(chunk, '\n', self.delimiter, self.event_buffer);

        // Process events with zero-copy field extraction
        var field_start: usize = 0;
        var current_row: usize = 0;
        var field_count: usize = 0;

        // Process events in batches for better cache performance
        const batch_size = 64; // Process 64 events at once
        var event_idx: usize = 0;

        while (event_idx < event_count) {
            const batch_end = @min(event_idx + batch_size, event_count);

            // Process current batch
            while (event_idx < batch_end and current_row < row_count) {
                const event = self.event_buffer[event_idx];
                const pos: usize = event.pos;

                if (event.is_newline == 1) {
                    // Add final field
                    if (field_count < self.field_buffer.len) {
                        self.field_buffer[field_count] = chunk[field_start..pos];
                        field_count += 1;
                    }

                    // Assign all fields for this row using optimized dispatcher
                    self.assignFieldsDispatch(columns, self.field_buffer[0..field_count], current_row);

                    // Reset for next row
                    current_row += 1;
                    field_count = 0;
                    field_start = pos + 1;
                } else {
                    // Add field to buffer
                    if (field_count < self.field_buffer.len) {
                        self.field_buffer[field_count] = chunk[field_start..pos];
                        field_count += 1;
                    }
                    field_start = pos + 1;
                }

                event_idx += 1;
            }
        }

        // Handle final row
        if (field_start < chunk.len and current_row < row_count) {
            if (field_count < self.field_buffer.len) {
                self.field_buffer[field_count] = chunk[field_start..];
                field_count += 1;
            }
            self.assignFieldsDispatch(columns, self.field_buffer[0..field_count], current_row);
        }

        return .{ .columns = columns };
    }
};

// Hyper-optimized newline counting with larger vectors
fn countNewlinesSIMDOptimized(data: []const u8) usize {
    var count: usize = 0;
    const vector_size = 32; // Use 32-byte vectors
    const chunks = data.len / vector_size;

    // Process multiple chunks in parallel
    var chunk_idx: usize = 0;
    while (chunk_idx + 4 <= chunks) {
        // Process 4 chunks simultaneously
        const base1 = chunk_idx * vector_size;
        const base2 = (chunk_idx + 1) * vector_size;
        const base3 = (chunk_idx + 2) * vector_size;
        const base4 = (chunk_idx + 3) * vector_size;

        const vec1: @Vector(vector_size, u8) = data[base1..][0..vector_size].*;
        const vec2: @Vector(vector_size, u8) = data[base2..][0..vector_size].*;
        const vec3: @Vector(vector_size, u8) = data[base3..][0..vector_size].*;
        const vec4: @Vector(vector_size, u8) = data[base4..][0..vector_size].*;

        const pattern: @Vector(vector_size, u8) = @splat('\n');

        const mask1 = vec1 == pattern;
        const mask2 = vec2 == pattern;
        const mask3 = vec3 == pattern;
        const mask4 = vec4 == pattern;

        count += @popCount(@as(u32, @bitCast(mask1)));
        count += @popCount(@as(u32, @bitCast(mask2)));
        count += @popCount(@as(u32, @bitCast(mask3)));
        count += @popCount(@as(u32, @bitCast(mask4)));

        chunk_idx += 4;
    }

    // Handle remaining chunks
    while (chunk_idx < chunks) {
        const base = chunk_idx * vector_size;
        const vec: @Vector(vector_size, u8) = data[base..][0..vector_size].*;
        const pattern: @Vector(vector_size, u8) = @splat('\n');
        const mask = vec == pattern;
        count += @popCount(@as(u32, @bitCast(mask)));
        chunk_idx += 1;
    }

    // Handle remainder bytes
    for (data[chunks * vector_size ..]) |byte| {
        if (byte == '\n') count += 1;
    }

    return count;
}

// Test with GeneralPurposeAllocator to detect leaks
pub fn main() !void {
    // Use GPA for leak detection
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer {
        const leaked = gpa.deinit();
        if (leaked == .leak) {
            std.debug.print("Memory leak detected!\n", .{});
        }
    }
    const allocator = gpa.allocator();

    var timer = try std.time.Timer.start();
    var loader = try CSVLoader.init(allocator, "data/sample.csv", ',', 1);
    defer loader.deinit();

    const result = try loader.load();
    const elapsed_time = @as(f64, @floatFromInt(timer.read()));
    std.debug.print("Loading completed in {d} milli seconds.\n\n", .{elapsed_time / 1000000});

    // Use the cleanup method - note that CSVResult is now a top-level type
    var result_copy = result;
    defer result_copy.deinit(allocator);

    // Print all data
    if (result.columns.len == 0) {
        std.debug.print("No data loaded.\n", .{});
        return;
    }

    const num_rows = result.columns[0].len;
    const num_cols = result.columns.len;

    std.debug.print("CSV Data ({d} rows, {d} columns):\n\n", .{ num_rows, num_cols });

    // Print first few rows as sample
    const sample_rows = @min(5, num_rows);
    for (0..sample_rows) |row| {
        std.debug.print("Row {d}: ", .{row + 1});
        for (0..num_cols) |col| {
            const cell_value = if (row < result.columns[col].len)
                result.columns[col][row]
            else
                "";
            if (col == num_cols - 1) {
                std.debug.print("'{s}'", .{cell_value});
            } else {
                std.debug.print("'{s}', ", .{cell_value});
            }
        }
        std.debug.print("\n", .{});
    }

    if (num_rows > sample_rows) {
        std.debug.print("... ({d} more rows)\n", .{num_rows - sample_rows});
    }

    std.debug.print("\nSummary: {d} rows × {d} columns\n", .{ num_rows, num_cols });
}
// probabl the best..
