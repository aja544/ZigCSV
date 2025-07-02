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

pub const CSVLoader = struct {
    alloc: Allocator,
    filename: []const u8,
    n_columns: usize,
    delimiter: u8,
    num_threads: usize,

    pub fn init(alloc: Allocator, filename: []const u8, delimiter: u8, num_threads: usize) !*CSVLoader {
        var self = try alloc.create(CSVLoader);
        self.alloc = alloc;
        self.filename = filename;
        self.delimiter = delimiter;
        self.num_threads = num_threads;
        self.n_columns = 0;
        return self;
    }

    pub fn deinit(self: *CSVLoader) void {
        self.alloc.destroy(self);
    }

    pub fn load(self: *CSVLoader) !CSVResult {
        const file = try fs.cwd().openFile(self.filename, .{});
        defer file.close();
        const file_size = try file.getEndPos();
        const mapped_len = math.cast(usize, file_size) orelse return error.FileTooBig;

        // Read file and store in result for lifetime management
        const file_data = try self.alloc.alloc(u8, mapped_len);
        errdefer self.alloc.free(file_data); // Clean up on error

        _ = try file.readAll(file_data);

        // Detect column count from first row
        self.n_columns = try self.detectColumnCount(file_data);

        // Process file data
        const result = try self.processChunk(file_data, true);
        return .{ .columns = result.columns, .file_data = file_data };
    }

    fn detectColumnCount(self: *CSVLoader, data: []const u8) !usize {
        if (mem.indexOfScalar(u8, data, '\n')) |first_newline| {
            const first_row = data[0..first_newline];
            var count: usize = 1;
            for (first_row) |byte| {
                if (byte == self.delimiter) count += 1;
            }
            return count;
        }
        return error.NoNewlineFound;
    }

    fn processChunk(self: *CSVLoader, chunk: []const u8, is_last_chunk: bool) !struct { columns: [][][]const u8 } {
        const newline_count = countNewlinesSIMD(chunk);
        const row_count = if (chunk.len == 0) 0 else if (is_last_chunk and chunk[chunk.len - 1] != '\n')
            newline_count + 1
        else
            newline_count;

        // Allocate column storage
        var columns = try self.alloc.alloc([][]const u8, self.n_columns);
        errdefer self.alloc.free(columns);

        // Initialize columns array to track which ones are allocated
        var allocated_columns: usize = 0;
        errdefer {
            // Only free the columns that were successfully allocated
            for (columns[0..allocated_columns]) |col| {
                self.alloc.free(col);
            }
        }

        for (columns) |*col| {
            col.* = try self.alloc.alloc([]const u8, row_count);
            allocated_columns += 1; // Track successful allocations
        }

        if (chunk.len == 0) return .{ .columns = columns };

        var field_start: usize = 0;
        var current_row: usize = 0;
        var current_col: usize = 0;

        const vector_size = 64;
        const simd_end = chunk.len - (chunk.len % vector_size);

        // SIMD processing
        var base: usize = 0;
        while (base < simd_end) {
            const vec: @Vector(vector_size, u8) = chunk[base..][0..vector_size].*;
            const newline_pattern: @Vector(vector_size, u8) = @splat('\n');
            const delim_pattern: @Vector(vector_size, u8) = @splat(self.delimiter);
            const newline_mask = vec == newline_pattern;
            const delim_mask = vec == delim_pattern;
            const event_bits = @as(u64, @bitCast(newline_mask)) | @as(u64, @bitCast(delim_mask));

            var bits = event_bits;
            while (bits != 0) {
                const i = @ctz(bits);
                const shift_amount: u6 = @intCast(i);
                bits ^= @as(u64, 1) << shift_amount;
                const pos = base + i;

                if (newline_mask[i]) {
                    // Only write if within column bounds
                    if (current_col < self.n_columns and current_row < row_count) {
                        columns[current_col][current_row] = chunk[field_start..pos];
                    }
                    current_col += 1;
                    // Pad the row if it has fewer columns
                    while (current_col < self.n_columns and current_row < row_count) : (current_col += 1) {
                        columns[current_col][current_row] = "";
                    }
                    current_row += 1;
                    current_col = 0;
                    field_start = pos + 1;
                } else {
                    if (current_col < self.n_columns and current_row < row_count) {
                        columns[current_col][current_row] = chunk[field_start..pos];
                    }
                    current_col += 1;
                    field_start = pos + 1;
                }
            }
            base += vector_size;
        }

        // Process remainder
        for (chunk[simd_end..], simd_end..) |byte, pos| {
            if (byte == '\n') {
                if (current_col < self.n_columns and current_row < row_count) {
                    columns[current_col][current_row] = chunk[field_start..pos];
                }
                current_col += 1;
                while (current_col < self.n_columns and current_row < row_count) : (current_col += 1) {
                    columns[current_col][current_row] = "";
                }
                current_row += 1;
                current_col = 0;
                field_start = pos + 1;
            } else if (byte == self.delimiter) {
                if (current_col < self.n_columns and current_row < row_count) {
                    columns[current_col][current_row] = chunk[field_start..pos];
                }
                current_col += 1;
                field_start = pos + 1;
            }
        }

        // Handle last field
        if (field_start < chunk.len) {
            if (current_col < self.n_columns and current_row < row_count) {
                columns[current_col][current_row] = chunk[field_start..];
            }
            current_col += 1;
        }

        // Pad last row
        while (current_col < self.n_columns and current_row < row_count) : (current_col += 1) {
            columns[current_col][current_row] = "";
        }

        return .{ .columns = columns };
    }
};

fn countNewlinesSIMD(data: []const u8) usize {
    var count: usize = 0;
    const vector_size = 64;
    const simd_end = data.len - (data.len % vector_size);

    var base: usize = 0;
    while (base < simd_end) {
        const vec: @Vector(vector_size, u8) = data[base..][0..vector_size].*;
        const pattern: @Vector(vector_size, u8) = @splat('\n');
        const mask = vec == pattern;
        count += @popCount(@as(u64, @bitCast(mask)));
        base += vector_size;
    }

    for (data[simd_end..]) |byte| {
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
    var loader = try CSVLoader.init(allocator, "data/test_large_1m.csv", ',', 1);
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

    // const num_rows = result.columns[0].len;
    // const num_cols = result.columns.len;

    // std.debug.print("CSV Data ({d} rows, {d} columns):\n\n", .{ num_rows, num_cols });

    // // Print all rows in simple format
    // for (0..num_rows) |row| {
    //     std.debug.print("Row {d}: ", .{row + 1});
    //     for (0..num_cols) |col| {
    //         const cell_value = if (row < result.columns[col].len)
    //             result.columns[col][row]
    //         else
    //             "";
    //         if (col == num_cols - 1) {
    //             std.debug.print("'{s}'", .{cell_value});
    //         } else {
    //             std.debug.print("'{s}', ", .{cell_value});
    //         }
    //     }
    //     std.debug.print("\n", .{});
    // }

    // std.debug.print("\nSummary: {d} rows × {d} columns\n", .{ num_rows, num_cols });
}
