const std = @import("std");
const Allocator = std.mem.Allocator;
const math = std.math;
const mem = std.mem;
const fs = std.fs;

pub const CSVLoader = struct {
    alloc: Allocator,
    filename: []const u8,
    n_columns: usize,
    delimiter: u8,
    num_threads: usize,

    pub const CSVResult = struct {
        columns: [][][]const u8,
        file_data: []const u8,
    };

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
        errdefer {
            for (columns) |col| self.alloc.free(col);
            self.alloc.free(columns);
        }

        for (columns) |*col| {
            col.* = try self.alloc.alloc([]const u8, row_count);
            errdefer {
                for (columns[0..columns.len]) |c| {
                    if (c.ptr == col.ptr) break;
                    self.alloc.free(c);
                }
                self.alloc.free(columns);
            }
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

pub fn main() !void {
    const allocator = std.heap.page_allocator;
    var timer = try std.time.Timer.start();
    var loader = try CSVLoader.init(allocator, "data/test.csv", ',', 2);
    defer loader.deinit();

    const result = try loader.load();
    const elapsed_time = @as(f64, @floatFromInt(timer.read()));
    std.debug.print("{} milli seconds.\n\n", .{elapsed_time / 1000000});
    defer {
        for (result.columns) |col| {
            allocator.free(col);
        }
        allocator.free(result.columns);
        allocator.free(result.file_data);
    }

    std.debug.print("First value: {s}\n", .{result.columns[0][0]});
    std.debug.print("First value: {s}\n", .{result.columns[2][0]});
}
