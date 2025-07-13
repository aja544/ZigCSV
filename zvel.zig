const std = @import("std");
const Allocator = std.mem.Allocator;
const math = std.math;
const mem = std.mem;
const fs = std.fs;
const posix = std.posix;
const Thread = std.Thread;
const Mutex = std.Thread.Mutex;
const Condition = std.Thread.Condition;
const atomic = std.atomic;

// Custom error types for better error handling
pub const CSVError = error{
    InvalidFilename,
    InvalidDelimiter,
    InvalidThreadCount,
    FileTooBig,
    ChunkProcessingFailed,
    MemoryMappingNotSupported,
    OutOfMemory,
    InvalidData,
};

/// CSV parsing result containing parsed data and metadata
///
/// Memory management:
/// - If using arena allocator: no manual cleanup needed
/// - If using general-purpose allocator: call deinit() to free memory
pub const CSVResult = struct {
    columns: [][][]const u8,
    file_data: []const u8,
    is_memory_mapped: bool,
    file_handle: ?std.fs.File,

    pub fn deinit(self: *CSVResult, allocator: Allocator) void {
        // Handle file_data cleanup
        if (self.is_memory_mapped) {
            if (self.file_data.len > 0) {
                std.posix.munmap(@alignCast(self.file_data));
            }
            if (self.file_handle) |file| {
                // Try to close, but ignore errors (file may already be closed)
                file.close();
                self.file_handle = null;
            }
        } else {
            if (self.file_data.len > 0) {
                allocator.free(self.file_data);
            }
        }

        // Handle columns cleanup (only needed for non-arena allocators)
        // Note: If using arena allocator, this is redundant but safe
        if (self.columns.len > 0) {
            for (self.columns) |col| {
                if (col.len > 0) {
                    allocator.free(col);
                }
            }
            allocator.free(self.columns);
        }
    }
};

// Chunk processing result
pub const ChunkResult = struct {
    columns: [][][]const u8,
    rows_processed: usize,

    pub fn deinit() void {
        // No-op: arena allocator handles all memory cleanup automatically
        // This function exists for API consistency but is not needed
    }
};

// Chunk processing data with enhanced metadata
const ChunkData = struct {
    chunk: []const u8,
    start_offset: usize,
    end_offset: usize,
    is_first_chunk: bool,
    is_last_chunk: bool,
    result: ?ChunkResult,
    error_occurred: bool,
    error_msg: ?[]const u8,
    loader: *CSVLoader,
    chunk_index: usize,
    expected_rows: usize,
};

// Configuration options for CSV parsing
pub const CSVConfig = struct {
    delimiter: u8 = ',',
    num_threads: usize = 0, // 0 = auto-detect
    min_chunk_size: usize = 64 * 1024, // 64KB
    use_memory_mapping: bool = true,
    skip_header: bool = false,
    max_file_size: usize = std.math.maxInt(usize), // No limit by default
};

// High-performance CSV loader with memory mapping and SIMD
pub const CSVLoader = struct {
    const Self = @This();

    alloc: Allocator,
    filename: []const u8,
    n_columns: usize,
    config: CSVConfig,

    pub fn init(alloc: Allocator, filename: []const u8, config: CSVConfig) !*Self {
        // Input validation
        if (filename.len == 0) return error.InvalidFilename;
        if (config.delimiter == '\n' or config.delimiter == '\r') return error.InvalidDelimiter;

        var self = try alloc.create(Self);
        self.alloc = alloc;
        self.filename = filename;
        self.config = config;
        self.n_columns = 0;

        // Auto-detect thread count if not specified
        if (self.config.num_threads == 0) {
            self.config.num_threads = @max(std.Thread.getCpuCount() catch 2, 1);
        }

        return self;
    }

    pub fn deinit(self: *Self) void {
        self.alloc.destroy(self);
    }

    pub fn load(self: *Self) !CSVResult {
        const file = try fs.cwd().openFile(self.filename, .{});
        defer file.close(); // Always close file to prevent resource leaks

        const file_size = try file.getEndPos();

        // Check file size limits
        if (file_size > self.config.max_file_size) {
            return error.FileTooBig;
        }

        const mapped_len = math.cast(usize, file_size) orelse return error.FileTooBig;

        // Try memory mapping for files larger than 1MB
        const should_use_mmap = self.config.use_memory_mapping and file_size > 1024 * 1024;

        var file_data: []const u8 = undefined;
        var is_memory_mapped = false;
        var file_handle: ?std.fs.File = null;

        if (should_use_mmap) {
            // Try memory mapping
            const fd = file.handle;
            if (posix.mmap(
                null,
                mapped_len,
                posix.PROT.READ,
                std.posix.MAP{ .TYPE = .PRIVATE },
                fd,
                0,
            )) |mapped_ptr| {
                file_data = @as([*]const u8, @ptrCast(mapped_ptr))[0..mapped_len];
                is_memory_mapped = true;
                // Do NOT store file handle for memory-mapped files (it will be closed by Zig/OS)
                file_handle = null;
            } else |err| switch (err) {
                error.MemoryMappingNotSupported => {
                    // Fall back to regular file reading
                    const allocated_data = try self.alloc.alloc(u8, mapped_len);
                    errdefer self.alloc.free(allocated_data);
                    _ = try file.readAll(@constCast(allocated_data));
                    file_data = allocated_data;
                    is_memory_mapped = false;
                    file_handle = null;
                },
                else => return err,
            }
        } else {
            // Regular file reading for smaller files
            const allocated_data = try self.alloc.alloc(u8, mapped_len);
            errdefer self.alloc.free(allocated_data);
            _ = try file.readAll(@constCast(allocated_data));
            file_data = allocated_data;
            is_memory_mapped = false;
            // Only store file handle for non-memory-mapped files (but we already defer file.close())
            file_handle = null;
        }

        // Validate file data
        if (file_data.len == 0) {
            if (!is_memory_mapped) {
                self.alloc.free(file_data);
            }
            return error.InvalidData;
        }

        // Detect column count from first row
        self.n_columns = try self.detectColumnCount(file_data);

        // Process file data with multithreading (atomic chunk index)
        const result = try self.processFileMultithreaded(file_data);

        return CSVResult{
            .columns = result.columns,
            .file_data = file_data,
            .is_memory_mapped = is_memory_mapped,
            .file_handle = file_handle,
        };
    }

    fn detectColumnCount(self: *Self, data: []const u8) !usize {
        if (data.len == 0) return error.InvalidData;

        if (mem.indexOfScalar(u8, data, '\n')) |first_newline| {
            const first_row = data[0..first_newline];
            var count: usize = 1;

            // Use SIMD to count delimiters in first row
            const vector_size = 32;
            const simd_end = first_row.len - (first_row.len % vector_size);

            var base: usize = 0;
            while (base < simd_end) {
                const vec: @Vector(vector_size, u8) = first_row[base..][0..vector_size].*;
                const delim_pattern: @Vector(vector_size, u8) = @splat(self.config.delimiter);
                const mask = vec == delim_pattern;
                count += @popCount(@as(u32, @bitCast(mask)));
                base += vector_size;
            }

            // Handle remainder
            for (first_row[simd_end..]) |byte| {
                if (byte == self.config.delimiter) count += 1;
            }

            return count;
        }
        return 1; // Single row, single column
    }

    fn processFileMultithreaded(self: *Self, file_data: []const u8) !ChunkResult {
        if (file_data.len == 0) {
            const columns = try self.alloc.alloc([][]const u8, self.n_columns);
            for (columns) |*col| {
                col.* = try self.alloc.alloc([]const u8, 0);
            }
            return ChunkResult{ .columns = columns, .rows_processed = 0 };
        }

        // If file is small or only one thread, process sequentially
        if (self.config.num_threads == 1 or file_data.len < self.config.min_chunk_size * 4) {
            return try self.processChunk(file_data, true, true);
        }

        // Calculate optimal chunk size (more chunks than threads)
        const chunk_count = self.config.num_threads * 4;
        const target_chunk_size = file_data.len / chunk_count;
        const chunk_size = @max(target_chunk_size, self.config.min_chunk_size);

        var chunks = std.ArrayList(ChunkData).init(self.alloc);
        defer chunks.deinit();

        // Create chunks with proper boundaries (split at newlines)
        var offset: usize = 0;
        var chunk_index: usize = 0;
        const file_len = file_data.len;

        while (offset < file_len) {
            var end_offset = @min(offset + chunk_size, file_len);

            // Find next newline boundary - consolidated bounds check
            if (end_offset < file_len) {
                const search_start = end_offset;
                const search_end = file_len;

                // Use SIMD-optimized newline search for better performance
                var pos = search_start;
                const vector_size = 64;
                const simd_end = search_end - (search_end - search_start) % vector_size;

                while (pos < simd_end) {
                    const vec: @Vector(vector_size, u8) = file_data[pos..][0..vector_size].*;
                    const newline_pattern: @Vector(vector_size, u8) = @splat('\n');
                    const mask = vec == newline_pattern;
                    const newline_bits = @as(u64, @bitCast(mask));

                    if (newline_bits != 0) {
                        const first_newline = pos + @ctz(newline_bits);
                        end_offset = first_newline + 1;
                        break;
                    }
                    pos += vector_size;
                }

                // Handle remainder if no newline found in SIMD section
                if (pos >= simd_end) {
                    while (pos < search_end and file_data[pos] != '\n') {
                        pos += 1;
                    }
                    if (pos < search_end) {
                        end_offset = pos + 1;
                    }
                }
            }

            const chunk = file_data[offset..end_offset];
            const is_first = chunk_index == 0;
            const is_last = end_offset >= file_len;
            const estimated_rows = self.estimateRowsInChunk(chunk);
            try chunks.append(ChunkData{
                .chunk = chunk,
                .start_offset = offset,
                .end_offset = end_offset,
                .is_first_chunk = is_first,
                .is_last_chunk = is_last,
                .result = null,
                .error_occurred = false,
                .error_msg = null,
                .loader = self,
                .chunk_index = chunk_index,
                .expected_rows = estimated_rows,
            });
            offset = end_offset;
            chunk_index += 1;
        }

        // Atomic chunk index for lock-free work distribution
        var next_chunk = atomic.Value(usize).init(0);
        const threads = try self.alloc.alloc(Thread, self.config.num_threads);
        defer self.alloc.free(threads);

        // Launch threads
        for (threads) |*thread| {
            thread.* = try Thread.spawn(.{}, Self.worker, .{ self, chunks.items, &next_chunk });
        }
        // Run one worker on the main thread as well
        Self.worker(self, chunks.items, &next_chunk);
        // Wait for all threads to finish
        for (threads) |thread| {
            thread.join();
        }

        // Check for errors
        for (chunks.items) |*chunk_data| {
            if (chunk_data.error_occurred) {
                std.debug.print("Error in chunk {}: {s}\n", .{ chunk_data.chunk_index, chunk_data.error_msg orelse "Unknown error" });
                return error.ChunkProcessingFailed;
            }
        }

        // Merge results
        return try self.mergeChunkResults(chunks.items);
    }

    fn worker(self: *Self, chunks: []ChunkData, next_chunk: *atomic.Value(usize)) void {
        while (true) {
            const idx = next_chunk.fetchAdd(1, .acq_rel);
            if (idx >= chunks.len) break;
            const chunk_data = &chunks[idx];

            // Process chunk directly with proper error handling
            chunk_data.result = self.processChunk(chunk_data.chunk, chunk_data.is_first_chunk, chunk_data.is_last_chunk) catch |err| {
                std.debug.print("Error processing chunk {}: {}\n", .{ chunk_data.chunk_index, err });
                chunk_data.error_occurred = true;
                chunk_data.error_msg = @errorName(err);
                return;
            };
        }
    }

    fn estimateRowsInChunk(self: *Self, chunk: []const u8) usize {
        _ = self; // Suppress unused parameter warning
        return countNewlinesSIMD(chunk) + 1; // +1 for potential last row without newline
    }

    fn mergeChunkResults(self: *Self, chunk_results: []ChunkData) !ChunkResult {
        // Calculate total rows
        var total_rows: usize = 0;
        for (chunk_results) |chunk_data| {
            if (chunk_data.result) |result| {
                total_rows += result.rows_processed;
            }
        }

        // Allocate merged columns using arena allocator (no manual cleanup needed)
        var merged_columns = try self.alloc.alloc([][]const u8, self.n_columns);
        for (merged_columns) |*col| {
            col.* = try self.alloc.alloc([]const u8, total_rows);
        }

        // Pre-compute bounds for efficiency
        const max_cols = self.n_columns;

        // Merge data from all chunks with consolidated bounds checks
        var current_row: usize = 0;
        for (chunk_results) |chunk_data| {
            if (chunk_data.result) |result| {
                const chunk_rows = result.rows_processed;
                const result_cols = result.columns.len;

                // Process columns with consolidated bounds check
                for (0..max_cols) |col_idx| {
                    if (col_idx < result_cols) {
                        const src_col = result.columns[col_idx];
                        // Single bounds check for memcpy
                        if (src_col.len >= chunk_rows and current_row + chunk_rows <= total_rows) {
                            @memcpy(merged_columns[col_idx][current_row .. current_row + chunk_rows], src_col[0..chunk_rows]);
                        }
                    } else {
                        // Fill missing columns with empty strings - no bounds check needed
                        for (current_row..current_row + chunk_rows) |row_idx| {
                            if (row_idx < total_rows) {
                                merged_columns[col_idx][row_idx] = "";
                            }
                        }
                    }
                }

                current_row += chunk_rows;
            }
        }

        // Clean up individual chunk results to prevent memory leaks
        for (chunk_results) |*chunk_data| {
            if (chunk_data.result) |*result| {
                // Free individual column arrays
                for (result.columns) |col| {
                    self.alloc.free(col);
                }
                // Free the columns array itself
                self.alloc.free(result.columns);
            }
        }

        // Arena allocator handles all cleanup automatically
        return ChunkResult{ .columns = merged_columns, .rows_processed = total_rows };
    }

    fn processChunk(self: *Self, chunk: []const u8, is_first_chunk: bool, is_last_chunk: bool) !ChunkResult {
        _ = is_first_chunk; // May be used for header processing in future

        if (chunk.len == 0) {
            const columns = try self.alloc.alloc([][]const u8, self.n_columns);
            for (columns) |*col| {
                col.* = try self.alloc.alloc([]const u8, 0);
            }
            return ChunkResult{ .columns = columns, .rows_processed = 0 };
        }

        const newline_count = countNewlinesSIMD(chunk);
        const row_count = if (is_last_chunk and chunk.len > 0 and chunk[chunk.len - 1] != '\n')
            newline_count + 1
        else
            newline_count;

        // Allocate column storage using arena allocator (no manual cleanup needed)
        const columns = try self.alloc.alloc([][]const u8, self.n_columns);
        for (columns) |*col| {
            col.* = try self.alloc.alloc([]const u8, row_count);
        }

        if (row_count == 0) {
            return ChunkResult{ .columns = columns, .rows_processed = 0 };
        }

        // Parse CSV data using SIMD-optimized parsing
        var rows_processed: usize = 0;
        try self.parseCSVDataSIMD(chunk, columns, &rows_processed);

        return ChunkResult{ .columns = columns, .rows_processed = rows_processed };
    }

    fn parseCSVDataSIMD(self: *Self, chunk: []const u8, columns: [][][]const u8, rows_processed: *usize) !void {
        var field_start: usize = 0;
        var current_row: usize = 0;
        var current_col: usize = 0;

        const vector_size = 64;
        const simd_end = chunk.len - (chunk.len % vector_size);
        const max_rows = columns[0].len;
        const max_cols = self.n_columns;

        // SIMD processing for bulk of data
        var base: usize = 0;
        while (base < simd_end) {
            const vec: @Vector(vector_size, u8) = chunk[base..][0..vector_size].*;
            const newline_pattern: @Vector(vector_size, u8) = @splat('\n');
            const delim_pattern: @Vector(vector_size, u8) = @splat(self.config.delimiter);

            const newline_mask = vec == newline_pattern;
            const delim_mask = vec == delim_pattern;

            // Use bitcast to efficiently process matches
            const newline_bits = @as(u64, @bitCast(newline_mask));
            const delim_bits = @as(u64, @bitCast(delim_mask));
            const event_bits = newline_bits | delim_bits;

            var bits = event_bits;
            while (bits != 0) {
                const i = @ctz(bits);
                const shift_amount: u6 = @intCast(i);
                bits ^= @as(u64, 1) << shift_amount;
                const pos = base + i;

                if (newline_mask[i]) {
                    // End of row - consolidated bounds check
                    if (current_col < max_cols and current_row < max_rows) {
                        columns[current_col][current_row] = chunk[field_start..pos];
                    }
                    current_col += 1;

                    // Pad missing columns with empty strings - single bounds check
                    if (current_row < max_rows) {
                        while (current_col < max_cols) : (current_col += 1) {
                            columns[current_col][current_row] = "";
                        }
                    }

                    current_row += 1;
                    current_col = 0;
                    field_start = pos + 1;
                } else if (delim_mask[i]) {
                    // Field separator - consolidated bounds check
                    if (current_col < max_cols and current_row < max_rows) {
                        columns[current_col][current_row] = chunk[field_start..pos];
                    }
                    current_col += 1;
                    field_start = pos + 1;
                }
            }
            base += vector_size;
        }

        // Process remainder sequentially with consolidated bounds checks
        for (chunk[simd_end..], simd_end..) |byte, pos| {
            if (byte == '\n') {
                // End of row - single bounds check
                if (current_col < max_cols and current_row < max_rows) {
                    columns[current_col][current_row] = chunk[field_start..pos];
                }
                current_col += 1;

                // Pad missing columns - single bounds check
                if (current_row < max_rows) {
                    while (current_col < max_cols) : (current_col += 1) {
                        columns[current_col][current_row] = "";
                    }
                }

                current_row += 1;
                current_col = 0;
                field_start = pos + 1;
            } else if (byte == self.config.delimiter) {
                // Field separator - single bounds check
                if (current_col < max_cols and current_row < max_rows) {
                    columns[current_col][current_row] = chunk[field_start..pos];
                }
                current_col += 1;
                field_start = pos + 1;
            }
        }

        // Handle last field if chunk doesn't end with newline - consolidated bounds check
        if (field_start < chunk.len and current_col < max_cols and current_row < max_rows) {
            columns[current_col][current_row] = chunk[field_start..];
            current_col += 1;

            // Pad last row - single bounds check
            while (current_col < max_cols) : (current_col += 1) {
                columns[current_col][current_row] = "";
            }

            current_row += 1;
        }

        rows_processed.* = current_row;
    }
};

// High-performance SIMD newline counting
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

    // Handle remainder
    for (data[simd_end..]) |byte| {
        if (byte == '\n') count += 1;
    }

    return count;
}

// Main function with comprehensive testing
pub fn main() !void {
    // Use GPA for leak detection
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer {
        const leaked = gpa.deinit();
        if (leaked == .leak) {
            std.debug.print("Memory leak detected!\n", .{});
        }
    }
    const gpa_allocator = gpa.allocator();

    // Create a single arena allocator for all per-parse allocations
    var arena = std.heap.ArenaAllocator.init(gpa_allocator);
    defer arena.deinit();
    const arena_allocator = arena.allocator();

    var timer = try std.time.Timer.start();
    const num_threads = @max(std.Thread.getCpuCount() catch 8, 2);
    const config = CSVConfig{
        .delimiter = ',',
        .num_threads = num_threads,
        // You can set other config options here if needed
    };
    var loader = try CSVLoader.init(arena_allocator, "/home/yash/csvParser/data/data_large.csv", config);
    defer loader.deinit();

    const result = try loader.load();
    const elapsed_time = @as(f64, @floatFromInt(timer.read())) / 1000000;
    std.debug.print("Loading completed in {d:.2} milliseconds.\n\n", .{elapsed_time});

    // Print comprehensive summary
    if (result.columns.len == 0) {
        std.debug.print("No data loaded.\n", .{});
        return;
    }

    // Clean up the result properly
    var result_mut = result;
    defer result_mut.deinit(arena_allocator);

    const num_rows = result.columns[0].len;
    const num_cols = result.columns.len;
    const file_size_mb = @as(f64, @floatFromInt(result.file_data.len)) / (1024.0 * 1024.0);
    const throughput_mb_per_sec = file_size_mb / (elapsed_time / 1000.0);

    std.debug.print("CSV Data loaded successfully:\n", .{});
    std.debug.print("  Rows: {}\n", .{num_rows});
    std.debug.print("  Columns: {}\n", .{num_cols});
    std.debug.print("  File size: {d:.2} MB\n", .{file_size_mb});
    std.debug.print("  Memory mapped: {}\n", .{result.is_memory_mapped});
    std.debug.print("  Throughput: {d:.2} MB/s\n", .{throughput_mb_per_sec});
    std.debug.print("  Rows/second: {d:.0}\n", .{@as(f64, @floatFromInt(num_rows)) / (elapsed_time / 1000.0)});

    // Optional: Print sample data
    const sample_rows = @min(5, num_rows);
    if (sample_rows > 0) {
        std.debug.print("\nFirst {} rows (sample):\n", .{sample_rows});
        for (0..sample_rows) |row| {
            std.debug.print("Row {}: ", .{row + 1});
            for (0..@min(num_cols, 5)) |col| {
                const cell_value = result.columns[col][row];
                if (col == @min(num_cols, 5) - 1) {
                    std.debug.print("{s}", .{cell_value});
                } else {
                    std.debug.print("{s}, ", .{cell_value});
                }
            }
            if (num_cols > 5) {
                std.debug.print(" ... (and {} more columns)", .{num_cols - 5});
            }
            std.debug.print("\n", .{});
        }
    }
}
