const std = @import("std");
const Allocator = std.mem.Allocator;
const math = std.math;
const mem = std.mem;
const fs = std.fs;
const posix = std.posix;
const Thread = std.Thread;
const Mutex = std.Thread.Mutex;
const Condition = std.Thread.Condition;

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

        // Unmap the memory-mapped file
        posix.munmap(@alignCast(self.file_data));
    }
};

// Thread pool for parallel processing
pub const ThreadPool = struct {
    const Self = @This();

    const Task = struct {
        func: *const fn (ctx: *anyopaque) void,
        ctx: *anyopaque,
    };

    threads: []Thread,
    tasks: std.ArrayList(Task),
    mutex: Mutex,
    condition: Condition,
    shutdown: bool,
    allocator: Allocator,

    pub fn init(allocator: Allocator, num_threads: usize) !*Self {
        const pool = try allocator.create(Self);
        pool.* = Self{
            .threads = try allocator.alloc(Thread, num_threads),
            .tasks = std.ArrayList(Task).init(allocator),
            .mutex = Mutex{},
            .condition = Condition{},
            .shutdown = false,
            .allocator = allocator,
        };

        // Start worker threads
        for (pool.threads, 0..) |*thread, i| {
            thread.* = try Thread.spawn(.{}, workerThread, .{ pool, i });
        }

        return pool;
    }

    pub fn deinit(self: *Self) void {
        // Signal shutdown
        self.mutex.lock();
        self.shutdown = true;
        self.condition.broadcast();
        self.mutex.unlock();

        // Wait for all threads to finish
        for (self.threads) |thread| {
            thread.join();
        }

        self.allocator.free(self.threads);
        self.tasks.deinit();
        self.allocator.destroy(self);
    }

    pub fn submit(self: *Self, func: *const fn (ctx: *anyopaque) void, ctx: *anyopaque) !void {
        self.mutex.lock();
        defer self.mutex.unlock();

        if (self.shutdown) return error.PoolShutdown;

        try self.tasks.append(.{ .func = func, .ctx = ctx });
        self.condition.signal();
    }

    pub fn waitForCompletion(self: *Self) void {
        while (true) {
            self.mutex.lock();
            const empty = self.tasks.items.len == 0;
            self.mutex.unlock();

            if (empty) break;
            std.time.sleep(1000000); // 1ms
        }

        // Give a bit more time for tasks to complete
        std.time.sleep(10000000); // 10ms
    }

    fn workerThread(self: *Self, thread_id: usize) void {
        _ = thread_id;
        while (true) {
            self.mutex.lock();

            // Wait for tasks or shutdown signal
            while (self.tasks.items.len == 0 and !self.shutdown) {
                self.condition.wait(&self.mutex);
            }

            if (self.shutdown) {
                self.mutex.unlock();
                break;
            }

            // Get a task
            const task = self.tasks.swapRemove(0);
            self.mutex.unlock();

            // Execute the task
            task.func(task.ctx);
        }
    }
};

// Memory mapping utilities
const MmapUtils = struct {
    const PAGE_SIZE = 4096; // Standard page size for most systems

    // Cross-platform memory mapping
    fn mmapFile(file: std.fs.File) ![]align(4096) const u8 {
        const file_size = try file.getEndPos();
        if (file_size == 0) return error.EmptyFile;

        const mapped_len = math.cast(usize, file_size) orelse return error.FileTooBig;

        // Get file descriptor
        const fd = file.handle;

        // Memory map the file with optimal flags
        const mapped_memory = try posix.mmap(null, // addr: let OS choose address
            mapped_len, // length: size of file
            posix.PROT.READ, // prot: read-only access
            .{ .TYPE = .PRIVATE }, // flags: private mapping
            fd, // fd: file descriptor
            0 // offset: start from beginning
        );

        return mapped_memory;
    }

    // Prefetch memory pages for better performance
    fn prefetchPages(data: []const u8) void {
        const page_size = PAGE_SIZE;
        const pages = (data.len + page_size - 1) / page_size;

        // Prefetch in chunks to avoid overwhelming the system
        const prefetch_chunk = 64; // 64 pages at a time
        var page: usize = 0;

        while (page < pages) {
            const chunk_end = @min(page + prefetch_chunk, pages);

            // Touch each page to bring it into memory
            while (page < chunk_end) : (page += 1) {
                const offset = page * page_size;
                if (offset < data.len) {
                    // Volatile read to prevent optimization
                    _ = @as(*volatile u8, @ptrCast(@constCast(&data[offset]))).*;
                }
            }

            // Small delay to avoid overwhelming memory subsystem
            if (page < pages) {
                std.time.sleep(1000); // 1 microsecond
            }
        }
    }

    // Advise the kernel about memory access patterns
    fn adviseSequentialAccess(data: []const u8) void {
        // Use madvise to hint sequential access pattern
        const ptr = @as([*]align(4096) u8, @ptrCast(@alignCast(@constCast(data.ptr))));
        posix.madvise(ptr, data.len, posix.MADV.SEQUENTIAL) catch {};

        // Also hint that we'll need the whole file
        posix.madvise(ptr, data.len, posix.MADV.WILLNEED) catch {};
    }
};

// Fast bit manipulation utilities with SIMD optimization
const BitUtils = struct {
    // SIMD-optimized byte comparison
    fn findBytesVectorized(data: []const u8, byte1: u8, byte2: u8, positions: *std.ArrayList(usize), newline_flags: *std.ArrayList(bool)) !void {
        const vector_size = 32; // Use 32-byte vectors for better performance
        const chunks = data.len / vector_size;

        var chunk_idx: usize = 0;
        while (chunk_idx < chunks) {
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
            while (combined_bits != 0) {
                const bit_pos = @ctz(combined_bits);
                const actual_pos = base + bit_pos;
                const is_newline = (bits1 & (@as(u32, 1) << @intCast(bit_pos))) != 0;

                try positions.append(actual_pos);
                try newline_flags.append(is_newline);
                combined_bits &= combined_bits - 1; // Clear lowest bit
            }

            chunk_idx += 1;
        }

        // Handle remainder
        const remainder_start = chunks * vector_size;
        for (data[remainder_start..], remainder_start..) |byte, pos| {
            if (byte == byte1) {
                try positions.append(pos);
                try newline_flags.append(true);
            } else if (byte == byte2) {
                try positions.append(pos);
                try newline_flags.append(false);
            }
        }
    }
};

// Chunk processing task for parallel execution
const ChunkTask = struct {
    loader: *CSVLoader,
    chunk_data: []const u8,
    columns: [][][]const u8,
    row_offset: usize,
    result: *usize, // Store number of rows processed
    mutex: *Mutex,

    pub fn process(ctx: *anyopaque) void {
        const task: *ChunkTask = @ptrCast(@alignCast(ctx));
        const rows_processed = task.loader.processChunkOptimized(task.chunk_data, task.columns, task.row_offset) catch 0;

        task.mutex.lock();
        task.result.* = rows_processed;
        task.mutex.unlock();
    }
};

pub const CSVLoader = struct {
    alloc: Allocator,
    filename: []const u8,
    n_columns: usize,
    delimiter: u8,
    thread_pool: *ThreadPool,

    // Pre-allocated buffers for reuse
    field_buffer: [][]const u8,

    pub fn init(alloc: Allocator, filename: []const u8, delimiter: u8, num_threads: usize) !*CSVLoader {
        var self = try alloc.create(CSVLoader);
        self.alloc = alloc;
        self.filename = filename;
        self.delimiter = delimiter;
        self.n_columns = 0;
        self.thread_pool = try ThreadPool.init(alloc, num_threads);

        // Pre-allocate field buffer for better performance
        self.field_buffer = try alloc.alloc([]const u8, 1024); // 1K fields per row

        return self;
    }

    pub fn deinit(self: *CSVLoader) void {
        self.thread_pool.deinit();
        self.alloc.free(self.field_buffer);
        self.alloc.destroy(self);
    }

    pub fn load(self: *CSVLoader) !CSVResult {
        const file = try fs.cwd().openFile(self.filename, .{});
        defer file.close();

        // Memory map the file
        const file_data = try MmapUtils.mmapFile(file);

        // Optimize memory access patterns
        MmapUtils.adviseSequentialAccess(file_data);

        // Prefetch pages in background for better performance
        MmapUtils.prefetchPages(file_data);

        std.debug.print("Successfully memory-mapped file: {d} bytes\n", .{file_data.len});

        // Detect column count from first row
        self.n_columns = try self.detectColumnCount(file_data);

        // Process file data with parallel chunking
        const result = try self.processWithParallelChunking(file_data);
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

    // FIXED: Added missing estimateRowsInChunk method
    fn estimateRowsInChunk(self: *CSVLoader, chunk_data: []const u8) usize {
        _ = self;
        // Simple estimation based on newline count
        return countNewlinesSIMDOptimized(chunk_data);
    }

    // New method using parallel chunking
    fn processWithParallelChunking(self: *CSVLoader, data: []const u8) !struct { columns: [][][]const u8 } {
        // First pass: count total rows using SIMD
        const total_rows = self.countTotalRows(data);

        std.debug.print("Total rows detected: {d}\n", .{total_rows});

        // Allocate column storage
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
            col.* = try self.alloc.alloc([]const u8, total_rows);
            allocated_columns += 1;
        }

        if (data.len == 0) return .{ .columns = columns };

        // For smaller files, process sequentially to avoid overhead
        if (data.len < 10 * 1024 * 1024) { // Less than 10MB
            _ = try self.processChunkOptimized(data, columns, 0);
            return .{ .columns = columns };
        }

        // Process data in parallel chunks for large files
        const num_threads = self.thread_pool.threads.len;
        const chunk_size = @max(1024 * 1024, data.len / num_threads); // At least 1MB per chunk

        var data_offset: usize = 0;
        var chunk_tasks = std.ArrayList(ChunkTask).init(self.alloc);
        defer chunk_tasks.deinit();

        var chunk_results = std.ArrayList(usize).init(self.alloc);
        defer chunk_results.deinit();

        var chunk_mutex = Mutex{};
        var current_row_offset: usize = 0;

        // Create chunks and submit to thread pool
        while (data_offset < data.len) {
            const chunk_end = @min(data_offset + chunk_size, data.len);

            // Find a safe chunk boundary (end at newline)
            var safe_end = chunk_end;
            if (chunk_end < data.len) {
                // Find the last newline in the chunk
                while (safe_end > data_offset and data[safe_end - 1] != '\n') {
                    safe_end -= 1;
                }

                // If no newline found, extend to find one
                if (safe_end == data_offset) {
                    safe_end = chunk_end;
                    while (safe_end < data.len and data[safe_end] != '\n') {
                        safe_end += 1;
                    }
                    if (safe_end < data.len) safe_end += 1; // Include the newline
                }
            }

            const chunk_data = data[data_offset..safe_end];

            // Create task for this chunk
            try chunk_results.append(0);
            const result_ptr = &chunk_results.items[chunk_results.items.len - 1];

            const task = ChunkTask{
                .loader = self,
                .chunk_data = chunk_data,
                .columns = columns,
                .row_offset = current_row_offset,
                .result = result_ptr,
                .mutex = &chunk_mutex,
            };

            try chunk_tasks.append(task);

            // Submit task to thread pool
            try self.thread_pool.submit(ChunkTask.process, &chunk_tasks.items[chunk_tasks.items.len - 1]);

            // Update counters for next chunk
            const estimated_rows = self.estimateRowsInChunk(chunk_data);
            current_row_offset += estimated_rows;
            data_offset = safe_end;
        }

        // Wait for all tasks to complete
        self.thread_pool.waitForCompletion();

        // Sum up actual results
        var actual_total_rows: usize = 0;
        for (chunk_results.items) |rows| {
            actual_total_rows += rows;
        }

        std.debug.print("Processed {d} chunks with {d} total rows\n", .{ chunk_tasks.items.len, actual_total_rows });

        return .{ .columns = columns };
    }

    // Count total rows using SIMD optimization
    fn countTotalRows(self: *CSVLoader, data: []const u8) usize {
        _ = self;
        return countNewlinesSIMDOptimized(data);
    }

    // Process a single chunk with SIMD optimization
    fn processChunkOptimized(self: *CSVLoader, chunk: []const u8, columns: [][][]const u8, row_offset: usize) !usize {
        if (chunk.len == 0) return 0;

        // Find all delimiters and newlines using SIMD
        var positions = std.ArrayList(usize).init(self.alloc);
        defer positions.deinit();

        var newline_flags = std.ArrayList(bool).init(self.alloc);
        defer newline_flags.deinit();

        try BitUtils.findBytesVectorized(chunk, '\n', self.delimiter, &positions, &newline_flags);

        // Process events with zero-copy field extraction
        var field_start: usize = 0;
        var current_row: usize = 0;
        var field_count: usize = 0;

        // Process events efficiently
        for (positions.items, newline_flags.items) |pos, is_newline| {
            if (is_newline) {
                // Add final field
                if (field_count < self.field_buffer.len) {
                    self.field_buffer[field_count] = chunk[field_start..pos];
                    field_count += 1;
                }

                // Assign all fields for this row using optimized dispatcher
                const actual_row = row_offset + current_row;
                if (actual_row < columns[0].len) {
                    self.assignFieldsDispatch(columns, self.field_buffer[0..field_count], actual_row);
                }

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
        }

        // Handle final row if no trailing newline
        if (field_start < chunk.len) {
            if (field_count < self.field_buffer.len) {
                self.field_buffer[field_count] = chunk[field_start..];
                field_count += 1;
            }
            const actual_row = row_offset + current_row;
            if (actual_row < columns[0].len) {
                self.assignFieldsDispatch(columns, self.field_buffer[0..field_count], actual_row);
            }
            current_row += 1;
        }

        return current_row;
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

    const filename = "data/test.csv";
    const num_threads = 4; // Adjust based on your CPU cores

    // Main execution with memory mapping and parallel processing
    var timer = try std.time.Timer.start();
    var loader = try CSVLoader.init(allocator, filename, ',', num_threads);
    defer loader.deinit();

    var result = try loader.load();
    const elapsed_time = @as(f64, @floatFromInt(timer.read()));

    // Use the cleanup method - CSVResult handles mmap cleanup
    defer result.deinit(allocator);

    // Print all data
    if (result.columns.len == 0) {
        std.debug.print("No data loaded.\n", .{});
        return;
    }

    const num_rows = result.columns[0].len;
    const num_cols = result.columns.len;

    std.debug.print("CSV Data ({d} rows, {d} columns) - Memory mapped with parallel processing\n\n", .{ num_rows, num_cols });

    // Print first few rows as sample
    const sample_rows = @min(5, num_rows); // Show max 10 rows
    for (0..sample_rows) |row| {
        std.debug.print("Row {d}: ", .{row + 1});
        for (0..num_cols) |col| {
            const cell_value = if (row < result.columns[col].len)
                result.columns[col][row]
            else
                "";
            if (col == num_cols - 1) {
                std.debug.print("{s}", .{cell_value});
            } else {
                std.debug.print("{s}, ", .{cell_value});
            }
        }
        std.debug.print("\n", .{});
    }

    if (num_rows > sample_rows) {
        std.debug.print("... ({d} more rows)\n", .{num_rows - sample_rows});
    }

    std.debug.print("\nSummary: {d} rows × {d} columns processed with {d} threads\n", .{ num_rows, num_cols, num_threads });
    std.debug.print("\nLoading completed in {d:.2} milliseconds.\n", .{elapsed_time / 1000000});
}
