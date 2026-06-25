// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <atomic>
#include <array>
#include <chrono>
#include <cstdint>
#include <latch>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>

#include <tt-metalium/experimental/realtime_profiler.hpp>

#include "tt_metal/common/broadcast_ring.hpp"

namespace tt::tt_metal {
namespace {

using tt::tt_metal::experimental::ProgramRealtimeRecord;

ProgramRealtimeRecord make_record(uint32_t runtime_id, std::span<const std::string_view> sources = {}) {
    return ProgramRealtimeRecord{
        .runtime_id = runtime_id,
        .chip_id = 7,
        .start_timestamp = 100 + runtime_id,
        .end_timestamp = 200 + runtime_id,
        .frequency = 1.5,
        .kernel_sources = sources,
    };
}

TEST(BroadcastRing, RoundsCapacityAndReaderStartsAtTail) {
    BroadcastRing<uint32_t> ring(3);
    EXPECT_EQ(ring.capacity(), 4u);

    ring.writer().publish(1);
    auto reader = ring.make_reader();

    uint32_t value = 0;
    EXPECT_FALSE(reader.read(value));

    ring.writer().publish(2);
    EXPECT_TRUE(reader.read(value));
    EXPECT_EQ(value, 2u);
    EXPECT_FALSE(reader.read(value));
}

TEST(BroadcastRing, CarriesProgramRealtimeRecordBatches) {
    std::string_view sources[] = {"kernel_a.cpp", "kernel_b.cpp"};
    BroadcastRing<ProgramRealtimeRecord> ring(8);
    auto reader_a = ring.make_reader();
    auto reader_b = ring.make_reader();

    ProgramRealtimeRecord records[] = {
        make_record(1, sources),
        make_record(2, sources),
        make_record(3, sources),
    };
    ring.writer().publish_batch(records);

    ProgramRealtimeRecord out_a[3];
    ProgramRealtimeRecord out_b[3];
    auto batch_a = reader_a.read_batch(out_a);
    auto batch_b = reader_b.read_batch(out_b);

    ASSERT_EQ(batch_a.size(), 3u);
    ASSERT_EQ(batch_b.size(), 3u);
    for (size_t i = 0; i < batch_a.size(); ++i) {
        const auto& record_a = batch_a[i];
        const auto& record_b = batch_b[i];
        EXPECT_EQ(record_a.runtime_id, i + 1);
        EXPECT_EQ(record_b.runtime_id, i + 1);
        ASSERT_EQ(record_a.kernel_sources.size(), 2u);
        EXPECT_EQ(record_a.kernel_sources[0], "kernel_a.cpp");
        EXPECT_EQ(record_a.kernel_sources[1], "kernel_b.cpp");
    }
}

TEST(BroadcastRing, OversizedBatchKeepsLastCapacityItems) {
    BroadcastRing<uint32_t> ring(4);
    auto reader = ring.make_reader();

    uint32_t records[] = {0, 1, 2, 3, 4, 5};
    ring.writer().publish_batch(records);

    uint32_t out[4] = {};
    auto batch = reader.read_batch(out);

    ASSERT_EQ(batch.size(), 4u);
    EXPECT_EQ(reader.dropped(), 2u);
    EXPECT_EQ(batch[0], 2u);
    EXPECT_EQ(batch[1], 3u);
    EXPECT_EQ(batch[2], 4u);
    EXPECT_EQ(batch[3], 5u);
}

TEST(BroadcastRing, ReadersTrackLagIndependently) {
    BroadcastRing<uint32_t> ring(4);
    auto fast_reader = ring.make_reader();
    auto slow_reader = ring.make_reader();

    uint32_t first_batch[] = {1, 2};
    ring.writer().publish_batch(first_batch);

    uint32_t out[4] = {};
    ASSERT_EQ(fast_reader.read_batch(out).size(), 2u);
    EXPECT_EQ(out[0], 1u);
    EXPECT_EQ(out[1], 2u);
    EXPECT_EQ(fast_reader.dropped(), 0u);

    uint32_t second_batch[] = {3, 4, 5, 6};
    ring.writer().publish_batch(second_batch);

    auto fast_batch = fast_reader.read_batch(out);
    ASSERT_EQ(fast_batch.size(), 4u);
    EXPECT_EQ(fast_reader.dropped(), 0u);
    EXPECT_EQ(fast_batch.front(), 3u);
    EXPECT_EQ(fast_batch.back(), 6u);

    auto slow_batch = slow_reader.read_batch(out);
    ASSERT_EQ(slow_batch.size(), 4u);
    EXPECT_EQ(slow_reader.dropped(), 2u);
    EXPECT_EQ(slow_batch.front(), 3u);
    EXPECT_EQ(slow_batch.back(), 6u);
}

TEST(BroadcastRing, WakeReadersAfterPublishWakesWaitingReader) {
    BroadcastRing<uint32_t> ring(4);
    auto reader = ring.make_reader();
    std::latch waiting{1};
    std::atomic<bool> received{false};

    std::thread reader_thread([&]() {
        uint32_t out = 0;
        EXPECT_FALSE(reader.read(out));
        waiting.count_down();
        reader.wait();
        received.store(reader.read(out) && out == 42, std::memory_order_release);
    });

    waiting.wait();
    ring.writer().publish(42);
    ring.writer().wake_readers();

    reader_thread.join();
    EXPECT_TRUE(received.load(std::memory_order_acquire));
}

TEST(BroadcastRing, WakeReadersWithoutPublishWakesWaitingReader) {
    BroadcastRing<uint32_t> ring(4);
    auto reader = ring.make_reader();
    std::latch waiting{1};
    std::atomic<bool> woke{false};

    std::thread reader_thread([&]() {
        uint32_t out = 0;
        EXPECT_FALSE(reader.read(out));
        waiting.count_down();
        reader.wait();
        EXPECT_FALSE(reader.read(out));
        woke.store(true, std::memory_order_release);
    });

    waiting.wait();
    ring.writer().wake_readers();

    reader_thread.join();
    EXPECT_TRUE(woke.load(std::memory_order_acquire));
}

TEST(BroadcastRing, WakeBeforeWaitDoesNotBlockReader) {
    BroadcastRing<uint32_t> ring(4);
    auto reader = ring.make_reader();
    uint32_t out = 0;

    EXPECT_FALSE(reader.read(out));
    ring.writer().publish(42);
    ring.writer().wake_readers();

    reader.wait();
    EXPECT_TRUE(reader.read(out));
    EXPECT_EQ(out, 42u);
}

TEST(BroadcastRing, ConcurrentProducerAndReadersMaintainOrderUnderPressure) {
    constexpr uint64_t kTotalRecords = 1u << 20;
    constexpr size_t kCapacity = 4096;
    constexpr size_t kReadBatchSize = 257;
    constexpr size_t kMaxPublishBatchSize = 513;

    struct ReaderStats {
        uint64_t received = 0;
        uint64_t dropped = 0;
        uint64_t last = 0;
        bool has_last = false;
        bool in_range = true;
        bool strictly_increasing = true;
    };

    BroadcastRing<uint64_t> ring(kCapacity);
    auto fast_reader = ring.make_reader();
    auto slow_reader = ring.make_reader();
    ReaderStats fast_stats;
    ReaderStats slow_stats;
    std::latch ready{3};
    std::latch start{1};
    std::atomic<bool> writer_done{false};

    auto run_reader = [&](BroadcastRing<uint64_t>::Reader& reader, ReaderStats& stats, bool throttle) {
        std::array<uint64_t, kReadBatchSize> out = {};
        uint32_t reads = 0;
        ready.count_down();
        start.wait();
        while (true) {
            auto batch = reader.read_batch(std::span<uint64_t>(out));
            if (batch.empty()) {
                if (writer_done.load(std::memory_order_acquire)) {
                    break;
                }
                std::this_thread::yield();
                continue;
            }
            for (uint64_t value : batch) {
                stats.in_range &= value < kTotalRecords;
                stats.strictly_increasing &= !stats.has_last || value > stats.last;
                stats.last = value;
                stats.has_last = true;
                stats.received++;
            }
            if (throttle && (++reads % 8 == 0)) {
                std::this_thread::sleep_for(std::chrono::microseconds(50));
            }
        }
        stats.dropped = reader.dropped();
    };

    std::thread fast_thread(run_reader, std::ref(fast_reader), std::ref(fast_stats), false);
    std::thread slow_thread(run_reader, std::ref(slow_reader), std::ref(slow_stats), true);
    std::thread writer_thread([&]() {
        ready.count_down();
        start.wait();
        std::array<uint64_t, kMaxPublishBatchSize> records = {};
        uint64_t next = 0;
        while (next < kTotalRecords) {
            const uint64_t base = next;
            const size_t batch_size =
                std::min<uint64_t>(1 + ((base * 1103515245u + 12345u) % kMaxPublishBatchSize), kTotalRecords - base);
            for (size_t k = 0; k < batch_size; ++k) {
                records[k] = base + k;
            }
            ring.writer().publish_batch(std::span<const uint64_t>(records.data(), batch_size));
            next += batch_size;
            if ((next & 0x3fffu) == 0) {
                std::this_thread::yield();
            }
        }
        writer_done.store(true, std::memory_order_release);
    });

    ready.wait();
    start.count_down();
    writer_thread.join();
    fast_thread.join();
    slow_thread.join();

    EXPECT_TRUE(fast_stats.in_range);
    EXPECT_TRUE(fast_stats.strictly_increasing);
    EXPECT_TRUE(slow_stats.in_range);
    EXPECT_TRUE(slow_stats.strictly_increasing);
    EXPECT_EQ(fast_stats.received + fast_stats.dropped, kTotalRecords);
    EXPECT_EQ(slow_stats.received + slow_stats.dropped, kTotalRecords);
    EXPECT_GT(fast_stats.received, 0u);
    EXPECT_GT(slow_stats.received, 0u);
    EXPECT_GT(slow_stats.dropped, 0u);
}

TEST(BroadcastRing, SlowReaderDropsOldProgramRealtimeRecords) {
    BroadcastRing<ProgramRealtimeRecord> ring(4);
    auto reader = ring.make_reader();

    std::vector<ProgramRealtimeRecord> records;
    records.reserve(10);
    for (uint32_t i = 0; i < 10; ++i) {
        records.push_back(make_record(i));
    }
    ring.writer().publish_batch(records);

    ProgramRealtimeRecord out[4];
    auto batch = reader.read_batch(out);

    ASSERT_EQ(batch.size(), 4u);
    EXPECT_EQ(reader.dropped(), 6u);
    EXPECT_EQ(batch.front().runtime_id, 6u);
    EXPECT_EQ(batch.back().runtime_id, 9u);
}

TEST(BroadcastRing, SupportsNonTrivialRecords) {
    struct NonTrivialRecord {
        uint32_t runtime_id = 0;
        std::string name;
    };
    static_assert(!BroadcastRing<NonTrivialRecord>::is_always_lock_free);
    static_assert(!std::is_trivially_copyable_v<NonTrivialRecord>);

    BroadcastRing<NonTrivialRecord> ring(4);
    auto reader = ring.make_reader();

    std::vector<NonTrivialRecord> records;
    records.reserve(7);
    for (uint32_t i = 0; i < 7; ++i) {
        records.push_back({.runtime_id = i, .name = "kernel_" + std::to_string(i)});
    }
    ring.writer().publish_batch(records);

    NonTrivialRecord out[4];
    auto batch = reader.read_batch(out);

    ASSERT_EQ(batch.size(), 4u);
    EXPECT_EQ(reader.dropped(), 3u);
    EXPECT_EQ(batch.front().runtime_id, 3u);
    EXPECT_EQ(batch.front().name, "kernel_3");
    EXPECT_EQ(batch.back().runtime_id, 6u);
    EXPECT_EQ(batch.back().name, "kernel_6");
}

TEST(BroadcastRing, MovesNonTrivialRecordsFromStagingBuffer) {
    struct NonTrivialRecord {
        uint32_t runtime_id = 0;
        std::string name;
    };

    BroadcastRing<NonTrivialRecord> ring(4);
    auto reader = ring.make_reader();

    std::vector<NonTrivialRecord> records;
    records.reserve(4);
    for (uint32_t i = 0; i < 4; ++i) {
        records.push_back({.runtime_id = i, .name = std::string(256, static_cast<char>('a' + i))});
    }
    ring.writer().publish_batch_move(std::span<NonTrivialRecord>(records));

    NonTrivialRecord out[4];
    auto batch = reader.read_batch(out);

    ASSERT_EQ(batch.size(), 4u);
    for (uint32_t i = 0; i < 4; ++i) {
        EXPECT_EQ(batch[i].runtime_id, i);
        EXPECT_EQ(batch[i].name, std::string(256, static_cast<char>('a' + i)));
    }
}

}  // namespace
}  // namespace tt::tt_metal
