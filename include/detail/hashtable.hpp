#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <vector>

#include <omp.h>

// Sentinel Constants (Adjust these based on what you consider "Empty")
#define EMPTY_KEY -1
#define DELETED_KEY -2

namespace ProbingStrategy {
    struct LINEAR {
        static size_t next_slot(size_t start, int attempt, size_t capacity);
    };

    struct QUADRATIC {
        static size_t next_slot(size_t start, int attempt, size_t capacity);
    };
}

enum class ParallelBackend {
    CAS,
    MUTEX,
};

namespace openmp_hash_table_detail {
    constexpr size_t kLoadNumerator = 7;
    constexpr size_t kLoadDenominator = 10;
    constexpr size_t kMigrationChunkSize = 64;

    enum class SlotState : std::uint8_t {
        Empty = 0,
        Claimed = 1,
        Occupied = 2,
        Deleted = 3,
        Moving = 4,
        Moved = 5,
    };

    enum class InsertStatus {
        InsertedNew,
        InsertedDeleted,
        Duplicate,
        Full,
        Retry,
    };

    template <typename K>
    K empty_key() {
        return static_cast<K>(EMPTY_KEY);
    }

    template <typename K>
    K deleted_key() {
        return static_cast<K>(DELETED_KEY);
    }

    template <typename V>
    V deleted_value() {
        if constexpr (std::is_constructible_v<V, int>) {
            return static_cast<V>(DELETED_KEY);
        }
        return V{};
    }

    template <typename Slot, typename K, typename V>
    void reset_slot(Slot& slot) {
        slot.key = empty_key<K>();
        if constexpr (std::is_constructible_v<V, int>) {
            slot.value = static_cast<V>(EMPTY_KEY);
        } else {
            slot.value = V{};
        }
    }

    template <typename Slot, typename K, typename V>
    void mark_deleted_slot(Slot& slot) {
        slot.key = deleted_key<K>();
        slot.value = deleted_value<V>();
    }

    inline bool is_readable_state(SlotState state) {
        return state == SlotState::Occupied ||
            state == SlotState::Moving ||
            state == SlotState::Moved;
    }

    inline bool is_probe_terminal_state(SlotState state) {
        return state == SlotState::Empty;
    }

    template <typename K>
    size_t hash_key(K key, size_t capacity) {
        if constexpr (std::is_integral_v<K>) {
            std::uint32_t k = static_cast<std::uint32_t>(key);
            k = ((k >> 16) ^ k) * 0x45d9f3bu;
            k = ((k >> 16) ^ k) * 0x45d9f3bu;
            k = (k >> 16) ^ k;
            return static_cast<size_t>(k) % capacity;
        }

        return std::hash<K>{}(key) % capacity;
    }

    inline size_t min_capacity_for_items(size_t items) {
        if (items == 0) {
            return 1;
        }
        return std::max<size_t>(
            1,
            (items * kLoadDenominator + kLoadNumerator - 1) / kLoadNumerator);
    }

    inline bool should_grow(size_t occupied, size_t capacity) {
        return occupied * kLoadDenominator >= capacity * kLoadNumerator;
    }

    inline bool should_cleanup(size_t occupied, size_t deleted, size_t capacity) {
        if (deleted == 0) {
            return false;
        }
        return deleted > occupied / 2 ||
            (occupied + deleted) * kLoadDenominator >= capacity * kLoadNumerator;
    }

    inline size_t growth_capacity(size_t current_capacity, size_t occupied_after_insert) {
        const size_t doubled = current_capacity < 2 ? 2 : current_capacity * 2;
        return std::max(doubled, min_capacity_for_items(occupied_after_insert));
    }

    inline size_t cleanup_capacity(size_t current_capacity, size_t occupied) {
        return std::max(current_capacity, min_capacity_for_items(occupied));
    }

    inline size_t ceil_div(size_t numerator, size_t denominator) {
        return (numerator + denominator - 1) / denominator;
    }

    inline size_t largest_power_of_two_leq(size_t value) {
        if (value <= 1) {
            return 1;
        }

        size_t result = 1;
        while (result <= value / 2) {
            result *= 2;
        }
        return result;
    }

    inline size_t default_segment_count(size_t initial_capacity, size_t num_threads) {
        const size_t thread_hint = std::max<size_t>(1, num_threads);
        const size_t segment_limit = initial_capacity / 8;
        if (segment_limit == 0) {
            return 1;
        }

        return largest_power_of_two_leq(std::min(thread_hint * 2, segment_limit));
    }

    template <typename K>
    size_t mix_hash_key(K key) {
        std::uint64_t mixed = 0;
        if constexpr (std::is_integral_v<K>) {
            mixed = static_cast<std::uint64_t>(key);
        } else {
            mixed = static_cast<std::uint64_t>(std::hash<K>{}(key));
        }

        mixed += 0x9e3779b97f4a7c15ull;
        mixed = (mixed ^ (mixed >> 30)) * 0xbf58476d1ce4e5b9ull;
        mixed = (mixed ^ (mixed >> 27)) * 0x94d049bb133111ebull;
        mixed ^= mixed >> 31;
        return static_cast<size_t>(mixed);
    }

    inline size_t segment_index_from_hash(size_t mixed_hash, size_t segment_count) {
        return segment_count == 1 ? 0 : (mixed_hash & (segment_count - 1));
    }
}

template <typename K, typename V, typename Probing = ProbingStrategy::LINEAR>
class ParallelHashTable {
private:
    struct Slot {
        K key;
        V value;
    };

    struct TableState {
        Slot* table;
        std::atomic<std::uint8_t>* meta;
        omp_lock_t* locks;
        size_t capacity;
        std::atomic<size_t> occupied;
        std::atomic<size_t> deleted;
        std::atomic<size_t> active_refs;
        std::atomic<size_t> active_mutators;
        std::atomic<bool> sealed_for_new_writes;

        TableState(size_t capacity, ParallelBackend backend);
        ~TableState();

        TableState(const TableState&) = delete;
        TableState& operator=(const TableState&) = delete;
    };

    struct ResizeContext {
        TableState* source;
        TableState* target;
        std::atomic<size_t> claim_cursor;
        std::atomic<size_t> remaining_slots;
        std::atomic<size_t> active_refs;

        ResizeContext(TableState* source_table, TableState* target_table);

        ResizeContext(const ResizeContext&) = delete;
        ResizeContext& operator=(const ResizeContext&) = delete;
    };

    struct OperationSnapshot {
        TableState* state;
        ResizeContext* resize;
    };

    class OperationGuard {
    public:
        explicit OperationGuard(const ParallelHashTable& owner)
            : owner_(owner), snapshot_(owner.enter_operation()) {}

        ~OperationGuard() {
            owner_.leave_operation(snapshot_);
        }

        TableState* state() const {
            return snapshot_.state;
        }

        ResizeContext* resize() const {
            return snapshot_.resize;
        }

    private:
        const ParallelHashTable& owner_;
        OperationSnapshot snapshot_;
    };

    std::atomic<TableState*> state_;
    std::atomic<ResizeContext*> resize_ctx_;
    size_t num_threads_;
    ParallelBackend backend_;
    std::atomic<size_t> live_items_;
    std::atomic<size_t> pending_resize_capacity_;
    std::atomic<bool> maintenance_mode_;
    std::mutex resize_mutex_;
    std::vector<ResizeContext*> retired_resize_;

    size_t next_slot(size_t start, int attempt, size_t capacity) const;
    OperationSnapshot enter_operation() const;
    void leave_operation(const OperationSnapshot& snapshot) const;
    void reset_state(TableState* state);
    void reset_pending_resize_request(size_t minimum_capacity = 0);
    void record_pending_resize_request(size_t requested_capacity);
    size_t consume_pending_resize_request();
    void reclaim_retired_resize_contexts();
    void wait_for_write_quiescence(TableState* state) const;
    void wait_for_all_operations_to_finish() const;
    bool begin_mutation(TableState* state) const;
    void end_mutation(TableState* state) const;
    bool wait_for_claimed_slot(const TableState* state, size_t slot_index) const;
    openmp_hash_table_detail::SlotState load_state(const TableState* state, size_t slot_index) const;
    void store_state(TableState* state, size_t slot_index, openmp_hash_table_detail::SlotState state_value) const;
    bool compare_exchange_state(TableState* state,
                                size_t slot_index,
                                openmp_hash_table_detail::SlotState expected,
                                openmp_hash_table_detail::SlotState desired) const;
    size_t resolved_resize_capacity(const TableState* state,
                                    size_t requested_capacity,
                                    bool force_rebuild) const;
    void maybe_finish_resize(ResizeContext* ctx);
    void help_resize(ResizeContext* ctx, size_t chunk_size = openmp_hash_table_detail::kMigrationChunkSize);
    void move_slot(ResizeContext* ctx, size_t slot_index);
    bool claim_slot_for_migration(TableState* source, size_t slot_index);
    void publish_resize(TableState* source, TableState* target);
    openmp_hash_table_detail::InsertStatus insert_migrated_entry(TableState* state, K key, const V& value);
    void maybe_resize(size_t requested_capacity, bool force_rebuild);
    openmp_hash_table_detail::InsertStatus insert_into_state(TableState* state, K key, const V& value);
    openmp_hash_table_detail::InsertStatus insert_with_cas(TableState* state, K key, const V& value);
    openmp_hash_table_detail::InsertStatus insert_with_mutex(TableState* state, K key, const V& value);
    bool get_from_state(TableState* state, K key, V& out_value, bool include_moved_states) const;
    bool get_with_cas(TableState* state, K key, V& out_value, bool include_moved_states) const;
    bool get_with_mutex(TableState* state, K key, V& out_value, bool include_moved_states) const;
    bool contains_in_state(TableState* state, K key, bool include_moved_states) const;
    bool remove_from_state(TableState* state, K key, bool allow_moved_redirect);
    bool remove_with_cas(TableState* state, K key, bool allow_moved_redirect);
    bool remove_with_mutex(TableState* state, K key, bool allow_moved_redirect);
    void validate_insert_key(K key) const;

public:
    ParallelHashTable(size_t size,
                      size_t num_threads = 0,
                      ParallelBackend backend = ParallelBackend::MUTEX);
    ~ParallelHashTable();

    void clear();
    size_t hash(K key) const;
    size_t size() const;
    size_t capacity() const;
    float load_factor() const;
    void reserve(size_t desired_items);
    void rehash(size_t new_capacity);

    bool insert(K key, V value);
    bool get(K key, V& out_value);
    bool remove(K key);

    void insert_batch(const std::vector<K>& keys, const std::vector<V>& values);
    void get_batch(const std::vector<K>& keys,
                   std::vector<V>& out_values,
                   std::vector<bool>& out_found);
    void remove_batch(const std::vector<K>& keys);
};

template <typename K, typename V, typename Probing = ProbingStrategy::LINEAR>
class SegmentedHashTable {
private:
    using SegmentTable = ParallelHashTable<K, V, Probing>;

    class OperationGuard {
    public:
        explicit OperationGuard(const SegmentedHashTable& owner)
            : owner_(owner), engaged_(false) {
            owner_.enter_operation();
            engaged_ = true;
        }

        ~OperationGuard() {
            if (engaged_) {
                owner_.leave_operation();
            }
        }

        OperationGuard(const OperationGuard&) = delete;
        OperationGuard& operator=(const OperationGuard&) = delete;

    private:
        const SegmentedHashTable& owner_;
        bool engaged_;
    };

    std::vector<std::unique_ptr<SegmentTable>> segments_;
    size_t num_threads_;
    ParallelBackend backend_;
    size_t segment_count_;
    mutable std::atomic<bool> maintenance_mode_;
    mutable std::atomic<size_t> active_operations_;

    void enter_operation() const;
    void leave_operation() const;
    void wait_for_all_operations_to_finish() const;
    void validate_insert_key(K key) const;
    size_t mixed_hash(K key) const;
    size_t segment_index_for_hash(size_t mixed_hash) const;
    SegmentTable& segment_for_hash(size_t mixed_hash) const;
    size_t size_unlocked() const;
    size_t capacity_unlocked() const;
    size_t per_segment_target_capacity(size_t total_target_capacity, size_t segment_live_items) const;

public:
    SegmentedHashTable(size_t size,
                       size_t num_threads = 0,
                       ParallelBackend backend = ParallelBackend::MUTEX);
    ~SegmentedHashTable();

    void clear();
    size_t hash(K key) const;
    size_t size() const;
    size_t capacity() const;
    float load_factor() const;
    void reserve(size_t desired_items);
    void rehash(size_t new_capacity);

    bool insert(K key, V value);
    bool get(K key, V& out_value);
    bool remove(K key);

    void insert_batch(const std::vector<K>& keys, const std::vector<V>& values);
    void get_batch(const std::vector<K>& keys,
                   std::vector<V>& out_values,
                   std::vector<bool>& out_found);
    void remove_batch(const std::vector<K>& keys);
};

template <typename K, typename V, typename Probing = ProbingStrategy::LINEAR>
class SequentialHashTable {
private:
    struct Slot {
        K key;
        V value;
    };

    Slot* table_;
    size_t capacity_;
    size_t occupied_;
    size_t deleted_;

    size_t next_slot(size_t start, int attempt, size_t capacity) const;
    void reset_slots(Slot* table, size_t capacity);
    void rehash_to_capacity(size_t target_capacity);
    void reinsert_live_entry(Slot* table, size_t capacity, K key, const V& value);
    openmp_hash_table_detail::InsertStatus insert_into_table(Slot* table,
                                                             size_t capacity,
                                                             K key,
                                                             const V& value);
    bool get_from_table(const Slot* table, size_t capacity, K key, V& out_value) const;
    bool remove_from_table(Slot* table, size_t capacity, K key);
    void validate_insert_key(K key) const;

public:
    explicit SequentialHashTable(size_t size);
    ~SequentialHashTable();

    void clear();
    size_t hash(K key) const;
    size_t size() const;
    size_t capacity() const;
    float load_factor() const;
    void reserve(size_t desired_items);
    void rehash(size_t new_capacity);

    bool insert(K key, V value);
    bool get(K key, V& out_value);
    bool remove(K key);
};

namespace ProbingStrategy {
    inline size_t LINEAR::next_slot(size_t start, int attempt, size_t capacity) {
        return (start + static_cast<size_t>(attempt)) % capacity;
    }

    inline size_t QUADRATIC::next_slot(size_t start, int attempt, size_t capacity) {
        return (start + static_cast<size_t>(attempt) * static_cast<size_t>(attempt)) % capacity;
    }
}

template <typename K, typename V, typename P>
size_t SequentialHashTable<K, V, P>::next_slot(size_t start, int attempt, size_t capacity) const {
    return P::next_slot(start, attempt, capacity);
}

template <typename K, typename V, typename P>
void SequentialHashTable<K, V, P>::reset_slots(Slot* table, size_t capacity) {
    for (size_t i = 0; i < capacity; ++i) {
        openmp_hash_table_detail::reset_slot<Slot, K, V>(table[i]);
    }
}

template <typename K, typename V, typename P>
SequentialHashTable<K, V, P>::SequentialHashTable(size_t size)
    : table_(nullptr), capacity_(size), occupied_(0), deleted_(0) {
    if (size == 0) {
        throw std::invalid_argument("hash table size must be greater than zero");
    }

    table_ = new Slot[capacity_];
    reset_slots(table_, capacity_);
}

template <typename K, typename V, typename P>
SequentialHashTable<K, V, P>::~SequentialHashTable() {
    delete[] table_;
}

template <typename K, typename V, typename P>
void SequentialHashTable<K, V, P>::validate_insert_key(K key) const {
    if (key == openmp_hash_table_detail::empty_key<K>() ||
        key == openmp_hash_table_detail::deleted_key<K>()) {
        throw std::invalid_argument("insert key collides with reserved sentinel value");
    }
}

template <typename K, typename V, typename P>
void SequentialHashTable<K, V, P>::clear() {
    reset_slots(table_, capacity_);
    occupied_ = 0;
    deleted_ = 0;
}

template <typename K, typename V, typename P>
size_t SequentialHashTable<K, V, P>::hash(K key) const {
    return openmp_hash_table_detail::hash_key(key, capacity_);
}

template <typename K, typename V, typename P>
size_t SequentialHashTable<K, V, P>::size() const {
    return occupied_;
}

template <typename K, typename V, typename P>
size_t SequentialHashTable<K, V, P>::capacity() const {
    return capacity_;
}

template <typename K, typename V, typename P>
float SequentialHashTable<K, V, P>::load_factor() const {
    return static_cast<float>(occupied_) / static_cast<float>(capacity_);
}

template <typename K, typename V, typename P>
void SequentialHashTable<K, V, P>::reserve(size_t desired_items) {
    if (desired_items == 0) {
        return;
    }

    const size_t target_capacity =
        openmp_hash_table_detail::min_capacity_for_items(std::max(desired_items, occupied_));
    if (target_capacity > capacity_) {
        rehash_to_capacity(target_capacity);
    }
}

template <typename K, typename V, typename P>
void SequentialHashTable<K, V, P>::rehash(size_t new_capacity) {
    if (new_capacity == 0) {
        throw std::invalid_argument("rehash capacity must be greater than zero");
    }

    const size_t target_capacity = std::max(
        new_capacity, openmp_hash_table_detail::min_capacity_for_items(occupied_));
    rehash_to_capacity(target_capacity);
}

template <typename K, typename V, typename P>
void SequentialHashTable<K, V, P>::reinsert_live_entry(Slot* table,
                                                       size_t capacity,
                                                       K key,
                                                       const V& value) {
    const size_t start = openmp_hash_table_detail::hash_key(key, capacity);
    for (int attempt = 0; static_cast<size_t>(attempt) < capacity; ++attempt) {
        const size_t slot_index = next_slot(start, attempt, capacity);
        const K current = table[slot_index].key;
        if (current == openmp_hash_table_detail::empty_key<K>()) {
            table[slot_index].key = key;
            table[slot_index].value = value;
            return;
        }
    }

    throw std::runtime_error("rehash target capacity is too small");
}

template <typename K, typename V, typename P>
void SequentialHashTable<K, V, P>::rehash_to_capacity(size_t target_capacity) {
    Slot* replacement = new Slot[target_capacity];
    reset_slots(replacement, target_capacity);

    size_t occupied = 0;
    try {
        for (size_t i = 0; i < capacity_; ++i) {
            const K current = table_[i].key;
            if (current != openmp_hash_table_detail::empty_key<K>() &&
                current != openmp_hash_table_detail::deleted_key<K>()) {
                reinsert_live_entry(replacement, target_capacity, current, table_[i].value);
                ++occupied;
            }
        }
    } catch (...) {
        delete[] replacement;
        throw;
    }

    delete[] table_;
    table_ = replacement;
    capacity_ = target_capacity;
    occupied_ = occupied;
    deleted_ = 0;
}

template <typename K, typename V, typename P>
openmp_hash_table_detail::InsertStatus SequentialHashTable<K, V, P>::insert_into_table(
    Slot* table,
    size_t capacity,
    K key,
    const V& value) {
    const K empty_key = openmp_hash_table_detail::empty_key<K>();
    const K deleted_key = openmp_hash_table_detail::deleted_key<K>();
    const size_t start = openmp_hash_table_detail::hash_key(key, capacity);
    size_t first_deleted = capacity;

    for (int attempt = 0; static_cast<size_t>(attempt) < capacity; ++attempt) {
        const size_t slot_index = next_slot(start, attempt, capacity);
        const K current = table[slot_index].key;

        if (current == key) {
            return openmp_hash_table_detail::InsertStatus::Duplicate;
        }
        if (current == deleted_key) {
            if (first_deleted == capacity) {
                first_deleted = slot_index;
            }
            continue;
        }
        if (current == empty_key) {
            const size_t target = first_deleted == capacity ? slot_index : first_deleted;
            table[target].key = key;
            table[target].value = value;
            return first_deleted == capacity
                ? openmp_hash_table_detail::InsertStatus::InsertedNew
                : openmp_hash_table_detail::InsertStatus::InsertedDeleted;
        }
    }

    if (first_deleted != capacity) {
        table[first_deleted].key = key;
        table[first_deleted].value = value;
        return openmp_hash_table_detail::InsertStatus::InsertedDeleted;
    }

    return openmp_hash_table_detail::InsertStatus::Full;
}

template <typename K, typename V, typename P>
bool SequentialHashTable<K, V, P>::insert(K key, V value) {
    validate_insert_key(key);

    for (;;) {
        const auto status = insert_into_table(table_, capacity_, key, value);
        if (status == openmp_hash_table_detail::InsertStatus::Duplicate) {
            return true;
        }
        if (status == openmp_hash_table_detail::InsertStatus::InsertedNew) {
            ++occupied_;
            if (openmp_hash_table_detail::should_grow(occupied_, capacity_)) {
                rehash_to_capacity(openmp_hash_table_detail::growth_capacity(capacity_, occupied_));
            }
            return true;
        }
        if (status == openmp_hash_table_detail::InsertStatus::InsertedDeleted) {
            ++occupied_;
            --deleted_;
            if (openmp_hash_table_detail::should_grow(occupied_, capacity_)) {
                rehash_to_capacity(openmp_hash_table_detail::growth_capacity(capacity_, occupied_));
            }
            return true;
        }

        rehash_to_capacity(openmp_hash_table_detail::growth_capacity(capacity_, occupied_ + 1));
    }
}

template <typename K, typename V, typename P>
bool SequentialHashTable<K, V, P>::get_from_table(const Slot* table,
                                                  size_t capacity,
                                                  K key,
                                                  V& out_value) const {
    const K empty_key = openmp_hash_table_detail::empty_key<K>();
    const size_t start = openmp_hash_table_detail::hash_key(key, capacity);

    for (int attempt = 0; static_cast<size_t>(attempt) < capacity; ++attempt) {
        const size_t slot_index = next_slot(start, attempt, capacity);
        const K current = table[slot_index].key;
        if (current == empty_key) {
            return false;
        }
        if (current == key) {
            out_value = table[slot_index].value;
            return true;
        }
    }

    return false;
}

template <typename K, typename V, typename P>
bool SequentialHashTable<K, V, P>::get(K key, V& out_value) {
    return get_from_table(table_, capacity_, key, out_value);
}

template <typename K, typename V, typename P>
bool SequentialHashTable<K, V, P>::remove_from_table(Slot* table, size_t capacity, K key) {
    const K empty_key = openmp_hash_table_detail::empty_key<K>();
    const size_t start = openmp_hash_table_detail::hash_key(key, capacity);

    for (int attempt = 0; static_cast<size_t>(attempt) < capacity; ++attempt) {
        const size_t slot_index = next_slot(start, attempt, capacity);
        const K current = table[slot_index].key;
        if (current == empty_key) {
            return false;
        }
        if (current == key) {
            openmp_hash_table_detail::mark_deleted_slot<Slot, K, V>(table[slot_index]);
            return true;
        }
    }

    return false;
}

template <typename K, typename V, typename P>
bool SequentialHashTable<K, V, P>::remove(K key) {
    if (!remove_from_table(table_, capacity_, key)) {
        return false;
    }

    --occupied_;
    ++deleted_;
    if (openmp_hash_table_detail::should_cleanup(occupied_, deleted_, capacity_)) {
        rehash_to_capacity(openmp_hash_table_detail::cleanup_capacity(capacity_, occupied_));
    }
    return true;
}

template <typename K, typename V, typename P>
ParallelHashTable<K, V, P>::TableState::TableState(size_t requested_capacity, ParallelBackend backend)
    : table(new Slot[requested_capacity]),
      meta(new std::atomic<std::uint8_t>[requested_capacity]),
      locks(backend == ParallelBackend::MUTEX ? new omp_lock_t[requested_capacity] : nullptr),
      capacity(requested_capacity),
      occupied(0),
      deleted(0),
      active_refs(0),
      active_mutators(0),
      sealed_for_new_writes(false) {
    for (size_t i = 0; i < capacity; ++i) {
        openmp_hash_table_detail::reset_slot<Slot, K, V>(table[i]);
        meta[i].store(static_cast<std::uint8_t>(openmp_hash_table_detail::SlotState::Empty),
                      std::memory_order_relaxed);
        if (locks != nullptr) {
            omp_init_lock(&locks[i]);
        }
    }
}

template <typename K, typename V, typename P>
ParallelHashTable<K, V, P>::TableState::~TableState() {
    if (locks != nullptr) {
        for (size_t i = 0; i < capacity; ++i) {
            omp_destroy_lock(&locks[i]);
        }
        delete[] locks;
    }
    delete[] meta;
    delete[] table;
}

template <typename K, typename V, typename P>
ParallelHashTable<K, V, P>::ResizeContext::ResizeContext(TableState* source_table,
                                                         TableState* target_table)
    : source(source_table),
      target(target_table),
      claim_cursor(0),
      remaining_slots(source_table->capacity),
      active_refs(0) {}

template <typename K, typename V, typename P>
ParallelHashTable<K, V, P>::ParallelHashTable(size_t size,
                                              size_t num_threads,
                                              ParallelBackend backend)
    : state_(nullptr),
      resize_ctx_(nullptr),
      num_threads_(num_threads == 0 ? omp_get_max_threads() : num_threads),
      backend_(backend),
      live_items_(0),
      pending_resize_capacity_(0),
      maintenance_mode_(false) {
    if (size == 0) {
        throw std::invalid_argument("hash table size must be greater than zero");
    }

    state_.store(new TableState(size, backend_), std::memory_order_release);
}

template <typename K, typename V, typename P>
ParallelHashTable<K, V, P>::~ParallelHashTable() {
    maintenance_mode_.store(true, std::memory_order_release);
    wait_for_all_operations_to_finish();
    std::unique_lock<std::mutex> resize_lock(resize_mutex_);
    while (!retired_resize_.empty()) {
        reclaim_retired_resize_contexts();
        if (retired_resize_.empty()) {
            break;
        }
        resize_lock.unlock();
        std::this_thread::yield();
        resize_lock.lock();
    }

    ResizeContext* ctx = resize_ctx_.exchange(nullptr, std::memory_order_acq_rel);
    TableState* current = state_.exchange(nullptr, std::memory_order_acq_rel);
    if (ctx != nullptr) {
        if (ctx->source != current) {
            delete ctx->source;
        }
        delete ctx->target;
        delete ctx;
    }
    delete current;
}

template <typename K, typename V, typename P>
size_t ParallelHashTable<K, V, P>::next_slot(size_t start, int attempt, size_t capacity) const {
    return P::next_slot(start, attempt, capacity);
}

template <typename K, typename V, typename P>
void ParallelHashTable<K, V, P>::validate_insert_key(K key) const {
    if (key == openmp_hash_table_detail::empty_key<K>() ||
        key == openmp_hash_table_detail::deleted_key<K>()) {
        throw std::invalid_argument("insert key collides with reserved sentinel value");
    }
}

template <typename K, typename V, typename P>
typename ParallelHashTable<K, V, P>::OperationSnapshot ParallelHashTable<K, V, P>::enter_operation() const {
    for (;;) {
        while (maintenance_mode_.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }

        ResizeContext* ctx = resize_ctx_.load(std::memory_order_acquire);
        if (ctx != nullptr) {
            ctx->active_refs.fetch_add(1, std::memory_order_acq_rel);
            if (!maintenance_mode_.load(std::memory_order_acquire) &&
                resize_ctx_.load(std::memory_order_acquire) == ctx) {
                return OperationSnapshot{ctx->target, ctx};
            }
            ctx->active_refs.fetch_sub(1, std::memory_order_acq_rel);
            continue;
        }

        TableState* state = state_.load(std::memory_order_acquire);
        state->active_refs.fetch_add(1, std::memory_order_acq_rel);
        if (!maintenance_mode_.load(std::memory_order_acquire) &&
            resize_ctx_.load(std::memory_order_acquire) == nullptr &&
            state_.load(std::memory_order_acquire) == state) {
            return OperationSnapshot{state, nullptr};
        }
        state->active_refs.fetch_sub(1, std::memory_order_acq_rel);
    }
}

template <typename K, typename V, typename P>
void ParallelHashTable<K, V, P>::leave_operation(const OperationSnapshot& snapshot) const {
    if (snapshot.resize != nullptr) {
        snapshot.resize->active_refs.fetch_sub(1, std::memory_order_acq_rel);
        return;
    }

    snapshot.state->active_refs.fetch_sub(1, std::memory_order_acq_rel);
}

template <typename K, typename V, typename P>
void ParallelHashTable<K, V, P>::reset_state(TableState* state) {
    for (size_t i = 0; i < state->capacity; ++i) {
        openmp_hash_table_detail::reset_slot<Slot, K, V>(state->table[i]);
        state->meta[i].store(static_cast<std::uint8_t>(openmp_hash_table_detail::SlotState::Empty),
                             std::memory_order_relaxed);
    }
    state->occupied.store(0, std::memory_order_relaxed);
    state->deleted.store(0, std::memory_order_relaxed);
}

template <typename K, typename V, typename P>
void ParallelHashTable<K, V, P>::reset_pending_resize_request(size_t minimum_capacity) {
    pending_resize_capacity_.store(minimum_capacity, std::memory_order_release);
}

template <typename K, typename V, typename P>
void ParallelHashTable<K, V, P>::record_pending_resize_request(size_t requested_capacity) {
    size_t current = pending_resize_capacity_.load(std::memory_order_acquire);
    while (current < requested_capacity &&
           !pending_resize_capacity_.compare_exchange_weak(
               current,
               requested_capacity,
               std::memory_order_acq_rel,
               std::memory_order_acquire)) {
    }
}

template <typename K, typename V, typename P>
size_t ParallelHashTable<K, V, P>::consume_pending_resize_request() {
    return pending_resize_capacity_.exchange(0, std::memory_order_acq_rel);
}

template <typename K, typename V, typename P>
void ParallelHashTable<K, V, P>::clear() {
    maintenance_mode_.store(true, std::memory_order_release);
    wait_for_all_operations_to_finish();

    std::unique_lock<std::mutex> resize_lock(resize_mutex_);
    while (!retired_resize_.empty()) {
        reclaim_retired_resize_contexts();
        if (retired_resize_.empty()) {
            break;
        }
        resize_lock.unlock();
        std::this_thread::yield();
        resize_lock.lock();
    }

    ResizeContext* ctx = resize_ctx_.load(std::memory_order_acquire);
    TableState* current = state_.load(std::memory_order_acquire);
    const size_t target_capacity = ctx != nullptr ? ctx->target->capacity : current->capacity;
    TableState* replacement = nullptr;
    try {
        replacement = new TableState(target_capacity, backend_);
    } catch (...) {
        maintenance_mode_.store(false, std::memory_order_release);
        throw;
    }

    ctx = resize_ctx_.exchange(nullptr, std::memory_order_acq_rel);
    current = state_.exchange(replacement, std::memory_order_acq_rel);

    if (ctx != nullptr) {
        if (ctx->source != current) {
            delete ctx->source;
        }
        delete ctx->target;
        delete ctx;
    }
    delete current;
    for (ResizeContext* retired : retired_resize_) {
        delete retired->source;
        delete retired;
    }
    retired_resize_.clear();

    state_.store(replacement, std::memory_order_release);
    live_items_.store(0, std::memory_order_release);
    reset_pending_resize_request();
    maintenance_mode_.store(false, std::memory_order_release);
}

template <typename K, typename V, typename P>
size_t ParallelHashTable<K, V, P>::hash(K key) const {
    OperationGuard guard(*this);
    return openmp_hash_table_detail::hash_key(key, guard.state()->capacity);
}

template <typename K, typename V, typename P>
size_t ParallelHashTable<K, V, P>::size() const {
    return live_items_.load(std::memory_order_acquire);
}

template <typename K, typename V, typename P>
size_t ParallelHashTable<K, V, P>::capacity() const {
    OperationGuard guard(*this);
    return guard.state()->capacity;
}

template <typename K, typename V, typename P>
float ParallelHashTable<K, V, P>::load_factor() const {
    OperationGuard guard(*this);
    return static_cast<float>(live_items_.load(std::memory_order_acquire)) /
        static_cast<float>(guard.state()->capacity);
}

template <typename K, typename V, typename P>
void ParallelHashTable<K, V, P>::reserve(size_t desired_items) {
    if (desired_items == 0) {
        return;
    }

    const size_t target_capacity = openmp_hash_table_detail::min_capacity_for_items(desired_items);
    maybe_resize(target_capacity, false);
}

template <typename K, typename V, typename P>
void ParallelHashTable<K, V, P>::rehash(size_t new_capacity) {
    if (new_capacity == 0) {
        throw std::invalid_argument("rehash capacity must be greater than zero");
    }
    maybe_resize(new_capacity, true);
}

template <typename K, typename V, typename P>
void ParallelHashTable<K, V, P>::reclaim_retired_resize_contexts() {
    auto it = retired_resize_.begin();
    while (it != retired_resize_.end()) {
        ResizeContext* ctx = *it;
        if (ctx->active_refs.load(std::memory_order_acquire) == 0 &&
            ctx->source->active_refs.load(std::memory_order_acquire) == 0 &&
            ctx->source->active_mutators.load(std::memory_order_acquire) == 0) {
            delete ctx->source;
            delete ctx;
            it = retired_resize_.erase(it);
            continue;
        }
        ++it;
    }
}

template <typename K, typename V, typename P>
void ParallelHashTable<K, V, P>::wait_for_write_quiescence(TableState* state) const {
    while (state->active_mutators.load(std::memory_order_acquire) != 0) {
        std::this_thread::yield();
    }
}

template <typename K, typename V, typename P>
void ParallelHashTable<K, V, P>::wait_for_all_operations_to_finish() const {
    for (;;) {
        TableState* current = state_.load(std::memory_order_acquire);
        ResizeContext* active = resize_ctx_.load(std::memory_order_acquire);
        bool idle = true;

        if (current != nullptr &&
            (current->active_refs.load(std::memory_order_acquire) != 0 ||
             current->active_mutators.load(std::memory_order_acquire) != 0)) {
            idle = false;
        }

        if (active != nullptr &&
            (active->active_refs.load(std::memory_order_acquire) != 0 ||
             active->source->active_refs.load(std::memory_order_acquire) != 0 ||
             active->source->active_mutators.load(std::memory_order_acquire) != 0)) {
            idle = false;
        }

        if (idle) {
            break;
        }
        std::this_thread::yield();
    }
}

template <typename K, typename V, typename P>
bool ParallelHashTable<K, V, P>::begin_mutation(TableState* state) const {
    state->active_mutators.fetch_add(1, std::memory_order_acq_rel);
    if (state->sealed_for_new_writes.load(std::memory_order_acquire)) {
        state->active_mutators.fetch_sub(1, std::memory_order_acq_rel);
        return false;
    }
    return true;
}

template <typename K, typename V, typename P>
void ParallelHashTable<K, V, P>::end_mutation(TableState* state) const {
    state->active_mutators.fetch_sub(1, std::memory_order_acq_rel);
}

template <typename K, typename V, typename P>
bool ParallelHashTable<K, V, P>::wait_for_claimed_slot(const TableState* state, size_t slot_index) const {
    while (load_state(state, slot_index) == openmp_hash_table_detail::SlotState::Claimed) {
        std::this_thread::yield();
    }
    return true;
}

template <typename K, typename V, typename P>
openmp_hash_table_detail::SlotState ParallelHashTable<K, V, P>::load_state(const TableState* state,
                                                                           size_t slot_index) const {
    return static_cast<openmp_hash_table_detail::SlotState>(
        state->meta[slot_index].load(std::memory_order_acquire));
}

template <typename K, typename V, typename P>
void ParallelHashTable<K, V, P>::store_state(TableState* state,
                                             size_t slot_index,
                                             openmp_hash_table_detail::SlotState state_value) const {
    state->meta[slot_index].store(
        static_cast<std::uint8_t>(state_value),
        std::memory_order_release);
}

template <typename K, typename V, typename P>
bool ParallelHashTable<K, V, P>::compare_exchange_state(
    TableState* state,
    size_t slot_index,
    openmp_hash_table_detail::SlotState expected,
    openmp_hash_table_detail::SlotState desired) const {
    std::uint8_t expected_raw = static_cast<std::uint8_t>(expected);
    return state->meta[slot_index].compare_exchange_strong(
        expected_raw,
        static_cast<std::uint8_t>(desired),
        std::memory_order_acq_rel,
        std::memory_order_acquire);
}

template <typename K, typename V, typename P>
size_t ParallelHashTable<K, V, P>::resolved_resize_capacity(const TableState* state,
                                                            size_t requested_capacity,
                                                            bool force_rebuild) const {
    const size_t live_items = live_items_.load(std::memory_order_acquire);
    if (force_rebuild) {
        return std::max(
            requested_capacity,
            openmp_hash_table_detail::min_capacity_for_items(live_items));
    }
    if (requested_capacity > state->capacity) {
        return std::max(
            requested_capacity,
            openmp_hash_table_detail::min_capacity_for_items(live_items));
    }

    const size_t occupied = state->occupied.load(std::memory_order_acquire);
    const size_t deleted = state->deleted.load(std::memory_order_acquire);
    if (openmp_hash_table_detail::should_cleanup(occupied, deleted, state->capacity)) {
        return openmp_hash_table_detail::cleanup_capacity(state->capacity, occupied);
    }
    return state->capacity;
}

template <typename K, typename V, typename P>
void ParallelHashTable<K, V, P>::publish_resize(TableState* source, TableState* target) {
    resize_ctx_.store(new ResizeContext(source, target), std::memory_order_release);
}

template <typename K, typename V, typename P>
bool ParallelHashTable<K, V, P>::claim_slot_for_migration(TableState* source, size_t slot_index) {
    if (backend_ == ParallelBackend::CAS) {
        return compare_exchange_state(
            source,
            slot_index,
            openmp_hash_table_detail::SlotState::Occupied,
            openmp_hash_table_detail::SlotState::Moving);
    }

    omp_set_lock(&source->locks[slot_index]);
    const auto state_value = load_state(source, slot_index);
    if (state_value != openmp_hash_table_detail::SlotState::Occupied) {
        omp_unset_lock(&source->locks[slot_index]);
        return false;
    }
    store_state(source, slot_index, openmp_hash_table_detail::SlotState::Moving);
    omp_unset_lock(&source->locks[slot_index]);
    return true;
}

template <typename K, typename V, typename P>
openmp_hash_table_detail::InsertStatus ParallelHashTable<K, V, P>::insert_migrated_entry(
    TableState* state,
    K key,
    const V& value) {
    for (;;) {
        const auto status = insert_into_state(state, key, value);
        if (status == openmp_hash_table_detail::InsertStatus::Retry) {
            continue;
        }
        if (status == openmp_hash_table_detail::InsertStatus::InsertedNew) {
            state->occupied.fetch_add(1, std::memory_order_relaxed);
        } else if (status == openmp_hash_table_detail::InsertStatus::InsertedDeleted) {
            state->deleted.fetch_sub(1, std::memory_order_relaxed);
            state->occupied.fetch_add(1, std::memory_order_relaxed);
        }
        return status;
    }
}

template <typename K, typename V, typename P>
void ParallelHashTable<K, V, P>::move_slot(ResizeContext* ctx, size_t slot_index) {
    TableState* source = ctx->source;
    const auto state_value = load_state(source, slot_index);
    if (state_value == openmp_hash_table_detail::SlotState::Occupied &&
        claim_slot_for_migration(source, slot_index)) {
        const K key = source->table[slot_index].key;
        const V value = source->table[slot_index].value;
        const auto status = insert_migrated_entry(ctx->target, key, value);
        if (status == openmp_hash_table_detail::InsertStatus::Full) {
            throw std::runtime_error("resize target capacity is too small");
        }
        store_state(source, slot_index, openmp_hash_table_detail::SlotState::Moved);
    }

    ctx->remaining_slots.fetch_sub(1, std::memory_order_acq_rel);
}

template <typename K, typename V, typename P>
void ParallelHashTable<K, V, P>::help_resize(ResizeContext* ctx, size_t chunk_size) {
    TableState* source = ctx->source;
    const size_t begin = ctx->claim_cursor.fetch_add(chunk_size, std::memory_order_acq_rel);
    if (begin >= source->capacity) {
        maybe_finish_resize(ctx);
        return;
    }

    const size_t end = std::min(begin + chunk_size, source->capacity);
    for (size_t slot_index = begin; slot_index < end; ++slot_index) {
        move_slot(ctx, slot_index);
    }
    maybe_finish_resize(ctx);
}

template <typename K, typename V, typename P>
void ParallelHashTable<K, V, P>::maybe_finish_resize(ResizeContext* ctx) {
    if (ctx->remaining_slots.load(std::memory_order_acquire) != 0) {
        return;
    }

    size_t pending_target = 0;
    {
        std::lock_guard<std::mutex> resize_lock(resize_mutex_);
        if (resize_ctx_.load(std::memory_order_acquire) != ctx ||
            ctx->remaining_slots.load(std::memory_order_acquire) != 0) {
            return;
        }

        state_.store(ctx->target, std::memory_order_release);
        resize_ctx_.store(nullptr, std::memory_order_release);
        retired_resize_.push_back(ctx);
        reclaim_retired_resize_contexts();
        pending_target = consume_pending_resize_request();
    }

    if (pending_target != 0) {
        maybe_resize(pending_target, false);
    }
}

template <typename K, typename V, typename P>
void ParallelHashTable<K, V, P>::maybe_resize(size_t requested_capacity, bool force_rebuild) {
    if (maintenance_mode_.load(std::memory_order_acquire)) {
        return;
    }

    ResizeContext* active = resize_ctx_.load(std::memory_order_acquire);
    if (active != nullptr) {
        if (requested_capacity > active->target->capacity) {
            record_pending_resize_request(requested_capacity);
        }
        help_resize(active);
        return;
    }

    TableState* current = state_.load(std::memory_order_acquire);
    const bool should_rebuild = force_rebuild ||
        requested_capacity > current->capacity ||
        openmp_hash_table_detail::should_cleanup(
            current->occupied.load(std::memory_order_acquire),
            current->deleted.load(std::memory_order_acquire),
            current->capacity);
    if (!should_rebuild) {
        return;
    }

    ResizeContext* created_ctx = nullptr;
    {
        std::lock_guard<std::mutex> resize_lock(resize_mutex_);
        reclaim_retired_resize_contexts();

        active = resize_ctx_.load(std::memory_order_acquire);
        if (active != nullptr) {
            if (requested_capacity > active->target->capacity) {
                record_pending_resize_request(requested_capacity);
            }
        } else {
            current = state_.load(std::memory_order_acquire);
            const bool refreshed_should_rebuild = force_rebuild ||
                requested_capacity > current->capacity ||
                openmp_hash_table_detail::should_cleanup(
                    current->occupied.load(std::memory_order_acquire),
                    current->deleted.load(std::memory_order_acquire),
                    current->capacity);
            if (!refreshed_should_rebuild) {
                return;
            }

            const size_t target_capacity =
                resolved_resize_capacity(current, requested_capacity, force_rebuild);
            current->sealed_for_new_writes.store(true, std::memory_order_release);
            wait_for_write_quiescence(current);

            TableState* replacement = nullptr;
            try {
                replacement = new TableState(target_capacity, backend_);
                publish_resize(current, replacement);
            } catch (...) {
                delete replacement;
                current->sealed_for_new_writes.store(false, std::memory_order_release);
                throw;
            }
            created_ctx = resize_ctx_.load(std::memory_order_acquire);
            reset_pending_resize_request(target_capacity);
        }
    }

    help_resize(created_ctx != nullptr ? created_ctx : active);
}

template <typename K, typename V, typename P>
openmp_hash_table_detail::InsertStatus ParallelHashTable<K, V, P>::insert_with_cas(
    TableState* state,
    K key,
    const V& value) {
    const size_t start = openmp_hash_table_detail::hash_key(key, state->capacity);
    size_t first_deleted = state->capacity;

    for (int attempt = 0; static_cast<size_t>(attempt) < state->capacity; ++attempt) {
        const size_t slot_index = next_slot(start, attempt, state->capacity);
        const auto state_value = load_state(state, slot_index);
        if (state_value == openmp_hash_table_detail::SlotState::Claimed) {
            wait_for_claimed_slot(state, slot_index);
            --attempt;
            continue;
        }
        if (openmp_hash_table_detail::is_readable_state(state_value) &&
            state->table[slot_index].key == key) {
            return openmp_hash_table_detail::InsertStatus::Duplicate;
        }
        if (state_value == openmp_hash_table_detail::SlotState::Deleted) {
            if (first_deleted == state->capacity) {
                first_deleted = slot_index;
            }
            continue;
        }
        if (state_value == openmp_hash_table_detail::SlotState::Empty) {
            const size_t target_slot = first_deleted != state->capacity ? first_deleted : slot_index;
            const auto expected_state = first_deleted != state->capacity
                ? openmp_hash_table_detail::SlotState::Deleted
                : openmp_hash_table_detail::SlotState::Empty;
            if (!compare_exchange_state(
                    state,
                    target_slot,
                    expected_state,
                    openmp_hash_table_detail::SlotState::Claimed)) {
                return openmp_hash_table_detail::InsertStatus::Retry;
            }

            try {
                state->table[target_slot].key = key;
                state->table[target_slot].value = value;
            } catch (...) {
                store_state(state, target_slot, expected_state);
                throw;
            }
            store_state(state, target_slot, openmp_hash_table_detail::SlotState::Occupied);
            return expected_state == openmp_hash_table_detail::SlotState::Empty
                ? openmp_hash_table_detail::InsertStatus::InsertedNew
                : openmp_hash_table_detail::InsertStatus::InsertedDeleted;
        }
    }

    if (first_deleted != state->capacity) {
        if (!compare_exchange_state(
                state,
                first_deleted,
                openmp_hash_table_detail::SlotState::Deleted,
                openmp_hash_table_detail::SlotState::Claimed)) {
            return openmp_hash_table_detail::InsertStatus::Retry;
        }
        try {
            state->table[first_deleted].key = key;
            state->table[first_deleted].value = value;
        } catch (...) {
            store_state(state, first_deleted, openmp_hash_table_detail::SlotState::Deleted);
            throw;
        }
        store_state(state, first_deleted, openmp_hash_table_detail::SlotState::Occupied);
        return openmp_hash_table_detail::InsertStatus::InsertedDeleted;
    }

    return openmp_hash_table_detail::InsertStatus::Full;
}

template <typename K, typename V, typename P>
openmp_hash_table_detail::InsertStatus ParallelHashTable<K, V, P>::insert_with_mutex(
    TableState* state,
    K key,
    const V& value) {
    const size_t start = openmp_hash_table_detail::hash_key(key, state->capacity);
    size_t first_deleted = state->capacity;

    for (int attempt = 0; static_cast<size_t>(attempt) < state->capacity; ++attempt) {
        const size_t slot_index = next_slot(start, attempt, state->capacity);
        const auto state_value = load_state(state, slot_index);
        if (state_value == openmp_hash_table_detail::SlotState::Claimed) {
            wait_for_claimed_slot(state, slot_index);
            --attempt;
            continue;
        }

        omp_set_lock(&state->locks[slot_index]);
        const auto locked_state = load_state(state, slot_index);
        if (openmp_hash_table_detail::is_readable_state(locked_state) &&
            state->table[slot_index].key == key) {
            omp_unset_lock(&state->locks[slot_index]);
            return openmp_hash_table_detail::InsertStatus::Duplicate;
        }
        if (locked_state == openmp_hash_table_detail::SlotState::Deleted) {
            if (first_deleted == state->capacity) {
                first_deleted = slot_index;
            }
            omp_unset_lock(&state->locks[slot_index]);
            continue;
        }
        if (locked_state == openmp_hash_table_detail::SlotState::Empty) {
            if (first_deleted == state->capacity) {
                try {
                    state->table[slot_index].key = key;
                    state->table[slot_index].value = value;
                } catch (...) {
                    omp_unset_lock(&state->locks[slot_index]);
                    throw;
                }
                store_state(state, slot_index, openmp_hash_table_detail::SlotState::Occupied);
                omp_unset_lock(&state->locks[slot_index]);
                return openmp_hash_table_detail::InsertStatus::InsertedNew;
            }

            omp_unset_lock(&state->locks[slot_index]);
            omp_set_lock(&state->locks[first_deleted]);
            if (load_state(state, first_deleted) == openmp_hash_table_detail::SlotState::Deleted) {
                try {
                    state->table[first_deleted].key = key;
                    state->table[first_deleted].value = value;
                } catch (...) {
                    omp_unset_lock(&state->locks[first_deleted]);
                    throw;
                }
                store_state(state, first_deleted, openmp_hash_table_detail::SlotState::Occupied);
                omp_unset_lock(&state->locks[first_deleted]);
                return openmp_hash_table_detail::InsertStatus::InsertedDeleted;
            }
            if (openmp_hash_table_detail::is_readable_state(load_state(state, first_deleted)) &&
                state->table[first_deleted].key == key) {
                omp_unset_lock(&state->locks[first_deleted]);
                return openmp_hash_table_detail::InsertStatus::Duplicate;
            }
            omp_unset_lock(&state->locks[first_deleted]);
            return openmp_hash_table_detail::InsertStatus::Retry;
        }
        omp_unset_lock(&state->locks[slot_index]);
    }

    if (first_deleted != state->capacity) {
        omp_set_lock(&state->locks[first_deleted]);
        if (load_state(state, first_deleted) == openmp_hash_table_detail::SlotState::Deleted) {
            try {
                state->table[first_deleted].key = key;
                state->table[first_deleted].value = value;
            } catch (...) {
                omp_unset_lock(&state->locks[first_deleted]);
                throw;
            }
            store_state(state, first_deleted, openmp_hash_table_detail::SlotState::Occupied);
            omp_unset_lock(&state->locks[first_deleted]);
            return openmp_hash_table_detail::InsertStatus::InsertedDeleted;
        }
        if (openmp_hash_table_detail::is_readable_state(load_state(state, first_deleted)) &&
            state->table[first_deleted].key == key) {
            omp_unset_lock(&state->locks[first_deleted]);
            return openmp_hash_table_detail::InsertStatus::Duplicate;
        }
        omp_unset_lock(&state->locks[first_deleted]);
        return openmp_hash_table_detail::InsertStatus::Retry;
    }

    return openmp_hash_table_detail::InsertStatus::Full;
}

template <typename K, typename V, typename P>
openmp_hash_table_detail::InsertStatus ParallelHashTable<K, V, P>::insert_into_state(
    TableState* state,
    K key,
    const V& value) {
    return backend_ == ParallelBackend::CAS
        ? insert_with_cas(state, key, value)
        : insert_with_mutex(state, key, value);
}

template <typename K, typename V, typename P>
bool ParallelHashTable<K, V, P>::insert(K key, V value) {
    validate_insert_key(key);

    for (;;) {
        size_t resize_target = 0;
        bool inserted = false;
        bool duplicate = false;
        bool retry = false;

        {
            OperationGuard guard(*this);
            if (guard.resize() == nullptr) {
                TableState* state = guard.state();
                if (!begin_mutation(state)) {
                    retry = true;
                } else {
                    try {
                        for (;;) {
                            const auto status = insert_into_state(state, key, value);
                            if (status == openmp_hash_table_detail::InsertStatus::Retry) {
                                continue;
                            }
                            if (status == openmp_hash_table_detail::InsertStatus::Duplicate) {
                                duplicate = true;
                            } else if (status == openmp_hash_table_detail::InsertStatus::InsertedNew) {
                                const size_t occupied =
                                    state->occupied.fetch_add(1, std::memory_order_relaxed) + 1;
                                live_items_.fetch_add(1, std::memory_order_relaxed);
                                inserted = true;
                                if (openmp_hash_table_detail::should_grow(occupied, state->capacity)) {
                                    resize_target = openmp_hash_table_detail::growth_capacity(
                                        state->capacity,
                                        occupied);
                                }
                            } else if (status == openmp_hash_table_detail::InsertStatus::InsertedDeleted) {
                                state->deleted.fetch_sub(1, std::memory_order_relaxed);
                                const size_t occupied =
                                    state->occupied.fetch_add(1, std::memory_order_relaxed) + 1;
                                live_items_.fetch_add(1, std::memory_order_relaxed);
                                inserted = true;
                                if (openmp_hash_table_detail::should_grow(occupied, state->capacity)) {
                                    resize_target = openmp_hash_table_detail::growth_capacity(
                                        state->capacity,
                                        occupied);
                                }
                            } else {
                                resize_target = openmp_hash_table_detail::growth_capacity(
                                    state->capacity,
                                    state->occupied.load(std::memory_order_acquire) + 1);
                            }
                            break;
                        }
                    } catch (...) {
                        end_mutation(state);
                        throw;
                    }
                    end_mutation(state);
                }
            } else {
                ResizeContext* ctx = guard.resize();
                TableState* target = ctx->target;
                TableState* source = ctx->source;
                if (contains_in_state(target, key, false) ||
                    contains_in_state(source, key, false)) {
                    duplicate = true;
                } else if (contains_in_state(source, key, true)) {
                    help_resize(ctx);
                    retry = true;
                } else {
                    for (;;) {
                        const auto status = insert_into_state(target, key, value);
                        if (status == openmp_hash_table_detail::InsertStatus::Retry) {
                            continue;
                        }
                        if (status == openmp_hash_table_detail::InsertStatus::Duplicate) {
                            duplicate = true;
                        } else if (status == openmp_hash_table_detail::InsertStatus::InsertedNew) {
                            const size_t occupied =
                                target->occupied.fetch_add(1, std::memory_order_relaxed) + 1;
                            live_items_.fetch_add(1, std::memory_order_relaxed);
                            inserted = true;
                            if (openmp_hash_table_detail::should_grow(occupied, target->capacity)) {
                                resize_target = openmp_hash_table_detail::growth_capacity(
                                    target->capacity,
                                    occupied);
                            }
                        } else if (status == openmp_hash_table_detail::InsertStatus::InsertedDeleted) {
                            target->deleted.fetch_sub(1, std::memory_order_relaxed);
                            const size_t occupied =
                                target->occupied.fetch_add(1, std::memory_order_relaxed) + 1;
                            live_items_.fetch_add(1, std::memory_order_relaxed);
                            inserted = true;
                            if (openmp_hash_table_detail::should_grow(occupied, target->capacity)) {
                                resize_target = openmp_hash_table_detail::growth_capacity(
                                    target->capacity,
                                    occupied);
                            }
                        } else {
                            resize_target = openmp_hash_table_detail::growth_capacity(
                                target->capacity,
                                target->occupied.load(std::memory_order_acquire) + 1);
                            record_pending_resize_request(resize_target);
                        }
                        break;
                    }
                }
                help_resize(ctx);
            }
        }

        if (retry) {
            continue;
        }
        if (resize_target != 0) {
            maybe_resize(resize_target, false);
        }
        if (duplicate || inserted) {
            return true;
        }
    }
}

template <typename K, typename V, typename P>
bool ParallelHashTable<K, V, P>::get_with_cas(TableState* state,
                                              K key,
                                              V& out_value,
                                              bool include_moved_states) const {
    const size_t start = openmp_hash_table_detail::hash_key(key, state->capacity);

    for (int attempt = 0; static_cast<size_t>(attempt) < state->capacity; ++attempt) {
        const size_t slot_index = next_slot(start, attempt, state->capacity);
        const auto state_value = load_state(state, slot_index);
        if (state_value == openmp_hash_table_detail::SlotState::Claimed) {
            wait_for_claimed_slot(state, slot_index);
            --attempt;
            continue;
        }
        if (openmp_hash_table_detail::is_probe_terminal_state(state_value)) {
            return false;
        }
        const bool readable = state_value == openmp_hash_table_detail::SlotState::Occupied ||
            (include_moved_states &&
             (state_value == openmp_hash_table_detail::SlotState::Moving ||
              state_value == openmp_hash_table_detail::SlotState::Moved));
        if (readable && state->table[slot_index].key == key) {
            out_value = state->table[slot_index].value;
            return true;
        }
    }

    return false;
}

template <typename K, typename V, typename P>
bool ParallelHashTable<K, V, P>::get_with_mutex(TableState* state,
                                                K key,
                                                V& out_value,
                                                bool include_moved_states) const {
    const size_t start = openmp_hash_table_detail::hash_key(key, state->capacity);

    for (int attempt = 0; static_cast<size_t>(attempt) < state->capacity; ++attempt) {
        const size_t slot_index = next_slot(start, attempt, state->capacity);
        const auto state_value = load_state(state, slot_index);
        if (state_value == openmp_hash_table_detail::SlotState::Claimed) {
            wait_for_claimed_slot(state, slot_index);
            --attempt;
            continue;
        }

        omp_set_lock(&state->locks[slot_index]);
        const auto locked_state = load_state(state, slot_index);
        if (openmp_hash_table_detail::is_probe_terminal_state(locked_state)) {
            omp_unset_lock(&state->locks[slot_index]);
            return false;
        }
        const bool readable = locked_state == openmp_hash_table_detail::SlotState::Occupied ||
            (include_moved_states &&
             (locked_state == openmp_hash_table_detail::SlotState::Moving ||
              locked_state == openmp_hash_table_detail::SlotState::Moved));
        if (readable && state->table[slot_index].key == key) {
            out_value = state->table[slot_index].value;
            omp_unset_lock(&state->locks[slot_index]);
            return true;
        }
        omp_unset_lock(&state->locks[slot_index]);
    }

    return false;
}

template <typename K, typename V, typename P>
bool ParallelHashTable<K, V, P>::get_from_state(TableState* state,
                                                K key,
                                                V& out_value,
                                                bool include_moved_states) const {
    return backend_ == ParallelBackend::CAS
        ? get_with_cas(state, key, out_value, include_moved_states)
        : get_with_mutex(state, key, out_value, include_moved_states);
}

template <typename K, typename V, typename P>
bool ParallelHashTable<K, V, P>::contains_in_state(TableState* state,
                                                   K key,
                                                   bool include_moved_states) const {
    const size_t start = openmp_hash_table_detail::hash_key(key, state->capacity);
    for (int attempt = 0; static_cast<size_t>(attempt) < state->capacity; ++attempt) {
        const size_t slot_index = next_slot(start, attempt, state->capacity);
        const auto state_value = load_state(state, slot_index);
        if (state_value == openmp_hash_table_detail::SlotState::Claimed) {
            wait_for_claimed_slot(state, slot_index);
            --attempt;
            continue;
        }
        if (openmp_hash_table_detail::is_probe_terminal_state(state_value)) {
            return false;
        }
        const bool readable = state_value == openmp_hash_table_detail::SlotState::Occupied ||
            (include_moved_states &&
             (state_value == openmp_hash_table_detail::SlotState::Moving ||
              state_value == openmp_hash_table_detail::SlotState::Moved));
        if (readable && state->table[slot_index].key == key) {
            return true;
        }
    }
    return false;
}

template <typename K, typename V, typename P>
bool ParallelHashTable<K, V, P>::get(K key, V& out_value) {
    for (int attempt = 0; attempt < 3; ++attempt) {
        OperationGuard guard(*this);
        if (guard.resize() == nullptr) {
            return get_from_state(guard.state(), key, out_value, false);
        }

        ResizeContext* ctx = guard.resize();
        if (get_from_state(ctx->target, key, out_value, false)) {
            help_resize(ctx);
            return true;
        }
        if (get_from_state(ctx->source, key, out_value, false)) {
            help_resize(ctx);
            return true;
        }
        if (!contains_in_state(ctx->source, key, true)) {
            help_resize(ctx);
            return false;
        }
        help_resize(ctx);
    }

    return false;
}

template <typename K, typename V, typename P>
bool ParallelHashTable<K, V, P>::remove_with_cas(TableState* state,
                                                 K key,
                                                 bool allow_moved_redirect) {
    const size_t start = openmp_hash_table_detail::hash_key(key, state->capacity);

    for (int attempt = 0; static_cast<size_t>(attempt) < state->capacity; ++attempt) {
        const size_t slot_index = next_slot(start, attempt, state->capacity);
        const auto state_value = load_state(state, slot_index);
        if (state_value == openmp_hash_table_detail::SlotState::Claimed) {
            wait_for_claimed_slot(state, slot_index);
            --attempt;
            continue;
        }
        if (openmp_hash_table_detail::is_probe_terminal_state(state_value)) {
            return false;
        }
        if (state->table[slot_index].key != key) {
            continue;
        }
        if (state_value == openmp_hash_table_detail::SlotState::Occupied) {
            if (compare_exchange_state(
                    state,
                    slot_index,
                    openmp_hash_table_detail::SlotState::Occupied,
                    openmp_hash_table_detail::SlotState::Deleted)) {
                return true;
            }
            const auto actual = load_state(state, slot_index);
            if (allow_moved_redirect &&
                (actual == openmp_hash_table_detail::SlotState::Moving ||
                 actual == openmp_hash_table_detail::SlotState::Moved)) {
                return false;
            }
            if (actual == openmp_hash_table_detail::SlotState::Deleted) {
                return false;
            }
            --attempt;
            continue;
        }
        if (allow_moved_redirect &&
            (state_value == openmp_hash_table_detail::SlotState::Moving ||
             state_value == openmp_hash_table_detail::SlotState::Moved)) {
            return false;
        }
    }

    return false;
}

template <typename K, typename V, typename P>
bool ParallelHashTable<K, V, P>::remove_with_mutex(TableState* state,
                                                   K key,
                                                   bool allow_moved_redirect) {
    const size_t start = openmp_hash_table_detail::hash_key(key, state->capacity);

    for (int attempt = 0; static_cast<size_t>(attempt) < state->capacity; ++attempt) {
        const size_t slot_index = next_slot(start, attempt, state->capacity);
        omp_set_lock(&state->locks[slot_index]);
        const auto state_value = load_state(state, slot_index);
        if (openmp_hash_table_detail::is_probe_terminal_state(state_value)) {
            omp_unset_lock(&state->locks[slot_index]);
            return false;
        }
        if (state->table[slot_index].key != key) {
            omp_unset_lock(&state->locks[slot_index]);
            continue;
        }
        if (state_value == openmp_hash_table_detail::SlotState::Occupied) {
            store_state(state, slot_index, openmp_hash_table_detail::SlotState::Deleted);
            omp_unset_lock(&state->locks[slot_index]);
            return true;
        }
        if (allow_moved_redirect &&
            (state_value == openmp_hash_table_detail::SlotState::Moving ||
             state_value == openmp_hash_table_detail::SlotState::Moved)) {
            omp_unset_lock(&state->locks[slot_index]);
            return false;
        }
        omp_unset_lock(&state->locks[slot_index]);
    }

    return false;
}

template <typename K, typename V, typename P>
bool ParallelHashTable<K, V, P>::remove_from_state(TableState* state,
                                                   K key,
                                                   bool allow_moved_redirect) {
    return backend_ == ParallelBackend::CAS
        ? remove_with_cas(state, key, allow_moved_redirect)
        : remove_with_mutex(state, key, allow_moved_redirect);
}

template <typename K, typename V, typename P>
bool ParallelHashTable<K, V, P>::remove(K key) {
    int resize_redirect_retries = 0;
    for (;;) {
        size_t cleanup_target = 0;
        bool removed = false;
        bool retry = false;
        bool resize_retry = false;

        {
            OperationGuard guard(*this);
            if (guard.resize() == nullptr) {
                TableState* state = guard.state();
                if (!begin_mutation(state)) {
                    retry = true;
                } else {
                    removed = remove_from_state(state, key, false);
                    if (removed) {
                        live_items_.fetch_sub(1, std::memory_order_relaxed);
                        const size_t occupied =
                            state->occupied.fetch_sub(1, std::memory_order_relaxed) - 1;
                        const size_t deleted =
                            state->deleted.fetch_add(1, std::memory_order_relaxed) + 1;
                        if (openmp_hash_table_detail::should_cleanup(occupied, deleted, state->capacity)) {
                            cleanup_target =
                                openmp_hash_table_detail::cleanup_capacity(state->capacity, occupied);
                        }
                    }
                    end_mutation(state);
                }
            } else {
                ResizeContext* ctx = guard.resize();
                TableState* target = ctx->target;
                TableState* source = ctx->source;

                removed = remove_from_state(target, key, false);
                if (removed) {
                    live_items_.fetch_sub(1, std::memory_order_relaxed);
                    const size_t occupied =
                        target->occupied.fetch_sub(1, std::memory_order_relaxed) - 1;
                    const size_t deleted =
                        target->deleted.fetch_add(1, std::memory_order_relaxed) + 1;
                    if (openmp_hash_table_detail::should_cleanup(occupied, deleted, target->capacity)) {
                        cleanup_target =
                            openmp_hash_table_detail::cleanup_capacity(target->capacity, occupied);
                    }
                } else {
                    removed = remove_from_state(source, key, true);
                    if (removed) {
                        live_items_.fetch_sub(1, std::memory_order_relaxed);
                    } else {
                        const bool source_redirect_hint =
                            !contains_in_state(source, key, false) &&
                            contains_in_state(source, key, true);
                        help_resize(ctx);
                        removed = remove_from_state(target, key, false);
                        resize_retry = !removed && source_redirect_hint;
                        retry = resize_retry;
                        if (removed) {
                            live_items_.fetch_sub(1, std::memory_order_relaxed);
                            const size_t occupied =
                                target->occupied.fetch_sub(1, std::memory_order_relaxed) - 1;
                            const size_t deleted =
                                target->deleted.fetch_add(1, std::memory_order_relaxed) + 1;
                            if (openmp_hash_table_detail::should_cleanup(
                                    occupied,
                                    deleted,
                                    target->capacity)) {
                                cleanup_target =
                                    openmp_hash_table_detail::cleanup_capacity(target->capacity, occupied);
                            }
                        }
                    }
                }
                help_resize(ctx);
            }
        }

        if (retry) {
            if (resize_retry) {
                ++resize_redirect_retries;
                if (resize_redirect_retries >= 4) {
                    return false;
                }
            }
            continue;
        }
        resize_redirect_retries = 0;
        if (cleanup_target != 0) {
            maybe_resize(cleanup_target, false);
        }
        return removed;
    }
}

template <typename K, typename V, typename P>
void ParallelHashTable<K, V, P>::insert_batch(const std::vector<K>& keys,
                                              const std::vector<V>& values) {
    if (keys.size() != values.size()) {
        throw std::invalid_argument("keys and values must have the same length");
    }

    #pragma omp parallel for schedule(static) num_threads(num_threads_)
    for (size_t i = 0; i < keys.size(); ++i) {
        insert(keys[i], values[i]);
    }
}

template <typename K, typename V, typename P>
void ParallelHashTable<K, V, P>::get_batch(const std::vector<K>& keys,
                                           std::vector<V>& out_values,
                                           std::vector<bool>& out_found) {
    out_values.resize(keys.size());
    std::vector<unsigned char> found_bits(keys.size(), 0);

    #pragma omp parallel for schedule(static) num_threads(num_threads_)
    for (size_t i = 0; i < keys.size(); ++i) {
        V value{};
        const bool found = get(keys[i], value);
        found_bits[i] = found ? 1U : 0U;
        if (found) {
            out_values[i] = value;
        }
    }

    out_found.assign(keys.size(), false);
    for (size_t i = 0; i < keys.size(); ++i) {
        out_found[i] = found_bits[i] != 0;
    }
}

template <typename K, typename V, typename P>
void ParallelHashTable<K, V, P>::remove_batch(const std::vector<K>& keys) {
    #pragma omp parallel for schedule(static) num_threads(num_threads_)
    for (size_t i = 0; i < keys.size(); ++i) {
        remove(keys[i]);
    }
}

template <typename K, typename V, typename P>
SegmentedHashTable<K, V, P>::SegmentedHashTable(size_t size,
                                                size_t num_threads,
                                                ParallelBackend backend)
    : segments_(),
      num_threads_(num_threads == 0 ? omp_get_max_threads() : num_threads),
      backend_(backend),
      segment_count_(openmp_hash_table_detail::default_segment_count(
          size,
          num_threads == 0 ? omp_get_max_threads() : num_threads)),
      maintenance_mode_(false),
      active_operations_(0) {
    if (size == 0) {
        throw std::invalid_argument("hash table size must be greater than zero");
    }

    const size_t initial_segment_capacity = std::max<size_t>(
        8,
        openmp_hash_table_detail::ceil_div(size, segment_count_));
    segments_.reserve(segment_count_);
    try {
        for (size_t segment_index = 0; segment_index < segment_count_; ++segment_index) {
            segments_.push_back(
                std::make_unique<SegmentTable>(initial_segment_capacity, num_threads_, backend_));
        }
    } catch (...) {
        segments_.clear();
        throw;
    }
}

template <typename K, typename V, typename P>
SegmentedHashTable<K, V, P>::~SegmentedHashTable() {
    maintenance_mode_.store(true, std::memory_order_release);
    wait_for_all_operations_to_finish();
    segments_.clear();
}

template <typename K, typename V, typename P>
void SegmentedHashTable<K, V, P>::enter_operation() const {
    for (;;) {
        while (maintenance_mode_.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }

        active_operations_.fetch_add(1, std::memory_order_acq_rel);
        if (!maintenance_mode_.load(std::memory_order_acquire)) {
            return;
        }
        active_operations_.fetch_sub(1, std::memory_order_acq_rel);
    }
}

template <typename K, typename V, typename P>
void SegmentedHashTable<K, V, P>::leave_operation() const {
    active_operations_.fetch_sub(1, std::memory_order_acq_rel);
}

template <typename K, typename V, typename P>
void SegmentedHashTable<K, V, P>::wait_for_all_operations_to_finish() const {
    while (active_operations_.load(std::memory_order_acquire) != 0) {
        std::this_thread::yield();
    }
}

template <typename K, typename V, typename P>
void SegmentedHashTable<K, V, P>::validate_insert_key(K key) const {
    if (key == openmp_hash_table_detail::empty_key<K>() ||
        key == openmp_hash_table_detail::deleted_key<K>()) {
        throw std::invalid_argument("insert key collides with reserved sentinel value");
    }
}

template <typename K, typename V, typename P>
size_t SegmentedHashTable<K, V, P>::mixed_hash(K key) const {
    return openmp_hash_table_detail::mix_hash_key(key);
}

template <typename K, typename V, typename P>
size_t SegmentedHashTable<K, V, P>::segment_index_for_hash(size_t mixed_hash_value) const {
    return openmp_hash_table_detail::segment_index_from_hash(mixed_hash_value, segment_count_);
}

template <typename K, typename V, typename P>
typename SegmentedHashTable<K, V, P>::SegmentTable&
SegmentedHashTable<K, V, P>::segment_for_hash(size_t mixed_hash_value) const {
    return *segments_[segment_index_for_hash(mixed_hash_value)];
}

template <typename K, typename V, typename P>
size_t SegmentedHashTable<K, V, P>::size_unlocked() const {
    size_t total = 0;
    for (const auto& segment : segments_) {
        total += segment->size();
    }
    return total;
}

template <typename K, typename V, typename P>
size_t SegmentedHashTable<K, V, P>::capacity_unlocked() const {
    size_t total = 0;
    for (const auto& segment : segments_) {
        total += segment->capacity();
    }
    return total;
}

template <typename K, typename V, typename P>
size_t SegmentedHashTable<K, V, P>::per_segment_target_capacity(size_t total_target_capacity,
                                                                size_t segment_live_items) const {
    return std::max(
        openmp_hash_table_detail::ceil_div(total_target_capacity, segment_count_),
        openmp_hash_table_detail::min_capacity_for_items(segment_live_items));
}

template <typename K, typename V, typename P>
void SegmentedHashTable<K, V, P>::clear() {
    maintenance_mode_.store(true, std::memory_order_release);
    wait_for_all_operations_to_finish();

    try {
        for (const auto& segment : segments_) {
            segment->clear();
        }
    } catch (...) {
        maintenance_mode_.store(false, std::memory_order_release);
        throw;
    }

    maintenance_mode_.store(false, std::memory_order_release);
}

template <typename K, typename V, typename P>
size_t SegmentedHashTable<K, V, P>::hash(K key) const {
    OperationGuard guard(*this);
    const size_t mixed_hash_value = mixed_hash(key);
    const size_t segment_index = segment_index_for_hash(mixed_hash_value);

    size_t global_offset = 0;
    for (size_t i = 0; i < segment_index; ++i) {
        global_offset += segments_[i]->capacity();
    }
    return global_offset + segments_[segment_index]->hash(key);
}

template <typename K, typename V, typename P>
size_t SegmentedHashTable<K, V, P>::size() const {
    OperationGuard guard(*this);
    return size_unlocked();
}

template <typename K, typename V, typename P>
size_t SegmentedHashTable<K, V, P>::capacity() const {
    OperationGuard guard(*this);
    return capacity_unlocked();
}

template <typename K, typename V, typename P>
float SegmentedHashTable<K, V, P>::load_factor() const {
    OperationGuard guard(*this);
    const size_t total_capacity = capacity_unlocked();
    if (total_capacity == 0) {
        return 0.0f;
    }
    return static_cast<float>(size_unlocked()) / static_cast<float>(total_capacity);
}

template <typename K, typename V, typename P>
void SegmentedHashTable<K, V, P>::reserve(size_t desired_items) {
    if (desired_items == 0) {
        return;
    }

    OperationGuard guard(*this);
    const size_t total_target_capacity =
        openmp_hash_table_detail::min_capacity_for_items(desired_items);

    for (const auto& segment : segments_) {
        const size_t local_target_capacity =
            per_segment_target_capacity(total_target_capacity, segment->size());
        segment->rehash(std::max(segment->capacity(), local_target_capacity));
    }
}

template <typename K, typename V, typename P>
void SegmentedHashTable<K, V, P>::rehash(size_t new_capacity) {
    if (new_capacity == 0) {
        throw std::invalid_argument("rehash capacity must be greater than zero");
    }

    OperationGuard guard(*this);
    const size_t total_live_items = size_unlocked();
    const size_t total_target_capacity = std::max(
        new_capacity,
        openmp_hash_table_detail::min_capacity_for_items(total_live_items));

    for (const auto& segment : segments_) {
        segment->rehash(per_segment_target_capacity(total_target_capacity, segment->size()));
    }
}

template <typename K, typename V, typename P>
bool SegmentedHashTable<K, V, P>::insert(K key, V value) {
    validate_insert_key(key);
    OperationGuard guard(*this);
    return segment_for_hash(mixed_hash(key)).insert(key, value);
}

template <typename K, typename V, typename P>
bool SegmentedHashTable<K, V, P>::get(K key, V& out_value) {
    OperationGuard guard(*this);
    return segment_for_hash(mixed_hash(key)).get(key, out_value);
}

template <typename K, typename V, typename P>
bool SegmentedHashTable<K, V, P>::remove(K key) {
    OperationGuard guard(*this);
    return segment_for_hash(mixed_hash(key)).remove(key);
}

template <typename K, typename V, typename P>
void SegmentedHashTable<K, V, P>::insert_batch(const std::vector<K>& keys,
                                               const std::vector<V>& values) {
    if (keys.size() != values.size()) {
        throw std::invalid_argument("keys and values must have the same length");
    }

    #pragma omp parallel for schedule(static) num_threads(num_threads_)
    for (size_t i = 0; i < keys.size(); ++i) {
        insert(keys[i], values[i]);
    }
}

template <typename K, typename V, typename P>
void SegmentedHashTable<K, V, P>::get_batch(const std::vector<K>& keys,
                                            std::vector<V>& out_values,
                                            std::vector<bool>& out_found) {
    out_values.resize(keys.size());
    std::vector<unsigned char> found_bits(keys.size(), 0);

    #pragma omp parallel for schedule(static) num_threads(num_threads_)
    for (size_t i = 0; i < keys.size(); ++i) {
        V value{};
        const bool found = get(keys[i], value);
        found_bits[i] = found ? 1U : 0U;
        if (found) {
            out_values[i] = value;
        }
    }

    out_found.assign(keys.size(), false);
    for (size_t i = 0; i < keys.size(); ++i) {
        out_found[i] = found_bits[i] != 0;
    }
}

template <typename K, typename V, typename P>
void SegmentedHashTable<K, V, P>::remove_batch(const std::vector<K>& keys) {
    #pragma omp parallel for schedule(static) num_threads(num_threads_)
    for (size_t i = 0; i < keys.size(); ++i) {
        remove(keys[i]);
    }
}
