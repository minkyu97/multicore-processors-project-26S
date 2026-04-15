#pragma once

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <stdexcept>
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
    constexpr int kBusyKey = -3;

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

    template <typename K>
    K busy_key() {
        return static_cast<K>(kBusyKey);
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
        omp_lock_t* locks;
        size_t capacity;
        std::atomic<size_t> occupied;
        std::atomic<size_t> deleted;

        TableState(size_t capacity, ParallelBackend backend);
        ~TableState();

        TableState(const TableState&) = delete;
        TableState& operator=(const TableState&) = delete;
    };

    class OperationGuard {
    public:
        explicit OperationGuard(const ParallelHashTable& owner)
            : owner_(owner), state_(owner.enter_operation()) {}

        ~OperationGuard() {
            owner_.leave_operation();
        }

        TableState* state() const {
            return state_;
        }

    private:
        const ParallelHashTable& owner_;
        TableState* state_;
    };

    std::atomic<TableState*> state_;
    size_t num_threads_;
    ParallelBackend backend_;
    mutable size_t active_operations_;
    mutable bool resize_gate_closed_;
    mutable std::mutex gate_mutex_;
    mutable std::condition_variable gate_cv_;
    mutable std::condition_variable idle_cv_;
    std::mutex resize_mutex_;

    size_t next_slot(size_t start, int attempt, size_t capacity) const;
    TableState* enter_operation() const;
    void leave_operation() const;
    void close_gate_and_wait_for_quiescence();
    void open_gate();
    void reset_state(TableState* state);
    void rebuild_state(TableState* source, TableState* target);
    void maybe_resize(size_t requested_capacity, bool force_rebuild);
    void reinsert_live_entry(TableState* state, K key, const V& value);
    bool try_claim_slot(TableState* state, size_t slot_index, K expected_marker, K key, const V& value);
    openmp_hash_table_detail::InsertStatus insert_into_state(TableState* state, K key, const V& value);
    openmp_hash_table_detail::InsertStatus insert_with_cas(TableState* state, K key, const V& value);
    openmp_hash_table_detail::InsertStatus insert_with_mutex(TableState* state, K key, const V& value);
    bool get_from_state(TableState* state, K key, V& out_value) const;
    bool get_with_cas(TableState* state, K key, V& out_value) const;
    bool get_with_mutex(TableState* state, K key, V& out_value) const;
    bool remove_from_state(TableState* state, K key);
    bool remove_with_cas(TableState* state, K key);
    bool remove_with_mutex(TableState* state, K key);
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
ParallelHashTable<K, V, P>::TableState::TableState(size_t requested_capacity, ParallelBackend /*backend*/)
    : table(new Slot[requested_capacity]),
      locks(new omp_lock_t[requested_capacity]),
      capacity(requested_capacity),
      occupied(0),
      deleted(0) {
    for (size_t i = 0; i < capacity; ++i) {
        openmp_hash_table_detail::reset_slot<Slot, K, V>(table[i]);
        omp_init_lock(&locks[i]);
    }
}

template <typename K, typename V, typename P>
ParallelHashTable<K, V, P>::TableState::~TableState() {
    for (size_t i = 0; i < capacity; ++i) {
        omp_destroy_lock(&locks[i]);
    }
    delete[] locks;
    delete[] table;
}

template <typename K, typename V, typename P>
ParallelHashTable<K, V, P>::ParallelHashTable(size_t size,
                                              size_t num_threads,
                                              ParallelBackend backend)
    : state_(nullptr),
      num_threads_(num_threads == 0 ? omp_get_max_threads() : num_threads),
      backend_(backend),
      active_operations_(0),
      resize_gate_closed_(false) {
    if (size == 0) {
        throw std::invalid_argument("hash table size must be greater than zero");
    }

    if (backend_ == ParallelBackend::CAS) {
        if constexpr (!std::is_trivially_copyable_v<K> || !std::is_constructible_v<K, int>) {
            throw std::invalid_argument(
                "CAS backend requires trivially copyable keys that support sentinel values");
        }
    }

    state_.store(new TableState(size, backend_), std::memory_order_release);
}

template <typename K, typename V, typename P>
ParallelHashTable<K, V, P>::~ParallelHashTable() {
    delete state_.load(std::memory_order_acquire);
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
    if (backend_ == ParallelBackend::CAS && key == openmp_hash_table_detail::busy_key<K>()) {
        throw std::invalid_argument("insert key collides with reserved CAS marker");
    }
}

template <typename K, typename V, typename P>
typename ParallelHashTable<K, V, P>::TableState* ParallelHashTable<K, V, P>::enter_operation() const {
    std::unique_lock<std::mutex> lock(gate_mutex_);
    gate_cv_.wait(lock, [this] {
        return !resize_gate_closed_;
    });

    ++active_operations_;
    return state_.load(std::memory_order_acquire);
}

template <typename K, typename V, typename P>
void ParallelHashTable<K, V, P>::leave_operation() const {
    std::lock_guard<std::mutex> lock(gate_mutex_);
    --active_operations_;
    if (resize_gate_closed_ && active_operations_ == 0) {
        idle_cv_.notify_all();
    }
}

template <typename K, typename V, typename P>
void ParallelHashTable<K, V, P>::close_gate_and_wait_for_quiescence() {
    std::unique_lock<std::mutex> lock(gate_mutex_);
    resize_gate_closed_ = true;
    idle_cv_.wait(lock, [this] {
        return active_operations_ == 0;
    });
}

template <typename K, typename V, typename P>
void ParallelHashTable<K, V, P>::open_gate() {
    {
        std::lock_guard<std::mutex> lock(gate_mutex_);
        resize_gate_closed_ = false;
    }
    gate_cv_.notify_all();
}

template <typename K, typename V, typename P>
void ParallelHashTable<K, V, P>::reset_state(TableState* state) {
    for (size_t i = 0; i < state->capacity; ++i) {
        openmp_hash_table_detail::reset_slot<Slot, K, V>(state->table[i]);
    }
    state->occupied.store(0, std::memory_order_relaxed);
    state->deleted.store(0, std::memory_order_relaxed);
}

template <typename K, typename V, typename P>
void ParallelHashTable<K, V, P>::reinsert_live_entry(TableState* state, K key, const V& value) {
    const K empty_key = openmp_hash_table_detail::empty_key<K>();
    const size_t start = openmp_hash_table_detail::hash_key(key, state->capacity);

    for (int attempt = 0; static_cast<size_t>(attempt) < state->capacity; ++attempt) {
        const size_t slot_index = next_slot(start, attempt, state->capacity);
        if (state->table[slot_index].key == empty_key) {
            state->table[slot_index].key = key;
            state->table[slot_index].value = value;
            return;
        }
    }

    throw std::runtime_error("rehash target capacity is too small");
}

template <typename K, typename V, typename P>
void ParallelHashTable<K, V, P>::rebuild_state(TableState* source, TableState* target) {
    size_t occupied = 0;
    const K empty_key = openmp_hash_table_detail::empty_key<K>();
    const K deleted_key = openmp_hash_table_detail::deleted_key<K>();
    const K busy_key = openmp_hash_table_detail::busy_key<K>();

    for (size_t i = 0; i < source->capacity; ++i) {
        K current = source->table[i].key;
        while (backend_ == ParallelBackend::CAS && current == busy_key) {
            auto key_ref = std::atomic_ref<K>(source->table[i].key);
            current = key_ref.load(std::memory_order_acquire);
        }
        if (current != empty_key && current != deleted_key) {
            reinsert_live_entry(target, current, source->table[i].value);
            ++occupied;
        }
    }

    target->occupied.store(occupied, std::memory_order_relaxed);
    target->deleted.store(0, std::memory_order_relaxed);
}

template <typename K, typename V, typename P>
void ParallelHashTable<K, V, P>::maybe_resize(size_t requested_capacity, bool force_rebuild) {
    TableState* snapshot = state_.load(std::memory_order_acquire);
    size_t occupied = snapshot->occupied.load(std::memory_order_relaxed);
    size_t deleted = snapshot->deleted.load(std::memory_order_relaxed);
    size_t target_capacity = snapshot->capacity;
    bool should_rebuild = force_rebuild;

    if (force_rebuild) {
        target_capacity = std::max(
            requested_capacity, openmp_hash_table_detail::min_capacity_for_items(occupied));
    } else {
        if (requested_capacity > snapshot->capacity) {
            target_capacity = requested_capacity;
            should_rebuild = true;
        }
        if (!should_rebuild &&
            openmp_hash_table_detail::should_cleanup(occupied, deleted, snapshot->capacity)) {
            target_capacity = openmp_hash_table_detail::cleanup_capacity(snapshot->capacity, occupied);
            should_rebuild = true;
        }
    }

    if (!should_rebuild) {
        return;
    }

    std::unique_lock<std::mutex> resize_lock(resize_mutex_);

    snapshot = state_.load(std::memory_order_acquire);
    occupied = snapshot->occupied.load(std::memory_order_relaxed);
    deleted = snapshot->deleted.load(std::memory_order_relaxed);
    target_capacity = snapshot->capacity;
    should_rebuild = force_rebuild;

    if (force_rebuild) {
        target_capacity = std::max(
            requested_capacity, openmp_hash_table_detail::min_capacity_for_items(occupied));
    } else {
        if (requested_capacity > snapshot->capacity) {
            target_capacity = requested_capacity;
            should_rebuild = true;
        }
        if (!should_rebuild &&
            openmp_hash_table_detail::should_cleanup(occupied, deleted, snapshot->capacity)) {
            target_capacity = openmp_hash_table_detail::cleanup_capacity(snapshot->capacity, occupied);
            should_rebuild = true;
        }
    }

    if (!should_rebuild) {
        return;
    }

    close_gate_and_wait_for_quiescence();
    TableState* current = state_.load(std::memory_order_acquire);
    occupied = current->occupied.load(std::memory_order_relaxed);
    deleted = current->deleted.load(std::memory_order_relaxed);
    target_capacity = current->capacity;
    should_rebuild = force_rebuild;

    if (force_rebuild) {
        target_capacity = std::max(
            requested_capacity, openmp_hash_table_detail::min_capacity_for_items(occupied));
    } else {
        if (requested_capacity > current->capacity) {
            target_capacity = requested_capacity;
            should_rebuild = true;
        }
        if (!should_rebuild &&
            openmp_hash_table_detail::should_cleanup(occupied, deleted, current->capacity)) {
            target_capacity = openmp_hash_table_detail::cleanup_capacity(current->capacity, occupied);
            should_rebuild = true;
        }
    }

    if (!should_rebuild) {
        open_gate();
        return;
    }

    TableState* replacement = nullptr;
    try {
        replacement = new TableState(target_capacity, backend_);
        rebuild_state(current, replacement);
        state_.store(replacement, std::memory_order_release);
    } catch (...) {
        delete replacement;
        open_gate();
        throw;
    }

    delete current;
    open_gate();
}

template <typename K, typename V, typename P>
void ParallelHashTable<K, V, P>::clear() {
    std::unique_lock<std::mutex> resize_lock(resize_mutex_);
    close_gate_and_wait_for_quiescence();
    reset_state(state_.load(std::memory_order_acquire));
    open_gate();
}

template <typename K, typename V, typename P>
size_t ParallelHashTable<K, V, P>::hash(K key) const {
    OperationGuard guard(*this);
    return openmp_hash_table_detail::hash_key(key, guard.state()->capacity);
}

template <typename K, typename V, typename P>
size_t ParallelHashTable<K, V, P>::size() const {
    OperationGuard guard(*this);
    return guard.state()->occupied.load(std::memory_order_relaxed);
}

template <typename K, typename V, typename P>
size_t ParallelHashTable<K, V, P>::capacity() const {
    OperationGuard guard(*this);
    return guard.state()->capacity;
}

template <typename K, typename V, typename P>
float ParallelHashTable<K, V, P>::load_factor() const {
    OperationGuard guard(*this);
    return static_cast<float>(guard.state()->occupied.load(std::memory_order_relaxed)) /
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
bool ParallelHashTable<K, V, P>::try_claim_slot(TableState* state,
                                                size_t slot_index,
                                                K expected_marker,
                                                K key,
                                                const V& value) {
    auto key_ref = std::atomic_ref<K>(state->table[slot_index].key);
    K expected = expected_marker;
    if (!key_ref.compare_exchange_strong(
            expected,
            openmp_hash_table_detail::busy_key<K>(),
            std::memory_order_acq_rel,
            std::memory_order_acquire)) {
        return false;
    }

    omp_set_lock(&state->locks[slot_index]);
    try {
        state->table[slot_index].value = value;
        std::atomic_thread_fence(std::memory_order_release);
        key_ref.store(key, std::memory_order_release);
    } catch (...) {
        key_ref.store(expected_marker, std::memory_order_release);
        omp_unset_lock(&state->locks[slot_index]);
        throw;
    }
    omp_unset_lock(&state->locks[slot_index]);
    return true;
}

template <typename K, typename V, typename P>
openmp_hash_table_detail::InsertStatus ParallelHashTable<K, V, P>::insert_with_cas(
    TableState* state,
    K key,
    const V& value) {
    const K empty_key = openmp_hash_table_detail::empty_key<K>();
    const K deleted_key = openmp_hash_table_detail::deleted_key<K>();
    const K busy_key = openmp_hash_table_detail::busy_key<K>();
    const size_t start = openmp_hash_table_detail::hash_key(key, state->capacity);
    size_t first_deleted = state->capacity;

    for (int attempt = 0; static_cast<size_t>(attempt) < state->capacity; ++attempt) {
        const size_t slot_index = next_slot(start, attempt, state->capacity);
        auto key_ref = std::atomic_ref<K>(state->table[slot_index].key);
        K current = key_ref.load(std::memory_order_acquire);

        while (current == busy_key) {
            current = key_ref.load(std::memory_order_acquire);
        }

        if (current == key) {
            return openmp_hash_table_detail::InsertStatus::Duplicate;
        }
        if (current == deleted_key) {
            if (first_deleted == state->capacity) {
                first_deleted = slot_index;
            }
            continue;
        }
        if (current == empty_key) {
            if (first_deleted != state->capacity) {
                return try_claim_slot(state, first_deleted, deleted_key, key, value)
                    ? openmp_hash_table_detail::InsertStatus::InsertedDeleted
                    : openmp_hash_table_detail::InsertStatus::Retry;
            }

            return try_claim_slot(state, slot_index, empty_key, key, value)
                ? openmp_hash_table_detail::InsertStatus::InsertedNew
                : openmp_hash_table_detail::InsertStatus::Retry;
        }
    }

    if (first_deleted != state->capacity) {
        return try_claim_slot(state, first_deleted, deleted_key, key, value)
            ? openmp_hash_table_detail::InsertStatus::InsertedDeleted
            : openmp_hash_table_detail::InsertStatus::Retry;
    }

    return openmp_hash_table_detail::InsertStatus::Full;
}

template <typename K, typename V, typename P>
openmp_hash_table_detail::InsertStatus ParallelHashTable<K, V, P>::insert_with_mutex(
    TableState* state,
    K key,
    const V& value) {
    const K empty_key = openmp_hash_table_detail::empty_key<K>();
    const K deleted_key = openmp_hash_table_detail::deleted_key<K>();
    const size_t start = openmp_hash_table_detail::hash_key(key, state->capacity);
    size_t first_deleted = state->capacity;

    for (int attempt = 0; static_cast<size_t>(attempt) < state->capacity; ++attempt) {
        const size_t slot_index = next_slot(start, attempt, state->capacity);
        omp_set_lock(&state->locks[slot_index]);
        const K current = state->table[slot_index].key;

        if (current == key) {
            omp_unset_lock(&state->locks[slot_index]);
            return openmp_hash_table_detail::InsertStatus::Duplicate;
        }
        if (current == deleted_key) {
            if (first_deleted == state->capacity) {
                first_deleted = slot_index;
            }
            omp_unset_lock(&state->locks[slot_index]);
            continue;
        }
        if (current == empty_key) {
            if (first_deleted == state->capacity) {
                state->table[slot_index].key = key;
                state->table[slot_index].value = value;
                omp_unset_lock(&state->locks[slot_index]);
                return openmp_hash_table_detail::InsertStatus::InsertedNew;
            }

            omp_unset_lock(&state->locks[slot_index]);
            omp_set_lock(&state->locks[first_deleted]);
            const K target_key = state->table[first_deleted].key;
            if (target_key == deleted_key) {
                state->table[first_deleted].key = key;
                state->table[first_deleted].value = value;
                omp_unset_lock(&state->locks[first_deleted]);
                return openmp_hash_table_detail::InsertStatus::InsertedDeleted;
            }
            if (target_key == key) {
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
        const K target_key = state->table[first_deleted].key;
        if (target_key == deleted_key) {
            state->table[first_deleted].key = key;
            state->table[first_deleted].value = value;
            omp_unset_lock(&state->locks[first_deleted]);
            return openmp_hash_table_detail::InsertStatus::InsertedDeleted;
        }
        if (target_key == key) {
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

        {
            OperationGuard guard(*this);
            TableState* state = guard.state();

            for (;;) {
                const auto status = insert_into_state(state, key, value);
                if (status == openmp_hash_table_detail::InsertStatus::Retry) {
                    continue;
                }
                if (status == openmp_hash_table_detail::InsertStatus::Duplicate) {
                    return true;
                }
                if (status == openmp_hash_table_detail::InsertStatus::InsertedNew) {
                    const size_t occupied =
                        state->occupied.fetch_add(1, std::memory_order_relaxed) + 1;
                    inserted = true;
                    if (openmp_hash_table_detail::should_grow(occupied, state->capacity)) {
                        resize_target =
                            openmp_hash_table_detail::growth_capacity(state->capacity, occupied);
                    }
                    break;
                }
                if (status == openmp_hash_table_detail::InsertStatus::InsertedDeleted) {
                    state->deleted.fetch_sub(1, std::memory_order_relaxed);
                    const size_t occupied =
                        state->occupied.fetch_add(1, std::memory_order_relaxed) + 1;
                    inserted = true;
                    if (openmp_hash_table_detail::should_grow(occupied, state->capacity)) {
                        resize_target =
                            openmp_hash_table_detail::growth_capacity(state->capacity, occupied);
                    }
                    break;
                }

                resize_target =
                    openmp_hash_table_detail::growth_capacity(state->capacity, state->occupied.load() + 1);
                break;
            }
        }

        if (inserted) {
            if (resize_target != 0) {
                maybe_resize(resize_target, false);
            }
            return true;
        }

        maybe_resize(resize_target, false);
    }
}

template <typename K, typename V, typename P>
bool ParallelHashTable<K, V, P>::get_with_cas(TableState* state, K key, V& out_value) const {
    const K empty_key = openmp_hash_table_detail::empty_key<K>();
    const K busy_key = openmp_hash_table_detail::busy_key<K>();
    const size_t start = openmp_hash_table_detail::hash_key(key, state->capacity);

    for (int attempt = 0; static_cast<size_t>(attempt) < state->capacity; ++attempt) {
        const size_t slot_index = next_slot(start, attempt, state->capacity);
        auto key_ref = std::atomic_ref<K>(state->table[slot_index].key);
        K current = key_ref.load(std::memory_order_acquire);

        while (current == busy_key) {
            current = key_ref.load(std::memory_order_acquire);
        }

        if (current == empty_key) {
            return false;
        }
        if (current == key) {
            omp_set_lock(&state->locks[slot_index]);
            const K confirmed = key_ref.load(std::memory_order_acquire);
            if (confirmed == key) {
                out_value = state->table[slot_index].value;
                omp_unset_lock(&state->locks[slot_index]);
                return true;
            }
            omp_unset_lock(&state->locks[slot_index]);
            if (confirmed == empty_key) {
                return false;
            }
            if (confirmed == busy_key) {
                --attempt;
            }
        }
    }

    return false;
}

template <typename K, typename V, typename P>
bool ParallelHashTable<K, V, P>::get_with_mutex(TableState* state, K key, V& out_value) const {
    const K empty_key = openmp_hash_table_detail::empty_key<K>();
    const size_t start = openmp_hash_table_detail::hash_key(key, state->capacity);

    for (int attempt = 0; static_cast<size_t>(attempt) < state->capacity; ++attempt) {
        const size_t slot_index = next_slot(start, attempt, state->capacity);
        omp_set_lock(&state->locks[slot_index]);
        const K current = state->table[slot_index].key;

        if (current == empty_key) {
            omp_unset_lock(&state->locks[slot_index]);
            return false;
        }
        if (current == key) {
            out_value = state->table[slot_index].value;
            omp_unset_lock(&state->locks[slot_index]);
            return true;
        }

        omp_unset_lock(&state->locks[slot_index]);
    }

    return false;
}

template <typename K, typename V, typename P>
bool ParallelHashTable<K, V, P>::get_from_state(TableState* state, K key, V& out_value) const {
    return backend_ == ParallelBackend::CAS
        ? get_with_cas(state, key, out_value)
        : get_with_mutex(state, key, out_value);
}

template <typename K, typename V, typename P>
bool ParallelHashTable<K, V, P>::get(K key, V& out_value) {
    OperationGuard guard(*this);
    return get_from_state(guard.state(), key, out_value);
}

template <typename K, typename V, typename P>
bool ParallelHashTable<K, V, P>::remove_with_cas(TableState* state, K key) {
    const K empty_key = openmp_hash_table_detail::empty_key<K>();
    const K deleted_key = openmp_hash_table_detail::deleted_key<K>();
    const K busy_key = openmp_hash_table_detail::busy_key<K>();
    const size_t start = openmp_hash_table_detail::hash_key(key, state->capacity);

    for (int attempt = 0; static_cast<size_t>(attempt) < state->capacity; ++attempt) {
        const size_t slot_index = next_slot(start, attempt, state->capacity);
        auto key_ref = std::atomic_ref<K>(state->table[slot_index].key);
        K current = key_ref.load(std::memory_order_acquire);

        while (current == busy_key) {
            current = key_ref.load(std::memory_order_acquire);
        }

        if (current == empty_key) {
            return false;
        }
        if (current == key) {
            K expected = key;
            if (key_ref.compare_exchange_strong(
                    expected,
                    deleted_key,
                    std::memory_order_acq_rel,
                    std::memory_order_acquire)) {
                return true;
            }
            return false;
        }
    }

    return false;
}

template <typename K, typename V, typename P>
bool ParallelHashTable<K, V, P>::remove_with_mutex(TableState* state, K key) {
    const K empty_key = openmp_hash_table_detail::empty_key<K>();
    const size_t start = openmp_hash_table_detail::hash_key(key, state->capacity);

    for (int attempt = 0; static_cast<size_t>(attempt) < state->capacity; ++attempt) {
        const size_t slot_index = next_slot(start, attempt, state->capacity);
        omp_set_lock(&state->locks[slot_index]);
        const K current = state->table[slot_index].key;

        if (current == empty_key) {
            omp_unset_lock(&state->locks[slot_index]);
            return false;
        }
        if (current == key) {
            openmp_hash_table_detail::mark_deleted_slot<Slot, K, V>(state->table[slot_index]);
            omp_unset_lock(&state->locks[slot_index]);
            return true;
        }

        omp_unset_lock(&state->locks[slot_index]);
    }

    return false;
}

template <typename K, typename V, typename P>
bool ParallelHashTable<K, V, P>::remove_from_state(TableState* state, K key) {
    return backend_ == ParallelBackend::CAS
        ? remove_with_cas(state, key)
        : remove_with_mutex(state, key);
}

template <typename K, typename V, typename P>
bool ParallelHashTable<K, V, P>::remove(K key) {
    size_t cleanup_target = 0;
    bool removed = false;

    {
        OperationGuard guard(*this);
        TableState* state = guard.state();
        removed = remove_from_state(state, key);
        if (removed) {
            const size_t occupied =
                state->occupied.fetch_sub(1, std::memory_order_relaxed) - 1;
            const size_t deleted =
                state->deleted.fetch_add(1, std::memory_order_relaxed) + 1;
            if (openmp_hash_table_detail::should_cleanup(occupied, deleted, state->capacity)) {
                cleanup_target =
                    openmp_hash_table_detail::cleanup_capacity(state->capacity, occupied);
            }
        }
    }

    if (cleanup_target != 0) {
        maybe_resize(cleanup_target, false);
    }

    return removed;
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
