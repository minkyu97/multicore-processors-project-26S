#pragma once

#include <cstddef>
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

template <typename K, typename V, typename ProbingStrategy = ProbingStrategy::LINEAR>
class ParallelHashTable {
private:
    struct Slot {
        K key;
        V value;
    };

    Slot* table;
    omp_lock_t* locks;
    size_t capacity;
    size_t num_threads;
    ParallelBackend backend;

    size_t next_slot(size_t start, int attempt) const;

public:
    ParallelHashTable(size_t size,
                      size_t num_threads = 0,
                      ParallelBackend backend = ParallelBackend::MUTEX);
    ~ParallelHashTable();
    void clear();
    size_t hash(K key) const;
    size_t size() const;

    bool insert(K key, V value);
    bool get(K key, V& out_value);
    bool remove(K key);

    void insert_batch(const std::vector<K>& keys, const std::vector<V>& values);
    void get_batch(const std::vector<K>& keys,
                   std::vector<V>& out_values,
                   std::vector<bool>& out_found);
    void remove_batch(const std::vector<K>& keys);
};

template <typename K, typename V, typename ProbingStrategy = ProbingStrategy::LINEAR>
class SequentialHashTable {
private:
    struct Slot {
        K key;
        V value;
    };

    Slot* table;
    size_t capacity;

    size_t next_slot(size_t start, int attempt) const;

public:
    explicit SequentialHashTable(size_t size);
    ~SequentialHashTable();
    void clear();
    size_t hash(K key) const;
    size_t size() const;

    bool insert(K key, V value);
    bool get(K key, V& out_value);
    bool remove(K key);
};

#include "openmp_hash_table/detail/hashtable_impl.hpp"
