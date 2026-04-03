#pragma once
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

    // Internal Hashing Tool
    size_t hash(K key) const;
    
    // Internal Probing Tool (Linear or Quadratic)
    size_t next_slot(size_t start, int attempt) const;

public:
    // 1. Constructor & Destructor
    ParallelHashTable(size_t size,
                      size_t num_threads = 0,
                      ParallelBackend backend = ParallelBackend::MUTEX);
    ~ParallelHashTable();

    // 2. Thread-Safe Single Operations
    // (If 10 threads call this simultaneously, the CAS/Mutex inside protects it)
    bool insert(K key, V value);
    bool get(K key, V& out_value);
    bool remove(K key);

    // 3. OpenMP High-Performance "Batch" Operations
    // (The user hands you huge arrays, you use `#pragma omp parallel for` inside)
    void insert_batch(const std::vector<K>& keys, const std::vector<V>& values);
    void get_batch(const std::vector<K>& keys, std::vector<V>& out_values, std::vector<bool>& out_found);
    void remove_batch(const std::vector<K>& keys);
};
