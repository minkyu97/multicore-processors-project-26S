#include <omp.h>
#include "hashtable.hpp"

#include <atomic>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace ProbingStrategy {
    size_t LINEAR::next_slot(size_t start, int attempt, size_t capacity) {
        return (start + attempt) % capacity;
    }

    size_t QUADRATIC::next_slot(size_t start, int attempt, size_t capacity) {
        return (start + attempt * attempt) % capacity;
    }
}

template <typename K, typename V, typename P>
size_t ParallelHashTable<K, V, P>::hash(K key) const {
    if constexpr (std::is_integral_v<K>) {
        std::uint32_t k = static_cast<std::uint32_t>(key);
        k = ((k >> 16) ^ k) * 0x45d9f3bu;
        k = ((k >> 16) ^ k) * 0x45d9f3bu;
        k = (k >> 16) ^ k;
        return static_cast<size_t>(k) % capacity;
    }

    return std::hash<K>{}(key) % capacity;
}

template <typename K, typename V, typename P>
size_t ParallelHashTable<K, V, P>::next_slot(size_t start, int attempt) const {
    return P::next_slot(start, attempt, capacity);
}

template <typename K, typename V, typename P>
ParallelHashTable<K, V, P>::ParallelHashTable(size_t size,
                                              size_t num_threads,
                                              ParallelBackend backend) {
    if (size == 0) {
        throw std::invalid_argument("hash table size must be greater than zero");
    }

    if (backend == ParallelBackend::CAS) {
        if constexpr (!std::is_trivially_copyable_v<K>) {
            throw std::invalid_argument("CAS backend requires trivially copyable keys");
        }
    }

    this->capacity = size;
    this->num_threads = num_threads == 0 ? omp_get_max_threads() : num_threads;
    this->backend = backend;

    table = new Slot[capacity];
    locks = backend == ParallelBackend::MUTEX ? new omp_lock_t[capacity] : nullptr;
    for (size_t i = 0; i < capacity; ++i) {
        table[i].key = EMPTY_KEY;
        if constexpr (std::is_constructible_v<V, int>) {
            table[i].value = static_cast<V>(EMPTY_KEY);
        } else {
            table[i].value = V{};
        }
        if (locks != nullptr) {
            omp_init_lock(&locks[i]);
        }
    }
}

template <typename K, typename V, typename P>
ParallelHashTable<K, V, P>::~ParallelHashTable() {
    if (locks != nullptr) {
        for (size_t i = 0; i < capacity; ++i) {
            omp_destroy_lock(&locks[i]);
        }
        delete[] locks;
    }
    delete[] table;
}

template <typename K, typename V, typename P>
bool ParallelHashTable<K, V, P>::insert(K key, V value) {
    const K empty_key = static_cast<K>(EMPTY_KEY);
    const K deleted_key = static_cast<K>(DELETED_KEY);
    size_t index = hash(key);
    int attempt = 0;

    while (true) {
        size_t slot_index = next_slot(index, attempt);

        if (backend == ParallelBackend::CAS) {
            auto key_ref = std::atomic_ref<K>(table[slot_index].key);
            K current = key_ref.load(std::memory_order_acquire);

            if (current == empty_key || current == deleted_key) {
                K expected = current;
                if (key_ref.compare_exchange_strong(
                        expected, key, std::memory_order_acq_rel, std::memory_order_acquire)) {
                    table[slot_index].value = value;
                    return true;
                }
                continue;
            }
            if (current == key) {
                return true;
            }
        } else {
            omp_set_lock(&locks[slot_index]);

            if (table[slot_index].key == empty_key || table[slot_index].key == deleted_key) {
                table[slot_index].key = key;
                table[slot_index].value = value;
                omp_unset_lock(&locks[slot_index]);
                return true;
            }
            if (table[slot_index].key == key) {
                omp_unset_lock(&locks[slot_index]);
                return true;
            }

            omp_unset_lock(&locks[slot_index]);
        }

        attempt++;
        if (static_cast<size_t>(attempt) >= capacity) {
            return false;
        }
    }
}

template <typename K, typename V, typename P>
bool ParallelHashTable<K, V, P>::get(K key, V& out_value) {
    const K empty_key = static_cast<K>(EMPTY_KEY);
    size_t index = hash(key);
    int attempt = 0;

    while (true) {
        size_t slot_index = next_slot(index, attempt);

        if (backend == ParallelBackend::CAS) {
            auto key_ref = std::atomic_ref<K>(table[slot_index].key);
            const K current = key_ref.load(std::memory_order_acquire);
            if (current == empty_key) {
                return false;
            }
            if (current == key) {
                out_value = table[slot_index].value;
                return true;
            }
        } else {
            omp_set_lock(&locks[slot_index]);

            if (table[slot_index].key == empty_key) {
                omp_unset_lock(&locks[slot_index]);
                return false;
            }
            if (table[slot_index].key == key) {
                out_value = table[slot_index].value;
                omp_unset_lock(&locks[slot_index]);
                return true;
            }

            omp_unset_lock(&locks[slot_index]);
        }

        attempt++;
        if (static_cast<size_t>(attempt) >= capacity) {
            return false;
        }
    }
}

template <typename K, typename V, typename P>
bool ParallelHashTable<K, V, P>::remove(K key) {
    const K empty_key = static_cast<K>(EMPTY_KEY);
    const K deleted_key = static_cast<K>(DELETED_KEY);
    size_t index = hash(key);
    int attempt = 0;

    while (true) {
        size_t slot_index = next_slot(index, attempt);

        if (backend == ParallelBackend::CAS) {
            auto key_ref = std::atomic_ref<K>(table[slot_index].key);
            K current = key_ref.load(std::memory_order_acquire);
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
                    if constexpr (std::is_constructible_v<V, int>) {
                        table[slot_index].value = static_cast<V>(DELETED_KEY);
                    } else {
                        table[slot_index].value = V{};
                    }
                    return true;
                }
                continue;
            }
        } else {
            omp_set_lock(&locks[slot_index]);

            if (table[slot_index].key == empty_key) {
                omp_unset_lock(&locks[slot_index]);
                return false;
            }
            if (table[slot_index].key == key) {
                table[slot_index].key = deleted_key;
                if constexpr (std::is_constructible_v<V, int>) {
                    table[slot_index].value = static_cast<V>(DELETED_KEY);
                } else {
                    table[slot_index].value = V{};
                }
                omp_unset_lock(&locks[slot_index]);
                return true;
            }

            omp_unset_lock(&locks[slot_index]);
        }

        attempt++;
        if (static_cast<size_t>(attempt) >= capacity) {
            return false;
        }
    }
}


template <typename K, typename V, typename P>
void ParallelHashTable<K, V, P>::insert_batch(const std::vector<K>& keys, const std::vector<V>& values) {
    if (keys.size() != values.size()) {
        throw std::invalid_argument("keys and values must have the same length");
    }

    #pragma omp parallel for schedule(static) num_threads(num_threads)
    for (size_t i = 0; i < keys.size(); ++i) {
        insert(keys[i], values[i]);
    }
}

template <typename K, typename V, typename P>
void ParallelHashTable<K, V, P>::get_batch(const std::vector<K>& keys, std::vector<V>& out_values, std::vector<bool>& out_found) {
    out_values.resize(keys.size());
    std::vector<unsigned char> found_bits(keys.size(), 0);

    #pragma omp parallel for schedule(static) num_threads(num_threads)
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
    #pragma omp parallel for schedule(static) num_threads(num_threads)
    for (size_t i = 0; i < keys.size(); ++i) {
        remove(keys[i]);
    }
}

template class ParallelHashTable<int, int, ProbingStrategy::LINEAR>;
template class ParallelHashTable<int, int, ProbingStrategy::QUADRATIC>;
