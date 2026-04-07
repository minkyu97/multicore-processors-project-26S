/*
 * SeparateChainTable.h
 *
 * Parallel separate-chaining hash table library.
 * Companion to ParallelHashTable_final.h (open addressing).
 *
 * Design decisions:
 *   - Pre-allocated node pool with per-thread sub-pools
 *     (eliminates atomic pool_next bottleneck from C benchmark)
 *   - CAS on bucket head pointer (fastest sync strategy)
 *   - Epoch-based memory reclamation (safe resize)
 *   - Compile-time StatsEnabled / StatsDisabled (zero overhead)
 *   - Auto-resize when avg chain length > threshold (not load factor)
 *   - Works at ANY load factor including >100%
 *   - Matching API to ParallelHashTable_final.h for fair comparison
 *
 * Compile:
 *   g++ -std=c++17 -fopenmp -O2 -o myapp myapp.cpp
 *
 * Usage:
 *   ChainHashTable<int,int> table(1000000, 8);
 *   table.insert(42, 100);
 *   auto v = table.search(42);   // returns std::optional<int>
 *   table.remove(42);
 *   table.bulk_insert(keys, values, n);
 *
 *   // With stats disabled (zero overhead)
 *   ChainHashTable<int,int,StatsDisabled> table(1000000, 8);
 *
 *   // Supports load > 100%  (impossible in open addressing)
 *   table.bulk_insert(keys, values, 2000000); // 200% load — works fine
 */

#pragma once

#include <atomic>
#include <optional>
#include <vector>
#include <mutex>
#include <thread>
#include <functional>
#include <stdexcept>
#include <cstdint>
#include <cstring>
#include <omp.h>

// ══════════════════════════════════════════════════════════════════════════
// STATS POLICY (compile-time zero overhead when disabled)
// Same interface as ParallelHashTable_final.h for consistency
// ══════════════════════════════════════════════════════════════════════════

#ifndef CHAIN_STATS_POLICY_DEFINED
#define CHAIN_STATS_POLICY_DEFINED

struct ChainStatsEnabled  { static constexpr bool enabled = true;  };
struct ChainStatsDisabled { static constexpr bool enabled = false; };

// When used standalone, ChainStatsEnabled/ChainStatsDisabled are the names.
// When used alongside ParallelHashTable_final.h, StatsEnabled/StatsDisabled
// are already defined there and compatible.

#endif

// ══════════════════════════════════════════════════════════════════════════
// CHAIN STATS CONTAINER
// ══════════════════════════════════════════════════════════════════════════

template <typename SP>
struct ChainStats {
    std::atomic<long> inserts       {0};
    std::atomic<long> searches      {0};
    std::atomic<long> removes       {0};
    std::atomic<long> duplicates    {0};
    std::atomic<long> cas_retries   {0};  // CAS head retries
    std::atomic<long> total_hops    {0};  // total chain hops across all ops
    std::atomic<long> max_chain     {0};  // longest chain seen
    std::atomic<int>  resize_count  {0};
    std::atomic<long> nodes_moved   {0};

    void add_insert(int retries, long hops) {
        if constexpr (SP::enabled) {
            inserts    .fetch_add(1,       std::memory_order_relaxed);
            cas_retries.fetch_add(retries, std::memory_order_relaxed);
            total_hops .fetch_add(hops,    std::memory_order_relaxed);
        }
    }

    void add_bulk_insert(long n, long retries, long hops, long max_ch) {
        if constexpr (SP::enabled) {
            inserts    .fetch_add(n,       std::memory_order_relaxed);
            cas_retries.fetch_add(retries, std::memory_order_relaxed);
            total_hops .fetch_add(hops,    std::memory_order_relaxed);
            long cur = max_chain.load(std::memory_order_relaxed);
            while (max_ch > cur)
                if (max_chain.compare_exchange_weak(cur, max_ch,
                    std::memory_order_relaxed)) break;
        }
    }

    void add_search(long hops) {
        if constexpr (SP::enabled) {
            searches  .fetch_add(1,    std::memory_order_relaxed);
            total_hops.fetch_add(hops, std::memory_order_relaxed);
        }
    }

    void add_remove() {
        if constexpr (SP::enabled)
            removes.fetch_add(1, std::memory_order_relaxed);
    }

    void add_duplicate() {
        if constexpr (SP::enabled)
            duplicates.fetch_add(1, std::memory_order_relaxed);
    }

    void add_resize(long moved) {
        if constexpr (SP::enabled) {
            resize_count.fetch_add(1,     std::memory_order_relaxed);
            nodes_moved .fetch_add(moved, std::memory_order_relaxed);
        }
    }

    void reset() {
        inserts.store(0);  searches.store(0); removes.store(0);
        duplicates.store(0); cas_retries.store(0); total_hops.store(0);
        max_chain.store(0); resize_count.store(0); nodes_moved.store(0);
    }

    float avg_hops() const {
        long ops = inserts.load() + searches.load() + removes.load();
        return ops > 0 ? (float)total_hops.load() / (float)ops : 0.0f;
    }

    float cas_retry_rate() const {
        long ins = inserts.load();
        return ins > 0 ? (float)cas_retries.load() / (float)ins : 0.0f;
    }

    void print(const char* label = "") const {
        if constexpr (!SP::enabled) {
            printf("  Stats disabled (ChainStatsDisabled)\n");
            return;
        }
        printf("=== ChainStats [%s] ===\n", label);
        printf("  Inserts      : %ld\n",  inserts.load());
        printf("  Searches     : %ld\n",  searches.load());
        printf("  Removes      : %ld\n",  removes.load());
        printf("  Duplicates   : %ld\n",  duplicates.load());
        printf("  CAS retries  : %ld\n",  cas_retries.load());
        printf("  CAS rate     : %.6f\n", cas_retry_rate());
        printf("  Avg hops     : %.3f\n", avg_hops());
        printf("  Max chain    : %ld\n",  max_chain.load());
        printf("  Resizes      : %d\n",   resize_count.load());
        printf("  Nodes moved  : %ld\n",  nodes_moved.load());
        printf("=====================\n");
    }
};

// ══════════════════════════════════════════════════════════════════════════
// EPOCH-BASED MEMORY RECLAMATION
// Reuse the same design from ParallelHashTable_final.h
// ══════════════════════════════════════════════════════════════════════════

#ifndef CHAIN_EPOCH_DEFINED
#define CHAIN_EPOCH_DEFINED

static constexpr int    CHAIN_EPOCH_MAX = 128;
static constexpr uint64_t CHAIN_UNPINNED = UINT64_MAX;

struct ChainEpochMgr {
    std::atomic<uint64_t> epoch {0};
    std::atomic<uint64_t> ann[CHAIN_EPOCH_MAX];

    ChainEpochMgr() {
        for (int i=0;i<CHAIN_EPOCH_MAX;i++)
            ann[i].store(CHAIN_UNPINNED, std::memory_order_relaxed);
    }

    static int tid() {
        static std::atomic<int> ctr {0};
        thread_local int id = ctr.fetch_add(1) % CHAIN_EPOCH_MAX;
        return id;
    }

    void pin()   { ann[tid()].store(epoch.load(std::memory_order_acquire),
                                    std::memory_order_release); }
    void unpin() { ann[tid()].store(CHAIN_UNPINNED,
                                    std::memory_order_release); }
    uint64_t advance() {
        return epoch.fetch_add(1, std::memory_order_acq_rel) + 1;
    }
    uint64_t min_ann() const {
        uint64_t mn = CHAIN_UNPINNED;
        for (int i=0;i<CHAIN_EPOCH_MAX;i++) {
            uint64_t e = ann[i].load(std::memory_order_acquire);
            if (e < mn) mn = e;
        }
        return mn;
    }
};

static ChainEpochMgr g_chain_epoch;

struct ChainEpochPin {
    ChainEpochPin()  { g_chain_epoch.pin();   }
    ~ChainEpochPin() { g_chain_epoch.unpin(); }
};

template <typename T>
class ChainRetireList {
public:
    void retire(T* p) {
        uint64_t e = g_chain_epoch.advance();
        std::lock_guard<std::mutex> lk(mx_);
        list_.push_back({p, e});
        reclaim();
    }
    void reclaim() {
        uint64_t safe = g_chain_epoch.min_ann();
        for (auto it=list_.begin(); it!=list_.end();) {
            if (it->epoch < safe) { delete it->ptr; it=list_.erase(it); }
            else ++it;
        }
    }
    ~ChainRetireList() { for (auto& e : list_) delete e.ptr; }
private:
    struct Entry { T* ptr; uint64_t epoch; };
    std::mutex mx_;
    std::vector<Entry> list_;
};

#endif

// ══════════════════════════════════════════════════════════════════════════
// NODE (linked list node for chaining)
// ══════════════════════════════════════════════════════════════════════════

template <typename K, typename V>
struct ChainNode {
    K                    key;
    V                    value;
    std::atomic<ChainNode<K,V>*> next {nullptr};

    ChainNode() = default;
    ChainNode(const K& k, const V& v) : key(k), value(v) {}
};

// ══════════════════════════════════════════════════════════════════════════
// PER-THREAD NODE POOL
//
// The C benchmark had one global pool_next atomic shared by all threads.
// Every insert did atomic_fetch_add(&pool_next, 1) — 500,000 times,
// all 64 threads serializing on one cache line.
//
// Fix: each thread gets its own pool segment of CHUNK_SIZE nodes.
// A thread works through its chunk privately (no atomic needed).
// Only when the chunk is exhausted does it claim another chunk
// with one atomic operation. This reduces atomic pool contention
// by a factor of CHUNK_SIZE (default 256).
// ══════════════════════════════════════════════════════════════════════════

template <typename K, typename V>
struct NodePool {
    using Node = ChainNode<K,V>;

    static constexpr int CHUNK_SIZE = 256;  // nodes per per-thread chunk

    Node*                pool;         // flat pre-allocated array
    std::atomic<size_t>  next_chunk;   // next unclaimed chunk index
    size_t               capacity;     // total nodes in pool

    explicit NodePool(size_t cap) : capacity(cap), next_chunk(0) {
        pool = new Node[cap];
    }

    ~NodePool() { delete[] pool; }

    // Thread-local chunk state
    struct ThreadChunk {
        size_t start = 0;
        size_t pos   = 0;
        size_t end   = 0;
        bool   valid = false;
    };

    // Claim one node for this thread (nearly free — local increment)
    Node* alloc(ThreadChunk& chunk) {
        if (!chunk.valid || chunk.pos >= chunk.end) {
            // Claim a new chunk atomically (once per CHUNK_SIZE nodes)
            size_t chunk_idx = next_chunk.fetch_add(
                CHUNK_SIZE, std::memory_order_relaxed);
            if (chunk_idx + CHUNK_SIZE > capacity) return nullptr;
            chunk.start = chunk_idx;
            chunk.pos   = chunk_idx;
            chunk.end   = chunk_idx + CHUNK_SIZE;
            chunk.valid = true;
        }
        Node* n = &pool[chunk.pos++];
        return n;
    }

    void reset() { next_chunk.store(0, std::memory_order_relaxed); }

    NodePool(const NodePool&) = delete;
    NodePool& operator=(const NodePool&) = delete;
};

// ══════════════════════════════════════════════════════════════════════════
// BUCKET ARRAY (the actual table data — swapped during resize)
// ══════════════════════════════════════════════════════════════════════════

template <typename K, typename V>
struct ChainTableData {
    using Node = ChainNode<K,V>;

    size_t                        num_buckets;
    std::atomic<Node*>*           buckets;     // array of head pointers

    explicit ChainTableData(size_t nb) : num_buckets(nb) {
        buckets = new std::atomic<Node*>[nb];
        for (size_t i=0; i<nb; i++)
            buckets[i].store(nullptr, std::memory_order_relaxed);
    }

    ~ChainTableData() { delete[] buckets; }
    ChainTableData(const ChainTableData&) = delete;
};

// ══════════════════════════════════════════════════════════════════════════
// DEFAULT HASHER
// ══════════════════════════════════════════════════════════════════════════

template <typename K>
struct ChainHasher {
    size_t operator()(const K& key, size_t nb) const {
        size_t h = std::hash<K>{}(key);
        // Secondary mix to reduce clustering
        h ^= h >> 17;
        h *= 0x45d9f3bULL;
        h ^= h >> 11;
        return h % nb;
    }
};

// ══════════════════════════════════════════════════════════════════════════
// ChainHashTable
//
// Template parameters:
//   K      : key type (any type with operator== and std::hash)
//   V      : value type (any copyable type)
//   SP     : ChainStatsEnabled or ChainStatsDisabled
//   Hasher : hash function (defaults to ChainHasher<K>)
// ══════════════════════════════════════════════════════════════════════════

template <
    typename K,
    typename V,
    typename SP     = ChainStatsEnabled,
    typename Hasher = ChainHasher<K>
>
class ChainHashTable {
    using Node = ChainNode<K,V>;
    using TD   = ChainTableData<K,V>;
    using Pool = NodePool<K,V>;
    using Chunk= typename Pool::ThreadChunk;

public:

    // ── Constructor ────────────────────────────────────────────────────────
    // num_buckets      : initial bucket count (table grows automatically)
    // threads          : OpenMP worker count for bulk operations
    // chain_threshold  : trigger resize when avg chain len > this (default 2)
    // growth_factor    : new_buckets = old_buckets * growth (default 2.0)
    // pool_multiplier  : node pool size = num_ops * pool_multiplier
    explicit ChainHashTable(
        size_t num_buckets,
        int    threads         = 1,
        float  chain_threshold = 2.0f,
        float  growth_factor   = 2.0f,
        size_t pool_size       = 0)
        : threads_(threads)
        , chain_threshold_(chain_threshold)
        , growth_(growth_factor)
        , size_(0)
    {
        if (num_buckets == 0)
            throw std::invalid_argument("num_buckets must be > 0");

        current_.store(new TD(num_buckets), std::memory_order_relaxed);

        // Default pool: 4x bucket count (supports load factor up to 4.0)
        size_t ps = pool_size > 0 ? pool_size : num_buckets * 4;
        pool_ = new Pool(ps);
    }

    // ── Destructor (RAII) ──────────────────────────────────────────────────
    ~ChainHashTable() {
        TD* td = current_.load(std::memory_order_relaxed);
        // Nodes belong to the pool — do NOT individually delete them.
        // Just delete the bucket array and the pool itself.
        delete td;
        delete pool_;
    }

    ChainHashTable(const ChainHashTable&) = delete;
    ChainHashTable& operator=(const ChainHashTable&) = delete;

    // ── insert ─────────────────────────────────────────────────────────────
    /*
     * CAS on bucket head — lock-free insert.
     *
     * Steps:
     *   1. Compute bucket index
     *   2. Walk chain to check for duplicate
     *   3. Allocate node from per-thread pool chunk
     *   4. CAS the bucket head: head = new_node -> old_head
     *   5. If CAS fails (another thread changed head), retry
     */
    bool insert(const K& key, const V& value) {
        ChainEpochPin pin;
        TD* td = current_.load(std::memory_order_acquire);

        // Check avg chain length and resize if needed
        if (avg_chain_length() > chain_threshold_)
            do_resize(td);

        thread_local Chunk chunk;
        return cas_insert(key, value, td, chunk, true);
    }

    // ── search ─────────────────────────────────────────────────────────────
    std::optional<V> search(const K& key) const {
        ChainEpochPin pin;
        TD* td = current_.load(std::memory_order_acquire);
        return chain_search(key, td);
    }

    bool contains(const K& key) const { return search(key).has_value(); }

    // ── remove ─────────────────────────────────────────────────────────────
    /*
     * Walk chain to find key, then CAS-remove it from the list.
     * Uses head-removal for simplicity — if key is not the head,
     * we lock-remove mid-chain using a predecessor pointer and CAS.
     */
    bool remove(const K& key) {
        ChainEpochPin pin;
        TD* td = current_.load(std::memory_order_acquire);
        size_t b = hasher_(key, td->num_buckets);

        // Try to remove from head first (most common case)
        Node* head = td->buckets[b].load(std::memory_order_acquire);
        while (head && head->key == key) {
            Node* next = head->next.load(std::memory_order_acquire);
            if (td->buckets[b].compare_exchange_strong(
                    head, next, std::memory_order_acq_rel))
            {
                size_.fetch_sub(1, std::memory_order_relaxed);
                stats_.add_remove();
                // Note: node memory returned to pool on next reset
                return true;
            }
            // CAS failed — head changed, re-read
            head = td->buckets[b].load(std::memory_order_acquire);
        }

        // Walk chain for non-head node
        Node* prev = head;
        if (!prev) return false;
        Node* curr = prev->next.load(std::memory_order_acquire);

        while (curr) {
            if (curr->key == key) {
                Node* next = curr->next.load(std::memory_order_acquire);
                // CAS prev->next from curr to curr->next
                if (prev->next.compare_exchange_strong(
                        curr, next, std::memory_order_acq_rel))
                {
                    size_.fetch_sub(1, std::memory_order_relaxed);
                    stats_.add_remove();
                    return true;
                }
                // Retry from current position
                curr = prev->next.load(std::memory_order_acquire);
                continue;
            }
            prev = curr;
            curr = curr->next.load(std::memory_order_acquire);
        }
        return false;
    }

    // ── bulk_insert (parallel OpenMP) ─────────────────────────────────────
    /*
     * Key improvement over C benchmark:
     * Each thread uses its OWN local ThreadChunk to claim nodes from the pool.
     * This reduces atomic pool contention from one-per-insert
     * to one-per-CHUNK_SIZE (default 256x reduction).
     *
     * Pool resized if insufficient capacity detected before parallel region.
     */
    int bulk_insert(const K* keys, const V* values, int n) {
        // Ensure pool has enough capacity
        ensure_pool_capacity((size_t)n);

        // Resize buckets if needed
        {
            TD* td = current_.load(std::memory_order_acquire);
            float projected_avg =
                (float)(size_.load() + n) / (float)td->num_buckets;
            if (projected_avg > chain_threshold_) {
                size_t new_buckets = static_cast<size_t>(
                    (float)(size_.load() + n) / (chain_threshold_ * 0.8f));
                if (new_buckets > td->num_buckets)
                    do_resize(td, new_buckets);
            }
        }

        int  total   = 0;
        long retries = 0;
        long hops    = 0;
        long max_ch  = 0;

        #pragma omp parallel num_threads(threads_) \
                shared(keys, values, retries, hops, max_ch)
        {
            // Each thread has its own chunk — no sharing between threads
            Chunk local_chunk;
            long  lr = 0, lh = 0, lmx = 0;
            int   li = 0;

            #pragma omp for schedule(static)
            for (int i=0; i<n; i++) {
                TD* td = current_.load(std::memory_order_acquire);
                size_t b = hasher_(keys[i], td->num_buckets);

                // Check for duplicate first (walk chain)
                bool dup = false;
                long hops_here = 0;
                Node* cur = td->buckets[b].load(std::memory_order_acquire);
                while (cur) {
                    hops_here++;
                    if (cur->key == keys[i]) { dup = true; break; }
                    cur = cur->next.load(std::memory_order_acquire);
                }

                if (dup) { lh += hops_here; continue; }

                // Allocate from per-thread chunk
                Node* nn = pool_->alloc(local_chunk);
                if (!nn) continue;  // pool exhausted (shouldn't happen)
                nn->key   = keys[i];
                nn->value = values[i];

                // CAS insert at head
                int ret = 0;
                Node* head = td->buckets[b].load(std::memory_order_acquire);
                do {
                    nn->next.store(head, std::memory_order_relaxed);
                    if (td->buckets[b].compare_exchange_strong(
                            head, nn,
                            std::memory_order_acq_rel,
                            std::memory_order_relaxed))
                    {
                        size_.fetch_add(1, std::memory_order_relaxed);
                        li++;
                        break;
                    }
                    ret++;
                } while (true);

                lr  += ret;
                lh  += hops_here + 1;

                // Track max chain length
                long chain_len = hops_here + 1;
                if (chain_len > lmx) lmx = chain_len;
            }

            #pragma omp atomic
            total   += li;
            #pragma omp atomic
            retries += lr;
            #pragma omp atomic
            hops    += lh;
            #pragma omp critical
            { if (lmx > max_ch) max_ch = lmx; }
        }

        stats_.add_bulk_insert((long)total, retries, hops, max_ch);
        return total;
    }

    // ── bulk_search (parallel OpenMP) ─────────────────────────────────────
    void bulk_search(const K* keys, bool* results, int n) const {
        #pragma omp parallel for schedule(static) num_threads(threads_)
        for (int i=0; i<n; i++) results[i] = contains(keys[i]);
    }

    // ── clear ──────────────────────────────────────────────────────────────
    void clear() {
        TD* td = current_.load(std::memory_order_acquire);
        for (size_t i=0; i<td->num_buckets; i++) {
            // Return nodes to pool by resetting (not freeing individually)
            td->buckets[i].store(nullptr, std::memory_order_relaxed);
        }
        pool_->reset();
        size_.store(0, std::memory_order_relaxed);
    }

    // ── Accessors ──────────────────────────────────────────────────────────
    size_t size()        const { return size_.load(std::memory_order_relaxed); }
    size_t num_buckets() const {
        return current_.load(std::memory_order_relaxed)->num_buckets;
    }
    // load_factor() compatible with open addressing API
    float  load_factor() const {
        return (float)size() / (float)num_buckets();
    }
    // Unique to chaining: average chain length (more meaningful than load factor)
    float  avg_chain_length() const {
        return (float)size() / (float)num_buckets();
    }
    // Chaining supports load > 1.0 — alias for clarity
    float  load_factor_pct() const { return load_factor() * 100.0f; }

    int    threads()     const { return threads_; }
    void   set_threads(int t)  { threads_ = t; }

    const ChainStats<SP>& stats()       const { return stats_; }
    void                  reset_stats()       { stats_.reset(); }

    // Compute actual max chain length (O(n) scan)
    long max_chain_length() const {
        TD* td = current_.load(std::memory_order_acquire);
        long mx = 0;
        for (size_t i=0; i<td->num_buckets; i++) {
            long len = 0;
            Node* n = td->buckets[i].load(std::memory_order_relaxed);
            while (n) { len++; n=n->next.load(std::memory_order_relaxed); }
            if (len > mx) mx = len;
        }
        return mx;
    }

    void print_info(const char* label = "") const {
        printf("=== ChainHashTable [%s] ===\n", label);
        printf("  Type         : Separate Chaining\n");
        printf("  Stats        : %s\n", SP::enabled ? "enabled":"disabled");
        printf("  Buckets      : %zu\n", num_buckets());
        printf("  Size         : %zu\n", size());
        printf("  Load factor  : %.2f%% (can exceed 100%%)\n",
               load_factor_pct());
        printf("  Avg chain    : %.3f\n", avg_chain_length());
        printf("  Max chain    : %ld\n",  max_chain_length());
        printf("  Chain thresh : %.1f\n", chain_threshold_);
        printf("  Threads      : %d\n",   threads_);
        printf("  Pool size    : %zu\n",  pool_->capacity);
        if constexpr (SP::enabled) stats_.print();
        printf("===========================\n");
    }

private:

    int                    threads_;
    float                  chain_threshold_;
    float                  growth_;
    std::atomic<size_t>    size_;
    std::atomic<TD*>       current_;
    Pool*                  pool_;
    Hasher                 hasher_;
    mutable ChainStats<SP> stats_;
    ChainRetireList<TD>    retired_;
    std::mutex             resize_mx_;

    // ── CAS insert into a specific table ──────────────────────────────────
    bool cas_insert(const K& key, const V& value, TD* td,
                    Chunk& chunk, bool count_stats)
    {
        size_t b = hasher_(key, td->num_buckets);

        // Walk chain: check duplicate
        long hops = 0;
        Node* cur = td->buckets[b].load(std::memory_order_acquire);
        while (cur) {
            hops++;
            if (cur->key == key) {
                stats_.add_duplicate();
                return false;
            }
            cur = cur->next.load(std::memory_order_acquire);
        }

        // Allocate node
        Node* nn = pool_->alloc(chunk);
        if (!nn) return false;
        nn->key   = key;
        nn->value = value;

        // CAS head
        int retries = 0;
        Node* head = td->buckets[b].load(std::memory_order_acquire);
        do {
            nn->next.store(head, std::memory_order_relaxed);
            if (td->buckets[b].compare_exchange_strong(
                    head, nn,
                    std::memory_order_acq_rel,
                    std::memory_order_relaxed))
            {
                size_.fetch_add(1, std::memory_order_relaxed);
                if (count_stats) stats_.add_insert(retries, hops+1);
                return true;
            }
            retries++;
        } while (true);
    }

    // ── Chain search ───────────────────────────────────────────────────────
    std::optional<V> chain_search(const K& key, TD* td) const {
        size_t b = hasher_(key, td->num_buckets);
        long hops = 0;
        Node* cur = td->buckets[b].load(std::memory_order_acquire);
        while (cur) {
            hops++;
            if (cur->key == key) {
                stats_.add_search(hops);
                return cur->value;
            }
            cur = cur->next.load(std::memory_order_acquire);
        }
        stats_.add_search(hops);
        return std::nullopt;
    }

    // ── Resize ────────────────────────────────────────────────────────────
    void do_resize(TD* old_td, size_t forced_buckets = 0) {
        std::lock_guard<std::mutex> lk(resize_mx_);

        // Double-check after acquiring lock
        TD* current = current_.load(std::memory_order_acquire);
        if (current != old_td) return;  // someone else already resized

        size_t old_nb = old_td->num_buckets;
        size_t new_nb = forced_buckets > 0
            ? forced_buckets
            : static_cast<size_t>((float)old_nb * growth_);

        TD* new_td = new TD(new_nb);

        // Parallel rehash: each thread handles its own slice of buckets
        long moved = 0;

        #pragma omp parallel for schedule(static) num_threads(threads_) \
                reduction(+:moved)
        for (int i=0; i<(int)old_nb; i++) {
            Node* n = old_td->buckets[i].load(std::memory_order_relaxed);
            while (n) {
                Node* nx = n->next.load(std::memory_order_relaxed);
                // Re-insert into new table
                size_t nb = hasher_(n->key, new_nb);
                Node* head = new_td->buckets[nb].load(
                    std::memory_order_relaxed);
                do {
                    n->next.store(head, std::memory_order_relaxed);
                } while (!new_td->buckets[nb].compare_exchange_strong(
                    head, n,
                    std::memory_order_acq_rel,
                    std::memory_order_relaxed));
                moved++;
                n = nx;
            }
        }

        // Atomic pointer swap
        current_.store(new_td, std::memory_order_release);

        // Epoch-safe retirement of old bucket array
        // Note: nodes themselves are reused (moved to new table)
        // We only retire the bucket array, not the nodes
        retired_.retire(old_td);

        stats_.add_resize(moved);
    }

    // ── Ensure pool has enough capacity ───────────────────────────────────
    void ensure_pool_capacity(size_t needed) {
        size_t available = pool_->capacity -
                           pool_->next_chunk.load(std::memory_order_relaxed);
        if (available >= needed) return;

        // Allocate a bigger pool
        std::lock_guard<std::mutex> lk(resize_mx_);
        // Re-check after lock
        available = pool_->capacity -
                    pool_->next_chunk.load(std::memory_order_relaxed);
        if (available >= needed) return;

        size_t new_cap = pool_->capacity + needed * 2;
        Pool* new_pool = new Pool(new_cap);

        // Copy existing allocated nodes to new pool
        // (This is the only non-lock-free path — rare event)
        memcpy(new_pool->pool, pool_->pool,
               pool_->capacity * sizeof(Node));
        new_pool->next_chunk.store(
            pool_->next_chunk.load(), std::memory_order_relaxed);

        delete pool_;
        pool_ = new_pool;
    }
};
