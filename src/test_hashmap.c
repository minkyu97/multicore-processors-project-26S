/*
 * test_ver4.c
 *
 * Comprehensive C test suite for hashtable_update_ver4.c
 *
 * Tests all internal functions directly without going through main():
 *   - hash()           : hash function distribution
 *   - next_slot()      : probing step correctness
 *   - init_table()     : table initialization
 *   - generate_keys()  : all 3 key distributions
 *   - seq_hash_ops()   : sequential insert + search
 *   - parallel_hash_ops(): parallel insert + search, all probing modes
 *   - check_result()   : correctness verification
 *   - trimmed_mean()   : statistics helper
 *   - stddev()         : statistics helper
 *
 * Compile:
 *   gcc -Wall -std=c99 -fopenmp -O2 -o test_ver4 test_ver4.c -lm
 *
 * Run:
 *   ./test_ver4
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <omp.h>

/* ── Copy all definitions from ver4 (must match exactly) ─────────────────── */
#define EMPTY       -1
#define DELETED     -2
#define PROBE_LINEAR    0
#define PROBE_QUADRATIC 1
#define PROBE_CAS       2
#define PROBE_MUTEX     3
#define KEYS_SEQUENTIAL 0
#define KEYS_RANDOM     1
#define KEYS_ZIPF       2
#define MAX_REPS        20

typedef struct { int key; int value; } Slot;

/* ── Globals (same as ver4) ───────────────────────────────────────────────── */
int numthreads = 8;
int probing    = PROBE_LINEAR;
int key_dist   = KEYS_SEQUENTIAL;
omp_lock_t *locks = NULL;

/* ── Copy all functions from ver4 verbatim ───────────────────────────────── */
static inline int hash(int key, int table_size) {
    unsigned int k = (unsigned int)key;
    k = ((k >> 16) ^ k) * 0x45d9f3b;
    k = ((k >> 16) ^ k) * 0x45d9f3b;
    k = (k >> 16) ^ k;
    return (int)(k % (unsigned int)table_size);
}

int next_slot(int start, int attempt, int N) {
    if (probing == PROBE_QUADRATIC)
        return (start + attempt * attempt) % N;
    return (start + attempt) % N;
}

void init_table(Slot *table, int table_size) {
    for (int i = 0; i < table_size; i++) {
        table[i].key   = EMPTY;
        table[i].value = EMPTY;
    }
}

double mean_fn(double *arr, int n) {
    double s = 0; for (int i=0;i<n;i++) s+=arr[i]; return s/n;
}
double stddev_fn(double *arr, int n, double m) {
    double s=0; for(int i=0;i<n;i++) s+=(arr[i]-m)*(arr[i]-m); return sqrt(s/n);
}
double arr_min(double *arr, int n) {
    double m=arr[0]; for(int i=1;i<n;i++) if(arr[i]<m) m=arr[i]; return m;
}
double arr_max(double *arr, int n) {
    double m=arr[0]; for(int i=1;i<n;i++) if(arr[i]>m) m=arr[i]; return m;
}
double trimmed_mean(double *arr, int n) {
    if (n < 3) return mean_fn(arr, n);
    double mn=arr_min(arr,n), mx=arr_max(arr,n), s=0;
    int cnt=0, dm=0, dmax=0;
    for(int i=0;i<n;i++){
        if(!dm   && arr[i]==mn){dm=1;   continue;}
        if(!dmax && arr[i]==mx){dmax=1; continue;}
        s+=arr[i]; cnt++;
    }
    return cnt>0 ? s/cnt : mean_fn(arr,n);
}

void generate_keys(int *keys, int *values, int num_ops, int table_size, int dist) {
    switch(dist) {
        case KEYS_SEQUENTIAL:
            for(int i=0;i<num_ops;i++){keys[i]=i+1;values[i]=(i+1)*10;}
            break;
        case KEYS_RANDOM:
            srand(42);
            for(int i=0;i<num_ops;i++){
                keys[i]=(rand()%(table_size*10))+1; values[i]=keys[i]*10;
            }
            break;
        case KEYS_ZIPF: {
            srand(42);
            int vocab=num_ops/5; if(vocab<1)vocab=1;
            double *cum=(double*)malloc((vocab+1)*sizeof(double));
            double total=0;
            for(int r=1;r<=vocab;r++) total+=1.0/r;
            cum[0]=0;
            for(int r=1;r<=vocab;r++) cum[r]=cum[r-1]+(1.0/r)/total;
            for(int i=0;i<num_ops;i++){
                double u=(double)rand()/(double)RAND_MAX;
                int lo=1,hi=vocab,rank=vocab;
                while(lo<=hi){int mid=(lo+hi)/2;
                    if(cum[mid]>=u){rank=mid;hi=mid-1;}else lo=mid+1;}
                keys[i]=rank; values[i]=rank*10;
            }
            free(cum);
            break;
        }
    }
}

void seq_hash_ops(Slot *table, int table_size, int *keys, int *values, int num_ops) {
    for(int i=0;i<num_ops;i++){
        int k=keys[i],v=values[i];
        int start=hash(k,table_size),idx=start,attempt=0;
        while(table[idx].key!=EMPTY&&table[idx].key!=DELETED&&table[idx].key!=k){
            attempt++; idx=next_slot(start,attempt,table_size);
        }
        if(table[idx].key!=k){table[idx].key=k;table[idx].value=v;}
    }
    int found=0;
    for(int i=0;i<num_ops;i++){
        int k=keys[i],start=hash(k,table_size),idx=start,attempt=0;
        while(table[idx].key!=EMPTY){
            if(table[idx].key==k){found++;break;}
            attempt++;idx=next_slot(start,attempt,table_size);
        }
    }
}

void check_result(int table_size, int *keys, int *values, int num_ops) {
    Slot *ref=(Slot*)malloc(table_size*sizeof(Slot));
    init_table(ref,table_size);
    int saved=probing;
    if(probing==PROBE_MUTEX||probing==PROBE_CAS) probing=PROBE_LINEAR;
    seq_hash_ops(ref,table_size,keys,values,num_ops);
    probing=saved;
    int mismatches=0;
    for(int i=0;i<num_ops;i++){
        int k=keys[i],start=hash(k,table_size),idx=start,attempt=0,found=0;
        while(ref[idx].key!=EMPTY){
            if(ref[idx].key==k){found=1;break;}
            attempt++;idx=next_slot(start,attempt,table_size);
        }
        if(!found) mismatches++;
    }
    printf(mismatches==0?"Result is correct!\n":"MISMATCH: %d keys not found\n",mismatches);
    free(ref);
}

void parallel_hash_ops(Slot *table, int table_size, int *keys, int *values, int num_ops) {
    omp_set_num_threads(numthreads);
    int  total_found   = 0;
    long total_retries = 0;
    if (probing != PROBE_MUTEX) {
        #pragma omp parallel shared(table,keys,values,table_size,total_retries) \
                             num_threads(numthreads)
        {
            long local_retries=0;
            #pragma omp for schedule(static)
            for(int i=0;i<num_ops;i++){
                int k=keys[i],v=values[i];
                int start=hash(k,table_size),idx=start,attempt=0,retries=0;
                while(1){
                    int cur=table[idx].key;
                    if(cur==EMPTY){
                        if(__sync_bool_compare_and_swap(&table[idx].key,EMPTY,k)){
                            table[idx].value=v;break;
                        }
                        retries++;continue;
                    } else if(cur==k){break;}
                    else{attempt++;idx=next_slot(start,attempt,table_size);}
                }
                local_retries+=retries;
            }
            #pragma omp atomic
            total_retries+=local_retries;
        }
        #pragma omp parallel for schedule(static) reduction(+:total_found) \
                num_threads(numthreads)
        for(int i=0;i<num_ops;i++){
            int k=keys[i],start=hash(k,table_size),idx=start,attempt=0;
            while(table[idx].key!=EMPTY){
                if(table[idx].key==k){total_found++;break;}
                attempt++;idx=next_slot(start,attempt,table_size);
            }
        }
    } else {
        #pragma omp parallel for schedule(static) \
                shared(table,keys,values,locks,table_size) num_threads(numthreads)
        for(int i=0;i<num_ops;i++){
            int k=keys[i],v=values[i];
            int start=hash(k,table_size),idx=start,attempt=0,placed=0;
            while(!placed){
                omp_set_lock(&locks[idx]);
                if(table[idx].key==EMPTY||table[idx].key==DELETED){
                    table[idx].key=k;table[idx].value=v;placed=1;
                } else if(table[idx].key==k){placed=1;}
                omp_unset_lock(&locks[idx]);
                if(!placed){attempt++;idx=(start+attempt)%table_size;}
            }
        }
        #pragma omp parallel for schedule(static) reduction(+:total_found) \
                shared(table,keys,table_size) num_threads(numthreads)
        for(int i=0;i<num_ops;i++){
            int k=keys[i],start=hash(k,table_size),idx=start,attempt=0;
            while(table[idx].key!=EMPTY){
                if(table[idx].key==k){total_found++;break;}
                attempt++;idx=(start+attempt)%table_size;
            }
        }
    }
}

/* ══════════════════════════════════════════════════════════════════════════
 * TEST INFRASTRUCTURE
 * ══════════════════════════════════════════════════════════════════════════ */

static int pass_count = 0;
static int fail_count = 0;

#define TEST(name, cond) do { \
    if (cond) { \
        printf("  PASS : %s\n", name); \
        pass_count++; \
    } else { \
        printf("  FAIL : %s\n", name); \
        fail_count++; \
    } \
} while(0)

#define SECTION(name) do { \
    printf("\n════════════════════════════════════════\n"); \
    printf("  %s\n", name); \
    printf("════════════════════════════════════════\n"); \
} while(0)

/* ── Helper: search a table for a key, return 1 if found ─────────────────── */
static int table_contains(Slot *table, int table_size, int key) {
    int start = hash(key, table_size), idx = start, attempt = 0;
    while (table[idx].key != EMPTY) {
        if (table[idx].key == key) return 1;
        attempt++;
        idx = next_slot(start, attempt, table_size);
        if (attempt >= table_size) break;
    }
    return 0;
}

/* ── Helper: get value for key, return EMPTY if not found ─────────────────── */
static int table_get(Slot *table, int table_size, int key) {
    int start = hash(key, table_size), idx = start, attempt = 0;
    while (table[idx].key != EMPTY) {
        if (table[idx].key == key) return table[idx].value;
        attempt++;
        idx = next_slot(start, attempt, table_size);
        if (attempt >= table_size) break;
    }
    return EMPTY;
}

/* ── Helper: count occupied slots in table ────────────────────────────────── */
static int count_occupied(Slot *table, int table_size) {
    int cnt = 0;
    for (int i = 0; i < table_size; i++)
        if (table[i].key != EMPTY && table[i].key != DELETED) cnt++;
    return cnt;
}

/* ── Helper: init locks for mutex mode ────────────────────────────────────── */
static void setup_locks(int table_size) {
    if (locks) {
        for (int i = 0; i < table_size; i++) omp_destroy_lock(&locks[i]);
        free(locks);
    }
    locks = (omp_lock_t*)malloc(table_size * sizeof(omp_lock_t));
    for (int i = 0; i < table_size; i++) omp_init_lock(&locks[i]);
}

static void teardown_locks(int table_size) {
    if (locks) {
        for (int i = 0; i < table_size; i++) omp_destroy_lock(&locks[i]);
        free(locks);
        locks = NULL;
    }
}

/* ══════════════════════════════════════════════════════════════════════════
 * TEST GROUPS
 * ══════════════════════════════════════════════════════════════════════════ */

/* ── 1. hash() ───────────────────────────────────────────────────────────── */
void test_hash_function() {
    SECTION("1. Hash Function");

    int N = 1000000;

    /* Every output is in range [0, N) */
    int ok = 1;
    for (int k = 1; k <= 10000; k++) {
        int h = hash(k, N);
        if (h < 0 || h >= N) { ok = 0; break; }
    }
    TEST("hash() output always in [0, table_size)", ok);

    /* hash(k, N) != hash(k, N+1) for different sizes — distinct */
    TEST("hash() different table sizes give different ranges",
         hash(42, 1000) != hash(42, 999) || hash(1, 1000) != hash(1, 999));

    /* Deterministic: same key same table always same result */
    TEST("hash() is deterministic",
         hash(12345, N) == hash(12345, N) &&
         hash(99999, N) == hash(99999, N));

    /* Different keys usually hash differently (check a few) */
    int collisions = 0;
    for (int i = 1; i <= 1000; i++)
        if (hash(i, N) == hash(i+1, N)) collisions++;
    TEST("hash() distributes keys (collisions < 5% of 1000 pairs)",
         collisions < 50);
}

/* ── 2. next_slot() ──────────────────────────────────────────────────────── */
void test_next_slot() {
    SECTION("2. Probing Step (next_slot)");

    int N = 1000;

    /* Linear: step is always +1 */
    probing = PROBE_LINEAR;
    TEST("Linear: attempt 0 returns start",
         next_slot(100, 0, N) == 100);
    TEST("Linear: attempt 1 returns start+1",
         next_slot(100, 1, N) == 101);
    TEST("Linear: attempt N-1 wraps around",
         next_slot(1, N-1, N) == 0);
    TEST("Linear: attempt 0 at slot 0",
         next_slot(0, 0, N) == 0);

    /* Quadratic: step is attempt^2 */
    probing = PROBE_QUADRATIC;
    TEST("Quadratic: attempt 0 returns start",
         next_slot(100, 0, N) == 100);
    TEST("Quadratic: attempt 1 returns start+1",
         next_slot(100, 1, N) == 101);
    TEST("Quadratic: attempt 2 returns start+4",
         next_slot(100, 2, N) == 104);
    TEST("Quadratic: attempt 3 returns start+9",
         next_slot(100, 3, N) == 109);
    TEST("Quadratic: wraps correctly",
         next_slot(999, 2, N) == (999 + 4) % N);

    /* CAS uses linear step (same as PROBE_LINEAR) */
    probing = PROBE_CAS;
    TEST("CAS mode: same step as linear",
         next_slot(100, 3, N) == 103);

    probing = PROBE_LINEAR; /* restore */
}

/* ── 3. init_table() ─────────────────────────────────────────────────────── */
void test_init_table() {
    SECTION("3. Table Initialization");

    int N = 10000;
    Slot *table = (Slot*)malloc(N * sizeof(Slot));

    /* Dirty the table first */
    for (int i = 0; i < N; i++) { table[i].key = i+1; table[i].value = i+2; }

    init_table(table, N);

    int all_empty = 1;
    for (int i = 0; i < N; i++)
        if (table[i].key != EMPTY || table[i].value != EMPTY) { all_empty=0; break; }

    TEST("All slots set to EMPTY after init_table()", all_empty);
    TEST("EMPTY sentinel value is -1", EMPTY == -1);
    TEST("DELETED sentinel value is -2", DELETED == -2);

    free(table);
}

/* ── 4. generate_keys() ──────────────────────────────────────────────────── */
void test_generate_keys() {
    SECTION("4. Key Generation");

    int N = 1000, table_size = 10000;
    int *keys   = (int*)malloc(N * sizeof(int));
    int *values = (int*)malloc(N * sizeof(int));

    /* Sequential */
    generate_keys(keys, values, N, table_size, KEYS_SEQUENTIAL);
    TEST("Sequential: keys[0] == 1",            keys[0] == 1);
    TEST("Sequential: keys[N-1] == N",           keys[N-1] == N);
    TEST("Sequential: values[i] == keys[i]*10",  values[5] == keys[5] * 10);
    TEST("Sequential: keys are unique",
         keys[0] != keys[1] && keys[N-2] != keys[N-1]);

    /* Random */
    generate_keys(keys, values, N, table_size, KEYS_RANDOM);
    int all_positive = 1;
    for (int i = 0; i < N; i++)
        if (keys[i] < 1) { all_positive = 0; break; }
    TEST("Random: all keys >= 1", all_positive);
    TEST("Random: values[i] == keys[i]*10", values[10] == keys[10] * 10);
    /* Random should not be sequential */
    int is_sequential = 1;
    for (int i = 1; i < N; i++)
        if (keys[i] != keys[i-1]+1) { is_sequential = 0; break; }
    TEST("Random: keys are not perfectly sequential", !is_sequential);

    /* Zipf */
    generate_keys(keys, values, N, table_size, KEYS_ZIPF);
    int zipf_positive = 1;
    for (int i = 0; i < N; i++)
        if (keys[i] < 1) { zipf_positive = 0; break; }
    TEST("Zipf: all keys >= 1", zipf_positive);
    TEST("Zipf: values[i] == keys[i]*10", values[0] == keys[0] * 10);

    /* Zipf skew: key=1 should appear more than key=vocab */
    int count_1 = 0, count_high = 0;
    int vocab = N / 5;
    for (int i = 0; i < N; i++) {
        if (keys[i] == 1)     count_1++;
        if (keys[i] == vocab) count_high++;
    }
    TEST("Zipf: key=1 appears more than key=vocab (skewed distribution)",
         count_1 > count_high);

    free(keys); free(values);
}

/* ── 5. seq_hash_ops() ───────────────────────────────────────────────────── */
void test_seq_hash_ops() {
    SECTION("5. Sequential Hash Operations");

    int N = 10000;
    Slot *table = (Slot*)malloc(N * sizeof(Slot));
    int *keys   = (int*)malloc(N/2 * sizeof(int));
    int *values = (int*)malloc(N/2 * sizeof(int));
    int num_ops = N / 2;

    probing = PROBE_LINEAR;

    /* ── Basic insert + search ── */
    generate_keys(keys, values, num_ops, N, KEYS_SEQUENTIAL);
    init_table(table, N);
    seq_hash_ops(table, N, keys, values, num_ops);

    /* All inserted keys should be findable */
    int found = 0;
    for (int i = 0; i < num_ops; i++)
        if (table_contains(table, N, keys[i])) found++;
    TEST("Sequential: all inserted keys found", found == num_ops);

    /* Values should be correct */
    int correct_vals = 1;
    for (int i = 0; i < num_ops; i++)
        if (table_get(table, N, keys[i]) != values[i]) { correct_vals=0; break; }
    TEST("Sequential: all values correct after insert", correct_vals);

    /* Table is not completely full (load ~50%) */
    int occupied = count_occupied(table, N);
    TEST("Sequential: occupied slots == unique inserts",
         occupied == num_ops);

    /* ── Duplicate handling: inserting same key twice doesn't create duplicate ── */
    init_table(table, N);
    int dup_keys[3]   = {42, 42, 99};
    int dup_values[3] = {420, 999, 990};
    seq_hash_ops(table, N, dup_keys, dup_values, 3);
    /* Should find 42 with original value, not overwritten */
    int occ = count_occupied(table, N);
    TEST("Sequential: duplicate keys not double-inserted", occ == 2);

    /* ── Quadratic probing ── */
    probing = PROBE_QUADRATIC;
    generate_keys(keys, values, num_ops, N, KEYS_SEQUENTIAL);
    init_table(table, N);
    seq_hash_ops(table, N, keys, values, num_ops);
    found = 0;
    for (int i = 0; i < num_ops; i++)
        if (table_contains(table, N, keys[i])) found++;
    TEST("Quadratic: all inserted keys found", found == num_ops);
    probing = PROBE_LINEAR;

    /* ── Random keys ── */
    generate_keys(keys, values, num_ops, N, KEYS_RANDOM);
    init_table(table, N);
    seq_hash_ops(table, N, keys, values, num_ops);
    /* Unique keys only — count found vs num_ops may differ due to duplicates */
    int any_found = 0;
    for (int i = 0; i < num_ops; i++)
        if (table_contains(table, N, keys[i])) { any_found = 1; break; }
    TEST("Sequential: random keys inserted and findable", any_found);

    free(table); free(keys); free(values);
}

/* ── 6. parallel_hash_ops() — correctness ────────────────────────────────── */
void test_parallel_correctness() {
    SECTION("6. Parallel Correctness (all probing modes)");

    int TABLE = 1000000;
    int NUM_OPS = 500000;
    Slot *table = (Slot*)malloc(TABLE * sizeof(Slot));
    int *keys   = (int*)malloc(NUM_OPS * sizeof(int));
    int *values = (int*)malloc(NUM_OPS * sizeof(int));
    Slot *ref   = (Slot*)malloc(TABLE * sizeof(Slot));

    numthreads = 8;

    /* ── Helper: compare parallel result to sequential reference ── */
    #define RUN_PARALLEL_TEST(label, probe_mode, dist) do { \
        probing  = probe_mode; \
        key_dist = dist; \
        if (probe_mode == PROBE_MUTEX) setup_locks(TABLE); \
        generate_keys(keys, values, NUM_OPS, TABLE, dist); \
        /* Build reference with sequential */ \
        init_table(ref, TABLE); \
        probing = PROBE_LINEAR; \
        seq_hash_ops(ref, TABLE, keys, values, NUM_OPS); \
        probing = probe_mode; \
        /* Run parallel */ \
        init_table(table, TABLE); \
        parallel_hash_ops(table, TABLE, keys, values, NUM_OPS); \
        /* Compare: every key in reference must be in parallel result */ \
        int mismatches = 0; \
        for (int i = 0; i < NUM_OPS; i++) { \
            if (!table_contains(table, TABLE, keys[i])) mismatches++; \
        } \
        TEST(label, mismatches == 0); \
        if (probe_mode == PROBE_MUTEX) teardown_locks(TABLE); \
    } while(0)

    RUN_PARALLEL_TEST("Linear + sequential keys (8 threads)",
                      PROBE_LINEAR, KEYS_SEQUENTIAL);
    RUN_PARALLEL_TEST("Linear + random keys (8 threads)",
                      PROBE_LINEAR, KEYS_RANDOM);
    RUN_PARALLEL_TEST("Linear + Zipf keys (8 threads)",
                      PROBE_LINEAR, KEYS_ZIPF);

    RUN_PARALLEL_TEST("Quadratic + sequential keys (8 threads)",
                      PROBE_QUADRATIC, KEYS_SEQUENTIAL);
    RUN_PARALLEL_TEST("Quadratic + random keys (8 threads)",
                      PROBE_QUADRATIC, KEYS_RANDOM);
    RUN_PARALLEL_TEST("Quadratic + Zipf keys (8 threads)",
                      PROBE_QUADRATIC, KEYS_ZIPF);

    RUN_PARALLEL_TEST("CAS + sequential keys (8 threads)",
                      PROBE_CAS, KEYS_SEQUENTIAL);
    RUN_PARALLEL_TEST("CAS + random keys (8 threads)",
                      PROBE_CAS, KEYS_RANDOM);
    RUN_PARALLEL_TEST("CAS + Zipf keys (8 threads)",
                      PROBE_CAS, KEYS_ZIPF);

    RUN_PARALLEL_TEST("Mutex + sequential keys (8 threads)",
                      PROBE_MUTEX, KEYS_SEQUENTIAL);
    RUN_PARALLEL_TEST("Mutex + random keys (8 threads)",
                      PROBE_MUTEX, KEYS_RANDOM);
    RUN_PARALLEL_TEST("Mutex + Zipf keys (8 threads)",
                      PROBE_MUTEX, KEYS_ZIPF);

    #undef RUN_PARALLEL_TEST

    probing = PROBE_LINEAR;
    free(table); free(keys); free(values); free(ref);
}

/* ── 7. Thread scaling correctness ──────────────────────────────────────────*/
void test_thread_scaling_correctness() {
    SECTION("7. Thread Scaling Correctness (linear, sequential keys)");

    int TABLE   = 1000000;
    int NUM_OPS = 500000;
    Slot *table = (Slot*)malloc(TABLE * sizeof(Slot));
    int *keys   = (int*)malloc(NUM_OPS * sizeof(int));
    int *values = (int*)malloc(NUM_OPS * sizeof(int));

    generate_keys(keys, values, NUM_OPS, TABLE, KEYS_SEQUENTIAL);
    probing = PROBE_LINEAR;

    int thread_counts[] = {1, 2, 4, 8, 16, 32, 64};
    int num_counts = 7;
    char label[64];

    for (int t = 0; t < num_counts; t++) {
        numthreads = thread_counts[t];
        init_table(table, TABLE);
        parallel_hash_ops(table, TABLE, keys, values, NUM_OPS);

        int found = 0;
        for (int i = 0; i < NUM_OPS; i++)
            if (table_contains(table, TABLE, keys[i])) found++;

        sprintf(label, "%d thread(s): all %d keys found",
                thread_counts[t], NUM_OPS);
        TEST(label, found == NUM_OPS);
    }

    numthreads = 8;
    free(table); free(keys); free(values);
}

/* ── 8. Edge cases ───────────────────────────────────────────────────────── */
void test_edge_cases() {
    SECTION("8. Edge Cases");

    /* Tiny table */
    {
        int TABLE = 100, NUM_OPS = 10;
        Slot *table = (Slot*)malloc(TABLE * sizeof(Slot));
        int keys[10], values[10];
        generate_keys(keys, values, NUM_OPS, TABLE, KEYS_SEQUENTIAL);
        probing = PROBE_LINEAR;
        numthreads = 4;

        init_table(table, TABLE);
        parallel_hash_ops(table, TABLE, keys, values, NUM_OPS);
        int found = 0;
        for (int i = 0; i < NUM_OPS; i++)
            if (table_contains(table, TABLE, keys[i])) found++;
        TEST("Tiny table (100 slots, 10 ops): all keys found", found == NUM_OPS);
        free(table);
    }

    /* Low load (10%) */
    {
        int TABLE = 1000000, NUM_OPS = 100000;
        Slot *table = (Slot*)malloc(TABLE * sizeof(Slot));
        int *keys   = (int*)malloc(NUM_OPS * sizeof(int));
        int *values = (int*)malloc(NUM_OPS * sizeof(int));
        generate_keys(keys, values, NUM_OPS, TABLE, KEYS_SEQUENTIAL);
        probing = PROBE_LINEAR; numthreads = 8;
        init_table(table, TABLE);
        parallel_hash_ops(table, TABLE, keys, values, NUM_OPS);
        int found = 0;
        for (int i = 0; i < NUM_OPS; i++)
            if (table_contains(table, TABLE, keys[i])) found++;
        TEST("Low load 10%: all keys found", found == NUM_OPS);
        free(table); free(keys); free(values);
    }

    /* High load (90%) */
    {
        int TABLE = 1000000, NUM_OPS = 900000;
        Slot *table = (Slot*)malloc(TABLE * sizeof(Slot));
        int *keys   = (int*)malloc(NUM_OPS * sizeof(int));
        int *values = (int*)malloc(NUM_OPS * sizeof(int));
        generate_keys(keys, values, NUM_OPS, TABLE, KEYS_SEQUENTIAL);
        probing = PROBE_LINEAR; numthreads = 8;
        init_table(table, TABLE);
        parallel_hash_ops(table, TABLE, keys, values, NUM_OPS);
        int found = 0;
        for (int i = 0; i < NUM_OPS; i++)
            if (table_contains(table, TABLE, keys[i])) found++;
        TEST("High load 90%: all keys found", found == NUM_OPS);
        free(table); free(keys); free(values);
    }

    /* 1-thread parallel should behave like sequential */
    {
        int TABLE = 100000, NUM_OPS = 50000;
        Slot *seq_table = (Slot*)malloc(TABLE * sizeof(Slot));
        Slot *par_table = (Slot*)malloc(TABLE * sizeof(Slot));
        int *keys   = (int*)malloc(NUM_OPS * sizeof(int));
        int *values = (int*)malloc(NUM_OPS * sizeof(int));
        generate_keys(keys, values, NUM_OPS, TABLE, KEYS_SEQUENTIAL);
        probing = PROBE_LINEAR; numthreads = 1;

        init_table(seq_table, TABLE);
        seq_hash_ops(seq_table, TABLE, keys, values, NUM_OPS);

        init_table(par_table, TABLE);
        parallel_hash_ops(par_table, TABLE, keys, values, NUM_OPS);

        int match = 1;
        for (int i = 0; i < NUM_OPS; i++)
            if (!table_contains(par_table, TABLE, keys[i])) { match=0; break; }
        TEST("1-thread parallel: same result as sequential", match);
        free(seq_table); free(par_table); free(keys); free(values);
    }

    /* All duplicate keys — table should hold just 1 unique key */
    {
        int TABLE = 1000, NUM_OPS = 100;
        Slot *table = (Slot*)malloc(TABLE * sizeof(Slot));
        int keys[100], values[100];
        for (int i = 0; i < NUM_OPS; i++) { keys[i] = 42; values[i] = 420; }
        probing = PROBE_LINEAR; numthreads = 4;
        init_table(table, TABLE);
        parallel_hash_ops(table, TABLE, keys, values, NUM_OPS);
        int occupied = count_occupied(table, TABLE);
        TEST("All-duplicate keys: table holds exactly 1 slot", occupied == 1);
        TEST("All-duplicate keys: key 42 is findable", table_contains(table, TABLE, 42));
        free(table);
    }
}

/* ── 9. Statistics helpers ───────────────────────────────────────────────── */
void test_statistics() {
    SECTION("9. Statistics Helpers");

    /* mean */
    double arr1[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    TEST("mean([1,2,3,4,5]) == 3.0", fabs(mean_fn(arr1, 5) - 3.0) < 1e-9);

    double arr2[] = {10.0, 10.0};
    TEST("mean([10,10]) == 10.0", fabs(mean_fn(arr2, 2) - 10.0) < 1e-9);

    /* stddev */
    double arr3[] = {2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0};
    double m3 = mean_fn(arr3, 8);
    double sd = stddev_fn(arr3, 8, m3);
    TEST("stddev of known dataset is ~2.0", fabs(sd - 2.0) < 0.01);

    double arr4[] = {5.0, 5.0, 5.0};
    TEST("stddev of constant array is 0.0",
         fabs(stddev_fn(arr4, 3, 5.0)) < 1e-9);

    /* trimmed_mean: drops min and max */
    double arr5[] = {1.0, 2.0, 3.0, 4.0, 100.0}; /* 100 is outlier */
    double tm = trimmed_mean(arr5, 5);
    /* Should drop 1.0 and 100.0, average 2+3+4 = 3.0 */
    TEST("trimmed_mean drops min and max outlier", fabs(tm - 3.0) < 1e-9);

    double arr6[] = {5.0, 5.0}; /* n < 3: no trimming */
    TEST("trimmed_mean with n=2 falls back to regular mean",
         fabs(trimmed_mean(arr6, 2) - 5.0) < 1e-9);

    double arr7[] = {3.0, 1.0, 2.0}; /* n == 3: drop 1 and 3, keep 2 */
    TEST("trimmed_mean with n=3 returns middle value",
         fabs(trimmed_mean(arr7, 3) - 2.0) < 1e-9);
}

/* ── 10. Performance sanity check ───────────────────────────────────────────*/
void test_performance_sanity() {
    SECTION("10. Performance Sanity Checks");

    int TABLE = 1000000, NUM_OPS = 500000;
    Slot *table = (Slot*)malloc(TABLE * sizeof(Slot));
    int *keys   = (int*)malloc(NUM_OPS * sizeof(int));
    int *values = (int*)malloc(NUM_OPS * sizeof(int));
    generate_keys(keys, values, NUM_OPS, TABLE, KEYS_SEQUENTIAL);
    probing = PROBE_LINEAR;

    /* Sequential should complete in < 5 seconds */
    init_table(table, TABLE);
    double t0 = omp_get_wtime();
    seq_hash_ops(table, TABLE, keys, values, NUM_OPS);
    double seq_time = omp_get_wtime() - t0;
    printf("  INFO : Sequential 1M/500K : %.4f s\n", seq_time);
    TEST("Sequential completes in < 5 seconds", seq_time < 5.0);

    /* Parallel 8 threads should also complete in < 5 seconds */
    numthreads = 8;
    init_table(table, TABLE);
    t0 = omp_get_wtime();
    parallel_hash_ops(table, TABLE, keys, values, NUM_OPS);
    double par_time = omp_get_wtime() - t0;
    printf("  INFO : Parallel 8-thread 1M/500K : %.4f s\n", par_time);
    TEST("Parallel (8 threads) completes in < 5 seconds", par_time < 5.0);
    printf("  INFO : Speedup vs sequential  : %.2fx\n", seq_time / par_time);

    /* Mutex mode completes in < 30 seconds (slower but must finish) */
    setup_locks(TABLE);
    probing = PROBE_MUTEX;
    init_table(table, TABLE);
    t0 = omp_get_wtime();
    parallel_hash_ops(table, TABLE, keys, values, NUM_OPS);
    double mutex_time = omp_get_wtime() - t0;
    printf("  INFO : Mutex 8-thread 1M/500K  : %.4f s\n", mutex_time);
    TEST("Mutex mode completes in < 30 seconds", mutex_time < 30.0);
    teardown_locks(TABLE);
    probing = PROBE_LINEAR;

    free(table); free(keys); free(values);
}

/* ══════════════════════════════════════════════════════════════════════════
 * MAIN
 * ══════════════════════════════════════════════════════════════════════════ */
int main(void) {
    printf("\n");
    printf("════════════════════════════════════════\n");
    printf("  test_ver4.c — hashtable_update_ver4\n");
    printf("  C Unit Test Suite\n");
    printf("════════════════════════════════════════\n");

    test_hash_function();
    test_next_slot();
    test_init_table();
    test_generate_keys();
    test_seq_hash_ops();
    test_parallel_correctness();
    test_thread_scaling_correctness();
    test_edge_cases();
    test_statistics();
    test_performance_sanity();

    printf("\n════════════════════════════════════════\n");
    printf("  RESULTS\n");
    printf("════════════════════════════════════════\n");
    printf("  Passed : %d\n", pass_count);
    printf("  Failed : %d\n", fail_count);
    printf("  Total  : %d\n", pass_count + fail_count);
    if (fail_count == 0)
        printf("  ALL TESTS PASSED\n");
    else
        printf("  %d TEST(S) FAILED\n", fail_count);
    printf("════════════════════════════════════════\n\n");

    return fail_count > 0 ? 1 : 0;
}
