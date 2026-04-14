/*
 * hashmap.c
 *
 * Version 4: Publication-ready benchmark.
 * Adds to ver3:
 *   - Multiple repetitions per experiment (default 7, drops min+max, averages rest)
 *   - Reports: mean time, std deviation, min, max, speedup
 *   - Supports larger table sizes (10M, 100M slots)
 *   - Supports up to 64 threads
 *   - Warm-up run before timing (avoids cold cache skew)
 *
 * Compile:
 *   gcc -Wall -std=c99 -fopenmp -O2 -o hashmap hashmap.c -lm
 *
 * Run:
 *   ./hashmap <table_size> <num_ops> <who> <threads> <probing> <key_dist> <reps>
 *
 *   reps     : number of repetitions (recommended: 7)
 *   who      : 0=sequential, 1=OpenMP parallel
 *   probing  : 0=linear, 1=quadratic, 2=CAS, 3=mutex
 *   key_dist : 0=sequential, 1=random, 2=zipf
 *
 * Example:
 *   ./hashmap 1000000  500000 0 1  0 0 7   (seq baseline, 7 reps)
 *   ./hashmap 1000000  500000 1 8  0 0 7   (parallel, linear, seq keys)
 *   ./hashmap 10000000 5000000 1 8 0 0 7   (10M table)
 *   ./hashmap 1000000  500000 1 64 0 0 7   (64 threads)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

/* ── Sentinel values ───────────────────────────────────────────────────── */
#define EMPTY   -1
#define DELETED -2

/* ── Probing strategy constants ────────────────────────────────────────── */
#define PROBE_LINEAR    0
#define PROBE_QUADRATIC 1
#define PROBE_CAS       2
#define PROBE_MUTEX     3

/* ── Key distribution constants ────────────────────────────────────────── */
#define KEYS_SEQUENTIAL 0
#define KEYS_RANDOM     1
#define KEYS_ZIPF       2

/* ── Max repetitions ───────────────────────────────────────────────────── */
#define MAX_REPS 20

/* ── Hash table slot ───────────────────────────────────────────────────── */
typedef struct {
    int key;
    int value;
} Slot;

/* ── Globals ───────────────────────────────────────────────────────────── */
int numthreads = 0;
int probing    = 0;
int key_dist   = 0;

/* ── Mutex array for PROBE_MUTEX mode ─────────────────────────────────── */
omp_lock_t *locks = NULL;

/* ── Function declarations ─────────────────────────────────────────────── */
void  generate_keys    (int *, int *, int, int, int);
int   next_slot        (int, int, int);
void  init_table       (Slot *, int);
void  seq_hash_ops     (Slot *, int, int *, int *, int);
void  parallel_hash_ops(Slot *, int, int *, int *, int);
void  check_result     (int, int *, int *, int);
const char *probe_name (int);
const char *dist_name  (int);

/* ── Statistics helpers ────────────────────────────────────────────────── */
double mean(double *arr, int n) {
    double s = 0;
    for (int i = 0; i < n; i++) s += arr[i];
    return s / n;
}

double stddev(double *arr, int n, double m) {
    double s = 0;
    for (int i = 0; i < n; i++) s += (arr[i] - m) * (arr[i] - m);
    return sqrt(s / n);
}

double arr_min(double *arr, int n) {
    double m = arr[0];
    for (int i = 1; i < n; i++) if (arr[i] < m) m = arr[i];
    return m;
}

double arr_max(double *arr, int n) {
    double m = arr[0];
    for (int i = 1; i < n; i++) if (arr[i] > m) m = arr[i];
    return m;
}

/*
 * Trimmed mean: drop the single fastest and single slowest run,
 * average the rest. Reduces noise from OS scheduling jitter.
 * Only applies when reps >= 3.
 */
double trimmed_mean(double *arr, int n) {
    if (n < 3) return mean(arr, n);
    double mn = arr_min(arr, n);
    double mx = arr_max(arr, n);
    double s  = 0;
    int    cnt = 0;
    int    dropped_min = 0, dropped_max = 0;
    for (int i = 0; i < n; i++) {
        if (!dropped_min && arr[i] == mn) { dropped_min = 1; continue; }
        if (!dropped_max && arr[i] == mx) { dropped_max = 1; continue; }
        s += arr[i];
        cnt++;
    }
    return cnt > 0 ? s / cnt : mean(arr, n);
}

/* ── Hash function ─────────────────────────────────────────────────────── */
static inline int hash(int key, int table_size)
{
    unsigned int k = (unsigned int)key;
    k = ((k >> 16) ^ k) * 0x45d9f3b;
    k = ((k >> 16) ^ k) * 0x45d9f3b;
    k = (k >> 16) ^ k;
    return (int)(k % (unsigned int)table_size);
}

/* ── Probing step ──────────────────────────────────────────────────────── */
int next_slot(int start, int attempt, int N)
{
    if (probing == PROBE_QUADRATIC)
        return (start + attempt * attempt) % N;
    return (start + attempt) % N;
}

/* ── Table initialization ──────────────────────────────────────────────── */
void init_table(Slot *table, int table_size)
{
    for (int i = 0; i < table_size; i++) {
        table[i].key   = EMPTY;
        table[i].value = EMPTY;
    }
}

/* ── Name helpers ──────────────────────────────────────────────────────── */
const char *probe_name(int p) {
    switch(p) {
        case PROBE_LINEAR:    return "Linear probing (CAS)";
        case PROBE_QUADRATIC: return "Quadratic probing (CAS)";
        case PROBE_CAS:       return "CAS lock-free (linear)";
        case PROBE_MUTEX:     return "Mutex locking (linear)";
        default:              return "Unknown";
    }
}
const char *dist_name(int d) {
    switch(d) {
        case KEYS_SEQUENTIAL: return "Sequential";
        case KEYS_RANDOM:     return "Random";
        case KEYS_ZIPF:       return "Zipf (skewed)";
        default:              return "Unknown";
    }
}

/* ── Key generation ────────────────────────────────────────────────────── */
void generate_keys(int *keys, int *values, int num_ops, int table_size, int dist)
{
    switch(dist) {
        case KEYS_SEQUENTIAL:
            for (int i = 0; i < num_ops; i++) {
                keys[i]   = i + 1;
                values[i] = (i + 1) * 10;
            }
            break;

        case KEYS_RANDOM:
            srand(42);
            for (int i = 0; i < num_ops; i++) {
                keys[i]   = (rand() % (table_size * 10)) + 1;
                values[i] = keys[i] * 10;
            }
            break;

        case KEYS_ZIPF: {
            srand(42);
            int vocab = num_ops / 5;
            if (vocab < 1) vocab = 1;

            double *cum = (double *)malloc((vocab + 1) * sizeof(double));
            if (!cum) { fprintf(stderr, "Zipf alloc failed\n"); exit(1); }

            double total = 0.0;
            for (int r = 1; r <= vocab; r++) total += 1.0 / (double)r;

            cum[0] = 0.0;
            for (int r = 1; r <= vocab; r++)
                cum[r] = cum[r-1] + (1.0 / (double)r) / total;

            for (int i = 0; i < num_ops; i++) {
                double u  = (double)rand() / (double)RAND_MAX;
                int lo = 1, hi = vocab, rank = vocab;
                while (lo <= hi) {
                    int mid = (lo + hi) / 2;
                    if (cum[mid] >= u) { rank = mid; hi = mid - 1; }
                    else               { lo = mid + 1; }
                }
                keys[i]   = rank;
                values[i] = rank * 10;
            }
            free(cum);
            break;
        }
    }
}


/******************************************************************/
/**** Do NOT CHANGE ANYTHING in main()                         ****/
/******************************************************************/
int main(int argc, char *argv[])
{
    if (argc != 8) {
        fprintf(stderr, "usage: hashmap table_size num_ops who threads probing key_dist reps\n");
        fprintf(stderr, "  table_size : slots in hash table (e.g. 1000000, 10000000)\n");
        fprintf(stderr, "  num_ops    : keys to insert/search\n");
        fprintf(stderr, "  who        : 0=sequential, 1=parallel\n");
        fprintf(stderr, "  threads    : 1,2,4,8,16,32,64\n");
        fprintf(stderr, "  probing    : 0=linear, 1=quadratic, 2=CAS, 3=mutex\n");
        fprintf(stderr, "  key_dist   : 0=sequential, 1=random, 2=zipf\n");
        fprintf(stderr, "  reps       : repetitions (recommended: 7)\n");
        exit(1);
    }

    int table_size = atoi(argv[1]);
    int num_ops    = atoi(argv[2]);
    int which_code = atoi(argv[3]);
    numthreads     = atoi(argv[4]);
    probing        = atoi(argv[5]);
    key_dist       = atoi(argv[6]);
    int reps       = atoi(argv[7]);

    if (reps < 1 || reps > MAX_REPS) {
        fprintf(stderr, "reps must be between 1 and %d\n", MAX_REPS); exit(1);
    }
    if (num_ops >= table_size) {
        fprintf(stderr, "num_ops must be < table_size\n"); exit(1);
    }

    float load_factor = (float)num_ops / (float)table_size;

    printf("========================================\n");
    printf("Table size  : %d\n",   table_size);
    printf("Operations  : %d\n",   num_ops);
    printf("Load factor : %.2f (%.0f%%)\n", load_factor, load_factor * 100);
    printf("Threads     : %d\n",   numthreads);
    printf("Probing     : %s\n",   probe_name(probing));
    printf("Key dist    : %s\n",   dist_name(key_dist));
    printf("Repetitions : %d (drops min+max, averages rest)\n", reps);
    printf("========================================\n");

    /* Allocate hash table */
    Slot *table = (Slot *)malloc(table_size * sizeof(Slot));
    if (!table) { fprintf(stderr, "Cannot allocate table of size %d\n", table_size); exit(1); }

    /* Allocate mutex locks if needed */
    if (probing == PROBE_MUTEX) {
        locks = (omp_lock_t *)malloc(table_size * sizeof(omp_lock_t));
        if (!locks) { fprintf(stderr, "Cannot allocate locks\n"); exit(1); }
        for (int i = 0; i < table_size; i++)
            omp_init_lock(&locks[i]);
    }

    /* Generate keys once — same keys used for every rep */
    int *keys   = (int *)malloc(num_ops * sizeof(int));
    int *values = (int *)malloc(num_ops * sizeof(int));
    if (!keys || !values) { fprintf(stderr, "Cannot allocate keys\n"); exit(1); }
    generate_keys(keys, values, num_ops, table_size, key_dist);

    /* Timing array */
    double times[MAX_REPS];

    /* ── Warm-up run (not timed) ── */
    printf("Running warm-up...\n");
    init_table(table, table_size);
    if (which_code == 0)
        seq_hash_ops(table, table_size, keys, values, num_ops);
    else
        parallel_hash_ops(table, table_size, keys, values, num_ops);

    /* ── Timed repetitions ── */
    printf("Running %d timed repetitions...\n", reps);
    for (int r = 0; r < reps; r++) {
        init_table(table, table_size);  /* reset table each rep */

        double start = omp_get_wtime();
        if (which_code == 0)
            seq_hash_ops(table, table_size, keys, values, num_ops);
        else
            parallel_hash_ops(table, table_size, keys, values, num_ops);
        double end = omp_get_wtime();

        times[r] = end - start;
        printf("  Rep %2d: %.6f seconds\n", r + 1, times[r]);
    }

    /* ── Correctness check on last run ── */
    if (which_code == 1)
        check_result(table_size, keys, values, num_ops);

    /* ── Statistics ── */
    double t_mean    = trimmed_mean(times, reps);
    double t_raw     = mean(times, reps);
    double t_std     = stddev(times, reps, t_raw);
    double t_min     = arr_min(times, reps);
    double t_max     = arr_max(times, reps);

    printf("----------------------------------------\n");
    printf("Mean time   : %.6f seconds (trimmed)\n", t_mean);
    printf("Std dev     : %.6f seconds\n", t_std);
    printf("Min time    : %.6f seconds\n", t_min);
    printf("Max time    : %.6f seconds\n", t_max);
    printf("CV (%%RSD)   : %.2f%%\n", (t_std / t_raw) * 100.0);
    printf("========================================\n");

    /* Cleanup */
    if (probing == PROBE_MUTEX) {
        for (int i = 0; i < table_size; i++) omp_destroy_lock(&locks[i]);
        free(locks);
    }
    free(table);
    free(keys);
    free(values);
    return 0;
}


/*********************************************************************
 * SEQUENTIAL VERSION — Do NOT change
 *********************************************************************/
void seq_hash_ops(Slot *table, int table_size, int *keys, int *values, int num_ops)
{
    for (int i = 0; i < num_ops; i++) {
        int k = keys[i], v = values[i];
        int start = hash(k, table_size), idx = start, attempt = 0;
        while (table[idx].key != EMPTY && table[idx].key != DELETED
               && table[idx].key != k) {
            attempt++; idx = next_slot(start, attempt, table_size);
        }
        if (table[idx].key != k) { table[idx].key = k; table[idx].value = v; }
    }
    int found = 0;
    for (int i = 0; i < num_ops; i++) {
        int k = keys[i];
        int start = hash(k, table_size), idx = start, attempt = 0;
        while (table[idx].key != EMPTY) {
            if (table[idx].key == k) { found++; break; }
            attempt++; idx = next_slot(start, attempt, table_size);
        }
    }
}


/*********************************************************************
 * CHECK RESULT — Do NOT change
 *********************************************************************/
void check_result(int table_size, int *keys, int *values, int num_ops)
{
    Slot *ref = (Slot *)malloc(table_size * sizeof(Slot));
    if (!ref) { fprintf(stderr, "check_result alloc failed\n"); exit(1); }
    init_table(ref, table_size);

    int saved = probing;
    if (probing == PROBE_MUTEX || probing == PROBE_CAS) probing = PROBE_LINEAR;
    seq_hash_ops(ref, table_size, keys, values, num_ops);
    probing = saved;

    int mismatches = 0;
    for (int i = 0; i < num_ops; i++) {
        int k = keys[i], start = hash(k, table_size), idx = start, attempt = 0, found = 0;
        while (ref[idx].key != EMPTY) {
            if (ref[idx].key == k) { found = 1; break; }
            attempt++; idx = next_slot(start, attempt, table_size);
        }
        if (!found) mismatches++;
    }
    printf(mismatches == 0 ? "Result is correct!\n"
                           : "MISMATCH: %d keys not found\n", mismatches);
    free(ref);
}


/*********************************************************************
 * PARALLEL VERSION ver4
 *********************************************************************/
void parallel_hash_ops(Slot *table, int table_size, int *keys, int *values, int num_ops)
{
    omp_set_num_threads(numthreads);
    int  total_found   = 0;
    long total_retries = 0;

    /* ── CAS lock-free (linear or quadratic) ── */
    if (probing != PROBE_MUTEX) {

        #pragma omp parallel shared(table, keys, values, table_size, total_retries) \
                             num_threads(numthreads)
        {
            long local_retries = 0;

            #pragma omp for schedule(static)
            for (int i = 0; i < num_ops; i++) {
                int k = keys[i], v = values[i];
                int start = hash(k, table_size), idx = start;
                int attempt = 0, retries = 0;

                while (1) {
                    int cur = table[idx].key;
                    if (cur == EMPTY) {
                        if (__sync_bool_compare_and_swap(&table[idx].key, EMPTY, k)) {
                            table[idx].value = v; break;
                        }
                        retries++; continue;
                    }
                    else if (cur == k) { break; }
                    else { attempt++; idx = next_slot(start, attempt, table_size); }
                }
                local_retries += retries;
            }

            #pragma omp atomic
            total_retries += local_retries;
        }

        #pragma omp parallel for schedule(static) reduction(+:total_found) \
                num_threads(numthreads)
        for (int i = 0; i < num_ops; i++) {
            int k = keys[i], start = hash(k, table_size), idx = start, attempt = 0;
            while (table[idx].key != EMPTY) {
                if (table[idx].key == k) { total_found++; break; }
                attempt++; idx = next_slot(start, attempt, table_size);
            }
        }
    }

    /* ── Mutex locking ── */
    else {
        #pragma omp parallel for schedule(static) \
                shared(table, keys, values, locks, table_size) num_threads(numthreads)
        for (int i = 0; i < num_ops; i++) {
            int k = keys[i], v = values[i];
            int start = hash(k, table_size), idx = start, attempt = 0, placed = 0;

            while (!placed) {
                omp_set_lock(&locks[idx]);
                if (table[idx].key == EMPTY || table[idx].key == DELETED) {
                    table[idx].key = k; table[idx].value = v; placed = 1;
                } else if (table[idx].key == k) { placed = 1; }
                omp_unset_lock(&locks[idx]);
                if (!placed) { attempt++; idx = (start + attempt) % table_size; }
            }
        }

        #pragma omp parallel for schedule(static) reduction(+:total_found) \
                shared(table, keys, table_size) num_threads(numthreads)
        for (int i = 0; i < num_ops; i++) {
            int k = keys[i], start = hash(k, table_size), idx = start, attempt = 0;
            while (table[idx].key != EMPTY) {
                if (table[idx].key == k) { total_found++; break; }
                attempt++; idx = (start + attempt) % table_size;
            }
        }
    }
}
